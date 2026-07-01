#include "nllb_600m.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ncnn_llm_base.h"
#include "utils/tokenizer/bpe_tokenizer.h"

namespace {

constexpr int kNumDecoderLayers = 24;

}

class nllb_600m::Impl : public ncnn_llm_base {
public:
    Impl(std::string embed_param,
         std::string embed_bin,
         std::string encoder_param,
         std::string encoder_bin,
         std::string decoder_param,
         std::string decoder_bin,
         std::string vocab_file,
         std::string merges_file,
         bool use_vulkan)
        : ncnn_llm_base(use_vulkan, 4)
        , embed_param_(std::move(embed_param))
        , embed_bin_(std::move(embed_bin))
        , encoder_param_(std::move(encoder_param))
        , encoder_bin_(std::move(encoder_bin))
        , decoder_param_(std::move(decoder_param))
        , decoder_bin_(std::move(decoder_bin))
        , vocab_file_(std::move(vocab_file))
        , merges_file_(std::move(merges_file))
        , bpe_(BpeTokenizer::LoadFromFiles(
              vocab_file_,
              merges_file_,
              SpecialTokensConfig{
                  .bos_token = "</s>",
                  .eos_token = "</s>",
                  .unk_token = "<unk>",
                  .mask_token = "<mask>",
              }))
    {
        ncnn::Option opt = create_option();

        embed_net_.opt = opt;
        encoder_net_.opt = opt;
        decoder_net_.opt = opt;

        if (!load_net(embed_net_, embed_param_, embed_bin_)) {
            std::cerr << "Failed to load embedding model\n";
        }

        if (!load_net(encoder_net_, encoder_param_, encoder_bin_)) {
            std::cerr << "Failed to load encoder model\n";
        }

        if (!load_net(decoder_net_, decoder_param_, decoder_bin_)) {
            std::cerr << "Failed to load decoder model\n";
        }

        if (ok_) {
            const auto& t2i = bpe_.token_to_id();
            auto it = t2i.find("</s>");
            if (it == t2i.end()) {
                std::cerr << "Tokenizer missing </s> id\n";
                ok_ = false;
            } else {
                bos_eos_id_ = it->second;
            }
        }
    }

    std::string translate_sync(const std::string& input_text,
                               const std::string& source_lang,
                               const std::string& target_lang,
                               const NllbConfig& config) {
        std::string out;
        translate_stream(input_text, source_lang, target_lang, config,
                         [&](const std::string& delta) { out += delta; });
        return out;
    }

    bool translate_stream(const std::string& input_text,
                          const std::string& source_lang,
                          const std::string& target_lang,
                          const NllbConfig& config,
                          std::function<void(const std::string&)> callback) {
        if (!ok_) return false;

        const auto& t2i = bpe_.token_to_id();

        auto it_src = t2i.find(source_lang);
        auto it_tgt = t2i.find(target_lang);
        if (it_src == t2i.end() || it_tgt == t2i.end()) {
            std::cerr << "Unknown language tokens: src=" << source_lang
                      << " tgt=" << target_lang << "\n";
            return false;
        }

        int src_lang_id = it_src->second;
        int tgt_lang_id = it_tgt->second;

        std::vector<int> input_ids = bpe_.encode(input_text, false, true);
        input_ids.insert(input_ids.begin(), src_lang_id);

        ncnn::Mat embed_input = embedding_forward(input_ids, -1);
        ncnn::Mat encoder_output = encoder_forward(embed_input);

        std::vector<int> bos = {bos_eos_id_};
        ncnn::Mat bos_embed = embedding_forward(bos, -1);
        KVCache kv_cache = decoder_prefill(bos_embed, encoder_output);

        int last_index = tgt_lang_id;
        std::vector<int> output;
        std::string last_decoded;

        SampleConfig sample_cfg;
        sample_cfg.temperature = config.temperature;
        sample_cfg.top_k = config.top_k;
        sample_cfg.top_p = config.top_p;
        sample_cfg.do_sample = config.do_sample;

        for (int pos = 2; pos < config.max_steps; ++pos) {
            std::vector<int> step_ids = {last_index};
            ncnn::Mat step_embed = embedding_forward(step_ids, pos);

            auto [logits, new_cache] = decoder_decode(step_embed, encoder_output, kv_cache);
            kv_cache = std::move(new_cache);

            last_index = sample_logits(logits, sample_cfg);
            output.push_back(last_index);

            if (last_index == bos_eos_id_) {
                break;
            }

            std::string current = bpe_.decode(output, true);
            if (current.size() >= last_decoded.size()) {
                std::string delta = current.substr(last_decoded.size());
                if (!delta.empty() && callback) callback(delta);
                last_decoded.swap(current);
            } else {
                if (callback) callback(current);
                last_decoded = std::move(current);
            }
        }

        return true;
    }

private:
    ncnn::Mat embedding_forward(const std::vector<int>& input_ids, int pos) {
        ncnn::Extractor ex = embed_net_.create_extractor();

        ncnn::Mat in_mat = mat_from_int_vector(input_ids);
        ex.input("in0", in_mat);

        ncnn::Mat out0;
        ex.extract("out0", out0);

        if (pos == -1) {
            ncnn::Mat pos_emb = sinusoidal_positional_embedding(out0.h, out0.w);
            ncnn::Mat result = out0.clone();
            add_mats_inplace(result, pos_emb);
            return result;
        } else {
            ncnn::Mat pos_emb = sinusoidal_positional_embedding_for_pos(pos, out0.w);
            ncnn::Mat result = out0.clone();
            add_mats_inplace(result, pos_emb);
            return result;
        }
    }

    ncnn::Mat encoder_forward(const ncnn::Mat& hidden) {
        ncnn::Extractor ex = encoder_net_.create_extractor();
        ex.input("in0", hidden);
        ncnn::Mat out0;
        ex.extract("out0", out0);
        return out0;
    }

    KVCache decoder_prefill(const ncnn::Mat& hidden, const ncnn::Mat& encoder_out) {
        KVCache kv_cache;
        kv_cache.reserve(kNumDecoderLayers);

        ncnn::Extractor ex = decoder_net_.create_extractor();
        ex.input("in0", hidden);
        ex.input("in1", encoder_out);

        const int seq_len = hidden.h;
        ncnn::Mat attention_mask(seq_len, seq_len);
        attention_mask.fill(0.0f);
        for (int i = 0; i < seq_len; ++i) {
            float* row = attention_mask.row(i);
            for (int j = i + 1; j < seq_len; ++j) {
                row[j] = -std::numeric_limits<float>::infinity();
            }
        }
        ex.input("in2", attention_mask);

        for (int i = 0; i < kNumDecoderLayers; ++i) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);

            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }
        return kv_cache;
    }

    std::pair<ncnn::Mat, KVCache> decoder_decode(const ncnn::Mat& new_token,
                                                 const ncnn::Mat& encoder_out,
                                                 const KVCache& old_kv_cache) {
        KVCache new_kv_cache;
        new_kv_cache.reserve(kNumDecoderLayers);

        ncnn::Extractor ex = decoder_net_.create_extractor();
        ex.input("in0", new_token);
        ex.input("in1", encoder_out);

        ncnn::Mat attention_mask(1, 1);
        attention_mask.fill(0.0f);
        ex.input("in2", attention_mask);

        for (int i = 0; i < kNumDecoderLayers; ++i) {
            char name_k_in[32], name_v_in[32];
            std::snprintf(name_k_in, sizeof(name_k_in), "cache_k%d", i);
            std::snprintf(name_v_in, sizeof(name_v_in), "cache_v%d", i);
            ex.input(name_k_in, old_kv_cache[i].first);
            ex.input(name_v_in, old_kv_cache[i].second);
        }

        for (int i = 0; i < kNumDecoderLayers; ++i) {
            char name_k_out[32], name_v_out[32];
            std::snprintf(name_k_out, sizeof(name_k_out), "out_cache_k%d", i);
            std::snprintf(name_v_out, sizeof(name_v_out), "out_cache_v%d", i);
            ncnn::Mat k_cache, v_cache;
            ex.extract(name_k_out, k_cache);
            ex.extract(name_v_out, v_cache);
            new_kv_cache.emplace_back(std::move(k_cache), std::move(v_cache));
        }

        ncnn::Mat logits;
        ex.extract("out0", logits);
        return {std::move(logits), std::move(new_kv_cache)};
    }

private:
    std::string embed_param_, embed_bin_;
    std::string encoder_param_, encoder_bin_;
    std::string decoder_param_, decoder_bin_;
    std::string vocab_file_, merges_file_;

    ncnn::Net embed_net_;
    ncnn::Net encoder_net_;
    ncnn::Net decoder_net_;
    BpeTokenizer bpe_;

    int bos_eos_id_{2};
};

nllb_600m::nllb_600m(std::string embed_param,
                     std::string embed_bin,
                     std::string encoder_param,
                     std::string encoder_bin,
                     std::string decoder_param,
                     std::string decoder_bin,
                     std::string vocab_file,
                     std::string merges_file)
    : impl_(std::make_unique<Impl>(std::move(embed_param),
                                   std::move(embed_bin),
                                   std::move(encoder_param),
                                   std::move(encoder_bin),
                                   std::move(decoder_param),
                                   std::move(decoder_bin),
                                   std::move(vocab_file),
                                   std::move(merges_file),
                                   false)) {}

nllb_600m::nllb_600m(std::string embed_param,
                     std::string embed_bin,
                     std::string encoder_param,
                     std::string encoder_bin,
                     std::string decoder_param,
                     std::string decoder_bin,
                     std::string vocab_file,
                     std::string merges_file,
                     bool use_vulkan)
    : impl_(std::make_unique<Impl>(std::move(embed_param),
                                   std::move(embed_bin),
                                   std::move(encoder_param),
                                   std::move(encoder_bin),
                                   std::move(decoder_param),
                                   std::move(decoder_bin),
                                   std::move(vocab_file),
                                   std::move(merges_file),
                                   use_vulkan)) {}

nllb_600m::~nllb_600m() = default;

std::string nllb_600m::translate(const std::string& input_text,
                                 const std::string& source_lang,
                                 const std::string& target_lang) {
    if (!impl_ || !impl_->ok()) return {};
    return impl_->translate_sync(input_text, source_lang, target_lang, NllbConfig{});
}

std::string nllb_600m::translate(const std::string& input_text,
                                 const std::string& source_lang,
                                 const std::string& target_lang,
                                 const NllbConfig& config) {
    if (!impl_ || !impl_->ok()) return {};
    return impl_->translate_sync(input_text, source_lang, target_lang, config);
}

bool nllb_600m::translate(const std::string& input_text,
                          const std::string& source_lang,
                          const std::string& target_lang,
                          std::function<void(const std::string&)> callback) {
    if (!impl_ || !impl_->ok()) return false;
    return impl_->translate_stream(input_text, source_lang, target_lang, NllbConfig{}, std::move(callback));
}

bool nllb_600m::translate(const std::string& input_text,
                          const std::string& source_lang,
                          const std::string& target_lang,
                          const NllbConfig& config,
                          std::function<void(const std::string&)> callback) {
    if (!impl_ || !impl_->ok()) return false;
    return impl_->translate_stream(input_text, source_lang, target_lang, config, std::move(callback));
}