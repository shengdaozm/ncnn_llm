from __future__ import annotations

"""
Qwen2.5-TTS Export to ncnn

Exports the following sub-networks from a Qwen2.5-TTS model:
  - embed.ncnn.param/bin     (token embedding)
  - decoder.ncnn.param/bin   (transformer decoder with KV cache)
  - lm_head.ncnn.param/bin   (lm_head / output projection)
  - codec.ncnn.param/bin     (audio codec decoder, e.g. SNAC)
  - flow.ncnn.param/bin      (flow-matching decoder, optional for flow mode)
  - vocoder.ncnn.param/bin   (vocoder, optional for flow mode)

Also extracts tokenizer files and generates model.json.

Usage:
  python export/qwen_tts_export.py --model_id Qwen/Qwen2.5-TTS --out_dir assets/qwen_tts
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
except ImportError:
    print("Please install transformers: pip install transformers")
    sys.exit(1)

try:
    import ncnn
except ImportError:
    print("Please install pnnx or ncnn python binding for conversion")
    ncnn = None


class EmbedTS(nn.Module):
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self.embed = embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)


class LmHeadTS(nn.Module):
    def __init__(self, lm_head: nn.Linear):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class DecoderLayerTS(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, mask, cos_cache, sin_cache,
                cache_k, cache_v):
        out = self.layer(
            hidden_states,
            attention_mask=mask,
            position_ids=None,
            past_key_value=(cache_k, cache_v),
            use_cache=True,
        )
        return out[0], out[1][0], out[1][1]


class DecoderTS(nn.Module):
    def __init__(self, model: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayerTS(model.layers[i]) for i in range(num_layers)
        ])
        self.norm = model.norm if hasattr(model, 'norm') else nn.Identity()

    def forward(self, hidden_states, mask, cos_cache, sin_cache,
                *caches):
        new_ks = []
        new_vs = []
        x = hidden_states
        for i, layer in enumerate(self.layers):
            ck = caches[i * 2] if len(caches) > i * 2 else None
            cv = caches[i * 2 + 1] if len(caches) > i * 2 + 1 else None
            x, nk, nv = layer(x, mask, cos_cache, sin_cache, ck, cv)
            new_ks.append(nk)
            new_vs.append(nv)
        x = self.norm(x)
        return x, new_ks, new_vs


class CodecDecoderTS(nn.Module):
    def __init__(self, codec_model: nn.Module):
        super().__init__()
        self.codec = codec_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(tokens)


class FlowNetTS(nn.Module):
    def __init__(self, flow_model: nn.Module):
        super().__init__()
        self.flow = flow_model

    def forward(self, x, condition, x_mask, t):
        return self.flow(x, condition, x_mask, t)


class VocoderTS(nn.Module):
    def __init__(self, vocoder: nn.Module):
        super().__init__()
        self.vocoder = vocoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.vocoder(mel)


def export_to_ncnn(module: nn.Module, example_inputs, out_dir: str, name: str,
                   device: str = "cpu") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    param_path = os.path.join(out_dir, f"{name}.ncnn.param")
    bin_path = os.path.join(out_dir, f"{name}.ncnn.bin")

    module = module.to(device).eval()

    try:
        import pnnx
        pnnx.convert(module, example_inputs, outputdir=out_dir,
                     optlevel=2, pnnx_param=param_path, pnnx_bin=bin_path,
                     fp16=False)
    except (ImportError, Exception) as e:
        print(f"  pnnx convert failed for {name}: {e}")
        print(f"  Falling back to TorchScript export: {name}.pt")
        ts_path = os.path.join(out_dir, f"{name}.pt")
        with torch.no_grad():
            if isinstance(example_inputs, (list, tuple)):
                traced = torch.jit.trace(module, example_inputs)
            else:
                traced = torch.jit.trace(module, example_inputs)
        traced.save(ts_path)
        print(f"  Saved TorchScript: {ts_path}")
        print(f"  Use pnnx CLI to convert: pnnx {ts_path}")

    return param_path, bin_path


def extract_tokenizer(tokenizer, out_dir: str):
    vocab_path = os.path.join(out_dir, "vocab.txt")
    merges_path = os.path.join(out_dir, "merges.txt")

    vocab = tokenizer.get_vocab()
    id_to_token = sorted(vocab.items(), key=lambda x: x[1])

    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in id_to_token:
            f.write(token + "\n")

    if hasattr(tokenizer, 'get_merges'):
        merges = tokenizer.get_merges()
    elif hasattr(tokenizer, 'merges'):
        merges = tokenizer.merges
    else:
        merges = []

    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(" ".join(merge) + "\n")

    return "vocab.txt", "merges.txt"


def build_model_json(config, tokenizer, out_dir: str,
                     tts_mode: str = "codec",
                     num_codebooks: int = 4,
                     sample_rate: int = 24000,
                     codec_vocab_size: int = 4096,
                     mel_dim: int = 128,
                     has_codec: bool = True,
                     has_flow: bool = False,
                     has_vocoder: bool = False):
    model_json = {
        "model_type": "tts",
        "params": {
            "embed_token_param": "embed.ncnn.param",
            "embed_token_bin": "embed.ncnn.bin",
            "decoder_param": "decoder.ncnn.param",
            "decoder_bin": "decoder.ncnn.bin",
            "lm_head_param": "lm_head.ncnn.param",
            "lm_head_bin": "lm_head.ncnn.bin",
        },
        "tokenizer": {
            "type": "bbpe",
            "vocab_file": "vocab.txt",
            "merges_file": "merges.txt",
            "eos": tokenizer.eos_token if tokenizer.eos_token else "",
            "bos": tokenizer.bos_token if tokenizer.bos_token else "",
            "additional_special_tokens": tokenizer.additional_special_tokens if hasattr(tokenizer, 'additional_special_tokens') else [],
        },
        "setting": {
            "attn_cnt": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 24,
            "tts_mode": tts_mode,
            "rope": {
                "type": "RoPE",
                "rope_head_dim": config.head_dim if hasattr(config, 'head_dim') else 64,
                "rope_theta": config.rope_theta if hasattr(config, 'rope_theta') else 1000000.0,
            },
            "audio": {
                "num_codebooks": num_codebooks,
                "codec_vocab_size": codec_vocab_size,
                "sample_rate": sample_rate,
                "mel_dim": mel_dim,
            },
        },
    }

    if has_codec:
        model_json["params"]["codec_param"] = "codec.ncnn.param"
        model_json["params"]["codec_bin"] = "codec.ncnn.bin"

    if has_flow:
        model_json["params"]["flow_param"] = "flow.ncnn.param"
        model_json["params"]["flow_bin"] = "flow.ncnn.bin"
        model_json["setting"]["flow_matching"] = {
            "num_steps": 10,
            "sigma": 0.0,
        }

    if has_vocoder:
        model_json["params"]["vocoder_param"] = "vocoder.ncnn.param"
        model_json["params"]["vocoder_bin"] = "vocoder.ncnn.bin"

    json_path = os.path.join(out_dir, "model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2, ensure_ascii=False)

    print(f"  Saved model.json: {json_path}")
    return json_path


@torch.no_grad()
def export_qwen_tts(model_id: str, out_dir: str, device: Optional[str] = None,
                    tts_mode: str = "codec"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {model_id}")
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device).eval()
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device).eval()

    hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else 2048
    num_layers = config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 24
    vocab_size = config.vocab_size if hasattr(config, 'vocab_size') else 151936

    print(f"  hidden_size: {hidden_size}, num_layers: {num_layers}, vocab_size: {vocab_size}")

    embed = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
    embed_ts = EmbedTS(embed)
    ex_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    export_to_ncnn(embed_ts, ex_ids, out_dir, "embed", device)

    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head
    lm_head_ts = LmHeadTS(lm_head)
    ex_hidden = torch.randn(1, 1, hidden_size, device=device)
    export_to_ncnn(lm_head_ts, ex_hidden, out_dir, "lm_head", device)

    print("Exporting decoder (this may take a while)...")
    decoder_model = model.model if hasattr(model, 'model') else model
    decoder_ts = DecoderTS(decoder_model, num_layers)

    seq_len = 8
    ex_embed = torch.randn(1, seq_len, hidden_size, device=device)
    ex_mask = torch.zeros(1, 1, seq_len, seq_len, device=device)
    ex_cos = torch.randn(seq_len, 64, device=device)
    ex_sin = torch.randn(seq_len, 64, device=device)
    caches = []
    for _ in range(num_layers):
        caches.append(torch.randn(1, 32, 0, 128, device=device))
        caches.append(torch.randn(1, 32, 0, 128, device=device))

    try:
        export_to_ncnn(decoder_ts, [ex_embed, ex_mask, ex_cos, ex_sin] + caches,
                       out_dir, "decoder", device)
    except Exception as e:
        print(f"  Decoder export failed: {e}")
        print("  The decoder needs to be exported with proper KV cache support.")
        print("  Consider using pnnx directly on the model with --pnnx flag.")

    print("Extracting tokenizer...")
    vocab_file, merges_file = extract_tokenizer(tokenizer, out_dir)

    has_codec = False
    has_flow = False
    has_vocoder = False

    if tts_mode == "codec":
        codec_model = None
        if hasattr(model, 'codec'):
            codec_model = model.codec
        elif hasattr(model, 'audio_codec'):
            codec_model = model.audio_codec

        if codec_model is not None:
            print("Exporting codec decoder...")
            codec_ts = CodecDecoderTS(codec_model)
            ex_tokens = torch.randint(0, 4096, (4, 100), dtype=torch.long, device=device)
            try:
                export_to_ncnn(codec_ts, ex_tokens, out_dir, "codec", device)
                has_codec = True
            except Exception as e:
                print(f"  Codec export failed: {e}")
        else:
            print("  No codec model found, skipping codec export")

    if tts_mode == "flow":
        flow_model = None
        if hasattr(model, 'flow'):
            flow_model = model.flow
        elif hasattr(model, 'flow_matching'):
            flow_model = model.flow_matching

        if flow_model is not None:
            print("Exporting flow-matching network...")
            flow_ts = FlowNetTS(flow_model)
            ex_x = torch.randn(128, 100, device=device)
            ex_cond = torch.randn(128, 100, device=device)
            ex_xmask = torch.ones(100, device=device)
            ex_t = torch.tensor(0.5, device=device)
            try:
                export_to_ncnn(flow_ts, [ex_x, ex_cond, ex_xmask, ex_t],
                               out_dir, "flow", device)
                has_flow = True
            except Exception as e:
                print(f"  Flow export failed: {e}")

        vocoder = None
        if hasattr(model, 'vocoder'):
            vocoder = model.vocoder
        elif hasattr(model, 'hifigan'):
            vocoder = model.hifigan

        if vocoder is not None:
            print("Exporting vocoder...")
            vocoder_ts = VocoderTS(vocoder)
            ex_mel = torch.randn(128, 100, device=device)
            try:
                export_to_ncnn(vocoder_ts, ex_mel, out_dir, "vocoder", device)
                has_vocoder = True
            except Exception as e:
                print(f"  Vocoder export failed: {e}")

    print("Building model.json...")
    build_model_json(
        config, tokenizer, out_dir,
        tts_mode=tts_mode,
        has_codec=has_codec,
        has_flow=has_flow,
        has_vocoder=has_vocoder,
    )

    print(f"\nExport complete! Output directory: {out_dir}")
    print(f"Place this directory under assets/ and run:")
    print(f"  xmake run tts_main --model assets/{os.path.basename(out_dir)} --text \"Hello world\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen2.5-TTS to ncnn format")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-TTS",
                        help="HuggingFace model ID")
    parser.add_argument("--out_dir", type=str, default="assets/qwen_tts",
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu or cuda")
    parser.add_argument("--mode", type=str, default="codec",
                        choices=["codec", "flow"],
                        help="TTS mode: codec (LLM + codec decoder) or flow (LLM + flow-matching + vocoder)")

    args = parser.parse_args()

    export_qwen_tts(
        model_id=args.model_id,
        out_dir=args.out_dir,
        device=args.device,
        tts_mode=args.mode,
    )
