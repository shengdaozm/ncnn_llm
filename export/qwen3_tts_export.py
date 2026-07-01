from __future__ import annotations

"""
Qwen3-TTS Export to ncnn

Exports the following sub-networks:
  From Qwen3-TTS LLM model (e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base):
    - embed.ncnn.param/bin       (token embedding)
    - decoder.ncnn.param/bin     (transformer decoder with KV cache)
    - lm_head.ncnn.param/bin     (lm_head / output projection)

  From Qwen3-TTS-Tokenizer-12Hz (separate model):
    - tokenizer_decoder.ncnn.param/bin  (audio codes -> PCM decoder)

Also extracts tokenizer files and generates model.json.

Usage:
  # Export LLM
  python export/qwen3_tts_export.py --llm_model_id Qwen/Qwen3-TTS-12Hz-0.6B-Base \
      --tokenizer_model_id Qwen/Qwen3-TTS-Tokenizer-12Hz --out_dir assets/qwen3_tts

  # Export only LLM (skip tokenizer decoder)
  python export/qwen3_tts_export.py --llm_model_id Qwen/Qwen3-TTS-12Hz-0.6B-Base \
      --out_dir assets/qwen3_tts --skip-tokenizer
"""

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoConfig, AutoTokenizer, AutoModel
except ImportError:
    print("Please install transformers: pip install transformers")
    sys.exit(1)

try:
    import pnnx
except ImportError:
    pnnx = None
    print("Warning: pnnx not found. Will use TorchScript + pnnx CLI fallback.")


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

    def forward(self, hidden_states, cos_cache, sin_cache,
                cache_k, cache_v):
        out = self.layer(
            hidden_states,
            position_ids=None,
            past_key_value=(cache_k, cache_v) if cache_k is not None else None,
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

    def forward(self, hidden_states, cos_cache, sin_cache, *caches):
        new_ks = []
        new_vs = []
        x = hidden_states
        for i, layer in enumerate(self.layers):
            ck = caches[i * 2] if len(caches) > i * 2 else None
            cv = caches[i * 2 + 1] if len(caches) > i * 2 + 1 else None
            x, nk, nv = layer(x, cos_cache, sin_cache, ck, cv)
            new_ks.append(nk)
            new_vs.append(nv)
        x = self.norm(x)
        return x, new_ks, new_vs


class TokenizerDecoderTS(nn.Module):
    """Wrapper for Qwen3-TTS-Tokenizer-12Hz decoder.
    
    The 12Hz tokenizer takes multi-codebook audio codes of shape (T, Q)
    and decodes them to PCM waveform.
    """
    def __init__(self, tokenizer_model: nn.Module):
        super().__init__()
        self.model = tokenizer_model

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        # audio_codes: (batch, T, Q) or (T, Q)
        # Returns: PCM waveform
        return self.model.decode(audio_codes)


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
            if isinstance(merge, (list, tuple)):
                f.write(" ".join(merge) + "\n")
            else:
                f.write(str(merge) + "\n")

    return "vocab.txt", "merges.txt"


def build_model_json(config, tokenizer, out_dir: str,
                     num_codebooks: int = 8,
                     sample_rate: int = 24000,
                     codec_vocab_size: int = 32768,
                     has_tokenizer_decoder: bool = False,
                     tts_model_type: str = "base"):
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
            "attn_cnt": getattr(config, 'talker_config', config).num_hidden_layers if hasattr(getattr(config, 'talker_config', config), 'num_hidden_layers') else 28,
            "tts_mode": "codec",
            "tts_model_type": tts_model_type,
            "rope": {
                "type": "RoPE",
                "rope_head_dim": getattr(config, 'talker_config', config).head_dim if hasattr(getattr(config, 'talker_config', config), 'head_dim') else 128,
                "rope_theta": getattr(config, 'talker_config', config).rope_theta if hasattr(getattr(config, 'talker_config', config), 'rope_theta') else 1000000.0,
            },
            "audio": {
                "num_codebooks": num_codebooks,
                "codec_vocab_size": codec_vocab_size,
                "sample_rate": sample_rate,
                "frame_rate": 12,
            },
        },
    }

    if has_tokenizer_decoder:
        model_json["params"]["tokenizer_decoder_param"] = "tokenizer_decoder.ncnn.param"
        model_json["params"]["tokenizer_decoder_bin"] = "tokenizer_decoder.ncnn.bin"

    json_path = os.path.join(out_dir, "model.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2, ensure_ascii=False)

    print(f"  Saved model.json: {json_path}")
    return json_path


@torch.no_grad()
def export_llm(model_id: str, out_dir: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting LLM: {model_id}")
    print(f"{'='*60}")

    # Register qwen3_tts model type (requires qwen-tts package)
    # The qwen_tts package does NOT auto-register on import;
    # we must explicitly call AutoConfig.register / AutoModel.register.
    try:
        from qwen_tts.core import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        print("  qwen3_tts model type registered via qwen_tts.core")
    except ImportError:
        print("  WARNING: qwen_tts.core not found, trying trust_remote_code fallback")
        try:
            import qwen_tts  # noqa: F401
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
            # Trigger registration by calling the class method that does it
            # Qwen3TTSModel.from_pretrained registers, but we don't want to load yet
            # Instead, manually register using the classes from the package
            from qwen_tts.core.models import Qwen3TTSConfig as _Cfg, Qwen3TTSForConditionalGeneration as _Mdl
            AutoConfig.register("qwen3_tts", _Cfg)
            AutoModel.register(_Cfg, _Mdl)
            print("  qwen3_tts model type registered via qwen_tts.core.models")
        except Exception as e:
            print(f"  WARNING: Could not register qwen3_tts: {e}")
            print("  Will rely on trust_remote_code=True")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    try:
        model = AutoModel.from_pretrained(model_id, dtype=torch.float32,
                                          trust_remote_code=True).to(device).eval()
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                                     trust_remote_code=True).to(device).eval()

    # Qwen3-TTS model structure:
    #   model.talker.model.text_embedding   → text token embedding
    #   model.talker.model.layers           → decoder layers
    #   model.talker.model.norm             → final norm
    #   model.talker.codec_head             → output head (codec logits)
    talker = getattr(model, 'talker', None)
    talker_model = getattr(talker, 'model', None) if talker else None

    if talker_model is not None:
        print("  Detected Qwen3-TTS model structure (talker.model)")
        embed = talker_model.text_embedding
        lm_head = talker.codec_head
        decoder_model = talker_model
    else:
        # Fallback: standard HF model structure
        print("  Using standard model structure")
        base_model = getattr(model, 'model', model)
        embed = getattr(base_model, 'embed_tokens', None) or getattr(model, 'embed_tokens', None)
        if embed is None:
            raise AttributeError("Cannot find embedding layer in model")
        lm_head = getattr(model, 'lm_head', None) or getattr(base_model, 'lm_head', None)
        if lm_head is None:
            raise AttributeError("Cannot find lm_head in model")
        decoder_model = base_model if hasattr(base_model, 'layers') else model

    # Get config parameters from talker config if available
    talker_config = getattr(config, 'talker_config', config)
    hidden_size = talker_config.hidden_size if hasattr(talker_config, 'hidden_size') else 1024
    num_layers = talker_config.num_hidden_layers if hasattr(talker_config, 'num_hidden_layers') else 28
    vocab_size = talker_config.vocab_size if hasattr(talker_config, 'vocab_size') else 151936
    num_kv_heads = talker_config.num_key_value_heads if hasattr(talker_config, 'num_key_value_heads') else 4
    num_heads = talker_config.num_attention_heads if hasattr(talker_config, 'num_attention_heads') else 16
    head_dim = talker_config.head_dim if hasattr(talker_config, 'head_dim') else (hidden_size // num_heads)

    print(f"  hidden_size: {hidden_size}, num_layers: {num_layers}")
    print(f"  vocab_size: {vocab_size}, num_heads: {num_heads}, num_kv_heads: {num_kv_heads}, head_dim: {head_dim}")

    # Export embedding
    print("\nExporting embed...")
    embed_ts = EmbedTS(embed)
    ex_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=device)
    export_to_ncnn(embed_ts, ex_ids, out_dir, "embed", device)

    # Export lm_head
    print("\nExporting lm_head...")
    lm_head_ts = LmHeadTS(lm_head)
    ex_hidden = torch.randn(1, 1, hidden_size, device=device)
    export_to_ncnn(lm_head_ts, ex_hidden, out_dir, "lm_head", device)

    # Export decoder
    print("\nExporting decoder (this may take a while)...")
    decoder_model = base_model if hasattr(base_model, 'layers') else model
    decoder_ts = DecoderTS(decoder_model, num_layers)

    seq_len = 8
    ex_embed = torch.randn(1, seq_len, hidden_size, device=device)
    ex_cos = torch.randn(seq_len, head_dim, device=device)
    ex_sin = torch.randn(seq_len, head_dim, device=device)
    caches = []
    for _ in range(num_layers):
        caches.append(torch.randn(1, num_kv_heads, 0, head_dim, device=device))
        caches.append(torch.randn(1, num_kv_heads, 0, head_dim, device=device))

    try:
        export_to_ncnn(decoder_ts, [ex_embed, ex_cos, ex_sin] + caches,
                       out_dir, "decoder", device)
    except Exception as e:
        print(f"  Decoder export failed: {e}")
        print("  Try using pnnx CLI directly on the model.")

    # Extract tokenizer
    print("\nExtracting tokenizer...")
    extract_tokenizer(tokenizer, out_dir)

    return config, tokenizer


@torch.no_grad()
def export_tokenizer_decoder(model_id: str, out_dir: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Exporting Tokenizer-12Hz: {model_id}")
    print(f"{'='*60}")

    from transformers import AutoConfig, AutoModel

    # Register qwen3_tts_tokenizer model types
    try:
        from qwen_tts.core import (
            Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model,
            Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model,
        )
        AutoConfig.register("qwen3_tts_tokenizer_25hz", Qwen3TTSTokenizerV1Config)
        AutoModel.register(Qwen3TTSTokenizerV1Config, Qwen3TTSTokenizerV1Model)
        AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
        AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)
        print("  qwen3_tts_tokenizer model types registered")
    except ImportError:
        print("  WARNING: qwen_tts.core not found, relying on trust_remote_code")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32,
                                      trust_remote_code=True).to(device).eval()

    # The tokenizer-12Hz decoder takes audio codes (T, Q) and returns PCM
    # Q (num_quantizers) is typically 8 for 12Hz tokenizer
    num_quantizers = getattr(config, 'num_quantizers', 8)

    print(f"  num_quantizers: {num_quantizers}")

    # Try to access the decode method
    if hasattr(model, 'decode'):
        print("  Model has decode() method, wrapping for export...")
        tokenizer_ts = TokenizerDecoderTS(model)

        # Example input: (1, T, Q) audio codes
        ex_codes = torch.randint(0, 32768, (1, 100, num_quantizers), dtype=torch.long, device=device)
        try:
            export_to_ncnn(tokenizer_ts, ex_codes, out_dir, "tokenizer_decoder", device)
            return True
        except Exception as e:
            print(f"  Tokenizer decoder export failed: {e}")
            print("  The tokenizer decoder may need manual export via pnnx CLI.")
            return False
    else:
        print("  Model does not have decode() method. Skipping.")
        print("  Manual export required: inspect model structure and export decoder sub-module.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS to ncnn format")
    parser.add_argument("--llm_model_id", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="HuggingFace model ID for the TTS LLM")
    parser.add_argument("--tokenizer_model_id", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                        help="HuggingFace model ID for the Tokenizer-12Hz")
    parser.add_argument("--out_dir", type=str, default="assets/qwen3_tts",
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu or cuda")
    parser.add_argument("--skip-tokenizer", action="store_true",
                        help="Skip exporting the tokenizer decoder")
    parser.add_argument("--num_codebooks", type=int, default=8,
                        help="Number of audio codebooks (quantizers)")
    parser.add_argument("--tts_model_type", type=str, default="base",
                        choices=["base", "custom_voice", "voice_design"],
                        help="TTS model type")

    args = parser.parse_args()

    config, tokenizer = export_llm(args.llm_model_id, args.out_dir, args.device)

    has_tokenizer_decoder = False
    if not args.skip_tokenizer:
        has_tokenizer_decoder = export_tokenizer_decoder(
            args.tokenizer_model_id, args.out_dir, args.device)

    print("\nBuilding model.json...")
    build_model_json(
        config, tokenizer, args.out_dir,
        num_codebooks=args.num_codebooks,
        has_tokenizer_decoder=has_tokenizer_decoder,
        tts_model_type=args.tts_model_type,
    )

    print(f"\n{'='*60}")
    print(f"Export complete! Output directory: {args.out_dir}")
    print(f"{'='*60}")
    print(f"\nPlace this directory under assets/ and run:")
    print(f"  xmake run tts_main --model assets/{os.path.basename(args.out_dir)} --text \"Hello world\"")
