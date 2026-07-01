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


class DecoderTS(nn.Module):
    """Wrapper for the TalkerModel decoder.
    
    Qwen3-TTS talker model has complex control flow (position_ids.ndim checks)
    that breaks torch.jit.trace. We bypass this by calling the internal
    layers directly, which gives a clean traceable graph.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.layers = model.layers
        self.norm = model.norm
        self.rotary_emb = model.rotary_emb

    def forward(self, inputs_embeds, position_ids):
        # Compute position embeddings once
        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings = (cos, sin)

        hidden = inputs_embeds
        for layer in self.layers:
            out = layer(
                hidden_states=hidden,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )
            hidden = out[0]

        hidden = self.norm(hidden)
        return hidden


class TokenizerDecoderTS(nn.Module):
    """Wrapper for Qwen3-TTS-Tokenizer-12Hz decoder.
    
    Wraps the internal decoder sub-module directly to avoid control flow
    issues in model.decode() that break torch.jit.trace.
    
    Input: audio codes (batch, num_quantizers, T) e.g. (1, 16, 100)
    Output: PCM waveform (batch, 1, samples)
    """
    def __init__(self, tokenizer_model: nn.Module):
        super().__init__()
        self.decoder = tokenizer_model.decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(codes)


def export_to_ncnn(module: nn.Module, example_inputs, out_dir: str, name: str,
                   device: str = "cpu") -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)

    param_path = os.path.join(out_dir, f"{name}.ncnn.param")
    bin_path = os.path.join(out_dir, f"{name}.ncnn.bin")
    ts_path = os.path.join(out_dir, f"{name}.pt")

    module = module.to(device).eval()

    # Step 1: Export to TorchScript first
    with torch.no_grad():
        try:
            if isinstance(example_inputs, (list, tuple)):
                traced = torch.jit.trace(module, example_inputs)
            else:
                traced = torch.jit.trace(module, example_inputs)
            traced.save(ts_path)
            print(f"  Saved TorchScript: {ts_path}")
        except Exception as e:
            print(f"  TorchScript trace failed for {name}: {e}")
            # Try ONNX as alternative intermediate format
            onnx_path = os.path.join(out_dir, f"{name}.onnx")
            try:
                with torch.no_grad():
                    torch.onnx.export(
                        module, example_inputs, onnx_path,
                        export_params=True, opset_version=17,
                        do_constant_folding=True,
                        dynamo=True,
                    )
                # Re-save without external data (inline all weights)
                import onnx
                from onnx.external_data_helper import convert_model_to_external_data
                onnx_model = onnx.load(onnx_path, load_external_data=True)
                for tensor in onnx_model.graph.initializer:
                    tensor.ClearField('data_location')
                onnx.save_model(onnx_model, onnx_path, save_as_external_data=False)
                # Clean up external data file
                ext_data = onnx_path + ".data"
                if os.path.exists(ext_data):
                    os.remove(ext_data)
                print(f"  Saved ONNX (inline): {onnx_path}")
                ts_path = onnx_path
            except Exception as e2:
                print(f"  ONNX export also failed for {name}: {e2}")
                print(f"  WARNING: Could not export {name}")
                return "", ""

    # Step 2: Convert to ncnn via pnnx CLI
    try:
        import pnnx
        # pnnx CLI: pnnx model.pt inputshape=[...] ncnnparam=... ncnnbin=...
        shapes = []
        for inp in (example_inputs if isinstance(example_inputs, (list, tuple)) else [example_inputs]):
            if isinstance(inp, torch.Tensor):
                shape_str = ",".join(str(s) for s in inp.shape)
                dtype_str = "f32" if inp.dtype == torch.float32 else "i64" if inp.dtype == torch.long else "f32"
                shapes.append(f"[{shape_str}]{dtype_str}")
        inputshape_str = ",".join(shapes)

        import subprocess
        cmd = [
            "pnnx", ts_path,
            f"inputshape={inputshape_str}",
            f"ncnnparam={param_path}",
            f"ncnnbin={bin_path}",
            "fp16=0",
            "optlevel=2",
            f"device={'gpu' if device != 'cpu' else 'cpu'}",
        ]
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(param_path):
            print(f"  Saved ncnn: {param_path}")
            return param_path, bin_path
        else:
            print(f"  pnnx CLI failed (exit {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            if result.stdout:
                print(f"  stdout: {result.stdout[:500]}")
            print(f"  Use pnnx CLI manually: pnnx {ts_path}")
            return ts_path, ""
    except Exception as e:
        print(f"  pnnx convert failed for {name}: {e}")
        print(f"  Use pnnx CLI manually: pnnx {ts_path}")
        return ts_path, ""


def extract_tokenizer(tokenizer, out_dir: str):
    # Copy vocab.json and merges.txt directly from the model directory
    # (Qwen2 tokenizer stores them as files, not accessible via Python API)
    import shutil

    model_dir = getattr(tokenizer, '_tokenizer', None)
    # Try to get the source directory from the tokenizer
    vocab_json_src = None
    merges_txt_src = None

    # Search in known locations
    for attr in ('name_or_path', '_name_or_path'):
        path = getattr(tokenizer, attr, None)
        if path and os.path.isdir(path):
            vocab_json_src = os.path.join(path, 'vocab.json')
            merges_txt_src = os.path.join(path, 'merges.txt')
            break

    vocab_path = os.path.join(out_dir, "vocab.txt")
    merges_path = os.path.join(out_dir, "merges.txt")

    if vocab_json_src and os.path.exists(vocab_json_src):
        # Convert vocab.json to vocab.txt (id-sorted token list)
        import json
        with open(vocab_json_src, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        id_to_token = sorted(vocab.items(), key=lambda x: x[1])
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in id_to_token:
                f.write(token + '\n')
        print(f"  Extracted vocab.txt ({len(id_to_token)} tokens)")
    else:
        # Fallback: use tokenizer API
        vocab = tokenizer.get_vocab()
        id_to_token = sorted(vocab.items(), key=lambda x: x[1])
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token, _ in id_to_token:
                f.write(token + '\n')
        print(f"  Extracted vocab.txt via API ({len(id_to_token)} tokens)")

    if merges_txt_src and os.path.exists(merges_txt_src):
        shutil.copy2(merges_txt_src, merges_path)
        print(f"  Copied merges.txt")
    else:
        # Create empty merges file as fallback
        with open(merges_path, 'w') as f:
            pass
        print(f"  WARNING: merges.txt not found, created empty file")

    return "vocab.txt", "merges.txt"


def build_model_json(config, tokenizer, out_dir: str,
                     num_codebooks: int = 16,
                     sample_rate: int = 24000,
                     codec_vocab_size: int = 2048,
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
        from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
        print("  qwen3_tts model type registered")
    except ImportError as e:
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
    decoder_ts = DecoderTS(decoder_model)

    seq_len = 8
    ex_embed = torch.randn(1, seq_len, hidden_size, device=device)
    # Qwen3-TTS uses mRoPE with 3D position_ids: (3, bs, seq)
    ex_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0).expand(3, 1, -1)

    try:
        export_to_ncnn(decoder_ts, [ex_embed, ex_position_ids],
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

    # Read actual config values
    decoder_cfg = getattr(config, 'decoder_config', None)
    if decoder_cfg is not None:
        num_quantizers = getattr(decoder_cfg, 'num_quantizers', 16)
        codebook_size = getattr(decoder_cfg, 'codebook_size', 2048)
    else:
        num_quantizers = 16
        codebook_size = 2048

    print(f"  num_quantizers: {num_quantizers}, codebook_size: {codebook_size}")

    # Try to access the decode method
    if hasattr(model, 'decode'):
        print("  Model has decode() method, wrapping for export...")
        tokenizer_ts = TokenizerDecoderTS(model)

        # Example input: (batch, num_quantizers, T) audio codes
        ex_codes = torch.randint(0, codebook_size, (1, num_quantizers, 100), dtype=torch.long, device=device)
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
    parser.add_argument("--num_codebooks", type=int, default=16,
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
