from __future__ import annotations

"""
Qwen3-TTS Export to ncnn

Exports the following sub-networks:
  From Qwen3-TTS LLM model (e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base):
    - embed.ncnn.param/bin              (text token embedding)
    - decoder.ncnn.param/bin            (talker transformer decoder, 20 layers, with KV cache)
    - lm_head.ncnn.param/bin            (codec_head: hidden → main codebook logits, vocab=3072)
    - text_projection.ncnn.param/bin    (text_hidden_size → hidden_size MLP)
    - speaker_encoder.ncnn.param/bin    (ECAPA-TDNN: mel → speaker embedding, dim=1024)
    - cp_decoder.ncnn.param/bin         (code predictor 5-layer transformer, with KV cache)
    - cp_lm_heads.ncnn.param/bin        (31 merged lm_heads: hidden → sub-codebook logits, vocab=2048)
    - cp_codec_embeds.ncnn.param/bin    (31 merged codec embeddings: sub-codebook token → hidden)

  From Qwen3-TTS-Tokenizer-12Hz (separate model):
    - tokenizer_decoder.ncnn.param/bin  (audio codes → PCM decoder)

Also extracts tokenizer files and generates model.json.

Usage:
  # Full export
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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor,
                           cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class DecoderLayerTS(nn.Module):
    """Single decoder layer wrapper with explicit KV cache tensors.

    Bypasses DynamicCache by manually managing KV cache as plain tensors.
    Calls the layer's sub-modules (layernorm, q/k/v/o projections, mlp)
    directly and implements attention + KV cache concatenation inline,
    so torch.jit.trace sees only plain tensor operations.

    This avoids:
      - DynamicCache (non-tensor object, untraceable)
      - Conditional branching on position_ids.ndim
      - FlashAttention dispatch
    """
    def __init__(self, layer: nn.Module, num_kv_heads: int, num_heads: int, head_dim: int):
        super().__init__()
        self.input_layernorm = layer.input_layernorm
        self.q_proj = layer.self_attn.q_proj
        self.k_proj = layer.self_attn.k_proj
        self.v_proj = layer.self_attn.v_proj
        self.o_proj = layer.self_attn.o_proj
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = layer.self_attn.scaling if hasattr(layer.self_attn, 'scaling') else (head_dim ** -0.5)
        self.num_key_value_groups = num_heads // num_kv_heads

    def forward(self, hidden_states, attention_mask,
                cos_cache, sin_cache,
                cache_k: torch.Tensor, cache_v: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        batch = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        query_states = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos_cache, sin_cache)

        key_states = torch.cat([cache_k, key_states], dim=2)
        value_states = torch.cat([cache_v, value_states], dim=2)
        new_k = key_states
        new_v = value_states

        key_states = _repeat_kv(key_states, self.num_key_value_groups)
        value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_k, new_v


class DecoderTS(nn.Module):
    """Wrapper for the TalkerModel decoder with explicit KV cache.

    Bypasses DynamicCache and complex control flow by calling sub-modules
    directly with explicit KV cache tensors.

    Interface (aligned with C++ ncnn_text_runtime llm_run_decoder_with_kv):
      Inputs:  in0=embeds, in1=mask, in2=cos, in3=sin, cache_k{i}, cache_v{i}
      Outputs: out0=hidden, out_cache_k{i}, out_cache_v{i}

    For prefill: pass zero-length cache tensors (shape [num_kv_heads, 0, head_dim]).
    For decode:  pass previous output cache tensors.
    """
    def __init__(self, model: nn.Module, num_layers: int, num_kv_heads: int, num_heads: int, head_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayerTS(model.layers[i], num_kv_heads, num_heads, head_dim) for i in range(num_layers)]
        )
        self.norm = model.norm

    def forward(self, inputs_embeds, mask, cos_cache, sin_cache, *caches):
        hidden = inputs_embeds
        new_ks = []
        new_vs = []
        for i, layer in enumerate(self.layers):
            ck = caches[i * 2]
            cv = caches[i * 2 + 1]
            hidden, nk, nv = layer(hidden, mask, cos_cache, sin_cache, ck, cv)
            new_ks.append(nk)
            new_vs.append(nv)

        hidden = self.norm(hidden)
        return (hidden, *new_ks, *new_vs)


class SpeakerEncoderTS(nn.Module):
    """Wrapper for Qwen3-TTS SpeakerEncoder (ECAPA-TDNN).

    Input: log-mel spectrogram [1, T_mel, mel_dim=128]
    Output: speaker embedding [1, enc_dim=1024]
    """
    def __init__(self, speaker_encoder: nn.Module):
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        out = self.speaker_encoder(mel)
        return out


class TextProjectionTS(nn.Module):
    """Wrapper for talker.text_projection (ResizeMLP).

    Input: text embeddings [B, T, text_hidden_size=2048]
    Output: projected embeddings [B, T, hidden_size=1024]
    """
    def __init__(self, text_projection: nn.Module):
        super().__init__()
        self.text_projection = text_projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.text_projection(x)


class CodePredictorDecoderTS(nn.Module):
    """Wrapper for CodePredictor 5-layer transformer decoder with explicit KV cache.

    Interface (aligned with C++ runtime):
      Inputs:  in0=embeds, in1=mask, in2=cos, in3=sin, cache_k{i}, cache_v{i}
      Outputs: out0=hidden, out_cache_k{i}, out_cache_v{i}
    """
    def __init__(self, cp_model: nn.Module, num_layers: int, num_kv_heads: int,
                 num_heads: int, head_dim: int):
        super().__init__()
        self.cp_model = cp_model
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_kv_heads

    def forward(self, inputs_embeds, mask, cos_cache, sin_cache, *caches):
        hidden = inputs_embeds
        new_ks = []
        new_vs = []
        for i in range(self.num_layers):
            ck = caches[i * 2] if len(caches) > 0 else None
            cv = caches[i * 2 + 1] if len(caches) > 0 else None
            hidden, nk, nv = self._forward_layer(i, hidden, mask, cos_cache, sin_cache, ck, cv)
            new_ks.append(nk)
            new_vs.append(nv)
        hidden = self.cp_model.norm(hidden)
        return (hidden, *new_ks, *new_vs)

    def _forward_layer(self, idx, hidden, mask, cos, sin, cache_k, cache_v):
        layer = self.cp_model.layers[idx]
        residual = hidden
        hidden = layer.input_layernorm(hidden)

        batch = hidden.size(0)
        seq_len = hidden.size(1)

        q = layer.self_attn.q_proj(hidden).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = layer.self_attn.k_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = layer.self_attn.v_proj(hidden).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if cache_k is not None and cache_k.numel() > 0:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)
        new_k, new_v = k, v

        k = _repeat_kv(k, self.num_key_value_groups)
        v = _repeat_kv(v, self.num_key_value_groups)

        scaling = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(2, 3)) * scaling
        if mask is not None:
            attn = attn + mask
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = layer.self_attn.o_proj(out)

        hidden = residual + out
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        return hidden + residual, new_k, new_v


class CodePredictorLmHeadsTS(nn.Module):
    """Merged 31 lm_heads for CodePredictor.

    Stacks 31 Linear(1024, 2048) into a single [1024, 31, 2048] weight tensor.
    Input: hidden [B, T, 1024], step_index (int)
    Output: logits [B, T, 2048] for the given step

    Since ncnn doesn't support dynamic index selection, we export all 31 heads
    as a single batched matmul and let C++ select the appropriate slice.
    """
    def __init__(self, lm_heads: nn.ModuleList, num_heads: int = 31):
        super().__init__()
        self.num_heads = num_heads
        weights = []
        for i in range(num_heads):
            w = lm_heads[i].weight.data  # [2048, 1024]
            weights.append(w)
        self.weight = nn.Parameter(torch.stack(weights, dim=0))  # [31, 2048, 1024]

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, 1024]
        # output: [31, B, T, 2048] — C++ selects the right step
        # For export simplicity, output all 31 at once
        B, T, D = hidden.shape
        # [31, 2048, 1024] x [B*T, 1024, 1] → [31, 2048, B*T] → [31, B, T, 2048]
        hidden_flat = hidden.reshape(B * T, D, 1)
        out = torch.matmul(self.weight, hidden_flat)  # [31, 2048, B*T]
        out = out.permute(0, 2, 1).reshape(self.num_heads, B, T, -1)
        return out


class CodePredictorCodecEmbedsTS(nn.Module):
    """Merged 31 codec embeddings for CodePredictor.

    Stacks 31 Embedding(2048, 1024) into a single weight tensor.
    Input: token_ids [B, T] (long), step_index (int)
    Output: embeddings [B, T, 1024]

    Exported as a single embedding lookup; C++ handles step selection.
    """
    def __init__(self, codec_embeddings: nn.ModuleList, num_heads: int = 31):
        super().__init__()
        self.num_heads = num_heads
        weights = []
        for i in range(num_heads):
            w = codec_embeddings[i].weight.data  # [2048, 1024]
            weights.append(w)
        self.weight = nn.Parameter(torch.stack(weights, dim=0))  # [31, 2048, 1024]

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, T] long
        # output: [31, B, T, 1024]
        B, T = token_ids.shape
        out = torch.zeros(self.num_heads, B, T, self.weight.shape[2])
        for i in range(self.num_heads):
            out[i] = nn.functional.embedding(token_ids, self.weight[i])
        return out


class TokenizerDecoderTS(nn.Module):
    """Wrapper for Qwen3-TTS-Tokenizer-12Hz decoder.

    Wraps the internal decoder sub-module directly to avoid control flow
    issues in model.decode() that break torch.jit.trace.

    Input: audio codes (batch, num_quantizers, T) e.g. (1, 32, 100)
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
                     num_codebooks: int = 32,
                     sample_rate: int = 24000,
                     codec_vocab_size: int = 3072,
                     cp_vocab_size: int = 2048,
                     has_tokenizer_decoder: bool = False,
                     has_speaker_encoder: bool = False,
                     has_code_predictor: bool = False,
                     has_text_projection: bool = False,
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
            "attn_cnt": getattr(config, 'talker_config', config).num_hidden_layers if hasattr(getattr(config, 'talker_config', config), 'num_hidden_layers') else 20,
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
                "cp_vocab_size": cp_vocab_size,
                "sample_rate": sample_rate,
                "frame_rate": 12,
            },
        },
    }

    if has_tokenizer_decoder:
        model_json["params"]["tokenizer_decoder_param"] = "tokenizer_decoder.ncnn.param"
        model_json["params"]["tokenizer_decoder_bin"] = "tokenizer_decoder.ncnn.bin"

    if has_speaker_encoder:
        model_json["params"]["speaker_encoder_param"] = "speaker_encoder.ncnn.param"
        model_json["params"]["speaker_encoder_bin"] = "speaker_encoder.ncnn.bin"

    if has_code_predictor:
        model_json["params"]["cp_decoder_param"] = "cp_decoder.ncnn.param"
        model_json["params"]["cp_decoder_bin"] = "cp_decoder.ncnn.bin"
        model_json["params"]["cp_lm_heads_param"] = "cp_lm_heads.ncnn.param"
        model_json["params"]["cp_lm_heads_bin"] = "cp_lm_heads.ncnn.bin"
        model_json["params"]["cp_codec_embeds_param"] = "cp_codec_embeds.ncnn.param"
        model_json["params"]["cp_codec_embeds_bin"] = "cp_codec_embeds.ncnn.bin"
        cp_config = getattr(config, 'talker_code_predictor_config', None)
        if cp_config is not None:
            model_json["setting"]["cp_attn_cnt"] = getattr(cp_config, 'num_hidden_layers', 5)
            model_json["setting"]["cp_num_kv_heads"] = getattr(cp_config, 'num_key_value_heads', 8)
            model_json["setting"]["cp_num_heads"] = getattr(cp_config, 'num_attention_heads', 16)
            model_json["setting"]["cp_head_dim"] = getattr(cp_config, 'head_dim', 128)

    if has_text_projection:
        model_json["params"]["text_projection_param"] = "text_projection.ncnn.param"
        model_json["params"]["text_projection_bin"] = "text_projection.ncnn.bin"

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

    # Export text_projection (ResizeMLP: text_hidden_size → hidden_size)
    has_text_projection = False
    text_projection = getattr(talker, 'text_projection', None)
    if text_projection is not None:
        print("\nExporting text_projection...")
        text_hidden_size = talker_config.text_hidden_size if hasattr(talker_config, 'text_hidden_size') else 2048
        tp_ts = TextProjectionTS(text_projection)
        ex_tp = torch.randn(1, 8, text_hidden_size, device=device)
        export_to_ncnn(tp_ts, ex_tp, out_dir, "text_projection", device)
        has_text_projection = True

    # Export speaker encoder (ECAPA-TDNN: mel → speaker embedding)
    has_speaker_encoder = False
    speaker_encoder = getattr(model, 'speaker_encoder', None)
    if speaker_encoder is not None:
        print("\nExporting speaker_encoder...")
        mel_dim = 128
        ex_mel = torch.randn(1, 375, mel_dim, device=device)
        se_ts = SpeakerEncoderTS(speaker_encoder)
        try:
            export_to_ncnn(se_ts, ex_mel, out_dir, "speaker_encoder", device)
            has_speaker_encoder = True
        except Exception as e:
            print(f"  Speaker encoder export failed: {e}")
    else:
        print("\n  No speaker_encoder found (non-base model?), skipping.")

    # Export code predictor (5-layer transformer + 31 lm_heads + 31 codec_embeddings)
    has_code_predictor = False
    code_predictor = getattr(talker, 'code_predictor', None)
    if code_predictor is not None:
        cp_model = getattr(code_predictor, 'model', code_predictor)
        cp_config = getattr(config, 'talker_code_predictor_config', None)

        cp_num_layers = 5
        cp_num_kv_heads = 8
        cp_num_heads = 16
        cp_head_dim = 128
        cp_hidden_size = 1024
        cp_vocab_size = 2048
        num_code_groups = 32

        if cp_config is not None:
            cp_num_layers = getattr(cp_config, 'num_hidden_layers', 5)
            cp_num_kv_heads = getattr(cp_config, 'num_key_value_heads', 8)
            cp_num_heads = getattr(cp_config, 'num_attention_heads', 16)
            cp_head_dim = getattr(cp_config, 'head_dim', 128)
            cp_hidden_size = getattr(cp_config, 'hidden_size', 1024)
            cp_vocab_size = getattr(cp_config, 'vocab_size', 2048)
            num_code_groups = getattr(cp_config, 'num_code_groups', 32)

        num_sub_codebooks = num_code_groups - 1

        # Export code predictor decoder (5-layer transformer with KV cache)
        print(f"\nExporting code_predictor decoder ({cp_num_layers} layers)...")
        cp_decoder_ts = CodePredictorDecoderTS(cp_model, cp_num_layers, cp_num_kv_heads, cp_num_heads, cp_head_dim)

        cp_seq_len = 2
        cp_past_len = 0
        cp_ex_embed = torch.randn(1, cp_seq_len, cp_hidden_size, device=device)
        cp_ex_mask = torch.zeros(cp_seq_len, cp_seq_len + cp_past_len, device=device)
        for i in range(cp_seq_len):
            for j in range(i + 1, cp_seq_len + cp_past_len):
                cp_ex_mask[i][j] = -1e38
        cp_ex_cos = torch.randn(cp_seq_len, cp_head_dim, device=device)
        cp_ex_sin = torch.randn(cp_seq_len, cp_head_dim, device=device)
        cp_caches = []
        for _ in range(cp_num_layers):
            cp_caches.append(torch.randn(cp_num_kv_heads, cp_past_len, cp_head_dim, device=device))
            cp_caches.append(torch.randn(cp_num_kv_heads, cp_past_len, cp_head_dim, device=device))

        try:
            export_to_ncnn(cp_decoder_ts, [cp_ex_embed, cp_ex_mask, cp_ex_cos, cp_ex_sin] + cp_caches,
                           out_dir, "cp_decoder", device)
        except Exception as e:
            print(f"  Code predictor decoder export failed: {e}")

        # Export merged 31 lm_heads
        print(f"\nExporting code_predictor lm_heads ({num_sub_codebooks} heads)...")
        cp_lm_heads = getattr(code_predictor, 'lm_head', None)
        if cp_lm_heads is not None:
            lm_heads_ts = CodePredictorLmHeadsTS(cp_lm_heads, num_heads=num_sub_codebooks)
            ex_cp_hidden = torch.randn(1, 1, cp_hidden_size, device=device)
            try:
                export_to_ncnn(lm_heads_ts, ex_cp_hidden, out_dir, "cp_lm_heads", device)
            except Exception as e:
                print(f"  Code predictor lm_heads export failed: {e}")

        # Export merged 31 codec embeddings
        print(f"\nExporting code_predictor codec_embeddings ({num_sub_codebooks} embeds)...")
        cp_codec_embeds = getattr(cp_model, 'codec_embedding', None)
        if cp_codec_embeds is not None:
            embeds_ts = CodePredictorCodecEmbedsTS(cp_codec_embeds, num_heads=num_sub_codebooks)
            ex_cp_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
            try:
                export_to_ncnn(embeds_ts, ex_cp_ids, out_dir, "cp_codec_embeds", device)
            except Exception as e:
                print(f"  Code predictor codec_embeddings export failed: {e}")

        has_code_predictor = True
    else:
        print("\n  No code_predictor found, skipping.")

    # Export decoder
    print("\nExporting decoder (this may take a while)...")
    decoder_ts = DecoderTS(decoder_model, num_layers, num_kv_heads, num_heads, head_dim)

    seq_len = 8
    past_len = 4  # non-zero so use_cache=True path is traced
    ex_embed = torch.randn(1, seq_len, hidden_size, device=device)
    ex_mask = torch.zeros(seq_len, seq_len + past_len, device=device)
    for i in range(seq_len):
        for j in range(i + 1, seq_len + past_len):
            ex_mask[i][j] = -1e38

    ex_cos = torch.randn(seq_len, head_dim, device=device)
    ex_sin = torch.randn(seq_len, head_dim, device=device)

    # Non-empty KV cache: [num_kv_heads, past_len, head_dim]
    caches = []
    for _ in range(num_layers):
        caches.append(torch.randn(num_kv_heads, past_len, head_dim, device=device))
        caches.append(torch.randn(num_kv_heads, past_len, head_dim, device=device))

    try:
        export_to_ncnn(decoder_ts, [ex_embed, ex_mask, ex_cos, ex_sin] + caches,
                       out_dir, "decoder", device)
    except Exception as e:
        print(f"  Decoder export failed: {e}")
        print("  Try using pnnx CLI directly on the model.")

    # Extract tokenizer
    print("\nExtracting tokenizer...")
    extract_tokenizer(tokenizer, out_dir)

    return config, tokenizer, has_speaker_encoder, has_code_predictor, has_text_projection


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
        ex_codes = torch.randint(0, codebook_size, (1, 32, 100), dtype=torch.long, device=device)
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
    parser.add_argument("--num_codebooks", type=int, default=32,
                        help="Number of audio codebooks (num_code_groups)")
    parser.add_argument("--tts_model_type", type=str, default="base",
                        choices=["base", "custom_voice", "voice_design"],
                        help="TTS model type")

    args = parser.parse_args()

    config, tokenizer, has_speaker_encoder, has_code_predictor, has_text_projection = \
        export_llm(args.llm_model_id, args.out_dir, args.device)

    has_tokenizer_decoder = False
    if not args.skip_tokenizer:
        has_tokenizer_decoder = export_tokenizer_decoder(
            args.tokenizer_model_id, args.out_dir, args.device)

    # Read actual vocab sizes from config
    talker_config = getattr(config, 'talker_config', config)
    codec_vocab_size = getattr(talker_config, 'vocab_size', 3072)
    cp_config = getattr(config, 'talker_code_predictor_config', None)
    cp_vocab_size = getattr(cp_config, 'vocab_size', 2048) if cp_config else 2048

    print("\nBuilding model.json...")
    build_model_json(
        config, tokenizer, args.out_dir,
        num_codebooks=args.num_codebooks,
        codec_vocab_size=codec_vocab_size,
        cp_vocab_size=cp_vocab_size,
        has_tokenizer_decoder=has_tokenizer_decoder,
        has_speaker_encoder=has_speaker_encoder,
        has_code_predictor=has_code_predictor,
        has_text_projection=has_text_projection,
        tts_model_type=args.tts_model_type,
    )

    print(f"\n{'='*60}")
    print(f"Export complete! Output directory: {args.out_dir}")
    print(f"{'='*60}")
    print(f"\nPlace this directory under assets/ and run:")
    print(f"  xmake run tts_main --model assets/{os.path.basename(args.out_dir)} --text \"Hello world\"")
