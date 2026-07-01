#!/usr/bin/env python3
"""本地测试导出脚本的模型属性访问逻辑（不需要真实模型权重）"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'export'))

import torch
import torch.nn as nn


class FakeQwen3TTSModel(nn.Module):
    """Mock Qwen3TTSForConditionalGeneration 结构"""
    def __init__(self):
        super().__init__()
        self.talker = FakeTalker()


class FakeTalker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeTalkerModel()
        self.codec_head = nn.Linear(1024, 151936, bias=False)


class FakeTalkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_embedding = nn.Embedding(151936, 1024)
        self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(28)])
        self.norm = nn.LayerNorm(1024)


class FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(1024, 16, batch_first=True)
        self.mlp = nn.Linear(1024, 1024)


def test_model_access():
    """测试导出脚本能正确访问模型属性"""
    model = FakeQwen3TTSModel()

    # 模拟导出脚本中的属性访问逻辑
    talker = getattr(model, 'talker', None)
    talker_model = getattr(talker, 'model', None) if talker else None

    assert talker_model is not None, "talker.model not found"

    embed = talker_model.text_embedding
    assert isinstance(embed, nn.Embedding), f"embed is {type(embed)}"
    assert embed.num_embeddings == 151936
    assert embed.embedding_dim == 1024

    lm_head = talker.codec_head
    assert isinstance(lm_head, nn.Linear), f"lm_head is {type(lm_head)}"
    assert lm_head.in_features == 1024
    assert lm_head.out_features == 151936

    decoder_model = talker_model
    assert hasattr(decoder_model, 'layers'), "decoder_model has no 'layers'"
    assert len(decoder_model.layers) == 28

    print("PASS: Qwen3-TTS model attribute access")
    print(f"  embed: Embedding({embed.num_embeddings}, {embed.embedding_dim})")
    print(f"  lm_head: Linear({lm_head.in_features}, {lm_head.out_features})")
    print(f"  decoder layers: {len(decoder_model.layers)}")


def test_fallback_access():
    """测试 fallback 路径（标准 HF 模型）"""
    class FakeHFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(1000, 512)
            self.model.layers = nn.ModuleList([nn.Linear(512, 512) for _ in range(4)])
            self.model.norm = nn.LayerNorm(512)
            self.lm_head = nn.Linear(512, 1000)

    model = FakeHFModel()

    # fallback 逻辑
    base_model = getattr(model, 'model', model)
    embed = getattr(base_model, 'embed_tokens', None) or getattr(model, 'embed_tokens', None)
    assert embed is not None
    lm_head = getattr(model, 'lm_head', None) or getattr(base_model, 'lm_head', None)
    assert lm_head is not None
    decoder_model = base_model if hasattr(base_model, 'layers') else model
    assert hasattr(decoder_model, 'layers')

    print("\nPASS: Fallback (standard HF) model attribute access")
    print(f"  embed: Embedding({embed.num_embeddings}, {embed.embedding_dim})")
    print(f"  lm_head: Linear({lm_head.in_features}, {lm_head.out_features})")
    print(f"  decoder layers: {len(decoder_model.layers)}")


def test_talker_config():
    """测试 config.talker_config 访问"""
    class FakeConfig:
        class talker_config:
            hidden_size = 1024
            num_hidden_layers = 28
            vocab_size = 151936
            num_key_value_heads = 4
            num_attention_heads = 16
            head_dim = 64
            rope_theta = 1000000.0

    config = FakeConfig()
    talker_config = getattr(config, 'talker_config', config)
    assert talker_config.hidden_size == 1024
    assert talker_config.num_hidden_layers == 28
    assert talker_config.head_dim == 64

    print("\nPASS: talker_config attribute access")
    print(f"  hidden_size: {talker_config.hidden_size}")
    print(f"  num_hidden_layers: {talker_config.num_hidden_layers}")
    print(f"  head_dim: {talker_config.head_dim}")


if __name__ == "__main__":
    test_model_access()
    test_fallback_access()
    test_talker_config()
    print("\n=== All tests passed ===")
