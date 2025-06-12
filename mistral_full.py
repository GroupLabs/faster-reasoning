import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

@dataclass
class MistralConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 4096

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

class MistralAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_probs = torch.softmax(attn_weights, dim=-1)
        attn_output = attn_probs @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

class MistralMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.up_proj(x)) * self.gate_proj(x))

class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.self_attn = MistralAttention(config.hidden_size, config.num_attention_heads)
        self.mlp = MistralMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states

class MistralModel(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.config = config

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class MistralForCausalLM(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = self.model(input_ids, mask)
        return self.lm_head(hidden_states)


def load_pretrained(model_name: str = MODEL_NAME) -> MistralForCausalLM:
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    cfg = hf_model.config
    config = MistralConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        max_position_embeddings=cfg.max_position_embeddings,
    )
    model = MistralForCausalLM(config)
    model.load_state_dict(hf_model.state_dict())
    return model


def print_layer_summary(model: MistralForCausalLM) -> None:
    for idx, layer in enumerate(model.model.layers):
        print(f"Layer {idx}: attn heads={layer.self_attn.num_heads}, hidden={model.model.config.hidden_size}")


if __name__ == "__main__":
    model = load_pretrained()
    print_layer_summary(model)
