from dataclasses import dataclass
from typing import List, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..blocks import FourierFeatures

@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None

# class FourierFeatures(nn.Module):
#     def __init__(self, out_features: int):
#         super().__init__()
#         # 256 features -> 128 frequencies (cos + sin)
#         self.register_buffer('weight', torch.randn(out_features // 2) * 2 * np.pi)

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         # Expected input shape: [B] or [B, 1]
#         if input.dim() == 1:
#             input = input.unsqueeze(-1)
#         f = input * self.weight
#         return torch.cat([f.cos(), f.sin()], dim=-1)


class DiT(nn.Module):
    def __init__(self, in_channels, out_channels, num_actions, num_steps_conditioning, 
                 depth=2, num_heads=2, mlp_ratio=2.0, hidden_dim=192, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        img_size = 64  # Added: Standard Atari resolution
        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d((num_steps_conditioning + 1) * in_channels, 
                                     hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Positional Embedding
        # New dynamic version
        num_patches = (img_size // patch_size) ** 2  # (64 // 4)**2 = 256
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.hidden_dim))
        
        # 3. Conditionings
        self.noise_emb = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU()
        )
        self.act_emb = nn.Embedding(num_actions, 256)
        
        self.cond_proj = nn.Sequential(
            nn.Linear(512, self.hidden_dim),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.unpatch = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=patch_size, stride=patch_size)
        
        self.initialize_weights()

    def forward(self, noisy_next_obs, c_noise, obs, act):
        B = noisy_next_obs.shape[0]
        
        # Patchify
        x = torch.cat([obs, noisy_next_obs], dim=1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        
        if c_noise.numel() == 1:
            # Case: Single scalar for the whole batch (Common in Samplers/AC loop)
            # Expand [1] -> [B, 1]
            c_noise_reshaped = c_noise.view(1, 1).expand(B, 1)
        else:
            # Case: One noise value per batch member (Common in Training loop)
            # Ensure shape is [B, 1]
            c_noise_reshaped = c_noise.view(B, 1)

        n_emb = self.noise_emb(c_noise_reshaped)  # Now guaranteed to be [B, 256]
        n_emb = n_emb.view(B, 256)               # Final safety flatten
        
        # Action Embedding - Always ensure [B, 256] output
        if act.dtype != torch.long:
            act = act.long()
        
        a_emb = self.act_emb(act)  # Embed actions
        while a_emb.dim() > 2:  # Reduce any extra dims
            a_emb = a_emb.mean(dim=1)
        
        # Ensure correct shape
        assert n_emb.shape == (B, 256), f"noise_emb shape {n_emb.shape}"
        assert a_emb.shape == (B, 256), f"act_emb shape {a_emb.shape}"
        
        # Concatenate Noise + Action
        cond_input = torch.cat([n_emb, a_emb], dim=1)  # [B, 512]
        cond = self.cond_proj(cond_input)
        
        for block in self.blocks:
            x = block(x, cond)
            
        x = self.norm(x).transpose(1, 2).reshape(B, -1, 64//self.patch_size, 64//self.patch_size)
        return self.unpatch(x)

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN[-1].weight, 0)
            nn.init.constant_(block.adaLN[-1].bias, 0)


class DiTBlock(nn.Module):
    """Transformer block with adaptive LayerNorm"""
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim)
        )
        
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim)
        )
        
    def forward(self, x, cond):
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(cond).chunk(6, dim=1)
        
        # Self-attention
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


# Dummy InnerModel (replaced by DiT in trainer.py)
class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.num_steps_conditioning = cfg.num_steps_conditioning
        self.dummy = nn.Linear(1, 1)
        
    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        return noisy_next_obs