# backend/levit_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# --- PatchEmbedding, MultiHeadSelfAttention, FeedForward, TransformerBlock ---
# (These classes remain unchanged from the version that added return_attention)
class PatchEmbedding(nn.Module):
    """Converts images into token embeddings."""
    def __init__(self, in_channels=3, embed_dim=128, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        self.H_feat, self.W_feat = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadSelfAttention(nn.Module):
    """Efficient Self-Attention Mechanism"""
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        x = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if return_attention:
            return x, attn_probs
        else:
            return x

class FeedForward(nn.Module):
    """Feedforward Network for feature transformation"""
    def __init__(self, embed_dim, expansion=4):
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Attention + FFN"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim)

    def forward(self, x, return_attention=False):
        processed_attn, attn_probs = self.attn(self.norm1(x), return_attention=True)
        x = x + processed_attn
        x = x + self.ffn(self.norm2(x))
        if return_attention:
            return x, attn_probs
        else:
            return x
# --- End of unchanged classes ---


class LeViT(nn.Module):
    """Custom LeViT Model with Dropout"""
    # *** MODIFICATION START ***
    # Added dropout_rate parameter with a default
    def __init__(self, img_size=224, num_classes=2, embed_dim=128, depth=4, num_heads=4, patch_size=16, dropout_rate=0.5): # Added dropout_rate
        super().__init__()
        self.patch_embed = PatchEmbedding(embed_dim=embed_dim, patch_size=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Define the dropout layer using the provided rate
        self.dropout = nn.Dropout(dropout_rate) # Defined dropout layer

        self.classifier = nn.Linear(embed_dim, num_classes)
    # *** MODIFICATION END ***

    def forward(self, x, return_attention=False):
        x = self.patch_embed(x)

        attn_map = None
        for i, blk in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                x, attn_map = blk(x, return_attention=True)
            else:
                x = blk(x, return_attention=False)

        x = self.norm(x)
        x = x.mean(dim=1)

        # *** MODIFICATION START ***
        # Apply dropout layer before the classifier during training
        x = self.dropout(x)
        # *** MODIFICATION END ***

        output = self.classifier(x)

        if return_attention:
            if attn_map is None:
                 print("Warning: Attention map was requested but not captured from the last block.")
            return output, attn_map
        else:
            return output