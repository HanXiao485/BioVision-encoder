import torch
import torch.nn as nn
from models.attention import SelfAttention

class PatchEmbedding(nn.Module):
    """
    Standard ViT Patch Embedding
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=512
    ):
        super().__init__()
        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)            # (B, D, H/P, W/P)
        x = x.flatten(2)            # (B, D, N)
        x = x.transpose(1, 2)       # (B, N, D)
        return x


# ====== MLP & TransformerBlock ======

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ====== Patch Embedding ======

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)          # (B, D, H/P, W/P)
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


# ====== Vision Transformer ======

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        embed_dim=512,
        num_layers=6,
        num_heads=8,
        mlp_dim=1024,
        dropout=0.1
    ):
        super().__init__()

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2. CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Positional Embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.pos_dropout = nn.Dropout(dropout)

        # 4. Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # 5. Final Norm
        self.norm = nn.LayerNorm(embed_dim)

        # 6. Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)

        x = self.patch_embed(x)           # (B, N, D)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, D)

        x = x + self.pos_embedding
        x = self.pos_dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        cls_out = x[:, 0]                # CLS token
        return self.head(cls_out)


