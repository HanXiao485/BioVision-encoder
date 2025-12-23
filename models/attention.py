import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=1):
        """
        hidden_dim: Dimension of the attention features
        num_heads: Number of attention heads
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # QKV projection
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        x: Input tensor of shape (B, T, D)
        return: Output tensor of shape (B, T, D)
        """
        B, T, D = x.shape

        # QKV Projection
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        
        # Reshape to (3, B, H, T, D_head) and permute for multi-head calculation
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv  # Each tensor has shape (B, H, T, D_head)

        # Scaled dot-product attention
        # (B, H, T, D_head) @ (B, H, D_head, T) -> (B, H, T, T)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # (B, H, T, T) @ (B, H, T, D_head) -> (B, H, T, D_head)
        x = torch.matmul(attn, V)  

        # Merge heads
        x = x.transpose(1, 2).contiguous()  # (B, T, H, D_head)
        x = x.view(B, T, self.hidden_dim)   # (B, T, D)

        return self.out_proj(x)


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tokens):
        super().__init__()
        assert input_dim % num_tokens == 0

        self.num_tokens = num_tokens
        self.token_dim = input_dim // num_tokens

        self.flatten = nn.Flatten()
        self.proj = nn.Linear(self.token_dim, hidden_dim)

    def forward(self, x):
        """
        x: Input tensor of shape (B, ..., input_dim)
        return: Output tensor of shape (B, T, D)
        """
        B = x.size(0)
        x = self.flatten(x)
        x = x.view(B, self.num_tokens, self.token_dim)
        return self.proj(x)


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tokens, num_heads):
        super().__init__()

        self.embedding = TokenEmbedding(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens
        )

        self.attn = SelfAttention(hidden_dim, num_heads=num_heads)

    def forward(self, x):
        x = self.embedding(x)     # (B, T, D)
        x = self.attn(x)          # (B, T, D)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, pooling="mean"):
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: Input tensor of shape (B, T, D)
        """
        if self.pooling == "mean":
            # Global Average Pooling across tokens
            x = x.mean(dim=1)
        elif self.pooling == "cls":
            # Use the first token for classification
            x = x[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.fc(x)
    

class AttentionNet(nn.Module):
    def __init__(
        self,
        input_dim=3072,
        hidden_dim=512,
        num_classes=100,
        num_tokens=32,
        num_heads=1
    ):
        super().__init__()

        self.encoder = AttentionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            num_heads=num_heads
        )

        self.head = ClassificationHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(self, x):
        x = self.encoder(x)   # (B, T, D)
        return self.head(x)