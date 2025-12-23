import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention, TokenEmbedding, ClassificationHead

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Standard Transformer uses GELU activation
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
        # Residual connection with Pre-Norm structure
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim=3072,      # Raw input dimension (e.g., 32x32x3 for images)
        hidden_dim=512,     # Latent dimension within Transformer
        num_tokens=64,      # Sequence length (T)
        num_heads=8,        # Number of attention heads
        num_layers=6,       # Number of stacked Transformer blocks
        mlp_dim=1024,       # Hidden dimension of FFN (usually 2x-4x hidden_dim)
        num_classes=100,    # Total target classes (Must match dataset labels)
        dropout=0.1
    ):
        super().__init__()
        
        # 1. Token Embedding Layer (utilizing your custom TokenEmbedding)
        self.embedding = TokenEmbedding(input_dim, hidden_dim, num_tokens)
        
        # 2. Positional Encoding (Learnable parameters to provide sequence order)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))
        
        # 3. Transformer Encoder Stack
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Final Normalization (applied before pooling)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 5. Output Head (utilizing your custom ClassificationHead)
        self.head = ClassificationHead(hidden_dim, num_classes, pooling="mean")

    def forward(self, x):
        # x: (B, input_dim) -> (B, T, D)
        x = self.embedding(x)
        
        # Add positional information to tokens
        x = x + self.pos_embedding
        
        # Pass through stacked Transformer Blocks
        for layer in self.layers:
            x = layer(x)
            
        # Global normalization
        x = self.norm(x)
        
        # Final classification logits
        return self.head(x)