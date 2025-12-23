import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=1):
        """
        hidden_dim: 注意力特征维度
        num_heads: 注意力头数
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
        x: (B, T, D)
        return: (B, T, D)
        """
        B, T, D = x.shape

        # QKV 投影
        qkv = self.qkv_proj(x)  # (B, T, 3*D)
        # reshape 成 (3, B, H, T, D_head)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv  # 每个都是 (B, H, T, D_head)

        # scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        x = torch.matmul(attn, V)  # (B, H, T, D_head)

        # merge heads
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
        x: (B, ..., input_dim)
        return: (B, T, D)
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

        self.attn = SelfAttention(hidden_dim, num_heads = num_heads)

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
        x: (B, T, D)
        """
        if self.pooling == "mean":
            x = x.mean(dim=1)
        elif self.pooling == "cls":
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
    



# class AttentionNet(nn.Module):
#     def __init__(self, input_dim=3072, hidden_dim=512, num_classes=100, num_tokens=32):
#         """
#         input_dim: 总输入维度（与 SimpleMLP 保持一致）
#         hidden_dim: attention 中的特征维度
#         num_classes: 分类数
#         num_tokens: 将 input_dim 切分成多少个 token
#         """
#         super(AttentionNet, self).__init__()

#         assert input_dim % num_tokens == 0
#         self.num_tokens = num_tokens
#         self.token_dim = input_dim // num_tokens

#         self.flatten = nn.Flatten()

#         # token embedding
#         self.token_proj = nn.Linear(self.token_dim, hidden_dim)

#         # self-attention (单头)
#         self.q_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.k_proj = nn.Linear(hidden_dim, hidden_dim)
#         self.v_proj = nn.Linear(hidden_dim, hidden_dim)

#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)

#         # classifier head
#         self.classifier = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         """
#         x: (B, ..., input_dim)
#         """
#         B = x.size(0)
#         x = self.flatten(x)                     # (B, input_dim)
#         x = x.view(B, self.num_tokens, -1)      # (B, T, token_dim)

#         x = self.token_proj(x)                  # (B, T, hidden_dim)

#         # self-attention
#         Q = self.q_proj(x)                      # (B, T, D)
#         K = self.k_proj(x)
#         V = self.v_proj(x)

#         attn = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
#         attn = F.softmax(attn, dim=-1)

#         x = torch.matmul(attn, V)               # (B, T, D)
#         x = self.out_proj(x)

#         # pooling (等价于 MLP 中的 flatten 后全连接)
#         x = x.mean(dim=1)                       # (B, D)

#         return self.classifier(x)







# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MultiHeadAttentionNet(nn.Module):
#     def __init__(
#         self,
#         input_dim=3072,
#         hidden_dim=512,
#         num_classes=100,
#         num_tokens=128,
#         num_heads=8
#     ):
#         super(MultiHeadAttentionNet, self).__init__()

#         assert input_dim % num_tokens == 0
#         assert hidden_dim % num_heads == 0

#         self.num_tokens = num_tokens
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads

#         self.flatten = nn.Flatten()

#         # token embedding
#         self.token_dim = input_dim // num_tokens
#         self.token_proj = nn.Linear(self.token_dim, hidden_dim)

#         # QKV projection（一次性投影，效率更高）
#         self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)

#         # output projection
#         self.out_proj = nn.Linear(hidden_dim, hidden_dim)

#         # classifier
#         self.classifier = nn.Linear(hidden_dim, num_classes)

#     def forward(self, x):
#         """
#         x: (B, ..., input_dim)
#         """
#         B = x.size(0)

#         # flatten + tokenize
#         x = self.flatten(x)                        # (B, input_dim)
#         x = x.view(B, self.num_tokens, -1)         # (B, T, token_dim)
#         x = self.token_proj(x)                     # (B, T, hidden_dim)

#         # QKV
#         qkv = self.qkv_proj(x)                     # (B, T, 3*hidden_dim)
#         qkv = qkv.view(
#             B, self.num_tokens, 3, self.num_heads, self.head_dim
#         ).permute(2, 0, 3, 1, 4)
#         # qkv: (3, B, H, T, D_head)

#         Q, K, V = qkv[0], qkv[1], qkv[2]

#         # scaled dot-product attention
#         attn = torch.matmul(Q, K.transpose(-2, -1))
#         attn = attn / (self.head_dim ** 0.5)
#         attn = F.softmax(attn, dim=-1)

#         x = torch.matmul(attn, V)                  # (B, H, T, D_head)

#         # merge heads
#         x = x.transpose(1, 2).contiguous()         # (B, T, H, D_head)
#         x = x.view(B, self.num_tokens, -1)         # (B, T, hidden_dim)

#         x = self.out_proj(x)                       # (B, T, hidden_dim)

#         # global pooling (对应 MLP 的 flatten + FC)
#         x = x.mean(dim=1)                          # (B, hidden_dim)

#         return self.classifier(x)
