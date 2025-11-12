import torch
import torch.nn as nn
import math

class AttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B*T, T, C) - input embeddings
            mask: Optional attention mask. Can be:
                - None: uses default causal mask (T, T)
                - (T, T): custom pattern, same for all examples (broadcasts)
                - (B*T, T): per-example padding mask (broadcasts to B*T, T, T)
                - (B*T, T, T): full per-example control
        Returns:
            out: (B*T, T, C) - output embeddings
        """
        B, T, C = x.shape
        
        Q = self.q(x)  # (B*T, T, C)
        K = self.k(x)  # (B*T, T, C)
        V = self.v(x)  # (B*T, T, C)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)  # (B*T, T, T)
        
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()  # (T, T)

        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)  # (B*T, T, T)
        out = torch.matmul(attn, V)  # (B*T, T, C)
        
        return out


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, context_length=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attention = AttentionHead(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        Args:
            x: (B*T, T) - token indices
        Returns:
            logits: (B*T, T, vocab_size)
        """
        x_emb = self.embedding(x)  # (B*T, T) -> (B*T, T, C)
        attn_out = self.attention(x_emb)  # (B*T, T, C) -> (B*T, T, C)
        logits = self.output(attn_out)  # (B*T, T, C) -> (B*T, T, vocab_size)
        
        return logits
