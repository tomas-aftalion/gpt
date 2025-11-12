import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, channel_size, head_size):
        super().__init__()
        assert channel_size % head_size == 0
        self.channel_size = channel_size
        self.head_size = head_size
        self.d_head = channel_size // head_size
        
        # Project to Q, K, V for all heads at once
        self.q = nn.Linear(channel_size, channel_size)
        self.k = nn.Linear(channel_size, channel_size)
        self.v = nn.Linear(channel_size, channel_size)
        self.output = nn.Linear(channel_size, channel_size)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize attention layer weights."""
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.q.bias)
        nn.init.zeros_(self.k.bias)
        nn.init.zeros_(self.v.bias)
        nn.init.zeros_(self.output.bias)
        
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
        
        # Project Q, K, V: (B*T, T, C) -> (B*T, T, C)
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        
        # Reshape for multi-head: (B*T, T, C) -> (B*T, head_size, T, d_head)
        Q = Q.view(B, T, self.head_size, self.d_head).transpose(1, 2)  # (B*T, head_size, T, d_head)
        K = K.view(B, T, self.head_size, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.head_size, self.d_head).transpose(1, 2)
        
        # Attention: (B*T, head_size, T, d_head) x (B*T, head_size, d_head, T) -> (B*T, head_size, T, T)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # Apply mask (broadcasts to all heads)
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)  # (B*T, head_size, T, T)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B*T, head_size, T, d_head)
        
        # Concatenate heads: (B*T, head_size, T, d_head) -> (B*T, T, head_size, d_head) -> (B*T, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        out = self.output(out)  # (B*T, T, C)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, channel_size, head_size, d_ff=None, dropout=0.1):
        super().__init__()
        self.channel_size = channel_size
        d_ff = d_ff or (4 * channel_size)  # Default: 4x channel_size
        
        # Multi-head attention
        self.attention = MultiHeadAttention(channel_size, head_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(channel_size, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, channel_size)
        )
        
        # Layer normalization (pre-norm: normalize before sub-layer)
        self.norm1 = nn.LayerNorm(channel_size)
        self.norm2 = nn.LayerNorm(channel_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize FFN layer weights."""
        nn.init.kaiming_normal_(self.ffn[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.xavier_uniform_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[0].bias)
        nn.init.zeros_(self.ffn[2].bias)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B*T, T, C) - input embeddings
            mask: Optional attention mask
        Returns:
            out: (B*T, T, C) - output embeddings
        """
        # Self-attention with residual connection (pre-norm)
        x_norm = self.norm1(x)  # (B*T, T, C)
        attn_out = self.attention(x_norm, mask)  # (B*T, T, C)
        x = x + self.dropout(attn_out)  # Residual connection
        
        # Feed-forward with residual connection (pre-norm)
        x_norm = self.norm2(x)  # (B*T, T, C)
        ffn_out = self.ffn(x_norm)  # (B*T, T, C)
        x = x + self.dropout(ffn_out)  # Residual connection
        
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, channel_size=128, head_size=8, layer_size=6, 
                 context_size=256, d_ff=None, dropout=0.1, criterion=None):
        super().__init__()
        self.channel_size = channel_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, channel_size)
        
        # Positional embedding
        self.pos_embedding = nn.Embedding(context_size, channel_size)
        
        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(channel_size, head_size, d_ff, dropout)
            for _ in range(layer_size)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(channel_size)
        
        # Output projection
        self.output = nn.Linear(channel_size, vocab_size)
        
        # Loss function (default CrossEntropyLoss with ignore_index=-1)
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=-1)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize output layer weights."""
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        
    def forward(self, x, y=None, mask=None):
        """
        Args:
            x: (B, T) - token indices
            y: (B, T) - target tokens for all positions (optional)
            mask: Optional attention mask
        Returns:
            If y provided: (loss, logits)
            If y not provided: logits
        """
        B, T = x.shape
        device = x.device
        
        # Token embeddings: (B, T) -> (B, T, C)
        tok_emb = self.embedding(x)
        
        # Position embeddings: (T,) -> (T, C)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.pos_embedding(pos)
        
        # Add and apply dropout: (B, T, C) + (T, C) -> (B, T, C)
        x = self.embed_dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)  # (B, T, C) -> (B, T, C)
        
        # Final layer norm
        x = self.norm(x)  # (B, T, C)
        
        # Output projection: (B, T, C) -> (B, T, vocab_size)
        logits = self.output(x)
        
        # If targets provided, compute loss at all positions
        if y is not None:
            # Compute loss at all positions (like Karpathy)
            loss = self.criterion(logits.view(-1, self.vocab_size), y.view(-1))
            return loss, logits
        
        return logits
