from . import data
from .model import GPT
import torch

if __name__ == "__main__":
    # Get and tokenize text
    CONTEXT = 8
    BATCH = 4
    CHANNELS = 4
    HEADS = 1
    LAYERS = 1

    text = data.get_shakespeare()
    tokenizer = data.Tokenizer(text)
    tokens = tokenizer.encode(text)
    vocab_size = tokenizer.size
    
    # Split train/test
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    test_tokens = tokens[split_idx:]
    
    # Create dataloaders
    train_loader = data.GPTDataLoader(train_tokens, context_length=CONTEXT, batch_size=BATCH)
    test_loader = data.GPTDataLoader(test_tokens, context_length=CONTEXT, batch_size=BATCH)
    
    # Create model
    model = GPT(
        vocab_size=vocab_size,
        channel_size=CHANNELS,
        head_size=HEADS,
        layer_size=LAYERS,
        context_size=CONTEXT,
        dropout=0.1
    )
    
    # Get a batch
    x, y = train_loader.get_batch()  # x: (B, T, T), y: (B, T)
    
    B, T, T = x.shape
    x = x.view(B * T, T)  # (B*T, T)
    y = y.view(B * T)  # (B*T,)
    
    # Forward pass
    logits = model(x)  # (B*T, T, vocab_size)
    print(f"Reshaped shapes - x: {x.shape}, y: {y.shape}")
    print(f"Logits shape: {logits.shape}")
