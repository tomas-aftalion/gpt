from . import data
from .model import GPT
import torch

if __name__ == "__main__":
    # Get and tokenize text
    CONTEXT = 20
    BATCH = 32
    CHANNELS = 8
    HEADS = 4
    LAYERS = 2

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
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = GPT(
        vocab_size=vocab_size,
        channel_size=CHANNELS,
        head_size=HEADS,
        layer_size=LAYERS,
        context_size=CONTEXT,
        dropout=0.1
    )
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Training loop
    num_steps = 1000
    for step in range(num_steps):
        # Get batch
        x, y = train_loader.get_batch()  # x: (B, T, T), y: (B, T)
        
        # Reshape (reuse x and y)
        B, T, _ = x.shape
        x = x.view(B * T, T).to(device)  # (B*T, T)
        y = y.view(B * T).to(device)  # (B*T,)
        
        # Forward pass - model returns loss and logits
        loss, logits = model(x, y=y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Print loss
        print(f"Step {step}: loss = {loss.item():.4f}")
