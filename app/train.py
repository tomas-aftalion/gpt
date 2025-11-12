from . import data
from .model import GPT
import torch

if __name__ == "__main__":
    # Get and tokenize text
    CONTEXT = 256
    BATCH = 4
    CHANNELS = 384
    HEADS = 6
    LAYERS = 6

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
        dropout=0.2
    )
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Gradient accumulation
    steps = 16  # Accumulate gradients over N mini-batches
    effective_batch_size = BATCH * steps  # 32 * 2 = 64 (matches Karpathy)
    
    # Training loop
    num_steps = 5000
    for step in range(num_steps):
        # Zero gradients at start of accumulation
        optimizer.zero_grad()
        
        total_loss = 0
        
        # Accumulate gradients over multiple mini-batches
        for micro_step in range(steps):
            # Get mini-batch
            x, y = train_loader.get_batch()  # x: (B, T), y: (B, T)
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            loss, logits = model(x, y=y)
            
            # Scale loss by accumulation steps
            loss = loss / steps
            total_loss += loss.item() * steps  # For logging (scale back up)
            
            # Backward pass (accumulates gradients)
            loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights (only after all accumulation steps)
        optimizer.step()
        
        # Print loss (average over accumulation steps)
        print(f"Step {step}: loss = {total_loss / steps:.4f}")
