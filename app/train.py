from . import data

if __name__ == "__main__":
    # Get and tokenize text
    text = data.get_shakespeare()
    tokenizer = data.Tokenizer(text)
    tokens = tokenizer.encode(text)
    
    # Split train/test
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    test_tokens = tokens[split_idx:]
    
    # Create dataloaders
    train_loader = data.GPTDataLoader(train_tokens, context_length=8, batch_size=4)
    
    # Get a batch
    x, y = train_loader.get_batch()  # x: (B, context_length, context_length), y: (B, context_length)
    
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")
    print(x)
