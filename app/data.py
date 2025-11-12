import urllib.request
import os
import random
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE_FILE = "shakespeare.txt"

def get_shakespeare():
    if not os.path.exists(CACHE_FILE):
        urllib.request.urlretrieve(URL, CACHE_FILE)
    with open(CACHE_FILE) as f:
        return f.read()

class Tokenizer:

    def __init__(self, words):
        chars = sorted(list(set(words)))
        stoi = { c:i for i,c in enumerate(chars) }
        itos = { i:c for i,c in enumerate(chars) }
        self.encode = lambda s: [ stoi[c] for c in s ]
        self.decode = lambda l: "".join([ itos[i] for i in l ])
        self.size = len(stoi)


class GPTDataLoader:
    def __init__(self, tokens, context_length=256, batch_size=64):
        self.tokens = tokens
        self.context_length = context_length
        self.batch_size = batch_size
        
    def get_batch(self):
        # Sample B sequences of length context_length + 1
        max_start = len(self.tokens) - self.context_length - 1
        starts = [random.randint(0, max_start) for _ in range(self.batch_size)]
        
        x = []
        y = []
        for start in starts:
            # Grab full sequence of length context_length + 1
            seq = self.tokens[start:start + self.context_length + 1]
            
            x_batch = []
            y_batch = []
            # Generate all pairs for t = 1, 2, ..., context_length
            for t in range(1, self.context_length + 1):
                x_seq = seq[:t]      # sequence of length t
                y_token = seq[t]      # single token (next character)
                
                # Pad x_seq to context_length
                x_padded = x_seq + [0] * (self.context_length - len(x_seq))
                x_batch.append(x_padded)
                y_batch.append(y_token)
            
            x.append(x_batch)
            y.append(y_batch)
        
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.long)  # (B, context_length, context_length)
        y_tensor = torch.tensor(y, dtype=torch.long)  # (B, context_length)
        
        return x_tensor, y_tensor


