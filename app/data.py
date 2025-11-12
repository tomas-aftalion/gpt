import urllib.request
import os

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

