from . import data

if __name__ == "__main__":
    words = data.get_shakespeare()
    tokenizer = data.Tokenizer(words)
    x = tokenizer.encode("hello")
    w = tokenizer.decode(x)
    print(w)
