"""Download TinyStories and train the shared BPE tokenizer -> data/*.npy, data/tok.json"""
from llmseries.data import prep_data

if __name__ == "__main__":
    prep_data()
