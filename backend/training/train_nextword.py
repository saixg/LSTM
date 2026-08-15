import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from backend.app.models.nextword_model import NextWordLSTM

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "tech_corpus.txt")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "checkpoints")

def build_vocab(text):
    words = text.lower().split()
    counter = Counter(words)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    # For a tiny corpus, keep everything
    for word, _ in counter.items():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab, words

def train_nextword():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    with open(DATA_FILE, "r") as f:
        text = f.read()
        
    vocab, words = build_vocab(text)
    vocab_size = len(vocab)
    
    vocab_path = os.path.join(CHECKPOINT_DIR, "nextword_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
        
    encoded = [vocab[w] for w in words]
    seq_length = 5 # small sequence length for fast training
    
    # Create dataset pairs
    X = []
    Y = []
    for i in range(len(encoded) - seq_length):
        X.append(encoded[i:i+seq_length])
        Y.append(encoded[i+1:i+seq_length+1])
        
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    
    model = NextWordLSTM(vocab_size=vocab_size, embedding_dim=128, hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 20
    model.train()
    
    batch_size = 64
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size].t() # (seq_len, batch_size)
            batch_y = Y[i:i+batch_size].t()
            
            optimizer.zero_grad()
            logits, _ = model(batch_x, tokens=[""]*batch_x.size(0))
            
            # logits: (seq_len, batch_size, vocab_size)
            # Flatten for CrossEntropy
            loss = criterion(logits.view(-1, vocab_size), batch_y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(1)
            
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(X):.4f}")
        
    save_path = os.path.join(CHECKPOINT_DIR, "nextword_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    print("Training Next-Word Predictor...")
    train_nextword()
