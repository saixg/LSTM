import os
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from backend.app.models.sentiment_model import SentimentLSTM

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "sentiment_samples.json")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "checkpoints")

def tokenize(text: str):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.items():
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    tokens = tokenize(text)
    return [vocab.get(w, vocab["<UNK>"]) for w in tokens]

def train_sentiment():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    with open(DATA_FILE, "r") as f:
        samples = json.load(f)
        
    texts = [s["text"] for s in samples]
    labels = [s["label"] for s in samples]
    trajectories = [s["trajectory"] for s in samples]
    
    vocab = build_vocab(texts)
    vocab_size = len(vocab)
    print(f"Built sentiment vocabulary with {vocab_size} tokens.", flush=True)
    
    # Save vocab for inference
    vocab_path = os.path.join(CHECKPOINT_DIR, "sentiment_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    
    model = SentimentLSTM(vocab_size=vocab_size, embedding_dim=64, hidden_size=128)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    epochs = 8
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        for text, label, traj in zip(texts, labels, trajectories):
            token_ids = encode(text, vocab)
            if len(token_ids) == 0: continue
            
            encoded = torch.tensor(token_ids, dtype=torch.long)
            
            # Ensure target trajectory length matches encoded length
            if len(traj) != len(token_ids):
                target_traj = torch.full((len(token_ids),), float(label), dtype=torch.float32)
            else:
                target_traj = torch.tensor(traj, dtype=torch.float32)
                
            optimizer.zero_grad()
            running_scores, _ = model(encoded, tokens=[""] * len(token_ids))
            
            # running_scores shape: (seq_len, 1) or (seq_len,)
            scores_flat = running_scores.view(-1)
            loss = criterion(scores_flat, target_traj)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            final_pred = scores_flat[-1].item()
            if (final_pred >= 0.5) == bool(label):
                correct += 1
                
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(texts):.4f}, Acc: {correct/len(texts):.4f}", flush=True)
        
    save_path = os.path.join(CHECKPOINT_DIR, "sentiment_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}", flush=True)

if __name__ == "__main__":
    print("Training Enhanced Sentiment Classifier...")
    train_sentiment()
