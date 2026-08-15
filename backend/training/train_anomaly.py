import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from backend.app.models.anomaly_model import AnomalyAutoencoder

DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "anomaly_samples.json")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "checkpoints")

def train_anomaly():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Load synthetic normal data for training
    with open(DATA_FILE, "r") as f:
        samples = json.load(f)
        
    normal_data = [torch.tensor(s["data"], dtype=torch.float32).unsqueeze(-1) for s in samples if s["type"] == "normal"]
    if not normal_data:
        print("No normal data found!")
        return

    # Train on normal samples only
    model = AnomalyAutoencoder(input_size=1, hidden_size=32)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    epochs = 150
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for seq in normal_data:
            # seq shape: (seq_len, 1)
            optimizer.zero_grad()
            reconstructed, _ = model(seq)
            loss = criterion(reconstructed.squeeze(1), seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(normal_data):.4f}")
            
    # Save checkpoint
    save_path = os.path.join(CHECKPOINT_DIR, "anomaly_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    print("Training Anomaly Autoencoder...")
    train_anomaly()
