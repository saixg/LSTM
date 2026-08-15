import os
import json
import torch

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "app", "checkpoints")

def export_model_weights(pt_name, json_name):
    pt_path = os.path.join(CHECKPOINT_DIR, pt_name)
    json_path = os.path.join(CHECKPOINT_DIR, json_name)
    if not os.path.exists(pt_path):
        print(f"File not found: {pt_path}")
        return
    
    state_dict = torch.load(pt_path, weights_only=True)
    converted = {}
    for k, v in state_dict.items():
        converted[k] = v.cpu().detach().tolist()
        
    with open(json_path, "w") as f:
        json.dump(converted, f)
    print(f"Exported {pt_name} -> {json_name} ({os.path.getsize(json_path)/1024:.1f} KB)")

if __name__ == "__main__":
    export_model_weights("nextword_model.pt", "nextword_weights.json")
    export_model_weights("anomaly_model.pt", "anomaly_weights.json")
    export_model_weights("sentiment_model.pt", "sentiment_weights.json")
