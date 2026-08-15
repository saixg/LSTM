import os
import json
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from ..schemas import GateStep
from ..models.fast_lstm_engine import FastAnomalyModel

router = APIRouter(prefix="/api/anomaly", tags=["anomaly"])

MODEL = None
SAMPLES = []

class AnomalyDetectRequest(BaseModel):
    sequence_id: str
    inject_anomaly: bool

class AnomalyDetectResponse(BaseModel):
    input: List[float]
    reconstruction: List[float]
    error: List[float]
    anomaly_flags: List[bool]
    steps: List[GateStep]

def get_model():
    global MODEL
    if MODEL is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        weights_path = os.path.join(checkpoint_dir, "anomaly_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                weights = json.load(f)
            MODEL = FastAnomalyModel(weights)
        else:
            try:
                import torch
                from ..models.anomaly_model import AnomalyAutoencoder
                model_path = os.path.join(checkpoint_dir, "anomaly_model.pt")
                pt_model = AnomalyAutoencoder(input_size=1, hidden_size=32)
                if os.path.exists(model_path):
                    pt_model.load_state_dict(torch.load(model_path, weights_only=True))
                pt_model.eval()
                MODEL = pt_model
            except Exception:
                pass
    return MODEL

def get_samples():
    global SAMPLES
    if not SAMPLES:
        data_path = os.path.join(os.path.dirname(__file__), "..", "..", "training", "data", "anomaly_samples.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                SAMPLES = json.load(f)
    return SAMPLES

@router.get("/samples")
def get_anomaly_samples():
    samples = get_samples()
    return [{"id": s["id"], "name": s["type"].capitalize() + " Sample", "description": "A sample sequence"} for s in samples]

@router.post("/detect", response_model=AnomalyDetectResponse)
def detect_anomaly(req: AnomalyDetectRequest):
    samples = get_samples()
    sample = next((s for s in samples if s["id"] == req.sequence_id), samples[0])
    
    sequence = list(sample["data"])
    if req.inject_anomaly:
        mid = len(sequence) // 2
        for t in range(mid, mid + 5):
            sequence[t] += 2.0
            
    model = get_model()
    
    if isinstance(model, FastAnomalyModel):
        reconstructed_list, steps = model.forward(sequence)
    else:
        import torch
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            reconstructed, steps = model(x)
        reconstructed_list = [float(v) for v in reconstructed.view(-1).tolist()]
        
    error = [float((sequence[i] - reconstructed_list[i])**2) for i in range(len(sequence))]
    threshold = 0.5
    anomaly_flags = [bool(e > threshold) for e in error]
    
    return AnomalyDetectResponse(
        input=sequence,
        reconstruction=reconstructed_list,
        error=error,
        anomaly_flags=anomaly_flags,
        steps=steps
    )
