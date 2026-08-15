import os
import json
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from ..schemas import GateStep
from ..models.fast_lstm_engine import FastNextWordModel, softmax

router = APIRouter(prefix="/api/nextword", tags=["nextword"])

MODEL = None
VOCAB = None
INV_VOCAB = None

class NextWordRequest(BaseModel):
    text: str
    level: str = "word"

class TokenProb(BaseModel):
    token: str
    prob: float

class NextWordResponse(BaseModel):
    top5: List[TokenProb]
    steps: List[GateStep]

def load_model():
    global MODEL, VOCAB, INV_VOCAB
    if MODEL is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        vocab_path = os.path.join(checkpoint_dir, "nextword_vocab.json")
        weights_path = os.path.join(checkpoint_dir, "nextword_weights.json")
        
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                VOCAB = json.load(f)
            INV_VOCAB = {v: k for k, v in VOCAB.items()}
        else:
            VOCAB = {"<PAD>": 0, "<UNK>": 1}
            INV_VOCAB = {0: "<PAD>", 1: "<UNK>"}
            
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                weights = json.load(f)
            MODEL = FastNextWordModel(weights)
        else:
            # Fallback to pytorch if pt exists
            try:
                import torch
                from ..models.nextword_model import NextWordLSTM
                pt_path = os.path.join(checkpoint_dir, "nextword_model.pt")
                pt_model = NextWordLSTM(vocab_size=len(VOCAB), embedding_dim=128, hidden_size=128)
                if os.path.exists(pt_path):
                    pt_model.load_state_dict(torch.load(pt_path, weights_only=True))
                pt_model.eval()
                MODEL = pt_model
            except Exception:
                pass

@router.post("/predict", response_model=NextWordResponse)
def predict_nextword(req: NextWordRequest):
    load_model()
    words = req.text.lower().split()
    if not words:
        return NextWordResponse(top5=[], steps=[])
        
    encoded = [VOCAB.get(w, VOCAB["<UNK>"]) for w in words]
    
    if isinstance(MODEL, FastNextWordModel):
        logits, steps = MODEL.forward(encoded, tokens=words)
        probs = softmax(logits)
        
        # Top 5 indices
        indexed_probs = list(enumerate(probs))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)
        top5_items = indexed_probs[:min(5, len(indexed_probs))]
        
        top5 = []
        for idx, p in top5_items:
            token = INV_VOCAB.get(idx, "<UNK>")
            top5.append(TokenProb(token=token, prob=float(p)))
            
        return NextWordResponse(top5=top5, steps=steps)
        
    # PyTorch fallback if needed
    import torch
    x = torch.tensor(encoded, dtype=torch.long)
    with torch.no_grad():
        logits_seq, steps = MODEL(x, tokens=words)
    final_logits = logits_seq[-1].squeeze(0)
    probs = torch.softmax(final_logits, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, min(5, len(VOCAB)))
    top5 = [TokenProb(token=INV_VOCAB.get(idx, "<UNK>"), prob=float(p)) for p, idx in zip(top5_probs.tolist(), top5_indices.tolist())]
    return NextWordResponse(top5=top5, steps=steps)
