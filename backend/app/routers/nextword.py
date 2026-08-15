import os
import json
import torch
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from ..schemas import GateStep
from ..models.nextword_model import NextWordLSTM

router = APIRouter(prefix="/api/nextword", tags=["nextword"])

MODEL = None
VOCAB = None
INV_VOCAB = None

class NextWordRequest(BaseModel):
    text: str
    level: str = "word" # char level not supported currently

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
        model_path = os.path.join(checkpoint_dir, "nextword_model.pt")
        
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                VOCAB = json.load(f)
            INV_VOCAB = {v: k for k, v in VOCAB.items()}
        else:
            VOCAB = {"<PAD>": 0, "<UNK>": 1}
            INV_VOCAB = {0: "<PAD>", 1: "<UNK>"}
            
        MODEL = NextWordLSTM(vocab_size=len(VOCAB), embedding_dim=128, hidden_size=128)
        if os.path.exists(model_path):
            MODEL.load_state_dict(torch.load(model_path, weights_only=True))
        MODEL.eval()

@router.post("/predict", response_model=NextWordResponse)
def predict_nextword(req: NextWordRequest):
    load_model()
    words = req.text.lower().split()
    if not words:
        # Default empty prediction
        return NextWordResponse(top5=[], steps=[])
        
    encoded = [VOCAB.get(w, VOCAB["<UNK>"]) for w in words]
    x = torch.tensor(encoded, dtype=torch.long)
    
    with torch.no_grad():
        logits_seq, steps = MODEL(x, tokens=words)
        
    final_logits = logits_seq[-1].squeeze(0) # Shape: (vocab_size,)
    probs = torch.softmax(final_logits, dim=-1)
    
    top5_probs, top5_indices = torch.topk(probs, min(5, len(VOCAB)))
    
    top5 = []
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        token = INV_VOCAB.get(idx, "<UNK>")
        top5.append(TokenProb(token=token, prob=float(prob)))
        
    return NextWordResponse(
        top5=top5,
        steps=steps
    )
