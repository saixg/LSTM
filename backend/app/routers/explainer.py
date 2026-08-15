import torch
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from ..schemas import GateStep
from . import nextword

router = APIRouter(prefix="/api/explainer", tags=["explainer"])

class ExplainerTraceRequest(BaseModel):
    text: str

class ExplainerTraceResponse(BaseModel):
    tokens: List[str]
    steps: List[GateStep]

@router.post("/trace", response_model=ExplainerTraceResponse)
def trace_explainer(req: ExplainerTraceRequest):
    # We reuse the NextWord model to generate a valid trace for the explainer UI
    nextword.load_model()
    
    words = req.text.lower().split()
    if not words:
        return ExplainerTraceResponse(tokens=[], steps=[])
        
    encoded = [nextword.VOCAB.get(w, nextword.VOCAB["<UNK>"]) for w in words]
    x = torch.tensor(encoded, dtype=torch.long)
    
    with torch.no_grad():
        _, steps = nextword.MODEL(x, tokens=words)
        
    return ExplainerTraceResponse(
        tokens=words,
        steps=steps
    )
