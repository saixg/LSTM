import os
import json
import re
import urllib.request
import urllib.parse
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ..schemas import GateStep
from ..models.fast_lstm_engine import FastSentimentModel

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

MODEL = None
VOCAB = None
API_NINJAS_KEY = os.environ.get("API_NINJAS_KEY", "ImySMWP8E8jFA6sP0MLDZBxEwFRPks4T9YzHck1I")

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    tokens: List[str]
    running_score: List[float]
    final_label: str
    steps: List[GateStep]
    api_ninjas_score: Optional[float] = None
    api_ninjas_label: Optional[str] = None

def load_model():
    global MODEL, VOCAB
    if MODEL is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
        vocab_path = os.path.join(checkpoint_dir, "sentiment_vocab.json")
        weights_path = os.path.join(checkpoint_dir, "sentiment_weights.json")
        
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                VOCAB = json.load(f)
        else:
            VOCAB = {"<PAD>": 0, "<UNK>": 1}
            
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                weights = json.load(f)
            MODEL = FastSentimentModel(weights)
        else:
            try:
                import torch
                from ..models.sentiment_model import SentimentLSTM
                model_path = os.path.join(checkpoint_dir, "sentiment_model.pt")
                pt_model = SentimentLSTM(vocab_size=len(VOCAB), embedding_dim=64, hidden_size=128)
                if os.path.exists(model_path):
                    pt_model.load_state_dict(torch.load(model_path, weights_only=True))
                pt_model.eval()
                MODEL = pt_model
            except Exception:
                pass

def resolve_token(w: str) -> int:
    if w in VOCAB:
        return VOCAB[w]
    for suffix in ["ing", "ed", "ly", "s", "es", "ness", "able", "ible", "ful", "less"]:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            base = w[:-len(suffix)]
            if base in VOCAB:
                return VOCAB[base]
            if base + "e" in VOCAB:
                return VOCAB[base + "e"]
    return VOCAB.get("<UNK>", 1)

def query_api_ninjas(text: str) -> Optional[Dict[str, Any]]:
    if not text.strip() or not API_NINJAS_KEY:
        return None
    try:
        url = "https://api.api-ninjas.com/v1/sentiment?text=" + urllib.parse.quote(text)
        req = urllib.request.Request(
            url,
            headers={
                "X-Api-Key": API_NINJAS_KEY,
                "User-Agent": "NeuroSeq-Sentiment-Benchmark/1.0"
            }
        )
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None

@router.post("/classify", response_model=SentimentResponse)
def classify_sentiment(req: SentimentRequest):
    load_model()
    words = re.findall(r"[a-zA-Z0-9']+", req.text.lower())
    if not words:
        words = [""]
        
    encoded = [resolve_token(w) for w in words]
    
    if isinstance(MODEL, FastSentimentModel):
        running_scores, steps = MODEL.forward(encoded, tokens=words)
        final_score = running_scores[-1] if running_scores else 0.5
        final_label = "Positive" if final_score >= 0.5 else "Negative"
    else:
        import torch
        x = torch.tensor(encoded, dtype=torch.long)
        with torch.no_grad():
            running_scores_t, steps = MODEL(x, tokens=words)
        running_scores = [float(v) for v in running_scores_t.view(-1).tolist()]
        final_score = running_scores[-1] if running_scores else 0.5
        final_label = "Positive" if final_score >= 0.5 else "Negative"
    
    # Dual Model Cloud Benchmark
    cloud_data = query_api_ninjas(req.text)
    api_ninjas_score = float(cloud_data["score"]) if cloud_data and "score" in cloud_data else None
    api_ninjas_label = str(cloud_data["sentiment"]) if cloud_data and "sentiment" in cloud_data else None
    
    return SentimentResponse(
        tokens=words,
        running_score=running_scores,
        final_label=final_label,
        steps=steps,
        api_ninjas_score=api_ninjas_score,
        api_ninjas_label=api_ninjas_label
    )
