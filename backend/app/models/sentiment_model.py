import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from .lstm_cell_loop import ExposedLSTMCell, unroll_exposed_lstm

class SentimentLSTM(nn.Module):
    """
    Sentiment Sequence Classifier.
    Predicts a running sentiment score per token.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = ExposedLSTMCell(embedding_dim, hidden_size)
        # Output a single score (logit) which will be squashed by sigmoid
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, tokens: List[str]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        x: shape (seq_len,) or (seq_len, batch_size) with integer token IDs
        Returns:
            running_scores: shape (seq_len, batch_size) containing the sentiment score at each timestep
            steps: list of GateStep dictionaries
        """
        if x.dim() == 1:
            x = x.unsqueeze(1) # (seq_len, 1)
            
        # (seq_len, batch_size, embedding_dim)
        embedded = self.embedding(x)
        
        # Unroll manually
        seq_len = embedded.size(0)
        batch_size = embedded.size(1)
        hx = torch.zeros(batch_size, self.hidden_size, device=embedded.device)
        cx = torch.zeros(batch_size, self.hidden_size, device=embedded.device)
        
        steps = []
        running_scores = []
        
        for t in range(seq_len):
            x_t = embedded[t]
            (hx, cx), (i_t, f_t, o_t, _) = self.lstm_cell(x_t, (hx, cx))
            
            score = torch.sigmoid(self.head(hx))
            running_scores.append(score)
            
            step = {
                "t": t,
                "token": tokens[t] if t < len(tokens) else str(t),
                "forget": float(f_t[0].mean().item()),
                "input": float(i_t[0].mean().item()),
                "output": float(o_t[0].mean().item()),
                "cell_state_norm": float(cx[0].norm().item())
            }
            steps.append(step)
            
        running_scores = torch.stack(running_scores).squeeze(-1) # (seq_len, batch_size)
        return running_scores, steps
