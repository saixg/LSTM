import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from .lstm_cell_loop import ExposedLSTMCell, unroll_exposed_lstm

class NextWordLSTM(nn.Module):
    """
    Next-Word Predictor Model.
    Word-level language model using ExposedLSTMCell.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_size: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = ExposedLSTMCell(embedding_dim, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, tokens: List[str]) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        x: shape (seq_len,) or (seq_len, batch_size) with integer token IDs
        Returns:
            logits: shape (seq_len, batch_size, vocab_size) predictions for the next word
            steps: list of GateStep dictionaries
        """
        if x.dim() == 1:
            x = x.unsqueeze(1) # (seq_len, 1)
            
        embedded = self.embedding(x)
        
        seq_len = embedded.size(0)
        batch_size = embedded.size(1)
        hx = torch.zeros(batch_size, self.hidden_size, device=embedded.device)
        cx = torch.zeros(batch_size, self.hidden_size, device=embedded.device)
        
        steps = []
        logits_seq = []
        
        for t in range(seq_len):
            x_t = embedded[t]
            (hx, cx), (i_t, f_t, o_t, _) = self.lstm_cell(x_t, (hx, cx))
            
            logits = self.head(hx)
            logits_seq.append(logits)
            
            step = {
                "t": t,
                "token": tokens[t] if t < len(tokens) else str(t),
                "forget": float(f_t[0].mean().item()),
                "input": float(i_t[0].mean().item()),
                "output": float(o_t[0].mean().item()),
                "cell_state_norm": float(cx[0].norm().item())
            }
            steps.append(step)
            
        logits_seq = torch.stack(logits_seq) # (seq_len, batch_size, vocab_size)
        return logits_seq, steps
