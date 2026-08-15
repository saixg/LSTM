import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

class ExposedLSTMCell(nn.Module):
    """
    Custom LSTM cell that exposes internal gate values.
    cuDNN-fused nn.LSTM hides per-gate values, so we implement an explicit python loop
    to capture f_t, i_t, o_t at every timestep for the explainer UI.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # We use a standard nn.LSTMCell to hold the weights and biases for compatibility
        self.cell = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        hx, cx = state
        
        # Manually compute gates to expose them
        gates = (torch.mm(x, self.cell.weight_ih.t()) + self.cell.bias_ih +
                 torch.mm(hx, self.cell.weight_hh.t()) + self.cell.bias_hh)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        i_t = torch.sigmoid(ingate)
        f_t = torch.sigmoid(forgetgate)
        g_t = torch.tanh(cellgate)
        o_t = torch.sigmoid(outgate)
        
        c_next = (f_t * cx) + (i_t * g_t)
        h_next = o_t * torch.tanh(c_next)
        
        return (h_next, c_next), (i_t, f_t, o_t, g_t)

def unroll_exposed_lstm(
    cell: ExposedLSTMCell, 
    embeddings: torch.Tensor, 
    tokens: List[str]
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """
    Unrolls the ExposedLSTMCell over a sequence of embeddings.
    Args:
        cell: ExposedLSTMCell instance
        embeddings: Tensor of shape (seq_len, 1, input_size) or (seq_len, input_size). We assume batch_size=1 for live inference.
        tokens: List of strings corresponding to the timesteps.
    Returns:
        final_hidden: (h_t, c_t) at the end.
        steps: List of GateStep dicts ready for Pydantic serialization.
    """
    seq_len = embeddings.size(0)
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(1) # (seq_len, 1, input_size)
    
    batch_size = embeddings.size(1)
    hx = torch.zeros(batch_size, cell.hidden_size, device=embeddings.device)
    cx = torch.zeros(batch_size, cell.hidden_size, device=embeddings.device)
    
    steps = []
    
    for t in range(seq_len):
        x_t = embeddings[t]
        (hx, cx), (i_t, f_t, o_t, _) = cell(x_t, (hx, cx))
        
        # We aggregate the gate values (mean across hidden dimension) for the frontend heatmap.
        # We assume batch_size=1 and take the 0th item.
        step = {
            "t": t,
            "token": tokens[t] if t < len(tokens) else str(t),
            "forget": float(f_t[0].mean().item()),
            "input": float(i_t[0].mean().item()),
            "output": float(o_t[0].mean().item()),
            "cell_state_norm": float(cx[0].norm().item())
        }
        steps.append(step)
        
    return hx, steps
