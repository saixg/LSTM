import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from .lstm_cell_loop import ExposedLSTMCell, unroll_exposed_lstm

class AnomalyAutoencoder(nn.Module):
    """
    LSTM Autoencoder for time-series anomaly detection.
    Encoder: ExposedLSTMCell (so we can visualize gate activations reading the signal).
    Decoder: Standard LSTMCell (we only care about the reconstruction output).
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 32):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = ExposedLSTMCell(input_size, hidden_size)
        self.decoder = nn.LSTMCell(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, tokens: List[str] = None) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        x: shape (seq_len, batch_size, input_size)
        Returns:
            reconstructed: shape (seq_len, batch_size, input_size)
            steps: list of GateStep dictionaries for the encoder
        """
        if x.dim() == 2:
            x = x.unsqueeze(1) # (seq_len, 1, input_size)
            
        seq_len, batch_size, _ = x.size()
        if tokens is None:
            tokens = [str(i) for i in range(seq_len)]
            
        # 1. Encoder pass
        final_hx, steps = unroll_exposed_lstm(self.encoder, x, tokens)
        
        # 2. Decoder pass
        # The decoder takes the final hidden state of the encoder and tries to reconstruct the sequence.
        # We feed the final_hx as input at every step of the decoder.
        reconstructed = []
        dec_hx = final_hx
        dec_cx = torch.zeros_like(final_hx)
        dec_input = final_hx
        
        for _ in range(seq_len):
            dec_hx, dec_cx = self.decoder(dec_input, (dec_hx, dec_cx))
            out = self.head(dec_hx)
            reconstructed.append(out)
            
        # Autoencoders typically reconstruct in reverse, but for visual simplicity we just map it 1:1 forward.
        reconstructed = torch.stack(reconstructed)
        return reconstructed, steps
