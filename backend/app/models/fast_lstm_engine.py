import os
import json
import math
from typing import List, Tuple, Dict, Any, Optional

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(x, 50.0), -50.0)))

def tanh(x: float) -> float:
    return math.tanh(max(min(x, 50.0), -50.0))

def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(a * b for a, b in zip(v1, v2))

def matrix_vector_mul(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [dot_product(row, vector) for row in matrix]

def softmax(logits: List[float]) -> List[float]:
    max_l = max(logits)
    exp_l = [math.exp(l - max_l) for l in logits]
    sum_e = sum(exp_l)
    return [e / sum_e for e in exp_l]

class FastExposedLSTMCell:
    def __init__(self, weight_ih: List[List[float]], weight_hh: List[List[float]], bias_ih: List[float], bias_hh: List[float]):
        self.weight_ih = weight_ih # (4*H, in_dim)
        self.weight_hh = weight_hh # (4*H, H)
        self.bias_ih = bias_ih     # (4*H,)
        self.bias_hh = bias_hh     # (4*H,)
        self.hidden_size = len(weight_hh[0])

    def forward(self, x: List[float], hx: List[float], cx: List[float]):
        H = self.hidden_size
        gates_ih = matrix_vector_mul(self.weight_ih, x)
        gates_hh = matrix_vector_mul(self.weight_hh, hx)
        
        gates = [g_ih + b_ih + g_hh + b_hh for g_ih, b_ih, g_hh, b_hh in zip(gates_ih, self.bias_ih, gates_hh, self.bias_hh)]
        
        i_gate = [sigmoid(gates[j]) for j in range(0, H)]
        f_gate = [sigmoid(gates[j]) for j in range(H, 2*H)]
        g_gate = [tanh(gates[j]) for j in range(2*H, 3*H)]
        o_gate = [sigmoid(gates[j]) for j in range(3*H, 4*H)]
        
        c_next = [f * c + i * g for f, c, i, g in zip(f_gate, cx, i_gate, g_gate)]
        h_next = [o * tanh(c) for o, c in zip(o_gate, c_next)]
        
        c_norm = math.sqrt(sum(c * c for c in c_next))
        f_mean = sum(f_gate) / H
        i_mean = sum(i_gate) / H
        o_mean = sum(o_gate) / H
        
        return (h_next, c_next), (i_mean, f_mean, o_mean, c_norm)

class FastNextWordModel:
    def __init__(self, weights: Dict[str, Any]):
        self.embedding = weights["embedding.weight"] # (vocab_size, emb_dim)
        self.cell = FastExposedLSTMCell(
            weight_ih=weights["lstm_cell.cell.weight_ih"],
            weight_hh=weights["lstm_cell.cell.weight_hh"],
            bias_ih=weights["lstm_cell.cell.bias_ih"],
            bias_hh=weights["lstm_cell.cell.bias_hh"]
        )
        self.head_w = weights.get("head.weight", weights.get("fc.weight")) # (vocab_size, H)
        self.head_b = weights.get("head.bias", weights.get("fc.bias"))     # (vocab_size,)
        self.hidden_size = self.cell.hidden_size

    def forward(self, token_ids: List[int], tokens: List[str]):
        hx = [0.0] * self.hidden_size
        cx = [0.0] * self.hidden_size
        steps = []
        
        for t, idx in enumerate(token_ids):
            emb_vec = self.embedding[idx] if idx < len(self.embedding) else self.embedding[1]
            (hx, cx), (i_m, f_m, o_m, c_norm) = self.cell.forward(emb_vec, hx, cx)
            
            steps.append({
                "t": t,
                "token": tokens[t] if t < len(tokens) else str(t),
                "forget": float(f_m),
                "input": float(i_m),
                "output": float(o_m),
                "cell_state_norm": float(c_norm)
            })
            
        logits = [dot_product(row, hx) + b for row, b in zip(self.head_w, self.head_b)]
        return logits, steps

class FastSentimentModel:
    def __init__(self, weights: Dict[str, Any]):
        self.embedding = weights["embedding.weight"]
        self.cell = FastExposedLSTMCell(
            weight_ih=weights["lstm_cell.cell.weight_ih"],
            weight_hh=weights["lstm_cell.cell.weight_hh"],
            bias_ih=weights["lstm_cell.cell.bias_ih"],
            bias_hh=weights["lstm_cell.cell.bias_hh"]
        )
        self.head_w = weights["head.weight"][0] # (H,)
        self.head_b = weights["head.bias"][0]   # float
        self.hidden_size = self.cell.hidden_size

    def forward(self, token_ids: List[int], tokens: List[str]):
        hx = [0.0] * self.hidden_size
        cx = [0.0] * self.hidden_size
        steps = []
        running_scores = []
        
        for t, idx in enumerate(token_ids):
            emb_vec = self.embedding[idx] if idx < len(self.embedding) else self.embedding[1]
            (hx, cx), (i_m, f_m, o_m, c_norm) = self.cell.forward(emb_vec, hx, cx)
            
            score_logit = dot_product(self.head_w, hx) + self.head_b
            score = sigmoid(score_logit)
            running_scores.append(score)
            
            steps.append({
                "t": t,
                "token": tokens[t] if t < len(tokens) else str(t),
                "forget": float(f_m),
                "input": float(i_m),
                "output": float(o_m),
                "cell_state_norm": float(c_norm)
            })
            
        return running_scores, steps

class FastAnomalyModel:
    def __init__(self, weights: Dict[str, Any]):
        self.encoder_cell = FastExposedLSTMCell(
            weight_ih=weights["encoder.cell.weight_ih"],
            weight_hh=weights["encoder.cell.weight_hh"],
            bias_ih=weights["encoder.cell.bias_ih"],
            bias_hh=weights["encoder.cell.bias_hh"]
        )
        self.decoder_cell = FastExposedLSTMCell(
            weight_ih=weights["decoder.weight_ih"],
            weight_hh=weights["decoder.weight_hh"],
            bias_ih=weights["decoder.bias_ih"],
            bias_hh=weights["decoder.bias_hh"]
        )
        self.head_w = weights["head.weight"][0] # (H,)
        self.head_b = weights["head.bias"][0]   # float
        self.hidden_size = self.encoder_cell.hidden_size

    def forward(self, sequence: List[float]):
        hx = [0.0] * self.hidden_size
        cx = [0.0] * self.hidden_size
        steps = []
        
        # 1. Encode
        for t, val in enumerate(sequence):
            x_t = [val]
            (hx, cx), (i_m, f_m, o_m, c_norm) = self.encoder_cell.forward(x_t, hx, cx)
            steps.append({
                "t": t,
                "token": f"{val:.2f}",
                "forget": float(f_m),
                "input": float(i_m),
                "output": float(o_m),
                "cell_state_norm": float(c_norm)
            })
            
        # 2. Decode
        reconstruction = []
        dec_hx = list(hx)
        dec_cx = list(cx)
        dec_in = [sequence[-1]]
        
        for _ in range(len(sequence)):
            (dec_hx, dec_cx), _ = self.decoder_cell.forward(dec_in, dec_hx, dec_cx)
            out_val = dot_product(self.head_w, dec_hx) + self.head_b
            reconstruction.append(out_val)
            dec_in = [out_val]
            
        reconstruction.reverse()
        return reconstruction, steps
