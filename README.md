# LexiFlow — LSTM Search Autocomplete + Animated LSTM Explainer

**implementation.md**
**Owner:** Sai
**Status:** Draft v1

---

## 1. What we're building

Two coupled deliverables in one project:

1. **Search bar with LSTM-powered postfix suggestions** — user types a prefix (partial word or phrase), a character/word-level LSTM predicts the most likely continuations, shown as live dropdown suggestions (like Google autocomplete, but the model is yours, trained and served by you — good hackathon/demo story).
2. **Animated "How LSTM Really Works" explainer** — an interactive visualization panel (separate route/tab) that walks through the LSTM cell step-by-step: gates, equations, and how a real input flows through the architecture, ideally animated using live data from your own trained model (not a static diagram).

The pitch angle: most autocomplete demos are black boxes. Yours lets the user open the box.

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + TS)                    │
│  ┌─────────────────┐        ┌──────────────────────────────┐   │
│  │  SearchBar.tsx   │        │   LSTMExplainer.tsx           │   │
│  │  - debounced      │        │   - step-through animation    │   │
│  │    input          │        │   - gate-by-gate math overlay │   │
│  │  - suggestion     │        │   - live activations from a   │   │
│  │    dropdown       │        │     real forward pass         │   │
│  └────────┬─────────┘        └───────────┬────────────────────┘ │
│           │  POST /predict                │  GET /trace           │
└───────────┼───────────────────────────────┼───────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND (FastAPI, Python)                    │
│  /predict   → top-k next-token predictions + probabilities       │
│  /trace     → per-timestep gate values (f_t, i_t, o_t, C_t, h_t) │
│  /health, /model-info                                            │
└───────────┬───────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER (PyTorch)                         │
│  Embedding → LSTM(1–2 layers) → Linear → Softmax over vocab      │
│  Trained on a word/char corpus, checkpointed, served via         │
│  TorchScript or plain nn.Module + state_dict                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Model | PyTorch (`nn.LSTM` + custom cell for the explainer trace) | Need raw gate values exposed, `nn.LSTM` alone won't give you those — see §5.4 |
| Serving | FastAPI + Uvicorn | Fast to stand up, async, easy CORS for local dev |
| Frontend | React + TypeScript + Vite | Fast dev loop, good for animation state management |
| Animation | Framer Motion (state transitions) + D3 or hand-rolled SVG (gate diagram) | Framer Motion for smooth easing, D3/SVG for precise node-link diagram |
| Styling | Tailwind CSS | Speed |
| Data | A domain corpus relevant to your search bar's use case (see §4) | Generic word corpora make weak/boring demos |

---

## 4. Data & vocabulary strategy

Two reasonable modes — pick one, don't try to do both for the hackathon:

**A. Word-level next-word prediction** (recommended for a "search postfix" feel)
- Corpus: pick something the demo can showcase meaningfully — e.g. product names, past search logs, Wikipedia sentence corpus, or a domain-specific dataset (movie titles, code identifiers, etc.)
- Vocabulary: top 8k–20k frequent tokens + `<unk>`, `<pad>`, `<eos>`
- Sequence framing: sliding window of last N tokens → predict next token

**B. Character-level completion** (better for "typing an unfinished word")
- Corpus: any large text corpus (Project Gutenberg subset, domain text)
- Vocabulary: ~70–100 chars (a–z, punctuation, space)
- Sequence framing: char sequence → predict next char, sample until whitespace to complete the word

**Recommendation:** use **word-level** for suggesting the *next word after* what's typed, and **char-level** for completing the *current partially-typed word*. Run both models, or one char-LSTM handles both cases well enough if you're time-constrained — pick char-level as the single MVP model since it demos more impressively ("watch it finish your word letter by letter").

### Preprocessing pipeline
```
raw text → lowercase/clean → tokenize (word or char)
        → build vocab (min freq cutoff) → int-encode
        → chunk into (input_seq, target) pairs → train/val split
```

---

## 5. Model architecture & the math (this is also what the explainer visualizes)

### 5.1 High-level pipeline
```
token ids → Embedding(vocab_size, emb_dim)
          → LSTM(emb_dim, hidden_dim, num_layers=1 or 2)
          → Linear(hidden_dim, vocab_size)
          → Softmax → probability distribution over next token
```

### 5.2 What's actually inside one LSTM cell

At each timestep `t`, the cell receives:
- `x_t` — the current input vector (embedded token)
- `h_{t-1}` — previous hidden state ("short-term memory")
- `C_{t-1}` — previous cell state ("long-term memory")

It produces `h_t` and `C_t`.

**Forget gate** — decides what to throw away from the cell state:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

**Input gate** — decides what new information to store:
```
i_t     = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t     = tanh(W_C · [h_{t-1}, x_t] + b_C)   # candidate values
```

**Cell state update** — combine forget + input:
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
```

**Output gate** — decides what to expose as the hidden state:
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

Where `σ` = sigmoid (squashes to 0–1, acts as a "gate" — how much to let through), `tanh` squashes to -1..1 (acts as content), `⊙` = elementwise multiply, `[h_{t-1}, x_t]` = concatenation.

**Why this matters conceptually (say this in the explainer):**
- The cell state `C_t` is a conveyor belt that information can ride along with minimal modification — this is *why* LSTMs solve vanishing gradients that plain RNNs suffer from.
- Each gate is itself a tiny neural network (linear layer + nonlinearity) that has *learned* when to open/close, not a hardcoded rule.
- Forget gate ≈ "what's no longer relevant", input gate ≈ "what's new and worth remembering", output gate ≈ "what's relevant to output *right now*, even if we keep remembering more internally".

### 5.3 Parameter count (for the explainer's "why is this heavy" moment)
For input size `n`, hidden size `h`:
```
params = 4 × [ h × (n + h) + h ]   # the 4 accounts for f, i, C̃, o gates
```

### 5.4 Important implementation note
`torch.nn.LSTM` is fast (cuDNN-fused) but does **not** expose intermediate gate values per timestep — you cannot animate `f_t`, `i_t`, `o_t` from it directly.

For the **explainer's `/trace` endpoint**, implement a manual `LSTMCell` forward pass (small, single-layer, pure PyTorch ops) so you can capture and return every gate's values per timestep as JSON. Use this manual cell *only* for the trace/demo model; use `nn.LSTM` for the production-speed autocomplete model. Keep both checkpoints compatible (train once, load weights into both implementations) so what the user sees in the animation is the real model, not a toy stand-in.

---

## 6. Backend API design

```
POST /predict
  body: { "text": "the quick brown fo" }
  resp: { "suggestions": [
            {"token": "fox", "prob": 0.71},
            {"token": "food", "prob": 0.09},
            {"token": "forest", "prob": 0.05}
          ] }

GET /trace?text=hello
  resp: { "tokens": ["h","e","l","l","o"],
          "steps": [
            { "t": 0, "input": "h",
              "f_t": [...], "i_t": [...], "C_tilde": [...],
              "o_t": [...], "C_t": [...], "h_t": [...] },
            ...
          ] }

GET /model-info
  resp: { "vocab_size": ..., "hidden_dim": ..., "num_layers": ...,
          "params": ..., "training_loss": ... }
```

- Debounce on the frontend (~150–250ms) before calling `/predict` so you're not hammering the API on every keystroke.
- Cache recent prefixes client-side (simple `Map`) to avoid duplicate calls when the user backspaces and retypes.

---

## 7. Frontend: Search bar

- `SearchBar.tsx`: controlled input, `onChange` → debounce → fetch `/predict` → render dropdown.
- Dropdown items ranked by probability; show probability as a small bar/percentage for transparency (ties into the "explainable AI" pitch).
- Keyboard nav: ↑/↓ to move through suggestions, Tab/→ to accept, Esc to dismiss.
- Empty/low-confidence state: if top prediction probability < threshold (e.g. 0.15), show "still learning this one" instead of a misleading suggestion.

---

## 8. Frontend: Animated LSTM explainer

Goal: don't just draw a static diagram — **replay a real forward pass** on user-provided (or example) input.

### 8.1 Structure
1. **Input strip** — the characters/words being fed in one at a time, with a moving "playhead."
2. **Cell diagram (SVG)** — the four gates as labeled boxes with arrows for data flow; highlight the active path as data moves through on each animation tick.
3. **Live math panel** — actual numeric values from `/trace` substituted into the equations, e.g. `f_t = σ(2.31) = 0.91` — not symbolic only, real numbers from the real model.
4. **Gate activation bars** — small bar charts (or heatmaps) for `f_t`, `i_t`, `o_t` vectors updating per timestep, so the user visually sees "forget gate mostly near 1 → cell is retaining most of its memory here."
5. **Cell state ribbon** — a horizontal ribbon representing `C_t` evolving left to right across timesteps — this is the single most important visual for building intuition (it's literally the "memory conveyor belt").

### 8.2 Animation implementation
- Framer Motion for: playhead movement, gate box pulse/highlight on activation, ribbon color intensity mapped to `C_t` magnitude.
- Step controls: Play / Pause / Step forward / Step back / Speed slider — this is a *walkthrough* tool, not a passive video.
- Color encoding: sigmoid outputs (0–1) → grayscale-to-blue intensity; tanh outputs (-1–1) → diverging red-blue scale. Keep this consistent and explain it once in a legend.

### 8.3 Suggested scene order (the "explanation script")
1. Plain RNN recap + why it fails (vanishing gradient) — 1 static diagram, no need to animate this part in depth.
2. Introduce the cell state ribbon as the fix.
3. Walk through one full timestep: forget gate → input gate → cell update → output gate, each with the live equation and real numbers.
4. Run the full sequence and show the ribbon evolve + suggestions appear at the end, closing the loop back to the search bar.

---

## 9. Suggested repo structure

```
lexiflow/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, routes
│   │   ├── model/
│   │   │   ├── lstm_fast.py        # nn.LSTM-based, used for /predict
│   │   │   ├── lstm_trace.py       # manual LSTMCell, used for /trace
│   │   │   └── checkpoint.pt
│   │   ├── data/
│   │   │   ├── vocab.json
│   │   │   └── preprocess.py
│   │   └── inference.py
│   ├── train.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchBar.tsx
│   │   │   ├── LSTMExplainer/
│   │   │   │   ├── CellDiagram.tsx
│   │   │   │   ├── MathPanel.tsx
│   │   │   │   ├── CellStateRibbon.tsx
│   │   │   │   └── PlaybackControls.tsx
│   │   ├── hooks/useDebouncedPredict.ts
│   │   └── App.tsx
│   └── package.json
└── implementation.md
```

---

## 10. Training checklist

- [ ] Build vocab + preprocessing pipeline
- [ ] Train baseline `nn.LSTM` model (1 layer, hidden_dim=128–256) — track train/val perplexity
- [ ] Overfit-test on a tiny subset first to confirm the pipeline works before full training
- [ ] Save checkpoint + vocab together (version them)
- [ ] Load same weights into manual `LSTMCell` trace model, verify outputs match `nn.LSTM` numerically (sanity check — this step is easy to get subtly wrong)
- [ ] Benchmark `/predict` latency (should feel instant, <100ms ideally)

---

## 11. Stretch goals (if time remains)

- Attention/analysis overlay comparing LSTM vs a tiny Transformer on the same prefix, for contrast.
- Let the user "edit" a gate value manually in the explainer (e.g. force forget gate to 0) and see how the prediction changes — turns it from a video into a real toy.
- Beam search / temperature slider on `/predict` for more diverse suggestions.
- Deploy: frontend on Vercel, backend on Render/Fly.io with model checkpoint bundled.

---

## 12. Known trade-offs to state upfront in any demo/pitch

- Word-level models need a large enough corpus to avoid a dominated `<unk>` — char-level avoids OOV but needs more steps to "learn" real words.
- The `/trace` model runs slower than `/predict` (unfused manual cell) — fine, since it's only invoked for the explainer, not on every keystroke.
- Suggestion quality is bounded by corpus relevance — pick a corpus that matches your demo's search domain, not a generic one, or the suggestions will look unimpressive live.
