# architecture.md — NeuroSeq

## 1. Stack decision

| Layer | Choice | Why |
|---|---|---|
| Frontend framework | React 18 + Vite + TypeScript | Fast dev loop, strict typing catches gate-shape bugs early |
| Styling | TailwindCSS | Speed of iteration for polish pass |
| Animation | Framer Motion (UI transitions) + custom SVG/Canvas for gate heatmaps | Framer for scroll/step transitions; raw Canvas for high-frequency live heatmap redraws (Framer is too heavy for per-keystroke redraw) |
| Equations | KaTeX (`react-katex`) | Fast render, no external CDN dependency at runtime |
| Charts | Recharts (reconstruction error, sentiment trace) + custom Canvas (gate heatmap grid) | Recharts is fine for line charts; heatmap needs pixel-level control Recharts doesn't give cheaply |
| Backend | Python 3.11 + FastAPI | Needed to run real PyTorch models and expose per-timestep internals cheaply |
| Model framework | PyTorch (custom `nn.LSTMCell` loop, NOT `nn.LSTM`) | `nn.LSTM` (cuDNN fused) does not expose per-gate values. Must manually loop `LSTMCell` (or hand-roll the 4 gate equations) to capture f_t, i_t, o_t, c_t, h_t at every timestep. |
| Serving | Uvicorn, local only (no cloud dependency for demo reliability) | |
| Package management | `uv` or `pip` (backend), `npm`/`pnpm` (frontend) | |

## 2. Why `nn.LSTMCell` and not `nn.LSTM`
This is the single most important architectural decision in the project. `torch.nn.LSTM` fuses all timesteps into a cuDNN kernel and only returns final hidden/cell state — it will not give you per-gate, per-timestep values, which is the entire point of the explainer. The agent must implement the forward pass as an explicit loop over `nn.LSTMCell` (or hand-rolled gate math using raw weight matrices) so that at every timestep it can capture and return:
```
f_t (forget gate, sigmoid output, vector)
i_t (input gate, sigmoid output, vector)
g_t (candidate cell state, tanh output, vector)
o_t (output gate, sigmoid output, vector)
c_t (cell state, vector)
h_t (hidden state, vector)
```
These are aggregated (e.g. mean or norm across the hidden dimension, plus the raw vector for optional deep-dive view) and returned per timestep in every API response used by the Explainer and the 3 Playground apps.

## 3. Models (one per app, kept intentionally small for fast local CPU inference)

### 3.1 Explainer demo model
- Small word-level or char-level LSTM (1 layer, hidden size 32–64), trained on a small curated corpus.
- Used only to generate the live walkthrough in Explainer §7 (real sentence → real gate trace).

### 3.2 Next-Word Predictor
- Word-level LSTM, embedding dim ~64–128, hidden size ~128, 1–2 layers.
- Trained on a modest corpus (domain-flavored text is fine — e.g. tech/hackathon-related corpus, or a classic public-domain text if corpus curation time is short).
- Output: softmax over vocabulary, return top-5 tokens + probabilities + gate trace for the input typed so far.

### 3.3 Anomaly Detector
- LSTM Autoencoder: encoder LSTM compresses a windowed sequence to a latent vector, decoder LSTM reconstructs it.
- Trained on "normal" synthetic or public sample sensor/time-series data (unsupervised — no anomaly labels needed for training).
- Anomaly score = reconstruction error (MSE) per timestep; threshold flagged in UI.
- "Inject anomaly" button perturbs a copy of the sample sequence client-side (or via a backend endpoint) before sending for reconstruction, so the error spike is visibly real.

### 3.4 Sentiment Sequence Classifier
- Word-level LSTM + embedding, hidden size ~128, final linear + sigmoid/softmax head.
- Trained on a standard public sentiment dataset (e.g. IMDB or similar — small subset acceptable for demo-grade accuracy; this is a demo of *mechanism*, not a SOTA claim).
- Returns per-token running sentiment score (apply the classifier head at every timestep on h_t, not just at the end) + gate trace.

## 4. API contract (FastAPI, all under `/api`)

```
POST /api/explainer/trace
  body: { "text": string }
  returns: { tokens: string[], steps: GateStep[] }

POST /api/nextword/predict
  body: { "text": string, "level": "word" | "char" }
  returns: { top5: {token: string, prob: number}[], steps: GateStep[] }

POST /api/anomaly/detect
  body: { "sequence_id": string, "inject_anomaly": boolean }
  returns: { input: number[], reconstruction: number[], error: number[], anomaly_flags: boolean[], steps: GateStep[] }

POST /api/sentiment/classify
  body: { "text": string }
  returns: { tokens: string[], running_score: number[], final_label: string, steps: GateStep[] }

GET  /api/anomaly/samples
  returns: { id: string, name: string, description: string }[]
```

`GateStep` shared shape (used by every endpoint, so the frontend heatmap component is fully reusable):
```ts
type GateStep = {
  t: number
  token: string          // the input token/char/timepoint label at this step
  forget: number         // aggregated (mean or norm) forget gate activation, 0-1
  input: number           // aggregated input gate activation, 0-1
  output: number          // aggregated output gate activation, 0-1
  cell_state_norm: number // magnitude of cell state vector at this step
}
```

## 5. Frontend structure

```
src/
  pages/
    Home.tsx
    Explainer.tsx
    playground/
      NextWordPredictor.tsx
      AnomalyDetector.tsx
      SentimentClassifier.tsx
    About.tsx
  components/
    nav/TopMenuBar.tsx
    gates/GateHeatmap.tsx        # shared canvas heatmap, used by Explainer + all 3 apps
    gates/CellStateTrace.tsx     # shared line chart of cell state magnitude over time
    equations/Equation.tsx       # thin KaTeX wrapper
    explainer/StepSection.tsx    # one scrollytelling section
  lib/
    api.ts                      # typed fetch wrappers for all /api endpoints
    types.ts                    # GateStep and other shared types (mirrors backend Pydantic models)
  App.tsx                       # router setup
```

## 6. Backend structure

```
backend/
  app/
    main.py                     # FastAPI app, CORS, route registration
    routers/
      explainer.py
      nextword.py
      anomaly.py
      sentiment.py
    models/
      lstm_cell_loop.py         # shared manual-loop LSTM forward pass returning GateStep[]
      nextword_model.py
      anomaly_model.py
      sentiment_model.py
    schemas.py                  # Pydantic request/response models, mirrors lib/types.ts
    checkpoints/                # trained .pt weights, committed or downloaded via setup script
  training/
    train_nextword.py
    train_anomaly.py
    train_sentiment.py
    prepare_corpus.py
  requirements.txt
```

## 7. Data flow (per keystroke, e.g. Next-Word Predictor)
```
User types character
  → debounced (150ms) frontend call to POST /api/nextword/predict
  → FastAPI loads cached model (already in memory, not reloaded per request)
  → manual LSTMCell loop over input tokens, capturing GateStep per timestep
  → response: top5 predictions + steps[]
  → GateHeatmap.tsx redraws canvas from steps[]
  → CellStateTrace.tsx updates line chart
```

## 8. Local dev / run
```
# backend
cd backend && uvicorn app.main:app --reload --port 8000

# frontend
cd frontend && npm run dev   # proxies /api to localhost:8000
```

## 9. Deployment for demo day
Run both processes locally on the presenting laptop. No external network dependency. Optionally containerize with a single `docker-compose up` for reproducibility, but local dev servers are acceptable and lower-risk for a live demo.
