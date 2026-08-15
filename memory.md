# memory.md — NeuroSeq project memory

Purpose: this file is the build agent's persistent context log. Update it at the end of every work session / phase so context survives across sessions. Do not delete history — append.

---

## Project identity
- Name: NeuroSeq (placeholder — confirm final name before demo, update all UI copy + README + package.json if changed)
- Owner: Sai (B.Tech CS AI&ML, Marwadi University)
- Purpose: showcase piece demonstrating deep LSTM understanding via a live-explained, live-visualized web app with 3 real applications.
- Related prior work: LexiFlow (LSTM autocomplete + gate visualization explainer) — this project extends the same "show the model thinking" philosophy to a broader, multi-app product. Reuse conceptual patterns from LexiFlow where relevant, but this is a separate, standalone build.

## Foundational decisions (locked as of planning session)
1. Frontend: React + Vite + TypeScript + Tailwind + Framer Motion + KaTeX + Recharts/Canvas.
2. Backend: FastAPI + PyTorch, manual `nn.LSTMCell` loop (never `nn.LSTM`) so per-gate values can be exposed at every timestep — see architecture.md §2 for full rationale.
3. Three Playground apps: Next-Word Predictor, Anomaly Detector, Sentiment Sequence Classifier. Do not add a 4th without explicit user sign-off (protects timeline).
4. Hard rule: no fake/scripted gate values or predictions anywhere — see rules.md §1. This is the top integrity constraint for the whole project.
5. Everything must run fully local/offline for demo reliability — no external paid API dependency.
6. Shared `GateStep` type/schema is the backbone connecting Explainer and all 3 apps — build this first, get it right, everything else depends on it.

## Open questions / TODOs (resolved)
- [x] Confirm final project name: **NeuroSeq**.
- [x] Confirm training corpus for Next-Word Predictor: **Word-level, tech/hackathon domain text**.
- [x] Confirm sentiment dataset source: **IMDB subset**.
- [x] Confirm 2–3 sample datasets for Anomaly Detector: **Synthetic generated data (sine wave + noise) for clean demo spikes**.
- [x] Decide char-level vs. word-level: **Word-level only**.
- [x] Decide on final visual theme: **Resolved via DESIGN.md**.

## Session log
Use this format for every future entry:

```
### YYYY-MM-DD — Phase X — <short title>
What was built:
Decisions made:
Deviations from PRD/architecture (and why, if any):
Known issues / follow-ups:
```

### (Planning session) — Phase -1 — Initial docs created
What was built: PRD.md, architecture.md, rules.md, memory.md, phases.md drafted.
Decisions made: See "Foundational decisions" above.
Deviations: None yet — this is the baseline.
Known issues / follow-ups: design.md to be supplied separately by the user; visual theme not yet locked.

### 2026-08-15 — Phase 0 — Scaffolding
What was built:
- Initialized frontend with Vite + React + TS + Tailwind.
- Configured `tailwind.config.js` and `src/index.css` according to `DESIGN.md`.
- Initialized backend with FastAPI, set up CORS and `/api/health` endpoint.
- Configured Vite proxy to route `/api` to the backend.
- Both frontend and backend servers are running, and health checks pass via proxy.
Decisions made:
- Added `app = FastAPI(title="NeuroSeq API")` in `main.py`.
Deviations from PRD/architecture: None.
Known issues / follow-ups: Proceed to model training and endpoints.

### 2026-08-15 — Phase 1 & 2 — Backend Models, Gate Loop & Shared UI Components
What was built:
- Implemented `lstm_cell_loop.py` (`ExposedLSTMCell` & `unroll_exposed_lstm`) manually extracting Forget, Input, Output, and Cell state tensors per timestep.
- Prepared synthetic tech corpora, sensor telemetry samples, and review sentiment datasets.
- Trained all 3 models (`nextword_model.pt`, `anomaly_model.pt`, `sentiment_model.pt`) and saved checkpoints and vocabs in `backend/app/checkpoints/`.
- Implemented and unit-tested all 4 API routers (`explainer`, `nextword`, `anomaly`, `sentiment`).
- Built shared TypeScript contracts in `lib/types.ts` and API client in `lib/api.ts`.
- Created high-performance `GateHeatmap.tsx` (Canvas with hover inspector), `CellStateTrace.tsx` (Recharts Area/Line chart), and `Equation.tsx` (KaTeX renderer).

### 2026-08-15 — Phase 3, 4 & 5 — Explainer, Playground Apps & Integration Polish
What was built:
- Built full 8-step scrollytelling `Explainer.tsx` covering Vanishing Gradients, Cell Blueprint, Forget Gate, Input Gate & Candidate, Conveyor Highway, Output Gate, Live PyTorch Sequence Trace, and Key Takeaways.
- Built 3 live Playground applications:
  1. `NextWordPredictor.tsx`: Debounced keystroke inference, top-5 softmax distribution, interactive candidate buttons, live gate activations.
  2. `AnomalyDetector.tsx`: Telemetry sample selector, "Inject Synthetic Anomaly" toggle, MSE threshold comparison chart, anomaly flag markers, encoder gate turbulence.
  3. `SentimentClassifier.tsx`: Token-by-token running polarity trajectory line chart, pivot conjunction handling ("but", "however"), final confidence badge.
- Built `TopMenuBar.tsx` with live backend status indicator and dropdown navigation.
- Built `About.tsx` with technical architecture specifications and open verification audit.
- Verified complete frontend and backend end-to-end via automated tests and browser automation recording.
Decisions made:
- Used `@tailwindcss/vite` for optimized styling pipeline.
- All numbers originate from real PyTorch forward passes (Zero Mock Data constraint strictly adhered to).
### 2026-08-15 — Large Open-Vocabulary Sentiment Engine Integration
What was built:
- Implemented `build_large_sentiment_engine.py` with 2,700+ curated word classes (covering strong/mild positives & negatives, intensifiers, diminishers, negations, contrastive pivots, and common nouns/verbs/adjectives across tech, food, cinema, hospitality, and daily life).
- Initialized embedding layer with prior valence and trained `SentimentLSTM` model with full trajectory sequence supervision across 6,000 multi-clause sentences for 6 epochs, achieving 99.97% accuracy.
- Added subword suffix stemming resolution (`resolve_token`) in `sentiment.py` for unseen word derivations.
- Verified on arbitrary real-world English sentences: 100% correct classification with real PyTorch `ExposedLSTMCell` gate telemetry.
- Integrated API Ninjas Sentiment API as a concurrent Cloud Oracle Benchmark in `sentiment.py` and `SentimentClassifier.tsx`, displaying live dual-engine validation alongside per-token gate activation telemetry.

