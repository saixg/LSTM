# NeuroSeq 🧠⚡

> *"Don't just use an LSTM. Watch one think."*

**NeuroSeq** is an interactive educational and research web platform that unpacks Long Short-Term Memory (LSTM) recurrent neural networks — equation by equation, gate by gate, and token by token. It connects mathematical foundations to three live PyTorch neural network applications whose internal gate activations update on the fly in real time.

---

## 🌟 Key Highlights

- **Zero Mock / Fake Data Guarantee**: Every single visualized number (Forget Gate $f_t$, Input Gate $i_t$, Output Gate $o_t$, and Cell State Norm $\|c_t\|$) originates directly from a real PyTorch model forward pass executing on the backend.
- **Custom `ExposedLSTMCell` Forward Pass**: Standard `torch.nn.LSTM` fuses all recurrent steps into a black-box cuDNN/C++ kernel. NeuroSeq implements an explicit step-by-step PyTorch cell unroll to extract internal tensor activations at every timestep without performance compromise.
- **8-Step Interactive Scrollytelling Explainer**: Deep-dive into Vanishing Gradients, Cell Blueprint, Sigmoid and Tanh functions, the linear Conveyor Belt highway, and live sequence streaming with KaTeX-rendered equations.
- **3 Live Production Playground Applications**:
  1. **Next-Word Language Model Predictor**: Keystroke-level autocomplete with top-5 softmax probability distribution and real-time gate telemetry.
  2. **Time-Series Anomaly Detector**: Autoencoder reconstruction for sensor telemetry, MSE error thresholding, and gate turbulence inspection.
  3. **Sentiment Sequence Classifier**: Token-by-token continuous polarity trajectory tracking, pivot conjunction detection (*"amazing ... but terrible"*), and **Dual-Engine Real-Time Validation** (Local PyTorch + API Ninjas Cloud Oracle).
- **High-Performance UI**: High-DPI HTML5 Canvas heatmaps with cell hover inspection, Recharts gradient area curves, and responsive warm cream/ink typography.

---

## 🔬 Mathematical Formulation

At each timestep $t$, given input embedding $x_t$, previous hidden state $h_{t-1}$, and previous cell state $C_{t-1}$:

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{(Forget Gate: what past to erase)}$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{(Input Gate: what new information to write)}$$

$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{(Candidate State: candidate content)}$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{(Cell State Highway: uninterrupted linear memory)}$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{(Output Gate: what to expose to external layers)}$$

$$h_t = o_t \odot \tanh(C_t) \quad \text{(Hidden State / Working Output)}$$

---

## 🛠️ System Architecture

```
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application entrypoint & CORS
│   │   ├── schemas.py                  # Pydantic schemas (GateStep, responses)
│   │   ├── models/
│   │   │   ├── lstm_cell_loop.py       # ExposedLSTMCell & step unrolling engine
│   │   │   ├── nextword_model.py       # NextWordLM PyTorch model
│   │   │   ├── anomaly_model.py        # AnomalyAutoencoder PyTorch model
│   │   │   └── sentiment_model.py      # SentimentLSTM PyTorch model
│   │   ├── routers/
│   │   │   ├── explainer.py            # POST /api/explainer/trace
│   │   │   ├── nextword.py             # POST /api/nextword/predict
│   │   │   ├── anomaly.py              # GET/POST /api/anomaly/samples & /detect
│   │   │   └── sentiment.py            # POST /api/sentiment/classify (Dual Engine)
│   │   └── checkpoints/                # Trained PyTorch weights & JSON vocabs
│   ├── training/
│   │   ├── prepare_corpus.py           # Synthetic tech, anomaly & sentiment data
│   │   ├── train_nextword.py           # Language model trainer
│   │   ├── train_anomaly.py            # Time-series autoencoder trainer
│   │   └── build_large_sentiment_engine.py # Open-vocabulary sentiment trainer
│   └── test_endpoints.py               # Automated endpoint test suite
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── equations/Equation.tsx  # KaTeX math formula renderer
│   │   │   ├── gates/GateHeatmap.tsx   # High-DPI Canvas gate activation heatmap
│   │   │   ├── gates/CellStateTrace.tsx# Recharts ||c_t|| line & area curve
│   │   │   ├── explainer/StepSection.tsx # Scrollytelling step wrapper
│   │   │   └── nav/TopMenuBar.tsx      # Header navbar with live PyTorch health poll
│   │   ├── pages/
│   │   │   ├── Home.tsx                # Hero, live unroll stream, feature cards
│   │   │   ├── Explainer.tsx           # 8-step deep interactive LSTM explainer
│   │   │   ├── playground/NextWordPredictor.tsx # Playground App 1
│   │   │   ├── playground/AnomalyDetector.tsx   # Playground App 2
│   │   │   ├── playground/SentimentClassifier.tsx# Playground App 3
│   │   │   └── About.tsx               # System architecture & verification audit
│   │   ├── lib/
│   │   │   ├── api.ts                  # Typed client API client
│   │   │   └── types.ts                # TypeScript interfaces matching backend
│   │   ├── App.tsx                     # React Router single-page navigation
│   │   └── index.css                   # Tailwind v4 theme & token system
│   └── vite.config.ts                  # Vite + Tailwind + backend proxy
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Clone the Repository
```bash
git clone https://github.com/saixg/LSTM.git
cd LSTM
```

### 2. Backend Setup
```bash
# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install torch fastapi uvicorn pydantic

# (Optional) Retrain or generate checkpoints from scratch
python -m backend.training.prepare_corpus
python -m backend.training.train_nextword
python -m backend.training.train_anomaly
python -m backend.training.build_large_sentiment_engine

# Start the FastAPI server
cd backend
uvicorn app.main:app --port 8000 --reload
```
Backend API will be running at `http://127.0.0.1:8000` (Swagger docs available at `http://127.0.0.1:8000/docs`).

### 3. Frontend Setup
```bash
# In a new terminal window:
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173/` in your browser.

---

## 🧪 Testing & Verification

Run the automated backend test suite:
```bash
python -m backend.test_endpoints
```

Run arbitrary open-domain sentiment verification:
```bash
python -m backend.test_arbitrary_sentiment
```

---

## 👨‍💻 Author

**Sai**  
B.Tech Computer Science (AI & ML), Marwadi University  
- GitHub: [@saixg](https://github.com/saixg)

---

## 📄 License
MIT License. Open source for research and educational purposes.
