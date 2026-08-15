# PRD.md — NeuroSeq (LSTM Explainer + Live Applications)

## 1. One-line pitch
An interactive website that teaches how LSTMs actually work — gate by gate, equation by equation, animation by animation — and then proves it by running three real, live LSTM applications whose internal gate activations you can watch in real time.

## 2. Problem
Almost every student-built "LSTM project" is a black box: input goes in, prediction comes out, accuracy number shown. Nobody shows *why* it works. Judges and reviewers have seen a thousand sentiment classifiers with a confusion matrix and nothing else. There is no differentiation in outcome quality, only in explainability and craft.

## 3. Target audience
- Primary: hackathon judges / evaluators (NextGen 2.0 and similar) — need to grasp architecture depth in under 2 minutes of demo time.
- Secondary: recruiters / GitHub visitors doing a portfolio review.
- Tertiary: students trying to actually learn LSTMs (this could double as a teaching tool later).

## 4. Core value proposition
"Don't just use an LSTM. Watch one think." The product's identity is the fusion of rigorous internals (real equations, real gate values from a real forward pass — never scripted/fake numbers) with polished, cinematic front-end animation.

## 5. Scope — Feature List

### 5.1 Home / Landing
- Hero section, one-line pitch, CTA buttons: "Learn how LSTMs work" → Explainer, "Try it live" → Playground.
- Top navigation bar (persistent across all pages): `Home | Explainer | Playground ▾ | About`
  - Playground dropdown/menu exposes the 3 apps directly: `Next-Word Predictor`, `Anomaly Detector`, `Sentiment Classifier`.

### 5.2 Explainer Module ("A–Z", scrollytelling format)
Ordered sections, each a distinct scroll-triggered or step-navigated segment:
1. **The problem RNNs have** — vanishing/exploding gradients, short animated diagram of gradient shrinking across timesteps.
2. **The LSTM cell, structurally** — labeled diagram: cell state line, hidden state line, three gates.
3. **Forget gate** — equation (KaTeX-rendered), plain-English meaning, animated sigmoid squashing a bar toward 0 or 1 over a toy example.
4. **Input gate + candidate cell state** — equation, animation of sigmoid × tanh combining.
5. **Cell state update** — animation showing old cell state being partially forgotten and partially updated, i.e. the "conveyor belt" visual.
6. **Output gate + hidden state** — equation, animation of cell state being filtered into the hidden state.
7. **Putting it together across a sequence** — a real short sentence (e.g. "the movie was not very good") streams token by token through a real trained LSTM; live heatmap of f_t, i_t, o_t per timestep; cell-state magnitude line chart evolving.
8. **Why this matters** — transition CTA into Playground.

Non-negotiable: every number shown in step 7 must come from an actual forward pass of a real trained model via the backend API — not a canned/hardcoded animation.

### 5.3 Playground — 3 Live Applications
Each app is reachable from the menu bar and shares a common "gate activation panel" component (heatmap of f/i/o gates per timestep + cell state trace), so the explainer's visual language carries through.

**App 1 — Next-Word Predictor**
- Text input box, user types, top-5 next-word predictions update live.
- Gate activation panel updates per character/word as they type.
- Toggle: char-level vs word-level model (if both are built; word-level is the minimum requirement).

**App 2 — Anomaly Detector (time-series)**
- Pre-loaded sample sensor/log sequences (2–3 curated datasets) + a "inject synthetic anomaly" button for live demo drama.
- LSTM autoencoder reconstructs the sequence; reconstruction-error plot updates live; anomalies above threshold are flagged/highlighted on the chart.
- Gate activation panel shows how gates react around the anomalous region vs normal region.

**App 3 — Sentiment Sequence Classifier**
- Text input, live sentiment score (not just final label — a running score per token).
- Demo hook: type a sentence that flips sentiment mid-way (e.g. "the food was amazing but the service ruined it") and watch the score and gate activations shift at the pivot word.

### 5.4 About
- Short project description, tech stack badges, link to GitHub, author credit.

## 6. Non-functional requirements
- **No fake data rule**: every visualized number (gate values, predictions, reconstruction error, sentiment score) must originate from a real model forward pass through the backend API. This is the single most important constraint — a judge asking "is that real?" must always get "yes."
- Live interactions (typing → gate update) must feel responsive: target < 150ms round trip for short sequences.
- Animations target 60fps; degrade gracefully on low-end demo laptops (reduce particle/heatmap resolution rather than drop frames).
- Equations rendered via KaTeX, not images/screenshots.
- Fully responsive down to a projector/laptop demo resolution (1366×768) — this is what most hackathon demo screens run.
- Must run fully offline/local (no dependency on external paid APIs) so demo doesn't break on venue wifi.

## 7. Success criteria
- A judge with zero ML background can explain what a forget gate does after 90 seconds on the Explainer page.
- A judge with ML background can verify, by inspecting the code/API, that gate values are real and not scripted.
- All 3 Playground apps run a full live demo (type → see prediction + gates update) with no visible lag on a standard laptop.
- Page-to-page navigation via the top menu bar works without full reloads (SPA routing).

## 8. Out of scope (explicitly, to protect build time)
- User accounts / auth / persistence of user sessions.
- Model training UI (training happens offline via scripts, not in the product).
- Mobile-first design (desktop/demo-screen first; mobile is "should work," not "must be beautiful").
- Multi-language support.
