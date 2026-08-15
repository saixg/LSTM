# rules.md — NeuroSeq (rules for the build agent)

These rules are binding for the entire build. If a rule and a convenience shortcut conflict, the rule wins.

## 1. Non-negotiable integrity rule
**Never fake a number.** Every gate activation, prediction, probability, reconstruction error, or sentiment score shown anywhere in the UI must come from a real forward pass of a real trained model through the backend API. No hardcoded arrays pretending to be "live" gate values, no client-side random-noise animations dressed up as model output. If a model isn't trained yet, the UI must show a clear "model not loaded" state — never a plausible-looking fake.

## 2. Model implementation rule
Use `nn.LSTMCell` in an explicit Python loop (or hand-rolled gate equations from raw weight matrices) — never `nn.LSTM` — anywhere gate-level values need to be exposed. Document this choice inline in `lstm_cell_loop.py` with a one-line comment explaining why (cuDNN-fused `nn.LSTM` hides per-gate values).

## 3. Equation accuracy rule
Every equation rendered in the Explainer must match standard LSTM formulation exactly (using the standard notation: f_t, i_t, o_t, g_t/C̃_t, C_t, h_t, with sigmoid σ and tanh). Cross-check against a primary reference before finalizing KaTeX strings. No simplified/approximate equations that read as illustrative-but-wrong.

## 4. Code style
- TypeScript: strict mode on, no `any` except at fetch/JSON boundaries (immediately narrowed to typed interfaces from `lib/types.ts`).
- Python: type hints everywhere, Pydantic models for all request/response bodies, `black` + `ruff` formatting.
- Shared types (`GateStep`, prediction shapes) must be kept in sync between `lib/types.ts` and `schemas.py` — if one changes, the other must be updated in the same commit/step.
- No inline styles in React except for values that must be computed at runtime (e.g. canvas dimensions); everything else via Tailwind classes.

## 5. Performance rules
- Debounce all "live as you type" API calls by 150ms minimum.
- Models load once into memory at backend startup, never reloaded per-request.
- Gate heatmap redraws use Canvas, not repeated DOM node creation, for anything updating faster than 2x/second.
- Keep every trained model small enough to run inference on CPU in well under 100ms for typical input lengths (≤ 40 tokens) — this is a demo constraint, not a research constraint. Prioritize latency over marginal accuracy gains.

## 6. Navigation / UX rules
- Top menu bar is present and identical on every page (`Home | Explainer | Playground ▾ | About`).
- Playground apps are reachable in ≤ 2 clicks from any page.
- Every page must load and be interactive in under 2 seconds on a typical demo laptop — no spinner-heavy cold starts during a live demo.
- Never let an API error produce a blank screen — always a visible, styled error/empty state.

## 7. Scope discipline
- Do not add features not listed in PRD.md without flagging it back to the user first (e.g. do not silently add auth, a 4th app, or a CMS).
- If a phase in phases.md is at risk of running over time, cut animation polish before cutting model correctness or the "no fake data" rule.

## 8. Testing minimums
- Backend: at least one test per endpoint verifying response shape matches the Pydantic schema, plus one test asserting gate values are within valid ranges (forget/input/output ∈ [0,1] post-sigmoid; cell state unbounded but finite).
- Frontend: smoke test that each route renders without throwing, and that GateHeatmap/CellStateTrace render given a sample `GateStep[]` fixture.

## 9. Documentation minimums
- Every model file (`nextword_model.py`, `anomaly_model.py`, `sentiment_model.py`) has a module-level docstring stating architecture (layers, hidden size), training data source, and known limitations.
- README at repo root with setup steps for both backend and frontend, matching architecture.md §8 exactly.

## 10. Git / commit hygiene
- Commit per logical unit of work (one model, one component, one endpoint) — not one giant commit per phase.
- Commit messages describe *what* and *why*, not just *what* (e.g. "Add LSTMCell manual loop to expose per-timestep gates for anomaly model" not "update model.py").
