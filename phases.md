# phases.md — NeuroSeq build phases

Each phase has an explicit exit criterion. Do not start a phase until the previous phase's exit criterion is met, except Phase 1 and Phase 3 which can run in parallel (backend model work vs. frontend shell work) if agent supports parallel work.

## Phase 0 — Scaffolding
- Init frontend (Vite + React + TS + Tailwind) and backend (FastAPI) repos/folders per architecture.md §5–6.
- Set up shared type contract stub: `lib/types.ts` and `schemas.py` with `GateStep` defined identically in both.
- Set up CORS, basic `/api/health` endpoint, frontend proxy to backend.
- **Exit criterion:** frontend dev server and backend dev server both run locally, frontend can successfully call `/api/health` and render the result.

## Phase 1 — Backend: models + gate-exposing forward pass
- Implement `lstm_cell_loop.py`: shared manual `nn.LSTMCell` loop utility returning `GateStep[]`.
- Implement and train the 3 app models + the small explainer demo model (training scripts in `training/`).
- Implement all 4 routers (`explainer`, `nextword`, `anomaly`, `sentiment`) per the API contract in architecture.md §4.
- Write backend tests per rules.md §8.
- **Exit criterion:** every endpoint in architecture.md §4 is callable (via curl/Postman) and returns real, schema-valid data including plausible gate values (forget/input/output within [0,1]).

## Phase 2 — Frontend shell + navigation + shared components
- Build `TopMenuBar.tsx` with routing to Home / Explainer / Playground (with 3-app submenu) / About.
- Build shared components: `GateHeatmap.tsx` (Canvas), `CellStateTrace.tsx` (Recharts line), `Equation.tsx` (KaTeX wrapper).
- Wire `lib/api.ts` typed fetch functions for all 4 endpoints.
- **Exit criterion:** navigating the app between all routes works with no full page reload; shared components render correctly against fixture/mock `GateStep[]` data (not yet wired to live backend).

## Phase 3 — Explainer module
- Build all 8 sections listed in PRD.md §5.2 as scroll/step-navigated `StepSection` components.
- Wire section 7 ("putting it together across a sequence") to the real `/api/explainer/trace` endpoint.
- Polish animations for gate squashing, cell-state conveyor-belt visual, per Framer Motion.
- **Exit criterion:** a first-time visitor can go through the Explainer top to bottom and reach the live gate trace at the end, backed by real API data.

## Phase 4 — Playground: 3 live applications
- Build `NextWordPredictor.tsx`, `AnomalyDetector.tsx`, `SentimentClassifier.tsx`, each wired to its real endpoint and reusing `GateHeatmap`/`CellStateTrace`.
- Implement debounced live typing behavior (rules.md §5).
- Implement Anomaly Detector's sample-sequence loader + "inject anomaly" interaction.
- **Exit criterion:** all 3 apps run a full live demo loop (type/select input → see prediction/score/reconstruction AND gate activations update) with no fake data and no visible lag.

## Phase 5 — Integration polish
- Apply final design system (colors/typography) once design.md is provided by the user.
- Cross-check every equation against rules.md §3.
- Add empty/error/loading states everywhere per rules.md §6.
- Performance pass: confirm every route loads interactive in < 2s, every live interaction responds in < 150ms debounce + model latency.
- **Exit criterion:** the full click-path Home → Explainer → each Playground app works end-to-end with production-quality visuals, no console errors.

## Phase 6 — Demo rehearsal + hardening
- Dry-run the exact demo script (which sentences/sequences will be typed/selected live) at least 3 times.
- Test on the actual presenting laptop / expected demo resolution (1366×768 per PRD §6).
- Prepare fallback: if live backend fails during demo, have a pre-recorded short clip as backup (not part of the product itself, just a safety net).
- Write final README with setup instructions matching architecture.md §8.
- **Exit criterion:** project is demo-ready; README allows a fresh machine to be set up and running in under 10 minutes.
