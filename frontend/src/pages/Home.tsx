import React, { useState, useEffect } from 'react';
import type { NavRoute } from '../components/nav/TopMenuBar';
import { fetchExplainerTrace } from '../lib/api';
import { GateHeatmap } from '../components/gates/GateHeatmap';
import { CellStateTrace } from '../components/gates/CellStateTrace';
import type { GateStep } from '../lib/types';

interface HomeProps {
  onNavigate: (route: NavRoute) => void;
}

export const Home: React.FC<HomeProps> = ({ onNavigate }) => {
  const [heroInput, setHeroInput] = useState('the neural network learns patterns');
  const [steps, setSteps] = useState<GateStep[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    const runTrace = async () => {
      if (!heroInput.trim()) {
        setSteps([]);
        return;
      }
      setLoading(true);
      try {
        const res = await fetchExplainerTrace(heroInput);
        if (active) setSteps(res.steps);
      } catch (err) {
        console.error('Trace error:', err);
      } finally {
        if (active) setLoading(false);
      }
    };

    const timer = setTimeout(runTrace, 150);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [heroInput]);

  return (
    <div className="min-h-screen bg-canvas text-body font-body">
      
      {/* Hero Section */}
      <section className="relative pt-12 pb-20 overflow-hidden border-b border-hairline">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-4xl mx-auto">
            
            {/* Top Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-surface-bone border border-hairline text-xs font-mono text-charcoal mb-6 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-primary animate-ping"></span>
              <span>NeuroSeq 1.0 • PyTorch nn.LSTMCell Internals</span>
            </div>

            {/* Main Headline */}
            <h1 className="font-display font-extrabold text-4xl sm:text-6xl md:text-7xl text-ink tracking-tight leading-[1.05]">
              Don't just use an LSTM.{' '}
              <span className="bg-gradient-to-r from-primary via-primary-deep to-rose-600 bg-clip-text text-transparent">
                Watch one think.
              </span>
            </h1>

            {/* Subhead */}
            <p className="mt-6 text-lg sm:text-xl text-charcoal font-body leading-relaxed max-w-2xl mx-auto">
              Most recurrent models are closed black boxes. NeuroSeq extracts real mathematical gate activations (<code className="font-mono text-sm text-ink bg-surface-bone px-1.5 py-0.5 rounded">f_t, i_t, o_t, c_t</code>) at every timestep from live PyTorch loops.
            </p>

            {/* CTA Buttons */}
            <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
              <button
                onClick={() => onNavigate('explainer')}
                className="btn-primary text-base px-8 py-3.5 h-[50px] shadow-lg shadow-primary/25 hover:scale-[1.02] transition-transform"
              >
                Launch 8-Step Interactive Explainer
              </button>
              <button
                onClick={() => onNavigate('playground-nextword')}
                className="btn-outline text-base px-7 py-3.5 h-[50px] hover:bg-surface-bone"
              >
                Open Live Playground Apps
              </button>
            </div>

          </div>

          {/* Live Hero Interactive Terminal */}
          <div className="mt-14 max-w-5xl mx-auto">
            <div className="rounded-3xl bg-surface-card border border-hairline shadow-2xl overflow-hidden p-6 sm:p-8">
              
              <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
                <div>
                  <span className="text-xs font-mono font-bold uppercase text-primary tracking-wider flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-primary"></span>
                    Live Forward-Pass Stream
                  </span>
                  <h3 className="font-display font-bold text-xl text-ink mt-0.5">
                    Type a sentence to unroll gates in real time:
                  </h3>
                </div>

                <div className="flex flex-wrap gap-2">
                  {[
                    'the neural network learns patterns',
                    'artificial intelligence will transform code',
                    'the quick brown fox jumps',
                  ].map((sample, idx) => (
                    <button
                      key={idx}
                      onClick={() => setHeroInput(sample)}
                      className="px-3 py-1 rounded-full text-xs font-body bg-surface-bone hover:bg-canvas border border-hairline transition-colors text-charcoal"
                    >
                      "{sample}"
                    </button>
                  ))}
                </div>
              </div>

              {/* Text Input */}
              <div className="relative mb-6">
                <input
                  type="text"
                  value={heroInput}
                  onChange={(e) => setHeroInput(e.target.value)}
                  placeholder="Type words here..."
                  className="w-full text-input text-lg font-medium pl-5 pr-12 h-[52px] bg-canvas border-hairline-strong/20 shadow-inner"
                />
                {loading && (
                  <div className="absolute right-4 top-1/2 -translate-y-1/2">
                    <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                  </div>
                )}
              </div>

              {/* Real-time Heatmap & Line Trace Grid */}
              <div className="space-y-6">
                <GateHeatmap
                  steps={steps}
                  title="Forward Pass Gate Activations"
                  subtitle="Values stream from manual nn.LSTMCell unroll (Forget ft, Input it, Output ot)"
                />
                <CellStateTrace
                  steps={steps}
                  title="Cell State L2 Magnitude (||c_t||)"
                  subtitle="Tracks cumulative conveyor-belt memory preservation"
                />
              </div>

            </div>
          </div>

        </div>
      </section>

      {/* 3 Live Applications Showcase */}
      <section className="py-20 bg-surface-bone/50 border-b border-hairline">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          
          <div className="text-center max-w-3xl mx-auto mb-16">
            <span className="text-xs font-mono font-bold uppercase text-primary tracking-wider">
              3 Production-Grade Testbeds
            </span>
            <h2 className="font-display font-extrabold text-3xl sm:text-4xl text-ink mt-2">
              Three Real Applications. Zero Mock Data.
            </h2>
            <p className="mt-3 text-base text-charcoal font-body">
              Every prediction, anomaly flag, and sentiment score is calculated on the fly by custom PyTorch neural networks executing locally.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            
            {/* App Card 1 */}
            <div
              onClick={() => onNavigate('playground-nextword')}
              className="rounded-2xl bg-surface-card border border-hairline p-7 shadow-sm hover:shadow-xl hover:border-primary/40 transition-all cursor-pointer group flex flex-col justify-between"
            >
              <div>
                <div className="w-12 h-12 rounded-xl bg-primary/10 text-primary flex items-center justify-center font-mono font-bold text-lg mb-5 group-hover:scale-110 transition-transform">
                  01
                </div>
                <h3 className="font-display font-bold text-xl text-ink group-hover:text-primary transition-colors">
                  Next-Word Predictor
                </h3>
                <p className="mt-2.5 text-sm text-charcoal leading-relaxed font-body">
                  Word-level language model unrolling vocabulary logits. Watch top-5 softmax probabilities evolve alongside internal token gate shifts.
                </p>
                <div className="mt-4 pt-4 border-t border-hairline flex items-center gap-2 text-xs font-mono text-mute">
                  <span className="px-2 py-0.5 rounded bg-surface-bone">Top-5 Softmax</span>
                  <span className="px-2 py-0.5 rounded bg-surface-bone">Word Vocab</span>
                </div>
              </div>
              <div className="mt-6 flex items-center text-sm font-semibold text-primary gap-1">
                <span>Launch Predictor</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </div>

            {/* App Card 2 */}
            <div
              onClick={() => onNavigate('playground-anomaly')}
              className="rounded-2xl bg-surface-card border border-hairline p-7 shadow-sm hover:shadow-xl hover:border-primary/40 transition-all cursor-pointer group flex flex-col justify-between"
            >
              <div>
                <div className="w-12 h-12 rounded-xl bg-emerald-500/10 text-emerald-600 flex items-center justify-center font-mono font-bold text-lg mb-5 group-hover:scale-110 transition-transform">
                  02
                </div>
                <h3 className="font-display font-bold text-xl text-ink group-hover:text-emerald-600 transition-colors">
                  Anomaly Detector
                </h3>
                <p className="mt-2.5 text-sm text-charcoal leading-relaxed font-body">
                  Time-series LSTM autoencoder reconstructing sensor telemetry. Inject artificial spikes to watch reconstruction MSE and gate turbulence surge.
                </p>
                <div className="mt-4 pt-4 border-t border-hairline flex items-center gap-2 text-xs font-mono text-mute">
                  <span className="px-2 py-0.5 rounded bg-surface-bone">Autoencoder MSE</span>
                  <span className="px-2 py-0.5 rounded bg-surface-bone">Spike Injection</span>
                </div>
              </div>
              <div className="mt-6 flex items-center text-sm font-semibold text-emerald-600 gap-1">
                <span>Launch Detector</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </div>

            {/* App Card 3 */}
            <div
              onClick={() => onNavigate('playground-sentiment')}
              className="rounded-2xl bg-surface-card border border-hairline p-7 shadow-sm hover:shadow-xl hover:border-primary/40 transition-all cursor-pointer group flex flex-col justify-between"
            >
              <div>
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 text-blue-600 flex items-center justify-center font-mono font-bold text-lg mb-5 group-hover:scale-110 transition-transform">
                  03
                </div>
                <h3 className="font-display font-bold text-xl text-ink group-hover:text-blue-600 transition-colors">
                  Sentiment Classifier
                </h3>
                <p className="mt-2.5 text-sm text-charcoal leading-relaxed font-body">
                  Token-by-token continuous sentiment classifier. Type pivotal sentences with "but" or "however" and observe how memory forgets prior polarity.
                </p>
                <div className="mt-4 pt-4 border-t border-hairline flex items-center gap-2 text-xs font-mono text-mute">
                  <span className="px-2 py-0.5 rounded bg-surface-bone">Running Polarity</span>
                  <span className="px-2 py-0.5 rounded bg-surface-bone">Pivot Detection</span>
                </div>
              </div>
              <div className="mt-6 flex items-center text-sm font-semibold text-blue-600 gap-1">
                <span>Launch Classifier</span>
                <span className="group-hover:translate-x-1 transition-transform">→</span>
              </div>
            </div>

          </div>

        </div>
      </section>

      {/* Why Architecture Matters Section */}
      <section className="py-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div>
            <span className="text-xs font-mono font-bold uppercase text-primary tracking-wider">
              Under The Hood
            </span>
            <h2 className="font-display font-extrabold text-3xl sm:text-4xl text-ink mt-2">
              Why We Rewrote the PyTorch Forward Pass
            </h2>
            <p className="mt-4 text-base text-charcoal leading-relaxed">
              Standard <code className="font-mono bg-surface-bone px-1.5 py-0.5 rounded text-ink">torch.nn.LSTM</code> delegates to cuDNN kernels that fuse timesteps together for throughput. While fast, this permanently hides per-gate values.
            </p>
            <p className="mt-3 text-base text-charcoal leading-relaxed">
              NeuroSeq implements custom <code className="font-mono bg-surface-bone px-1.5 py-0.5 rounded text-ink">ExposedLSTMCell</code> unrolling in Python. This exposes exact values for Forget (<InlineCode text="f_t" />), Input (<InlineCode text="i_t" />), Candidate (<InlineCode text="g_t" />), and Output (<InlineCode text="o_t" />) at every timestep without approximations.
            </p>

            <div className="mt-8 flex items-center gap-4">
              <button
                onClick={() => onNavigate('explainer')}
                className="btn-dark text-sm px-6 py-3"
              >
                Read the 8-Step Math Walkthrough
              </button>
              <button
                onClick={() => onNavigate('about')}
                className="btn-outline text-sm px-6 py-3"
              >
                View System Architecture
              </button>
            </div>
          </div>

          <div className="rounded-2xl bg-surface-dark text-on-dark p-6 sm:p-8 font-mono text-xs overflow-x-auto shadow-2xl border border-white/10">
            <div className="flex items-center justify-between pb-3 mb-4 border-b border-white/10 text-on-dark-mute text-[11px]">
              <span>backend/app/models/lstm_cell_loop.py</span>
              <span className="text-emerald-400">● PyTorch CPU Loop</span>
            </div>
            <pre className="text-green-300 leading-relaxed">
{`# Explicit gate unroll in ExposedLSTMCell
gates = (torch.mm(x_t, cell.weight_ih.t()) + cell.bias_ih +
         torch.mm(h_prev, cell.weight_hh.t()) + cell.bias_hh)

i_gate, f_gate, g_gate, o_gate = gates.chunk(4, 1)

f_t = torch.sigmoid(f_gate)    # Forget gate
i_t = torch.sigmoid(i_gate)    # Input gate
g_t = torch.tanh(g_gate)       # Candidate state
o_t = torch.sigmoid(o_gate)    # Output gate

c_next = (f_t * c_prev) + (i_t * g_t) # Conveyor update
h_next = o_t * torch.tanh(c_next)     # Hidden output`}
            </pre>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-10 border-t border-hairline bg-surface-bone/30 text-center text-xs text-mute font-body">
        <p>NeuroSeq — Interactive LSTM Internals Explainer & Live Applications</p>
        <p className="mt-1">Built with React 18, Vite, TypeScript, PyTorch, FastAPI, KaTeX & Recharts</p>
      </footer>

    </div>
  );
};

const InlineCode: React.FC<{ text: string }> = ({ text }) => (
  <code className="font-mono text-xs font-semibold px-1.5 py-0.5 rounded bg-surface-bone text-primary border border-hairline">
    {text}
  </code>
);
