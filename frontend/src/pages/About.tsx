import React from 'react';
import type { NavRoute } from '../components/nav/TopMenuBar';

interface AboutProps {
  onNavigate: (route: NavRoute) => void;
}

export const About: React.FC<AboutProps> = ({ onNavigate }) => {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
      
      {/* Header */}
      <div className="text-center max-w-3xl mx-auto space-y-3 pb-8 border-b border-hairline">
        <span className="text-xs font-mono font-bold uppercase text-primary tracking-wider">
          Project Philosophy & Architecture
        </span>
        <h1 className="font-display font-extrabold text-4xl sm:text-5xl text-ink">
          About NeuroSeq
        </h1>
        <p className="text-base sm:text-lg text-charcoal font-body leading-relaxed">
          A transparent machine learning explainer and multi-app playground built to dismantle the black-box nature of Recurrent Neural Networks.
        </p>
      </div>

      {/* Author & Context */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-2">
          <span className="text-xs font-mono text-mute uppercase">Project Author</span>
          <h3 className="font-display font-bold text-lg text-ink">Sai</h3>
          <p className="text-xs text-charcoal leading-relaxed font-body">
            B.Tech Computer Science (AI & ML), Marwadi University. Built as an advanced portfolio piece and technical capstone.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-2">
          <span className="text-xs font-mono text-mute uppercase">Design Philosophy</span>
          <h3 className="font-display font-bold text-lg text-ink">"Don't Just Use. Watch."</h3>
          <p className="text-xs text-charcoal leading-relaxed font-body">
            Replaces static confusion matrices with live, keystroke-by-keystroke neural activation telemetry.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-2">
          <span className="text-xs font-mono text-mute uppercase">Integrity Contract</span>
          <h3 className="font-display font-bold text-lg text-ink">Zero Mock Data</h3>
          <p className="text-xs text-charcoal leading-relaxed font-body">
            Every gate value (<code className="font-mono text-[11px] text-primary">f_t, i_t, o_t</code>) comes from a live PyTorch forward pass.
          </p>
        </div>
      </div>

      {/* System Architecture Section */}
      <div className="p-8 rounded-3xl bg-surface-card border border-hairline shadow-sm space-y-6">
        <div>
          <h2 className="font-display font-bold text-2xl text-ink">
            Full End-to-End System Architecture
          </h2>
          <p className="mt-1 text-sm text-charcoal font-body">
            Designed for sub-150ms round-trip latency on standard CPU hardware without external cloud dependencies.
          </p>
        </div>

        {/* Tech Stack Badges */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
          <div className="p-3 rounded-xl bg-surface-bone border border-hairline flex flex-col gap-1">
            <span className="text-mute text-[10px]">FRONTEND FRAMEWORK</span>
            <span className="font-bold text-ink">React 18 + Vite + TS</span>
          </div>
          <div className="p-3 rounded-xl bg-surface-bone border border-hairline flex flex-col gap-1">
            <span className="text-mute text-[10px]">NEURAL COMPUTATION</span>
            <span className="font-bold text-ink">PyTorch nn.LSTMCell</span>
          </div>
          <div className="p-3 rounded-xl bg-surface-bone border border-hairline flex flex-col gap-1">
            <span className="text-mute text-[10px]">API SERVING</span>
            <span className="font-bold text-ink">FastAPI (Python 3.11)</span>
          </div>
          <div className="p-3 rounded-xl bg-surface-bone border border-hairline flex flex-col gap-1">
            <span className="text-mute text-[10px]">VISUALIZATIONS</span>
            <span className="font-bold text-ink">Canvas + Recharts + KaTeX</span>
          </div>
        </div>

        {/* Technical Deep Dive Explanation */}
        <div className="space-y-3 text-sm text-charcoal leading-relaxed font-body pt-4 border-t border-hairline">
          <h4 className="font-display font-bold text-base text-ink">
            Why Standard cuDNN LSTM Was Incompatible
          </h4>
          <p>
            Standard <code className="font-mono text-xs bg-surface-bone px-1 rounded text-ink">torch.nn.LSTM</code> fuses all timesteps into an optimized C++/CUDA kernel. While performant for batch training, it creates a total black box: intermediate gate activations (<InlineCode text="f_t" />, <InlineCode text="i_t" />, <InlineCode text="o_t" />) are discarded immediately after memory updates.
          </p>
          <p>
            In NeuroSeq, we implement <code className="font-mono text-xs bg-surface-bone px-1 rounded text-ink">ExposedLSTMCell</code> with a manual Python unroll loop. This preserves exact per-timestep activations and packs them into our shared <code className="font-mono text-xs bg-surface-bone px-1 rounded text-ink">GateStep[]</code> schema, powering our HTML5 Canvas heatmap and Recharts state trace in real time.
          </p>
        </div>
      </div>

      {/* Code Verification Audit */}
      <div className="p-8 rounded-3xl bg-surface-dark text-on-dark shadow-xl border border-white/10 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="font-display font-bold text-xl text-white">
            Backend Verification Audit
          </h3>
          <span className="px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-400 font-mono text-xs font-bold">
            ✓ Verified Open Source
          </span>
        </div>
        <p className="text-sm text-on-dark-mute font-body">
          To verify that gate values are real and not pre-baked, any evaluator can inspect the running process or call the endpoints directly:
        </p>

        <div className="p-4 rounded-xl bg-surface-deep font-mono text-xs text-green-400 overflow-x-auto space-y-2">
          <div># 1. Query the live explainer gate trace endpoint via curl:</div>
          <div className="text-white">
            curl -X POST http://localhost:8000/api/explainer/trace \
              -H "Content-Type: application/json" \
              -d '{`{"text": "neural network memory"}`}'
          </div>
          <div className="pt-2 text-mute"># Returns real floating-point gate activation arrays from PyTorch forward pass:</div>
          <div className="text-on-dark-mute">
            {`{"tokens":["neural","network","memory"],"steps":[{"t":0,"token":"neural","forget":0.521,"input":0.684,"output":0.491,"cell_state_norm":1.24}, ...]}`}
          </div>
        </div>

        <div className="flex flex-wrap gap-4 pt-4">
          <button
            onClick={() => onNavigate('explainer')}
            className="btn-primary text-xs px-5 py-2.5"
          >
            Review 8-Step Math Walkthrough →
          </button>
          <button
            onClick={() => onNavigate('playground-nextword')}
            className="btn-outline text-xs px-5 py-2.5 text-white bg-transparent border-white/20 hover:bg-white/10"
          >
            Explore Next-Word Predictor →
          </button>
        </div>
      </div>

    </div>
  );
};

const InlineCode: React.FC<{ text: string }> = ({ text }) => (
  <code className="font-mono text-xs font-semibold px-1.5 py-0.5 rounded bg-surface-bone text-primary border border-hairline">
    {text}
  </code>
);
