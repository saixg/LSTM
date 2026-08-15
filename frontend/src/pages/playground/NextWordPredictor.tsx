import React, { useState, useEffect } from 'react';
import { fetchNextWordPredict } from '../../lib/api';
import { GateHeatmap } from '../../components/gates/GateHeatmap';
import { CellStateTrace } from '../../components/gates/CellStateTrace';
import type { GateStep, TokenProb } from '../../lib/types';

export const NextWordPredictor: React.FC = () => {
  const [inputVal, setInputVal] = useState('the AI model learns from a');
  const [top5, setTop5] = useState<TokenProb[]>([]);
  const [steps, setSteps] = useState<GateStep[]>([]);
  const [loading, setLoading] = useState(false);

  const samplePrompts = [
    'the AI model learns from a',
    'deep learning architectures like LSTM and',
    'training a neural network requires',
    'gradient descent is used to',
    'hackathons provide a great',
  ];

  useEffect(() => {
    let active = true;
    const runInference = async () => {
      if (!inputVal.trim()) {
        setTop5([]);
        setSteps([]);
        return;
      }
      setLoading(true);
      try {
        const res = await fetchNextWordPredict(inputVal, 'word');
        if (active) {
          setTop5(res.top5);
          setSteps(res.steps);
        }
      } catch (err) {
        console.error('NextWord prediction error:', err);
      } finally {
        if (active) setLoading(false);
      }
    };

    const timer = setTimeout(runInference, 150); // 150ms debounce
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [inputVal]);

  const handleAppendToken = (token: string) => {
    setInputVal((prev) => `${prev.trim()} ${token}`);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      
      {/* App Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-hairline">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-primary/10 text-primary font-mono text-xs font-bold">
              PLAYGROUND APP 01
            </span>
            <span className="text-xs font-mono text-mute">Word-Level Language Modeling</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-ink">
            Next-Word Predictor & Logit Distribution
          </h1>
          <p className="mt-1 text-sm sm:text-base text-charcoal max-w-2xl">
            Type any sequence or click suggestions to predict next tokens via PyTorch softmax, while live gates (<code className="font-mono text-xs text-primary">f_t, i_t, o_t</code>) visualize memory retention.
          </p>
        </div>

        {/* Quick presets */}
        <div className="flex flex-col items-end gap-1.5">
          <span className="text-xs font-mono text-mute">Quick Demo Prompts:</span>
          <div className="flex flex-wrap gap-1.5 justify-end">
            {samplePrompts.map((p, idx) => (
              <button
                key={idx}
                onClick={() => setInputVal(p)}
                className="text-xs px-2.5 py-1 rounded-lg bg-surface-card border border-hairline hover:bg-surface-bone text-charcoal transition-colors font-body"
              >
                "{p.slice(0, 24)}..."
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Interactive Input & Top-5 Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Input Box & Quick Add */}
        <div className="lg:col-span-7 space-y-4">
          <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <label className="font-display font-bold text-sm text-ink flex items-center gap-2">
                <span>Input Sequence:</span>
                {loading && <span className="text-xs font-mono text-primary animate-pulse">Computing PyTorch forward pass...</span>}
              </label>
              <button
                onClick={() => setInputVal('')}
                className="text-xs font-mono text-mute hover:text-ink transition-colors"
              >
                Clear
              </button>
            </div>

            <div className="relative">
              <textarea
                rows={3}
                value={inputVal}
                onChange={(e) => setInputVal(e.target.value)}
                placeholder="Type words here..."
                className="w-full text-input rounded-2xl p-4 h-auto text-base font-medium resize-none shadow-inner"
              />
            </div>

            {/* Click to Append Suggestions */}
            {top5.length > 0 && (
              <div className="space-y-2 pt-2">
                <span className="text-xs font-mono text-charcoal font-semibold">
                  Click to Append Candidate:
                </span>
                <div className="flex flex-wrap gap-2">
                  {top5.map((t, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleAppendToken(t.token)}
                      className="px-3 py-1.5 rounded-xl bg-surface-bone hover:bg-primary hover:text-white border border-hairline transition-all text-xs font-mono font-semibold flex items-center gap-1.5 shadow-sm group"
                    >
                      <span>"{t.token}"</span>
                      <span className="text-[10px] opacity-70 group-hover:opacity-100">
                        ({(t.prob * 100).toFixed(0)}%)
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Model Anatomy Context Note */}
          <div className="p-4 rounded-xl bg-surface-bone/60 border border-hairline text-xs text-charcoal space-y-1.5">
            <div className="font-mono font-bold text-ink">🔬 Forward Pass Telemetry:</div>
            <p>
              The model projects embedding vectors of size 128 through our custom <code className="font-mono text-primary font-semibold">ExposedLSTMCell</code> with hidden size 128, then computes linear softmax over the vocabulary logits.
            </p>
          </div>
        </div>

        {/* Right Top-5 Probability Distribution */}
        <div className="lg:col-span-5">
          <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-hairline">
              <h3 className="font-display font-bold text-base text-ink flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-primary"></span>
                Top-5 Softmax Predictions
              </h3>
              <span className="text-xs font-mono text-mute">P(w_{'{t+1}'} | w_{'{1:t}'})</span>
            </div>

            {top5.length === 0 ? (
              <div className="py-12 text-center text-xs text-mute font-body">
                Type in the input box to calculate vocabulary probabilities...
              </div>
            ) : (
              <div className="space-y-3 pt-1">
                {top5.map((item, idx) => {
                  const pct = Math.max(1, item.prob * 100);
                  const isTop1 = idx === 0;
                  return (
                    <div key={idx} className="space-y-1">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <div className="flex items-center gap-2">
                          <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                            isTop1 ? 'bg-primary text-white' : 'bg-surface-bone text-charcoal'
                          }`}>
                            {idx + 1}
                          </span>
                          <span className={`font-semibold text-sm ${isTop1 ? 'text-primary' : 'text-ink'}`}>
                            "{item.token}"
                          </span>
                        </div>
                        <span className="font-bold text-charcoal">{pct.toFixed(2)}%</span>
                      </div>
                      
                      {/* Probability Gauge Bar */}
                      <div className="w-full bg-surface-bone rounded-full h-3.5 overflow-hidden p-0.5 border border-hairline">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            isTop1 ? 'bg-gradient-to-r from-primary to-primary-deep' : 'bg-charcoal/70'
                          }`}
                          style={{ width: `${pct}%` }}
                        ></div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Shared Gate Heatmap & Line Trace Panels */}
      <div className="space-y-6 pt-4">
        <GateHeatmap
          steps={steps}
          title="Keystroke Gate Activation Heatmap"
          subtitle="Real-time Forget (ft), Input (it), and Output (ot) activations across every input token"
        />

        <CellStateTrace
          steps={steps}
          title="Memory Preservation Curve (||c_t||)"
          subtitle="Measures magnitude of information persisting down the LSTM cell highway"
        />
      </div>

    </div>
  );
};
