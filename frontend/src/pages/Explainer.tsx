import React, { useState, useEffect } from 'react';
import { InlineMath } from 'react-katex';
import type { NavRoute } from '../components/nav/TopMenuBar';
import { StepSection } from '../components/explainer/StepSection';
import { Equation } from '../components/equations/Equation';
import { GateHeatmap } from '../components/gates/GateHeatmap';
import { CellStateTrace } from '../components/gates/CellStateTrace';
import { fetchExplainerTrace } from '../lib/api';
import type { GateStep } from '../lib/types';

interface ExplainerProps {
  onNavigate: (route: NavRoute) => void;
}

export const Explainer: React.FC<ExplainerProps> = ({ onNavigate }) => {
  // Step 1 interactive state: gradient decay across timesteps
  const [rnnWeight, setRnnWeight] = useState(0.7);
  const [rnnSteps, setRnnSteps] = useState(6);

  // Step 3 interactive state: Forget Gate Sigmoid Squasher
  const [forgetLogit, setForgetLogit] = useState(1.2);
  const forgetSigmoid = 1 / (1 + Math.exp(-forgetLogit));

  // Step 4 interactive state: Input Gate & Candidate
  const [inputLogit, setInputLogit] = useState(0.8);
  const [candidateLogit, setCandidateLogit] = useState(-0.6);
  const inputSigmoid = 1 / (1 + Math.exp(-inputLogit));
  const candidateTanh = Math.tanh(candidateLogit);
  const inputUpdate = inputSigmoid * candidateTanh;

  // Step 5 interactive state: Conveyor belt update
  const [prevCellState, setPrevCellState] = useState(2.0);
  const updatedCellState = forgetSigmoid * prevCellState + inputUpdate;

  // Step 6 interactive state: Output gate & Hidden state
  const [outputLogit, setOutputLogit] = useState(1.5);
  const outputSigmoid = 1 / (1 + Math.exp(-outputLogit));
  const hiddenOutput = outputSigmoid * Math.tanh(updatedCellState);

  // Step 7 real PyTorch trace
  const [traceText, setTraceText] = useState('the movie was not very good');
  const [traceSteps, setTraceSteps] = useState<GateStep[]>([]);
  const [traceLoading, setTraceLoading] = useState(false);
  const [activeTokenIdx, setActiveTokenIdx] = useState<number | null>(null);

  useEffect(() => {
    let active = true;
    const loadTrace = async () => {
      if (!traceText.trim()) {
        setTraceSteps([]);
        return;
      }
      setTraceLoading(true);
      try {
        const res = await fetchExplainerTrace(traceText);
        if (active) setTraceSteps(res.steps);
      } catch (err) {
        console.error('Explainer trace error:', err);
      } finally {
        if (active) setTraceLoading(false);
      }
    };
    loadTrace();
    return () => {
      active = false;
    };
  }, [traceText]);

  // Jump to step helper
  const scrollToStep = (stepNumber: number) => {
    const el = document.getElementById(`step-${stepNumber}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-canvas text-body font-body pb-24">
      
      {/* Explainer Header / Hero Banner */}
      <div className="bg-surface-bone/60 border-b border-hairline py-12">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <span className="text-xs font-mono font-bold uppercase text-primary tracking-wider">
                Full Architectural Dissection
              </span>
              <h1 className="font-display font-extrabold text-3xl sm:text-5xl text-ink mt-1 tracking-tight">
                How LSTMs Actually Work
              </h1>
              <p className="mt-3 text-base sm:text-lg text-charcoal max-w-2xl">
                An 8-step journey from the vanishing gradient flaw in vanilla RNNs to exact mathematical gate mechanics, animated conveyor-belt memory, and live PyTorch token unrolling.
              </p>
            </div>

            {/* Quick Step Bar Navigation */}
            <div className="flex flex-wrap gap-1.5 p-1.5 rounded-2xl bg-surface-card border border-hairline shadow-sm">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((s) => (
                <button
                  key={s}
                  onClick={() => scrollToStep(s)}
                  className="w-8 h-8 rounded-xl font-mono text-xs font-semibold text-charcoal hover:bg-surface-dark hover:text-on-dark transition-colors flex items-center justify-center"
                  title={`Jump to Step ${s}`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* STEP 1: The Problem RNNs Have */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={1}
        title="The Vanishing Gradient Crisis in Standard RNNs"
        badge="Theoretical Motivation"
        subtitle="Why recurrent loops collapse when trying to remember long-range dependencies."
        interactiveSlot={
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-charcoal">Recurrent Weight Norm (W_hh):</span>
              <span className="font-bold text-primary">{rnnWeight.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.5"
              step="0.05"
              value={rnnWeight}
              onChange={(e) => setRnnWeight(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />
            
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-charcoal">Backprop Timesteps (T):</span>
              <span className="font-bold text-ink">{rnnSteps}</span>
            </div>
            <input
              type="range"
              min="2"
              max="10"
              step="1"
              value={rnnSteps}
              onChange={(e) => setRnnSteps(parseInt(e.target.value))}
              className="w-full accent-primary"
            />

            {/* Visual simulation of gradient decay */}
            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-2">
              <div className="text-xs font-mono font-semibold text-ink mb-2">
                Gradient Flow Magnitude <code className="text-primary">||∂L/∂h_0|| ∝ W_hh^T</code>:
              </div>
              <div className="space-y-1.5">
                {Array.from({ length: rnnSteps }).map((_, i) => {
                  const decay = Math.pow(rnnWeight, i);
                  const clampedWidth = Math.min(100, Math.max(1, decay * 100));
                  return (
                    <div key={i} className="flex items-center gap-2 text-[11px] font-mono">
                      <span className="w-10 text-mute">t-{i}:</span>
                      <div className="flex-1 bg-surface-bone rounded-full h-3 overflow-hidden">
                        <div
                          className={`h-full transition-all duration-300 ${
                            decay < 0.1 ? 'bg-rose-500' : decay > 1.2 ? 'bg-amber-500' : 'bg-primary'
                          }`}
                          style={{ width: `${clampedWidth}%` }}
                        ></div>
                      </div>
                      <span className="w-14 text-right text-charcoal">{decay.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
              <p className="text-[11px] text-mute pt-2 border-t border-hairline">
                {rnnWeight < 1.0
                  ? '⚠️ Vanishing Gradient: Repeated multiplication by weights < 1.0 diminishes signal to 0 after a few timesteps.'
                  : rnnWeight > 1.0
                  ? '⚡ Exploding Gradient: Repeated multiplication by weights > 1.0 causes unstable exponential blowup.'
                  : 'Stable unity flow (rare in unconstrained matrix chains).'}
              </p>
            </div>
          </div>
        }
      >
        <p>
          In a standard Recurrent Neural Network (RNN), the hidden state at timestep <InlineMath math="t" /> is computed as a direct non-linear function of the previous hidden state:
        </p>
        <Equation
          math="h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)"
          explanation="Standard RNN hidden recurrence"
          terms={[
            { symbol: 'h_{t-1}', meaning: 'Previous hidden state' },
            { symbol: 'W_{hh}', meaning: 'Recurrent transition weight' },
            { symbol: 'x_t', meaning: 'Current token vector' },
          ]}
        />
        <p>
          When computing gradients during Backpropagation Through Time (BPTT), the chain rule requires repeated multiplication across every step:
        </p>
        <Equation
          math="\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{k=2}^{T} \frac{\partial h_k}{\partial h_{k-1}} = \frac{\partial L}{\partial h_T} \prod_{k=2}^{T} \left( \operatorname{diag}(1 - \tanh^2(\cdot)) W_{hh}^T \right)"
          explanation="Repeated matrix product leading to exponential decay"
        />
        <p className="text-sm text-charcoal">
          Because <InlineMath math="\tanh'(z) \le 1" />, the product continuously shrinks. If the sequence has 50 words, gradients reaching the first words become practically zero. The network cannot connect words at the start with words at the end.
        </p>
      </StepSection>

      {/* ========================================================================= */}
      {/* STEP 2: The LSTM Cell Structurally */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={2}
        title="The LSTM Cell Blueprint"
        badge="Architectural Anatomy"
        subtitle="Hochreiter & Schmidhuber's breakthrough: separating memory into a linear cell state highway guarded by multiplicative gates."
        interactiveSlot={
          <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-4">
            <div className="font-mono text-xs font-bold text-ink mb-1">
              Interactive Cell Highway Diagram:
            </div>
            
            {/* SVG Cell Anatomy */}
            <svg viewBox="0 0 400 240" className="w-full h-auto rounded-lg bg-surface-bone/50 border border-hairline">
              {/* Cell state highway line */}
              <line x1="30" y1="50" x2="370" y2="50" stroke="#ea2804" strokeWidth="4" strokeDasharray="4 2" />
              <text x="35" y="38" fill="#ea2804" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">Cell State (C_t) Highway</text>

              {/* Hidden state line */}
              <line x1="30" y1="190" x2="370" y2="190" stroke="#3b82f6" strokeWidth="3" />
              <text x="35" y="210" fill="#3b82f6" fontSize="11" fontFamily="JetBrains Mono" fontWeight="bold">Hidden State (h_t)</text>

              {/* Forget Gate */}
              <rect x="70" y="90" width="60" height="50" rx="8" fill="#ea2804" fillOpacity="0.15" stroke="#ea2804" strokeWidth="2" />
              <text x="100" y="115" textAnchor="middle" fill="#ea2804" fontSize="11" fontWeight="bold" fontFamily="JetBrains Mono">f_t (σ)</text>
              <text x="100" y="130" textAnchor="middle" fill="#ea2804" fontSize="8" fontFamily="Space Grotesk">Forget</text>
              <line x1="100" y1="90" x2="100" y2="56" stroke="#ea2804" strokeWidth="2" />
              <circle cx="100" cy="50" r="8" fill="#ffffff" stroke="#ea2804" strokeWidth="2" />
              <text x="100" y="53" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#ea2804">×</text>

              {/* Input Gate + Candidate */}
              <rect x="170" y="90" width="60" height="50" rx="8" fill="#10b981" fillOpacity="0.15" stroke="#10b981" strokeWidth="2" />
              <text x="200" y="115" textAnchor="middle" fill="#10b981" fontSize="11" fontWeight="bold" fontFamily="JetBrains Mono">i_t ⊙ C̃</text>
              <text x="200" y="130" textAnchor="middle" fill="#10b981" fontSize="8" fontFamily="Space Grotesk">Write</text>
              <line x1="200" y1="90" x2="200" y2="56" stroke="#10b981" strokeWidth="2" />
              <circle cx="200" cy="50" r="8" fill="#ffffff" stroke="#10b981" strokeWidth="2" />
              <text x="200" y="53" textAnchor="middle" fontSize="10" fontWeight="bold" fill="#10b981">+</text>

              {/* Output Gate */}
              <rect x="270" y="90" width="60" height="50" rx="8" fill="#3b82f6" fillOpacity="0.15" stroke="#3b82f6" strokeWidth="2" />
              <text x="300" y="115" textAnchor="middle" fill="#3b82f6" fontSize="11" fontWeight="bold" fontFamily="JetBrains Mono">o_t (σ)</text>
              <text x="300" y="130" textAnchor="middle" fill="#3b82f6" fontSize="8" fontFamily="Space Grotesk">Output</text>
              <line x1="200" y1="50" x2="280" y2="185" stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="3 3" />
            </svg>

            <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono">
              <div className="p-2 rounded bg-primary/10 border border-primary/20 text-primary">
                1. Forget (f_t)
              </div>
              <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-700">
                2. Input (i_t)
              </div>
              <div className="p-2 rounded bg-blue-500/10 border border-blue-500/20 text-blue-700">
                3. Output (o_t)
              </div>
            </div>
          </div>
        }
      >
        <p>
          The LSTM solves vanishing gradients by splitting recurrent state into two distinct vectors:
        </p>
        <ul className="list-disc pl-5 space-y-2 text-sm text-charcoal">
          <li>
            <strong className="text-ink font-semibold">Cell State (<InlineMath math="C_t" />):</strong> The uninterrupted conveyor-belt highway. Information flows down it with only linear additive operations, avoiding continuous matrix product decays.
          </li>
          <li>
            <strong className="text-ink font-semibold">Hidden State (<InlineMath math="h_t" />):</strong> The filtered working memory exposed to the outside world and the next timestep.
          </li>
        </ul>
        <p className="text-sm text-charcoal mt-3">
          Three specialized neural gates act as valves regulating what enters, stays, and leaves this conveyor belt:
        </p>
        <div className="p-3 rounded-xl bg-surface-card border border-hairline font-mono text-xs space-y-1">
          <div><span className="text-primary font-bold">Forget Gate (f_t):</span> What percentage of old memory to erase.</div>
          <div><span className="text-emerald-600 font-bold">Input Gate (i_t):</span> What new candidate info to write into memory.</div>
          <div><span className="text-blue-600 font-bold">Output Gate (o_t):</span> What portion of memory to emit as the next prediction.</div>
        </div>
      </StepSection>

      {/* ========================================================================= */}
      {/* STEP 3: Forget Gate */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={3}
        title="The Forget Gate (f_t)"
        badge="Memory Eraser"
        subtitle="Decides what fraction of prior long-term memory should be discarded."
        interactiveSlot={
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-charcoal">Raw Linear Activation (z_f = W_f x + b_f):</span>
              <span className="font-bold text-primary">{forgetLogit.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-4.0"
              max="4.0"
              step="0.1"
              value={forgetLogit}
              onChange={(e) => setForgetLogit(parseFloat(e.target.value))}
              className="w-full accent-primary"
            />

            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-3">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-semibold text-ink">Sigmoid Output (f_t = σ(z_f)):</span>
                <span className="text-sm font-bold text-primary">{(forgetSigmoid * 100).toFixed(1)}%</span>
              </div>

              {/* Gauge Bar */}
              <div className="w-full bg-surface-bone rounded-full h-5 p-0.5 border border-hairline">
                <div
                  className="bg-primary h-full rounded-full transition-all duration-150 flex items-center justify-end pr-2 text-[10px] font-mono text-white font-bold"
                  style={{ width: `${Math.max(8, forgetSigmoid * 100)}%` }}
                >
                  {forgetSigmoid > 0.3 && `${forgetSigmoid.toFixed(2)}`}
                </div>
              </div>

              <div className="text-[11px] text-charcoal bg-canvas p-2.5 rounded-lg border border-hairline font-mono">
                {forgetSigmoid < 0.2
                  ? '🛑 Close to 0.0: Prior subject context is almost completely forgotten.'
                  : forgetSigmoid > 0.8
                  ? '🟢 Close to 1.0: Full retention. Memory passes forward untouched.'
                  : '⚖️ Partial retention: Old context is scaled down proportionally.'}
              </div>
            </div>
          </div>
        }
      >
        <p>
          The forget gate looks at the previous hidden state <InlineMath math="h_{t-1}" /> and current input <InlineMath math="x_t" />, producing a number between 0 and 1 for each number in the cell state <InlineMath math="C_{t-1}" />:
        </p>
        <Equation
          math="f_t = \sigma\left(W_f \cdot [h_{t-1}, x_t] + b_f\right)"
          explanation="Forget gate activation vector"
          terms={[
            { symbol: '\sigma(z)', meaning: 'Sigmoid activation 1/(1+e^{-z})' },
            { symbol: 'W_f', meaning: 'Forget gate weight matrix' },
            { symbol: 'b_f', meaning: 'Forget gate bias vector' },
          ]}
        />
        <div className="space-y-2 text-sm text-charcoal">
          <p>
            <strong>Plain English intuition:</strong> Imagine reading a story where a new paragraph starts with <em>"Meanwhile, Bob went to the store..."</em>. The forget gate sees the new subject "Bob" and drops the previous character's grammatical gender from the cell state.
          </p>
          <p>
            A value of <code className="font-mono text-xs bg-surface-bone px-1 rounded text-ink">1.0</code> means "completely keep this information," while <code className="font-mono text-xs bg-surface-bone px-1 rounded text-ink">0.0</code> means "completely forget this information."
          </p>
        </div>
      </StepSection>

      {/* ========================================================================= */}
      {/* STEP 4: Input Gate + Candidate Cell State */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={4}
        title="Input Gate (i_t) & Candidate Memory (C̃_t)"
        badge="Memory Writer"
        subtitle="Decides what brand new information to inject into the conveyor belt."
        interactiveSlot={
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-charcoal">Input Gate Filter (z_i):</span>
              <span className="font-bold text-emerald-600">{inputLogit.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-3.0"
              max="3.0"
              step="0.1"
              value={inputLogit}
              onChange={(e) => setInputLogit(parseFloat(e.target.value))}
              className="w-full accent-emerald-500"
            />

            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-charcoal">Candidate Info (z_c tanh):</span>
              <span className="font-bold text-ink">{candidateLogit.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-3.0"
              max="3.0"
              step="0.1"
              value={candidateLogit}
              onChange={(e) => setCandidateLogit(parseFloat(e.target.value))}
              className="w-full accent-charcoal"
            />

            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span>Filter i_t = σ(z_i):</span>
                <span className="font-bold text-emerald-600">{(inputSigmoid * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Candidate C̃_t = tanh(z_c):</span>
                <span className="font-bold text-ink">{candidateTanh.toFixed(3)}</span>
              </div>
              <div className="pt-2 border-t border-hairline flex justify-between font-semibold text-sm">
                <span>Injected Memory (i_t ⊙ C̃_t):</span>
                <span className="text-emerald-700 font-bold">{inputUpdate.toFixed(3)}</span>
              </div>
            </div>
          </div>
        }
      >
        <p>
          Updating memory is a two-part process:
        </p>
        <ol className="list-decimal pl-5 space-y-2 text-sm text-charcoal">
          <li>
            <strong className="text-ink font-semibold">The Input Gate (<InlineMath math="i_t" />):</strong> A sigmoid layer that decides which values will be updated.
          </li>
          <li>
            <strong className="text-ink font-semibold">The Candidate Vector (<InlineMath math="\tilde{C}_t" />):</strong> A tanh layer creating a vector of new candidate values that could be added to the state.
          </li>
        </ol>
        <Equation
          math="i_t = \sigma\left(W_i \cdot [h_{t-1}, x_t] + b_i\right)"
          explanation="Input gate activation"
        />
        <Equation
          math="\tilde{C}_t = \tanh\left(W_c \cdot [h_{t-1}, x_t] + b_c\right)"
          explanation="New candidate values bounded within [-1, +1]"
        />
        <p className="text-sm text-charcoal">
          The element-wise product <InlineMath math="i_t \odot \tilde{C}_t" /> scales the new information by how important the network believes it to be before adding it to the long-term memory.
        </p>
      </StepSection>

      {/* ========================================================================= */}
      {/* STEP 5: Cell State Update (Conveyor Belt) */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={5}
        title="The Cell State Conveyor Belt Update"
        badge="Core Highway Equation"
        subtitle="Combining forgotten past memory with filtered new candidate memory."
        interactiveSlot={
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-charcoal">Previous Cell State (C_t-1):</span>
              <span className="font-bold text-amber-600">{prevCellState.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-4.0"
              max="4.0"
              step="0.2"
              value={prevCellState}
              onChange={(e) => setPrevCellState(parseFloat(e.target.value))}
              className="w-full accent-amber-500"
            />

            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-3 font-mono text-xs">
              <div className="flex justify-between items-center text-primary">
                <span>Retained Past (f_t · C_t-1):</span>
                <span className="font-bold">{(forgetSigmoid * prevCellState).toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-center text-emerald-600">
                <span>Added Info (i_t · C̃_t):</span>
                <span className="font-bold">{inputUpdate.toFixed(3)}</span>
              </div>
              <div className="pt-2 border-t border-hairline flex justify-between items-center text-sm font-bold text-ink">
                <span>New Cell State (C_t):</span>
                <span className="px-2 py-0.5 rounded bg-surface-bone text-primary">{updatedCellState.toFixed(3)}</span>
              </div>
            </div>
          </div>
        }
      >
        <p>
          Now we execute the central update of the LSTM: multiplying the old state by <InlineMath math="f_t" /> (dropping what we decided to forget) and adding <InlineMath math="i_t \odot \tilde{C}_t" /> (writing new scaled candidates):
        </p>
        <Equation
          math="C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t"
          explanation="Linear combination on the cell highway"
          terms={[
            { symbol: 'f_t \odot C_{t-1}', meaning: 'Decayed old memory' },
            { symbol: 'i_t \odot \tilde{C}_t', meaning: 'Injected new memory' },
            { symbol: '\odot', meaning: 'Hadamard (element-wise) product' },
          ]}
        />
        <p className="text-sm text-charcoal">
          <strong>Why this prevents vanishing gradients:</strong> Notice that <InlineMath math="\frac{\partial C_t}{\partial C_{t-1}} = f_t" />. If the forget gate is saturated near 1.0, error gradients flow backwards along the conveyor belt with <em>zero decay</em> across arbitrarily long timesteps.
        </p>
      </StepSection>

      {/* ========================================================================= */}
      {/* STEP 6: Output Gate + Hidden State */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={6}
        title="Output Gate (o_t) & Hidden State (h_t)"
        badge="Memory Projector"
        subtitle="Filtering cell state through tanh to generate the emitted hidden state."
        interactiveSlot={
          <div className="space-y-4">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-charcoal">Output Gate Logit (z_o):</span>
              <span className="font-bold text-blue-600">{outputLogit.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="-3.0"
              max="3.0"
              step="0.1"
              value={outputLogit}
              onChange={(e) => setOutputLogit(parseFloat(e.target.value))}
              className="w-full accent-blue-500"
            />

            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-2 font-mono text-xs">
              <div className="flex justify-between">
                <span>Output Valve (o_t = σ(z_o)):</span>
                <span className="font-bold text-blue-600">{(outputSigmoid * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>Squashed Memory tanh(C_t):</span>
                <span className="font-bold text-ink">{Math.tanh(updatedCellState).toFixed(3)}</span>
              </div>
              <div className="pt-2 border-t border-hairline flex justify-between items-center text-sm font-bold">
                <span>Hidden Output (h_t):</span>
                <span className="px-2 py-0.5 rounded bg-surface-bone text-blue-600">{hiddenOutput.toFixed(3)}</span>
              </div>
            </div>
          </div>
        }
      >
        <p>
          Finally, the network decides what to output as working memory. The output is based on the cell state, but filtered:
        </p>
        <Equation
          math="o_t = \sigma\left(W_o \cdot [h_{t-1}, x_t] + b_o\right)"
          explanation="Output gate activation vector"
        />
        <Equation
          math="h_t = o_t \odot \tanh(C_t)"
          explanation="Projected hidden state exposed to outside layers"
        />
        <p className="text-sm text-charcoal">
          We pass the cell state through <InlineMath math="\tanh" /> (to squash the values between -1 and 1) and multiply it by the output gate <InlineMath math="o_t" />, so that we only output the parts we decided to. This <InlineMath math="h_t" /> is what feeds the classifier head or next token predictor.
        </p>
      </StepSection>

      {/* ========================================================================= */}
      {/* STEP 7: Putting It Together Across a Sequence */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={7}
        title="Putting It Together Across a Real Sequence"
        badge="Live PyTorch Forward Pass"
        subtitle="Watch actual gate activations computed token-by-token by a trained LSTM model."
        className="bg-surface-bone/30"
      >
        <div className="space-y-6">
          <p>
            Every number below is queried from our live FastAPI backend executing custom <code className="font-mono bg-surface-bone px-1 rounded text-ink">ExposedLSTMCell</code> loops in PyTorch. Click a preset sentence or type your own:
          </p>

          {/* Quick presets */}
          <div className="flex flex-wrap gap-2">
            {[
              'the movie was not very good',
              'artificial intelligence will transform software development',
              'the model learns patterns from data',
            ].map((preset, idx) => (
              <button
                key={idx}
                onClick={() => setTraceText(preset)}
                className={`px-3 py-1 rounded-full text-xs font-mono transition-colors border ${
                  traceText === preset
                    ? 'bg-primary text-white border-primary'
                    : 'bg-surface-card text-charcoal border-hairline hover:bg-canvas'
                }`}
              >
                "{preset}"
              </button>
            ))}
          </div>

          <div className="relative">
            <input
              type="text"
              value={traceText}
              onChange={(e) => setTraceText(e.target.value)}
              placeholder="Enter sentence for real-time gate unroll..."
              className="w-full text-input font-medium pr-10"
            />
            {traceLoading && (
              <div className="absolute right-4 top-1/2 -translate-y-1/2">
                <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
              </div>
            )}
          </div>

          {/* Real PyTorch Heatmap & Trace */}
          <div className="space-y-6 pt-2">
            <GateHeatmap
              steps={traceSteps}
              highlightIndex={activeTokenIdx}
              onHoverStep={(step) => setActiveTokenIdx(step ? step.t : null)}
              title="Live Unrolled Gate Activations (ExposedLSTMCell)"
              subtitle="Hover any column to inspect exact Forget, Input, Output values and Cell State magnitude"
            />

            <CellStateTrace
              steps={traceSteps}
              title="Conveyor Belt Accumulation (||c_t||)"
              subtitle="Tracks the L2-norm magnitude of the internal cell state across the sequence"
            />
          </div>
        </div>
      </StepSection>

      {/* ========================================================================= */}
      {/* STEP 8: Why This Matters & Transition to Playground */}
      {/* ========================================================================= */}
      <StepSection
        stepNumber={8}
        title="Why Understanding Internals Matters"
        badge="Synthesis & Next Steps"
        subtitle="Now that you know the math, see how this architecture performs in three real applications."
      >
        <div className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-1.5">
              <span className="font-mono text-xs text-primary font-bold">Key Insight 1</span>
              <h4 className="font-display font-bold text-sm text-ink">Additive Highway</h4>
              <p className="text-xs text-charcoal">
                Linear updates prevent the exponential vanishing of backpropagation gradients.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-1.5">
              <span className="font-mono text-xs text-emerald-600 font-bold">Key Insight 2</span>
              <h4 className="font-display font-bold text-sm text-ink">Dynamic Filtering</h4>
              <p className="text-xs text-charcoal">
                Gates dynamically adapt memory retention based on the current context word.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-surface-card border border-hairline space-y-1.5">
              <span className="font-mono text-xs text-blue-600 font-bold">Key Insight 3</span>
              <h4 className="font-display font-bold text-sm text-ink">Separated Memory</h4>
              <p className="text-xs text-charcoal">
                Long-term invariant state (<InlineMath math="C_t" />) remains decoupled from working state (<InlineMath math="h_t" />).
              </p>
            </div>
          </div>

          <div className="p-6 rounded-2xl bg-surface-dark text-on-dark space-y-4">
            <h3 className="font-display font-bold text-xl text-white">
              Ready to see real-world models in action?
            </h3>
            <p className="text-sm text-on-dark-mute leading-relaxed">
              We've trained 3 real PyTorch LSTM models on language modeling, anomaly telemetry, and sentiment sequences. Jump into the Playground to test them live with real-time gate telemetry.
            </p>
            <div className="flex flex-wrap gap-3 pt-2">
              <button
                onClick={() => onNavigate('playground-nextword')}
                className="btn-primary text-xs px-5 py-2.5"
              >
                1. Next-Word Predictor →
              </button>
              <button
                onClick={() => onNavigate('playground-anomaly')}
                className="btn-outline text-xs px-5 py-2.5 text-white bg-transparent border-white/20 hover:bg-white/10"
              >
                2. Anomaly Detector →
              </button>
              <button
                onClick={() => onNavigate('playground-sentiment')}
                className="btn-outline text-xs px-5 py-2.5 text-white bg-transparent border-white/20 hover:bg-white/10"
              >
                3. Sentiment Classifier →
              </button>
            </div>
          </div>
        </div>
      </StepSection>

    </div>
  );
};
