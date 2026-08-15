import React, { useState, useEffect } from 'react';
import { fetchSentimentClassify } from '../../lib/api';
import { GateHeatmap } from '../../components/gates/GateHeatmap';
import { CellStateTrace } from '../../components/gates/CellStateTrace';
import type { SentimentResponse } from '../../lib/types';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';

export const SentimentClassifier: React.FC = () => {
  const [text, setText] = useState('the food was amazing but the service was terrible');
  const [data, setData] = useState<SentimentResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const demoPhrases = [
    'the food was amazing but the service was terrible',
    'the movie started boring but the ending was fantastic',
    'great plot fantastic acting and wonderful direction',
    'terrible script awful acting and horrible dialogue',
  ];

  useEffect(() => {
    let active = true;
    const runClassify = async () => {
      if (!text.trim()) {
        setData(null);
        return;
      }
      setLoading(true);
      try {
        const res = await fetchSentimentClassify(text);
        if (active) setData(res);
      } catch (err) {
        console.error('Sentiment classify error:', err);
      } finally {
        if (active) setLoading(false);
      }
    };

    const timer = setTimeout(runClassify, 150);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [text]);

  const trajectoryData = data
    ? data.tokens.map((tok, idx) => ({
        token: tok,
        time: idx,
        score: parseFloat((data.running_score[idx] * 100).toFixed(1)),
        prob: data.running_score[idx],
      }))
    : [];

  const finalScore = data && data.running_score.length > 0
    ? data.running_score[data.running_score.length - 1]
    : 0.5;

  const isPositive = finalScore >= 0.5;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      
      {/* App Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-hairline">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-600 font-mono text-xs font-bold">
              PLAYGROUND APP 03
            </span>
            <span className="text-xs font-mono text-mute">Token-by-Token Sequence Classification</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-ink">
            Sentiment Sequence Classifier & Pivot Tracker
          </h1>
          <p className="mt-1 text-sm sm:text-base text-charcoal max-w-2xl">
            Watch the neural network update its sentiment probability in real time at every word. Pivot conjunctions like <code className="font-mono text-xs text-primary font-bold">"but"</code> trigger sudden forget-gate erasure of earlier polarity.
          </p>
        </div>

        {/* Quick Demo Sentences */}
        <div className="flex flex-col items-end gap-1.5">
          <span className="text-xs font-mono text-mute">Click Pivot Scenarios:</span>
          <div className="flex flex-wrap gap-1.5 justify-end">
            {demoPhrases.map((phrase, idx) => (
              <button
                key={idx}
                onClick={() => setText(phrase)}
                className="text-xs px-2.5 py-1 rounded-lg bg-surface-card border border-hairline hover:bg-surface-bone text-charcoal transition-colors font-body"
              >
                "{phrase.slice(0, 28)}..."
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Input & Result Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* Left Text Input */}
        <div className="lg:col-span-8 space-y-4">
          <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <label className="font-display font-bold text-sm text-ink flex items-center gap-2">
                <span>Sentence to Classify:</span>
                {loading && <span className="text-xs font-mono text-primary animate-pulse">Classifying...</span>}
              </label>
              <button
                onClick={() => setText('')}
                className="text-xs font-mono text-mute hover:text-ink transition-colors"
              >
                Clear
              </button>
            </div>

            <textarea
              rows={3}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type a review sentence with sentiment transitions..."
              className="w-full text-input rounded-2xl p-4 h-auto text-base font-medium resize-none shadow-inner"
            />
          </div>

          {/* Real-time Token Trajectory Line Chart */}
          <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-hairline">
              <div>
                <h3 className="font-display font-bold text-base text-ink flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-blue-500"></span>
                  Running Polarity Trajectory (Token by Token)
                </h3>
                <p className="text-xs text-mute font-body">
                  Evaluated at every intermediate hidden state h_t across the sequence.
                </p>
              </div>
              <span className="text-xs font-mono text-charcoal">Score: 0% (Negative) to 100% (Positive)</span>
            </div>

            <div className="h-56 w-full">
              {trajectoryData.length === 0 ? (
                <div className="h-full flex items-center justify-center text-xs text-mute">
                  Awaiting text to plot running sentiment trajectory...
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trajectoryData} margin={{ top: 15, right: 20, left: -10, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(32,32,32,0.06)" />
                    <XAxis dataKey="token" stroke="#8d8d8d" fontSize={11} tickLine={false} />
                    <YAxis stroke="#8d8d8d" fontSize={11} tickLine={false} domain={[0, 100]} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (active && payload && payload.length) {
                          const d = payload[0].payload;
                          return (
                            <div className="p-3 rounded-xl bg-surface-dark text-on-dark text-xs shadow-xl border border-white/10 font-mono">
                              <div className="font-sans font-bold text-white mb-1">Token: "{d.token}"</div>
                              <div>Running Probability: <span className={d.score >= 50 ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>{d.score}% Positive</span></div>
                              <div className="text-[10px] text-on-dark-mute mt-1">Sigmoid(W_s · h_{d.time})</div>
                            </div>
                          );
                        }
                        return null;
                      }}
                    />
                    <ReferenceLine y={50} stroke="#8d8d8d" strokeDasharray="3 3" label={{ value: 'Neutral 50%', fill: '#8d8d8d', fontSize: 10, position: 'insideTopLeft' }} />
                    <Line
                      type="monotone"
                      dataKey="score"
                      stroke={isPositive ? '#10b981' : '#ea2804'}
                      strokeWidth={3}
                      dot={{ r: 4, fill: isPositive ? '#10b981' : '#ea2804' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        </div>

        {/* Right Classification Summary Card */}
        <div className="lg:col-span-4 space-y-4">
          <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-4">
            <h3 className="font-display font-bold text-base text-ink pb-3 border-b border-hairline">
              Classification Output
            </h3>

            {data ? (
              <div className="space-y-4">
                
                {/* Local PyTorch LSTM Decision */}
                <div className="p-4 rounded-xl bg-canvas border border-hairline text-center space-y-2">
                  <div className="flex items-center justify-between text-[11px] font-mono text-mute">
                    <span>ENGINE 1: LOCAL PYTORCH LSTM</span>
                    <span className="text-primary font-semibold">ExposedLSTMCell</span>
                  </div>
                  <div className={`inline-flex items-center gap-2 px-5 py-2 rounded-full font-display font-extrabold text-lg shadow-sm ${
                    isPositive ? 'bg-emerald-500/10 text-emerald-600 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-600 border border-rose-500/20'
                  }`}>
                    <span className={`w-2.5 h-2.5 rounded-full ${isPositive ? 'bg-emerald-500' : 'bg-rose-500'}`}></span>
                    {data.final_label} ({ (finalScore * 100).toFixed(1) }%)
                  </div>
                  <div className="w-full bg-surface-bone rounded-full h-2.5 overflow-hidden p-0.5 border border-hairline">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        isPositive ? 'bg-emerald-500' : 'bg-rose-500'
                      }`}
                      style={{ width: `${Math.max(5, finalScore * 100)}%` }}
                    ></div>
                  </div>
                </div>

                {/* API Ninjas Cloud Oracle Benchmark */}
                {data.api_ninjas_label && (
                  <div className="p-4 rounded-xl bg-surface-dark text-on-dark border border-white/10 text-center space-y-2">
                    <div className="flex items-center justify-between text-[11px] font-mono text-on-dark-mute">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse"></span>
                        ENGINE 2: API NINJAS CLOUD
                      </span>
                      <span className="text-blue-400 font-semibold">Live Oracle</span>
                    </div>
                    <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/10 text-white font-mono font-bold text-sm border border-white/15">
                      <span>{data.api_ninjas_label}</span>
                      {data.api_ninjas_score !== undefined && data.api_ninjas_score !== null && (
                        <span className="text-xs text-on-dark-mute">({data.api_ninjas_score > 0 ? `+${data.api_ninjas_score.toFixed(2)}` : data.api_ninjas_score.toFixed(2)})</span>
                      )}
                    </div>
                    <div className="text-[10px] font-mono text-emerald-400 pt-1">
                      ✓ Dual-Engine Real-Time Validation
                    </div>
                  </div>
                )}

                <div className="p-3 rounded-xl bg-surface-bone text-xs text-charcoal space-y-1 font-body">
                  <div className="font-mono font-bold text-ink">💡 The Pivot Effect:</div>
                  <p>
                    Notice how words like "amazing" drive the score up, but a subsequent "terrible" causes the forget gate to scale down previous positive activation and write negative polarity.
                  </p>
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-xs text-mute">
                Awaiting input...
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Shared Gate Heatmap & Line Trace Panels */}
      {data && (
        <div className="space-y-6 pt-4">
          <GateHeatmap
            steps={data.steps}
            title="Sentiment Gate Activation Sequence"
            subtitle="Look at the forget gate (f_t) value when reading contrastive conjunctions"
          />

          <CellStateTrace
            steps={data.steps}
            title="Cell State Norm (||c_t||) Over Review Tokens"
            subtitle="Shows long-term memory magnitude along the sentence"
          />
        </div>
      )}

    </div>
  );
};
