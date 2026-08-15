import React, { useState, useEffect } from 'react';
import { fetchAnomalySamples, fetchAnomalyDetect } from '../../lib/api';
import { GateHeatmap } from '../../components/gates/GateHeatmap';
import type { AnomalySample, AnomalyDetectResponse } from '../../lib/types';
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

export const AnomalyDetector: React.FC = () => {
  const [samples, setSamples] = useState<AnomalySample[]>([]);
  const [selectedId, setSelectedId] = useState<string>('normal_0');
  const [injectAnomaly, setInjectAnomaly] = useState(false);
  const [data, setData] = useState<AnomalyDetectResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadSamples = async () => {
      try {
        const res = await fetchAnomalySamples();
        setSamples(res);
        if (res.length > 0) setSelectedId(res[0].id);
      } catch (err) {
        console.error('Failed to load anomaly samples:', err);
      }
    };
    loadSamples();
  }, []);

  useEffect(() => {
    let active = true;
    const runDetect = async () => {
      if (!selectedId) return;
      setLoading(true);
      try {
        const res = await fetchAnomalyDetect(selectedId, injectAnomaly);
        if (active) setData(res);
      } catch (err) {
        console.error('Anomaly detect error:', err);
      } finally {
        if (active) setLoading(false);
      }
    };
    runDetect();
    return () => {
      active = false;
    };
  }, [selectedId, injectAnomaly]);

  const chartData = data
    ? data.input.map((val, idx) => ({
        time: idx,
        input: parseFloat(val.toFixed(3)),
        reconstruction: parseFloat(data.reconstruction[idx].toFixed(3)),
        error: parseFloat(data.error[idx].toFixed(3)),
        isAnomaly: data.anomaly_flags[idx],
      }))
    : [];

  const anomalyCount = data ? data.anomaly_flags.filter(Boolean).length : 0;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      
      {/* App Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-hairline">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 font-mono text-xs font-bold">
              PLAYGROUND APP 02
            </span>
            <span className="text-xs font-mono text-mute">LSTM Autoencoder Time-Series</span>
          </div>
          <h1 className="font-display font-extrabold text-3xl sm:text-4xl text-ink">
            Time-Series Anomaly Detector
          </h1>
          <p className="mt-1 text-sm sm:text-base text-charcoal max-w-2xl">
            An encoder-decoder LSTM learns the normal manifold of sequential telemetry. Unexpected spikes cause reconstruction failure (MSE &gt; 0.5) and gate turbulence.
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Sample Sequence Selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs font-mono text-charcoal font-semibold">Sequence:</label>
            <select
              value={selectedId}
              onChange={(e) => setSelectedId(e.target.value)}
              className="text-xs font-mono bg-surface-card border border-hairline rounded-xl px-3 py-2 text-ink shadow-sm focus:outline-none focus:ring-2 focus:ring-primary"
            >
              {samples.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.id})
                </option>
              ))}
            </select>
          </div>

          {/* Inject Synthetic Anomaly Button */}
          <button
            onClick={() => setInjectAnomaly((prev) => !prev)}
            className={`text-xs font-semibold px-4 py-2 rounded-xl transition-all flex items-center gap-2 border shadow-sm ${
              injectAnomaly
                ? 'bg-rose-500 text-white border-rose-600 animate-pulse'
                : 'bg-surface-card text-charcoal border-hairline hover:bg-surface-bone'
            }`}
          >
            <span>{injectAnomaly ? '⚡ Anomaly Injected' : '💉 Inject Synthetic Anomaly'}</span>
          </button>
        </div>
      </div>

      {/* Metrics Banner */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-surface-card border border-hairline shadow-sm">
          <span className="text-xs font-mono text-mute">Sequence Length</span>
          <div className="font-display font-bold text-xl text-ink mt-1">
            {data ? data.input.length : 0} Timesteps
          </div>
        </div>

        <div className="p-4 rounded-xl bg-surface-card border border-hairline shadow-sm">
          <span className="text-xs font-mono text-mute">Anomaly Threshold</span>
          <div className="font-display font-bold text-xl text-ink mt-1 font-mono">
            0.50 MSE
          </div>
        </div>

        <div className="p-4 rounded-xl bg-surface-card border border-hairline shadow-sm">
          <span className="text-xs font-mono text-mute">Outliers Detected</span>
          <div className={`font-display font-bold text-xl mt-1 ${anomalyCount > 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
            {anomalyCount} Anomalous Steps
          </div>
        </div>

        <div className="p-4 rounded-xl bg-surface-card border border-hairline shadow-sm">
          <span className="text-xs font-mono text-mute">Model Architecture</span>
          <div className="font-display font-bold text-sm text-ink mt-1 font-mono">
            Encoder (32) → Latent → Decoder (32)
          </div>
        </div>
      </div>

      {/* Main Signal Reconstruction & Error Chart */}
      <div className="p-6 rounded-2xl bg-surface-card border border-hairline shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4 pb-3 border-b border-hairline">
          <div>
            <h3 className="font-display font-bold text-base text-ink flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
              Original Signal vs Autoencoder Reconstruction
            </h3>
            <p className="text-xs text-mute font-body">
              Reconstruction diverges dramatically in regions outside the training distribution.
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-1.5 text-charcoal">
              <span className="w-3 h-0.5 bg-blue-500"></span> Input Signal (x_t)
            </span>
            <span className="flex items-center gap-1.5 text-charcoal">
              <span className="w-3 h-0.5 bg-emerald-500"></span> Reconstructed (x̂_t)
            </span>
            <span className="flex items-center gap-1.5 text-rose-600 font-semibold">
              <span className="w-3 h-0.5 bg-rose-500"></span> MSE Error (e_t)
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center text-xs text-mute">
              Running LSTM Autoencoder reconstruction...
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(32,32,32,0.06)" />
                <XAxis dataKey="time" stroke="#8d8d8d" fontSize={11} tickLine={false} />
                <YAxis stroke="#8d8d8d" fontSize={11} tickLine={false} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="p-3 rounded-xl bg-surface-dark text-on-dark text-xs shadow-xl border border-white/10 font-mono">
                          <div className="font-sans font-bold text-white mb-1">Timestep t = {d.time}</div>
                          <div>Input: <span className="text-blue-400">{d.input}</span></div>
                          <div>Reconstructed: <span className="text-emerald-400">{d.reconstruction}</span></div>
                          <div>MSE Error: <span className={d.error > 0.5 ? 'text-rose-400 font-bold' : 'text-white'}>{d.error}</span></div>
                          {d.isAnomaly && (
                            <div className="mt-1.5 px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-400 font-bold text-[10px] inline-block">
                              ⚠️ ANOMALY DETECTED
                            </div>
                          )}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine y={0.5} stroke="#ea2804" strokeDasharray="4 4" label={{ value: 'Threshold 0.5', fill: '#ea2804', fontSize: 10, position: 'insideTopRight' }} />
                <Line type="monotone" dataKey="input" stroke="#3b82f6" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="reconstruction" stroke="#10b981" strokeWidth={2} strokeDasharray="4 2" dot={false} />
                <Line type="monotone" dataKey="error" stroke="#ea2804" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Shared Gate Heatmap for Encoder */}
      {data && (
        <div className="space-y-6 pt-4">
          <GateHeatmap
            steps={data.steps}
            title="Encoder Gate Activations"
            subtitle="Observe how the encoder's forget (f_t) and input (i_t) gates fluctuate when unexpected anomalies occur"
          />
        </div>
      )}

    </div>
  );
};
