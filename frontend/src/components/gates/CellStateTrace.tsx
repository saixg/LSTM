import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import type { GateStep } from '../../lib/types';

interface CellStateTraceProps {
  steps: GateStep[];
  title?: string;
  subtitle?: string;
  className?: string;
}

export const CellStateTrace: React.FC<CellStateTraceProps> = ({
  steps,
  title = "Cell State Magnitude (||c_t||) Trajectory",
  subtitle = "Vector L2-norm of the cell state vector across the sequence",
  className = "",
}) => {
  const data = steps.map((s) => ({
    time: `t=${s.t}`,
    token: s.token,
    norm: parseFloat(s.cell_state_norm.toFixed(3)),
    forget: parseFloat((s.forget * 100).toFixed(1)),
    input: parseFloat((s.input * 100).toFixed(1)),
    output: parseFloat((s.output * 100).toFixed(1)),
  }));

  return (
    <div className={`p-5 rounded-2xl bg-surface-card border border-hairline shadow-sm ${className}`}>
      <div className="flex items-center justify-between gap-2 mb-4 pb-3 border-b border-hairline">
        <div>
          <h4 className="font-display font-bold text-lg text-ink flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
            {title}
          </h4>
          <p className="text-xs text-mute font-body mt-0.5">{subtitle}</p>
        </div>
        <div className="px-2.5 py-1 rounded-full bg-surface-bone text-charcoal text-xs font-mono font-medium border border-hairline">
          {steps.length} Timesteps Unrolled
        </div>
      </div>

      <div className="h-48 w-full">
        {steps.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-mute font-body">
            Awaiting tokens to plot cell state trajectory...
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="cellNormGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ea2804" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ea2804" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(32,32,32,0.06)" />
              <XAxis
                dataKey="token"
                stroke="#8d8d8d"
                fontSize={11}
                tickLine={false}
                tickFormatter={(val) => (val.length > 6 ? val.slice(0, 5) + '…' : val)}
              />
              <YAxis
                stroke="#8d8d8d"
                fontSize={11}
                tickLine={false}
                domain={['auto', 'auto']}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="p-3 rounded-xl bg-surface-dark text-on-dark text-xs shadow-xl border border-white/10 font-mono">
                        <div className="font-sans font-bold text-sm text-white mb-1.5 flex items-center gap-2">
                          <span className="text-primary font-mono">{d.time}:</span> "{d.token}"
                        </div>
                        <div className="space-y-1 text-on-dark-mute">
                          <div>||c_t|| Magnitude: <span className="text-white font-bold">{d.norm}</span></div>
                          <div>Forget Rate: <span className="text-primary">{d.forget}%</span></div>
                          <div>Write Rate: <span className="text-emerald-400">{d.input}%</span></div>
                          <div>Emit Rate: <span className="text-blue-400">{d.output}%</span></div>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area
                type="monotone"
                dataKey="norm"
                stroke="#ea2804"
                strokeWidth={2.5}
                fillOpacity={1}
                fill="url(#cellNormGrad)"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
