import React, { useEffect, useRef, useState } from 'react';
import type { GateStep } from '../../lib/types';

interface GateHeatmapProps {
  steps: GateStep[];
  title?: string;
  subtitle?: string;
  highlightIndex?: number | null;
  onHoverStep?: (step: GateStep | null) => void;
  className?: string;
}

export const GateHeatmap: React.FC<GateHeatmapProps> = ({
  steps,
  title = "Real-time Gate Activation Heatmap",
  subtitle = "Internal σ activations (f_t, i_t, o_t) extracted from PyTorch ExposedLSTMCell",
  highlightIndex = null,
  onHoverStep,
  className = "",
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const gates = [
    { key: 'forget' as const, label: 'Forget Gate (f_t)', symbol: 'f_t', color: [234, 40, 4], desc: 'Retention factor [0=forget, 1=retain]' },
    { key: 'input' as const, label: 'Input Gate (i_t)', symbol: 'i_t', color: [16, 185, 129], desc: 'Write intensity [0=ignore, 1=write]' },
    { key: 'output' as const, label: 'Output Gate (o_t)', symbol: 'o_t', color: [59, 130, 246], desc: 'Exposure factor [0=mask, 1=emit]' },
  ];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.parentElement?.clientWidth || 700;
    const rowHeight = 36;
    const headerHeight = 28;
    const footerHeight = 44;
    const leftGutter = 140;
    const totalHeight = headerHeight + gates.length * rowHeight + footerHeight;

    canvas.width = width * dpr;
    canvas.height = totalHeight * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${totalHeight}px`;

    ctx.resetTransform?.();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, totalHeight);

    if (!steps || steps.length === 0) {
      ctx.fillStyle = '#8d8d8d';
      ctx.font = '14px "Space Grotesk", sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting input stream to unroll LSTM gates...', width / 2, totalHeight / 2);
      return;
    }

    const availableWidth = width - leftGutter - 20;
    const colWidth = Math.max(28, Math.min(64, availableWidth / steps.length));

    // Background card fill
    ctx.fillStyle = '#ffffff';
    ctx.roundRect?.(0, 0, width, totalHeight, 8);
    ctx.fill();

    // Draw row labels & grid
    gates.forEach((gate, rIdx) => {
      const y = headerHeight + rIdx * rowHeight;

      // Row background
      ctx.fillStyle = rIdx % 2 === 0 ? '#fdfdfc' : '#f9f7f3';
      ctx.fillRect(leftGutter, y, steps.length * colWidth, rowHeight);

      // Row label
      ctx.fillStyle = '#202020';
      ctx.font = '600 12px "Outfit", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(gate.label, 12, y + rowHeight / 2);

      // Gate cell squares
      steps.forEach((step, cIdx) => {
        const x = leftGutter + cIdx * colWidth;
        const val = Math.max(0, Math.min(1, step[gate.key]));
        const [r, g, b] = gate.color;

        // Fill color interpolated with opacity/brightness
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.15 + val * 0.85})`;
        ctx.fillRect(x + 1, y + 1, colWidth - 2, rowHeight - 2);

        // Highlight border if active or hovered
        const isHovered = hoveredIdx === cIdx || highlightIndex === cIdx;
        if (isHovered) {
          ctx.strokeStyle = '#202020';
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, colWidth - 2, rowHeight - 2);
        } else {
          ctx.strokeStyle = 'rgba(32,32,32,0.06)';
          ctx.lineWidth = 1;
          ctx.strokeRect(x, y, colWidth, rowHeight);
        }

        // Numerical readout inside cell if colWidth allows
        if (colWidth >= 36) {
          ctx.fillStyle = val > 0.6 ? '#ffffff' : '#202020';
          ctx.font = '500 10px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(val.toFixed(2), x + colWidth / 2, y + rowHeight / 2);
        }
      });
    });

    // Draw Column Headers / Footers (Tokens and Timesteps)
    steps.forEach((step, cIdx) => {
      const x = leftGutter + cIdx * colWidth + colWidth / 2;

      // Timestep above
      ctx.fillStyle = '#8d8d8d';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(`t=${step.t}`, x, headerHeight - 6);

      // Token label below
      const isHovered = hoveredIdx === cIdx || highlightIndex === cIdx;
      ctx.fillStyle = isHovered ? '#ea2804' : '#202020';
      ctx.font = isHovered ? '700 12px "Space Grotesk", sans-serif' : '500 12px "Space Grotesk", sans-serif';
      ctx.textBaseline = 'top';

      // Truncate long token string
      const tokenDisplay = step.token.length > 8 ? step.token.slice(0, 7) + '…' : step.token;
      ctx.fillText(`"${tokenDisplay}"`, x, headerHeight + gates.length * rowHeight + 8);

      // Norm subtext
      ctx.fillStyle = '#646464';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText(`||c||=${step.cell_state_norm.toFixed(1)}`, x, headerHeight + gates.length * rowHeight + 24);
    });

  }, [steps, hoveredIdx, highlightIndex]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !steps.length) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const leftGutter = 140;
    const availableWidth = canvas.clientWidth - leftGutter - 20;
    const colWidth = Math.max(28, Math.min(64, availableWidth / steps.length));

    if (x >= leftGutter && x <= leftGutter + steps.length * colWidth) {
      const idx = Math.floor((x - leftGutter) / colWidth);
      if (idx >= 0 && idx < steps.length) {
        setHoveredIdx(idx);
        onHoverStep?.(steps[idx]);
        return;
      }
    }
    setHoveredIdx(null);
    onHoverStep?.(null);
  };

  const handleMouseLeave = () => {
    setHoveredIdx(null);
    onHoverStep?.(null);
  };

  const activeStep = hoveredIdx !== null && steps[hoveredIdx] ? steps[hoveredIdx] : null;

  return (
    <div className={`p-5 rounded-2xl bg-surface-card border border-hairline shadow-sm overflow-hidden ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-hairline">
        <div>
          <h4 className="font-display font-bold text-lg text-ink flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse"></span>
            {title}
          </h4>
          <p className="text-xs text-mute font-body mt-0.5">{subtitle}</p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-medium">
          <span className="flex items-center gap-1.5 text-charcoal">
            <span className="w-3 h-3 rounded-sm bg-primary/80"></span> Forget (<code className="font-mono text-[11px]">f_t</code>)
          </span>
          <span className="flex items-center gap-1.5 text-charcoal">
            <span className="w-3 h-3 rounded-sm bg-emerald-500/80"></span> Input (<code className="font-mono text-[11px]">i_t</code>)
          </span>
          <span className="flex items-center gap-1.5 text-charcoal">
            <span className="w-3 h-3 rounded-sm bg-blue-500/80"></span> Output (<code className="font-mono text-[11px]">o_t</code>)
          </span>
        </div>
      </div>

      {/* Canvas Heatmap container */}
      <div className="overflow-x-auto pb-2 scrollbar-thin">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="cursor-crosshair rounded-lg block"
        />
      </div>

      {/* Inspection Tooltip Bar */}
      {activeStep && (
        <div className="mt-3 p-3 rounded-xl bg-canvas border border-hairline flex flex-wrap items-center justify-between gap-3 text-xs animate-fadeIn">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded bg-surface-dark text-on-dark font-mono font-semibold">
              t={activeStep.t}
            </span>
            <span className="font-display font-bold text-ink text-sm">
              "{activeStep.token}"
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-4 font-mono">
            <span className="text-primary font-semibold">
              Forget (f_t): {(activeStep.forget * 100).toFixed(1)}%
            </span>
            <span className="text-emerald-600 font-semibold">
              Input (i_t): {(activeStep.input * 100).toFixed(1)}%
            </span>
            <span className="text-blue-600 font-semibold">
              Output (o_t): {(activeStep.output * 100).toFixed(1)}%
            </span>
            <span className="text-charcoal font-medium">
              ||c_t||: {activeStep.cell_state_norm.toFixed(3)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
