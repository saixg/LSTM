import React from 'react';
import { InlineMath, BlockMath } from 'react-katex';

interface EquationProps {
  math: string;
  block?: boolean;
  explanation?: string;
  terms?: { symbol: string; meaning: string }[];
  className?: string;
}

export const Equation: React.FC<EquationProps> = ({
  math,
  block = true,
  explanation,
  terms,
  className = '',
}) => {
  return (
    <div className={`my-3 p-4 rounded-xl bg-surface-card border border-hairline shadow-sm ${className}`}>
      <div className="overflow-x-auto text-ink py-1 text-center font-medium">
        {block ? <BlockMath math={math} /> : <InlineMath math={math} />}
      </div>
      {explanation && (
        <p className="mt-2 text-sm text-charcoal text-center leading-relaxed">
          {explanation}
        </p>
      )}
      {terms && terms.length > 0 && (
        <div className="mt-3 pt-3 border-t border-hairline grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2 text-xs">
          {terms.map((t, idx) => (
            <div key={idx} className="flex items-center space-x-1.5 bg-canvas/60 px-2.5 py-1 rounded-md border border-hairline/60">
              <span className="font-mono font-semibold text-primary"><InlineMath math={t.symbol} /></span>
              <span className="text-charcoal">: {t.meaning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
