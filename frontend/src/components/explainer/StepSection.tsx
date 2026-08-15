import React from 'react';

interface StepSectionProps {
  stepNumber: number;
  totalSteps?: number;
  title: string;
  badge?: string;
  subtitle?: string;
  children: React.ReactNode;
  interactiveSlot?: React.ReactNode;
  id?: string;
  className?: string;
}

export const StepSection: React.FC<StepSectionProps> = ({
  stepNumber,
  totalSteps = 8,
  title,
  badge,
  subtitle,
  children,
  interactiveSlot,
  id,
  className = '',
}) => {
  return (
    <section
      id={id || `step-${stepNumber}`}
      className={`py-12 sm:py-16 border-b border-hairline/80 scroll-mt-20 ${className}`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        
        {/* Step Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <span className="font-mono text-xs font-bold px-2.5 py-1 rounded-full bg-primary/10 text-primary border border-primary/20">
              STEP {String(stepNumber).padStart(2, '0')} OF {String(totalSteps).padStart(2, '0')}
            </span>
            {badge && (
              <span className="text-xs font-mono font-medium px-2.5 py-1 rounded-full bg-surface-bone text-charcoal border border-hairline">
                {badge}
              </span>
            )}
          </div>
          
          <h2 className="font-display font-bold text-2xl sm:text-3xl text-ink tracking-tight">
            {title}
          </h2>
          
          {subtitle && (
            <p className="mt-2 text-base text-charcoal font-body leading-relaxed max-w-3xl">
              {subtitle}
            </p>
          )}
        </div>

        {/* 2-column or full layout: Left explanatory text + equations, Right interactive visual */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className={`space-y-4 text-body font-body leading-relaxed ${interactiveSlot ? 'lg:col-span-6' : 'lg:col-span-12'}`}>
            {children}
          </div>

          {interactiveSlot && (
            <div className="lg:col-span-6 sticky top-24">
              <div className="rounded-2xl bg-surface-bone/80 border border-hairline p-5 shadow-sm">
                <div className="flex items-center justify-between pb-3 mb-4 border-b border-hairline">
                  <span className="text-xs font-mono uppercase font-bold text-charcoal flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-primary"></span>
                    Interactive Mechanism Simulation
                  </span>
                  <span className="text-[11px] font-mono text-mute">Live React State</span>
                </div>
                {interactiveSlot}
              </div>
            </div>
          )}
        </div>

      </div>
    </section>
  );
};
