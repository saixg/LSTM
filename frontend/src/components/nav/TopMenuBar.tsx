import React, { useState, useEffect } from 'react';
import { fetchHealth } from '../../lib/api';

export type NavRoute = 'home' | 'explainer' | 'playground-nextword' | 'playground-anomaly' | 'playground-sentiment' | 'about';

interface TopMenuBarProps {
  currentRoute: NavRoute;
  onNavigate: (route: NavRoute) => void;
}

export const TopMenuBar: React.FC<TopMenuBarProps> = ({ currentRoute, onNavigate }) => {
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  useEffect(() => {
    let mounted = true;
    const checkStatus = async () => {
      try {
        await fetchHealth();
        if (mounted) setApiOnline(true);
      } catch {
        if (mounted) setApiOnline(false);
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const isPlaygroundActive = currentRoute.startsWith('playground');

  return (
    <header className="sticky top-0 z-50 bg-canvas/90 backdrop-blur-md border-b border-hairline transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand / Logo */}
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => onNavigate('home')}>
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-primary to-primary-deep flex items-center justify-center text-white shadow-md shadow-primary/20">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <span className="font-display font-bold text-xl tracking-tight text-ink">NeuroSeq</span>
              <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-surface-bone text-charcoal border border-hairline">
                PyTorch Exposed
              </span>
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="hidden md:flex items-center space-x-1 font-body text-sm font-medium">
          <button
            onClick={() => onNavigate('home')}
            className={`px-4 py-2 rounded-full transition-all ${
              currentRoute === 'home'
                ? 'bg-surface-dark text-on-dark font-semibold shadow-sm'
                : 'text-body hover:text-ink hover:bg-surface-bone'
            }`}
          >
            Home
          </button>

          <button
            onClick={() => onNavigate('explainer')}
            className={`px-4 py-2 rounded-full transition-all ${
              currentRoute === 'explainer'
                ? 'bg-surface-dark text-on-dark font-semibold shadow-sm'
                : 'text-body hover:text-ink hover:bg-surface-bone'
            }`}
          >
            Explainer
          </button>

          {/* Playground with Dropdown */}
          <div
            className="relative"
            onMouseEnter={() => setDropdownOpen(true)}
            onMouseLeave={() => setDropdownOpen(false)}
          >
            <button
              onClick={() => onNavigate('playground-nextword')}
              className={`px-4 py-2 rounded-full transition-all flex items-center gap-1.5 ${
                isPlaygroundActive
                  ? 'bg-surface-dark text-on-dark font-semibold shadow-sm'
                  : 'text-body hover:text-ink hover:bg-surface-bone'
              }`}
            >
              Playground
              <svg
                className={`w-3.5 h-3.5 transition-transform duration-200 ${dropdownOpen ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {/* Dropdown Menu */}
            {dropdownOpen && (
              <div className="absolute top-full left-0 mt-1 w-64 rounded-2xl bg-surface-card border border-hairline shadow-xl py-2 z-50 animate-fadeIn">
                <button
                  onClick={() => {
                    onNavigate('playground-nextword');
                    setDropdownOpen(false);
                  }}
                  className={`w-full px-4 py-2.5 text-left text-xs flex flex-col gap-0.5 hover:bg-canvas transition-colors ${
                    currentRoute === 'playground-nextword' ? 'bg-canvas text-primary font-bold' : 'text-ink'
                  }`}
                >
                  <span className="font-semibold text-sm">1. Next-Word Predictor</span>
                  <span className="text-mute text-[11px]">Real-time token probability distribution</span>
                </button>

                <button
                  onClick={() => {
                    onNavigate('playground-anomaly');
                    setDropdownOpen(false);
                  }}
                  className={`w-full px-4 py-2.5 text-left text-xs flex flex-col gap-0.5 hover:bg-canvas transition-colors ${
                    currentRoute === 'playground-anomaly' ? 'bg-canvas text-primary font-bold' : 'text-ink'
                  }`}
                >
                  <span className="font-semibold text-sm">2. Anomaly Detector</span>
                  <span className="text-mute text-[11px]">Time-series reconstruction autoencoder</span>
                </button>

                <button
                  onClick={() => {
                    onNavigate('playground-sentiment');
                    setDropdownOpen(false);
                  }}
                  className={`w-full px-4 py-2.5 text-left text-xs flex flex-col gap-0.5 hover:bg-canvas transition-colors ${
                    currentRoute === 'playground-sentiment' ? 'bg-canvas text-primary font-bold' : 'text-ink'
                  }`}
                >
                  <span className="font-semibold text-sm">3. Sentiment Classifier</span>
                  <span className="text-mute text-[11px]">Token-by-token sequence polarity flip</span>
                </button>
              </div>
            )}
          </div>

          <button
            onClick={() => onNavigate('about')}
            className={`px-4 py-2 rounded-full transition-all ${
              currentRoute === 'about'
                ? 'bg-surface-dark text-on-dark font-semibold shadow-sm'
                : 'text-body hover:text-ink hover:bg-surface-bone'
            }`}
          >
            About
          </button>
        </nav>

        {/* Right Status & Action */}
        <div className="flex items-center gap-3">
          {/* Backend Status Indicator */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-bone border border-hairline text-xs font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                apiOnline === true
                  ? 'bg-emerald-500 shadow-sm shadow-emerald-500/50'
                  : apiOnline === false
                  ? 'bg-rose-500 animate-pulse'
                  : 'bg-amber-400 animate-pulse'
              }`}
            ></span>
            <span className="text-charcoal text-[11px]">
              {apiOnline === true ? 'PyTorch: Live' : apiOnline === false ? 'API: Offline' : 'Connecting...'}
            </span>
          </div>

          <button
            onClick={() => onNavigate('playground-nextword')}
            className="btn-primary text-xs px-4 py-2 h-[36px] shadow-sm font-medium"
          >
            Try Live
          </button>
        </div>

      </div>

      {/* Mobile Bar sub-navigation */}
      <div className="md:hidden flex items-center justify-around px-2 py-2 border-t border-hairline bg-canvas/95 text-xs font-medium">
        <button
          onClick={() => onNavigate('home')}
          className={`px-2.5 py-1 rounded-md ${currentRoute === 'home' ? 'bg-surface-dark text-on-dark' : 'text-body'}`}
        >
          Home
        </button>
        <button
          onClick={() => onNavigate('explainer')}
          className={`px-2.5 py-1 rounded-md ${currentRoute === 'explainer' ? 'bg-surface-dark text-on-dark' : 'text-body'}`}
        >
          Explainer
        </button>
        <button
          onClick={() => onNavigate('playground-nextword')}
          className={`px-2.5 py-1 rounded-md ${isPlaygroundActive ? 'bg-primary text-on-primary' : 'text-body'}`}
        >
          Playground
        </button>
        <button
          onClick={() => onNavigate('about')}
          className={`px-2.5 py-1 rounded-md ${currentRoute === 'about' ? 'bg-surface-dark text-on-dark' : 'text-body'}`}
        >
          About
        </button>
      </div>
    </header>
  );
};
