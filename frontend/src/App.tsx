import { useState, useEffect } from 'react';
import { TopMenuBar } from './components/nav/TopMenuBar';
import type { NavRoute } from './components/nav/TopMenuBar';
import { Home } from './pages/Home';
import { Explainer } from './pages/Explainer';
import { NextWordPredictor } from './pages/playground/NextWordPredictor';
import { AnomalyDetector } from './pages/playground/AnomalyDetector';
import { SentimentClassifier } from './pages/playground/SentimentClassifier';
import { About } from './pages/About';

function App() {
  const [currentRoute, setCurrentRoute] = useState<NavRoute>('home');

  const handleNavigate = (route: NavRoute) => {
    setCurrentRoute(route);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  useEffect(() => {
    // Sync browser tab title based on active route
    const titles: Record<NavRoute, string> = {
      home: 'NeuroSeq — Real-time LSTM Internals & Live Gate Activation Engine',
      explainer: 'NeuroSeq — 8-Step Interactive LSTM Math & Architecture Explainer',
      'playground-nextword': 'NeuroSeq — Next-Word Predictor & Logit Distribution',
      'playground-anomaly': 'NeuroSeq — Time-Series Anomaly Detector',
      'playground-sentiment': 'NeuroSeq — Sentiment Sequence Classifier & Pivot Tracker',
      about: 'NeuroSeq — System Architecture & Philosophy',
    };
    document.title = titles[currentRoute] || 'NeuroSeq';
  }, [currentRoute]);

  return (
    <div className="min-h-screen flex flex-col bg-canvas text-body font-body selection:bg-primary/20 selection:text-primary">
      {/* Persistent Navigation Bar */}
      <TopMenuBar currentRoute={currentRoute} onNavigate={handleNavigate} />

      {/* Main Dynamic View */}
      <main className="flex-1">
        {currentRoute === 'home' && <Home onNavigate={handleNavigate} />}
        {currentRoute === 'explainer' && <Explainer onNavigate={handleNavigate} />}
        {currentRoute === 'playground-nextword' && <NextWordPredictor />}
        {currentRoute === 'playground-anomaly' && <AnomalyDetector />}
        {currentRoute === 'playground-sentiment' && <SentimentClassifier />}
        {currentRoute === 'about' && <About onNavigate={handleNavigate} />}
      </main>
    </div>
  );
}

export default App;
