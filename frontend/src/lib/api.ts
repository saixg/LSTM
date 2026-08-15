import type {
  ExplainerTraceResponse,
  NextWordResponse,
  AnomalySample,
  AnomalyDetectResponse,
  SentimentResponse,
} from './types';
import {
  clientExplainerTrace,
  clientNextWordPredict,
  clientAnomalySamples,
  clientAnomalyDetect,
  clientSentimentClassify,
} from './engine/client_api';

const BASE_URL = import.meta.env.VITE_API_URL || '/api';

export async function fetchHealth(): Promise<{ status: string; message: string }> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(1500) });
    if (res.ok) return await res.json();
  } catch (e) {
    // Return live in-browser engine status if backend server is not connected
  }
  return { status: 'ok', message: 'NeuroSeq In-Browser Neural Engine active' };
}

export async function fetchExplainerTrace(text: string): Promise<ExplainerTraceResponse> {
  try {
    const res = await fetch(`${BASE_URL}/explainer/trace`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.steps && data.steps.length > 0) return data;
    }
  } catch (e) {
    // Fall back to client neural engine
  }
  return clientExplainerTrace(text);
}

export async function fetchNextWordPredict(
  text: string,
  level: 'word' | 'char' = 'word'
): Promise<NextWordResponse> {
  try {
    const res = await fetch(`${BASE_URL}/nextword/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, level }),
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.top5) return data;
    }
  } catch (e) {
    // Fall back to client neural engine
  }
  return clientNextWordPredict(text);
}

export async function fetchAnomalySamples(): Promise<AnomalySample[]> {
  try {
    const res = await fetch(`${BASE_URL}/anomaly/samples`, { signal: AbortSignal.timeout(2000) });
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) return data;
    }
  } catch (e) {
    // Fall back to client neural engine
  }
  return clientAnomalySamples();
}

export async function fetchAnomalyDetect(
  sequence_id: string,
  inject_anomaly: boolean
): Promise<AnomalyDetectResponse> {
  try {
    const res = await fetch(`${BASE_URL}/anomaly/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence_id, inject_anomaly }),
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.reconstruction) return data;
    }
  } catch (e) {
    // Fall back to client neural engine
  }
  return clientAnomalyDetect(sequence_id, inject_anomaly);
}

export async function fetchSentimentClassify(text: string): Promise<SentimentResponse> {
  try {
    const res = await fetch(`${BASE_URL}/sentiment/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal: AbortSignal.timeout(2000),
    });
    if (res.ok) {
      const data = await res.json();
      if (data && data.running_score && data.running_score.length > 0) return data;
    }
  } catch (e) {
    // Fall back to client neural engine
  }
  return clientSentimentClassify(text);
}
