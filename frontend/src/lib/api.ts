import type {
  ExplainerTraceResponse,
  NextWordResponse,
  AnomalySample,
  AnomalyDetectResponse,
  SentimentResponse,
} from './types';

const BASE_URL = '/api';

export async function fetchHealth(): Promise<{ status: string; message: string }> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchExplainerTrace(text: string): Promise<ExplainerTraceResponse> {
  const res = await fetch(`${BASE_URL}/explainer/trace`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`Explainer trace failed: ${res.statusText}`);
  return res.json();
}

export async function fetchNextWordPredict(
  text: string,
  level: 'word' | 'char' = 'word'
): Promise<NextWordResponse> {
  const res = await fetch(`${BASE_URL}/nextword/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, level }),
  });
  if (!res.ok) throw new Error(`Next-word predict failed: ${res.statusText}`);
  return res.json();
}

export async function fetchAnomalySamples(): Promise<AnomalySample[]> {
  const res = await fetch(`${BASE_URL}/anomaly/samples`);
  if (!res.ok) throw new Error(`Failed to fetch anomaly samples: ${res.statusText}`);
  return res.json();
}

export async function fetchAnomalyDetect(
  sequence_id: string,
  inject_anomaly: boolean
): Promise<AnomalyDetectResponse> {
  const res = await fetch(`${BASE_URL}/anomaly/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sequence_id, inject_anomaly }),
  });
  if (!res.ok) throw new Error(`Anomaly detection failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSentimentClassify(text: string): Promise<SentimentResponse> {
  const res = await fetch(`${BASE_URL}/sentiment/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`Sentiment classification failed: ${res.statusText}`);
  return res.json();
}
