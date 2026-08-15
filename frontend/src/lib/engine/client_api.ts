import {
  ExposedLSTMCellEngine,
  dotProduct,
  sigmoid,
  type GateStep,
} from './lstm_engine';
import type {
  ExplainerTraceResponse,
  NextWordResponse,
  AnomalySample,
  AnomalyDetectResponse,
  SentimentResponse,
  TokenProb,
} from '../types';

let nextwordVocab: Record<string, number> | null = null;
let nextwordInvVocab: Record<number, string> | null = null;
let nextwordWeights: any = null;
let nextwordCell: ExposedLSTMCellEngine | null = null;

let sentimentVocab: Record<string, number> | null = null;
let sentimentWeights: any = null;
let sentimentCell: ExposedLSTMCellEngine | null = null;

let anomalyWeights: any = null;
let anomalyEncoderCell: ExposedLSTMCellEngine | null = null;
let anomalyDecoderCell: ExposedLSTMCellEngine | null = null;
let anomalySamples: any[] | null = null;

async function loadNextwordAssets() {
  if (!nextwordVocab) {
    const vRes = await fetch('/weights/nextword_vocab.json');
    nextwordVocab = await vRes.json();
    nextwordInvVocab = {};
    for (const [k, v] of Object.entries(nextwordVocab!)) {
      nextwordInvVocab[v] = k;
    }
  }
  if (!nextwordWeights) {
    const wRes = await fetch('/weights/nextword_weights.json');
    nextwordWeights = await wRes.json();
    nextwordCell = new ExposedLSTMCellEngine(
      nextwordWeights['lstm_cell.cell.weight_ih'],
      nextwordWeights['lstm_cell.cell.weight_hh'],
      nextwordWeights['lstm_cell.cell.bias_ih'],
      nextwordWeights['lstm_cell.cell.bias_hh']
    );
  }
}

async function loadSentimentAssets() {
  if (!sentimentVocab) {
    const vRes = await fetch('/weights/sentiment_vocab.json');
    sentimentVocab = await vRes.json();
  }
  if (!sentimentWeights) {
    const wRes = await fetch('/weights/sentiment_weights.json');
    sentimentWeights = await wRes.json();
    sentimentCell = new ExposedLSTMCellEngine(
      sentimentWeights['lstm_cell.cell.weight_ih'],
      sentimentWeights['lstm_cell.cell.weight_hh'],
      sentimentWeights['lstm_cell.cell.bias_ih'],
      sentimentWeights['lstm_cell.cell.bias_hh']
    );
  }
}

async function loadAnomalyAssets() {
  if (!anomalySamples) {
    const sRes = await fetch('/weights/anomaly_samples.json');
    anomalySamples = await sRes.json();
  }
  if (!anomalyWeights) {
    const wRes = await fetch('/weights/anomaly_weights.json');
    anomalyWeights = await wRes.json();
    anomalyEncoderCell = new ExposedLSTMCellEngine(
      anomalyWeights['encoder.cell.weight_ih'],
      anomalyWeights['encoder.cell.weight_hh'],
      anomalyWeights['encoder.cell.bias_ih'],
      anomalyWeights['encoder.cell.bias_hh']
    );
    anomalyDecoderCell = new ExposedLSTMCellEngine(
      anomalyWeights['decoder.weight_ih'],
      anomalyWeights['decoder.weight_hh'],
      anomalyWeights['decoder.bias_ih'],
      anomalyWeights['decoder.bias_hh']
    );
  }
}

function resolveSentimentToken(w: string): number {
  if (!sentimentVocab) return 1;
  if (w in sentimentVocab) return sentimentVocab[w];
  const suffixes = ['ing', 'ed', 'ly', 's', 'es', 'ness', 'able', 'ible', 'ful', 'less'];
  for (const suffix of suffixes) {
    if (w.endsWith(suffix) && w.length > suffix.length + 2) {
      const base = w.slice(0, -suffix.length);
      if (base in sentimentVocab) return sentimentVocab[base];
      if (base + 'e' in sentimentVocab) return sentimentVocab[base + 'e'];
    }
  }
  return sentimentVocab['<UNK>'] || 1;
}

export async function clientExplainerTrace(text: string): Promise<ExplainerTraceResponse> {
  await loadNextwordAssets();
  const words = text.toLowerCase().match(/[a-zA-Z0-9']+/g) || [];
  if (words.length === 0) return { tokens: [], steps: [] };

  const H = nextwordCell!.hidden_size;
  let hx = new Array(H).fill(0);
  let cx = new Array(H).fill(0);
  const steps: GateStep[] = [];

  for (let t = 0; t < words.length; t++) {
    const token = words[t];
    const idx = nextwordVocab![token] ?? nextwordVocab!['<UNK>'] ?? 1;
    const emb = nextwordWeights['embedding.weight'][idx] || nextwordWeights['embedding.weight'][1];

    const [[nextH, nextC], [i_m, f_m, o_m, c_norm]] = nextwordCell!.forward(emb, hx, cx);
    hx = nextH;
    cx = nextC;

    steps.push({
      t,
      token,
      forget: f_m,
      input: i_m,
      output: o_m,
      cell_state_norm: c_norm,
    });
  }

  return { tokens: words, steps };
}

export async function clientNextWordPredict(text: string): Promise<NextWordResponse> {
  await loadNextwordAssets();
  const words = text.toLowerCase().match(/[a-zA-Z0-9']+/g) || [];
  if (words.length === 0) return { top5: [], steps: [] };

  const H = nextwordCell!.hidden_size;
  let hx = new Array(H).fill(0);
  let cx = new Array(H).fill(0);
  const steps: GateStep[] = [];

  for (let t = 0; t < words.length; t++) {
    const token = words[t];
    const idx = nextwordVocab![token] ?? nextwordVocab!['<UNK>'] ?? 1;
    const emb = nextwordWeights['embedding.weight'][idx] || nextwordWeights['embedding.weight'][1];

    const [[nextH, nextC], [i_m, f_m, o_m, c_norm]] = nextwordCell!.forward(emb, hx, cx);
    hx = nextH;
    cx = nextC;

    steps.push({
      t,
      token,
      forget: f_m,
      input: i_m,
      output: o_m,
      cell_state_norm: c_norm,
    });
  }

  const headW: number[][] = nextwordWeights['head.weight'] || nextwordWeights['fc.weight'];
  const headB: number[] = nextwordWeights['head.bias'] || nextwordWeights['fc.bias'];

  const logits = new Array(headW.length);
  let maxLogit = -Infinity;
  for (let i = 0; i < headW.length; i++) {
    const l = dotProduct(headW[i], hx) + headB[i];
    logits[i] = l;
    if (l > maxLogit) maxLogit = l;
  }

  let sumExp = 0;
  const expLogits = new Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    expLogits[i] = e;
    sumExp += e;
  }

  const probs = expLogits.map((e, idx) => ({ prob: e / sumExp, idx }));
  probs.sort((a, b) => b.prob - a.prob);

  const top5: TokenProb[] = probs.slice(0, 5).map((p) => ({
    token: nextwordInvVocab![p.idx] || '<UNK>',
    prob: p.prob,
  }));

  return { top5, steps };
}

export async function clientAnomalySamples(): Promise<AnomalySample[]> {
  await loadAnomalyAssets();
  return (anomalySamples || []).map((s) => ({
    id: s.id,
    name: s.type.charAt(0).toUpperCase() + s.type.slice(1) + ' Sample',
    description: 'A telemetry sensor trace sequence',
  }));
}

export async function clientAnomalyDetect(
  sequence_id: string,
  inject_anomaly: boolean
): Promise<AnomalyDetectResponse> {
  await loadAnomalyAssets();
  const sample = (anomalySamples || []).find((s) => s.id === sequence_id) || anomalySamples![0];
  const sequence = [...sample.data];

  if (inject_anomaly) {
    const mid = Math.floor(sequence.length / 2);
    for (let t = mid; t < mid + 5; t++) {
      sequence[t] += 2.0;
    }
  }

  const H = anomalyEncoderCell!.hidden_size;
  let hx = new Array(H).fill(0);
  let cx = new Array(H).fill(0);
  const steps: GateStep[] = [];

  for (let t = 0; t < sequence.length; t++) {
    const val = sequence[t];
    const [[nextH, nextC], [i_m, f_m, o_m, c_norm]] = anomalyEncoderCell!.forward([val], hx, cx);
    hx = nextH;
    cx = nextC;

    steps.push({
      t,
      token: val.toFixed(2),
      forget: f_m,
      input: i_m,
      output: o_m,
      cell_state_norm: c_norm,
    });
  }

  const headW: number[] = anomalyWeights['head.weight'][0];
  const headB: number = anomalyWeights['head.bias'][0];

  const reconstruction: number[] = [];
  let decHx = [...hx];
  let decCx = [...cx];
  let decIn = [sequence[sequence.length - 1]];

  for (let t = 0; t < sequence.length; t++) {
    const [[nextH, nextC]] = anomalyDecoderCell!.forward(decIn, decHx, decCx);
    decHx = nextH;
    decCx = nextC;
    const outVal = dotProduct(headW, decHx) + headB;
    reconstruction.push(outVal);
    decIn = [outVal];
  }

  reconstruction.reverse();
  const error = sequence.map((v, i) => Math.pow(v - reconstruction[i], 2));
  const threshold = 0.5;
  const anomaly_flags = error.map((e) => e > threshold);

  return {
    input: sequence,
    reconstruction,
    error,
    anomaly_flags,
    steps,
  };
}

export async function clientSentimentClassify(text: string): Promise<SentimentResponse> {
  await loadSentimentAssets();
  const words = text.toLowerCase().match(/[a-zA-Z0-9']+/g) || [''];

  const H = sentimentCell!.hidden_size;
  let hx = new Array(H).fill(0);
  let cx = new Array(H).fill(0);
  const steps: GateStep[] = [];
  const running_score: number[] = [];

  const headW: number[] = sentimentWeights['head.weight'][0];
  const headB: number = sentimentWeights['head.bias'][0];

  for (let t = 0; t < words.length; t++) {
    const token = words[t];
    const idx = resolveSentimentToken(token);
    const emb = sentimentWeights['embedding.weight'][idx] || sentimentWeights['embedding.weight'][1];

    const [[nextH, nextC], [i_m, f_m, o_m, c_norm]] = sentimentCell!.forward(emb, hx, cx);
    hx = nextH;
    cx = nextC;

    const logit = dotProduct(headW, hx) + headB;
    const score = sigmoid(logit);
    running_score.push(score);

    steps.push({
      t,
      token,
      forget: f_m,
      input: i_m,
      output: o_m,
      cell_state_norm: c_norm,
    });
  }

  const finalScore = running_score[running_score.length - 1] ?? 0.5;
  const final_label = finalScore >= 0.5 ? 'Positive' : 'Negative';

  // API Ninjas direct call fallback from browser if online
  let api_ninjas_score: number | null = null;
  let api_ninjas_label: string | null = null;

  try {
    const apiKey = 'ImySMWP8E8jFA6sP0MLDZBxEwFRPks4T9YzHck1I';
    const cRes = await fetch(`https://api.api-ninjas.com/v1/sentiment?text=${encodeURIComponent(text)}`, {
      headers: { 'X-Api-Key': apiKey },
    });
    if (cRes.ok) {
      const cData = await cRes.json();
      api_ninjas_score = cData.score !== undefined ? Number(cData.score) : null;
      api_ninjas_label = cData.sentiment || null;
    }
  } catch (e) {
    // Ignore external network errors
  }

  return {
    tokens: words,
    running_score,
    final_label,
    steps,
    api_ninjas_score,
    api_ninjas_label,
  };
}
