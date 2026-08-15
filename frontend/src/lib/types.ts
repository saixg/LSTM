export type GateStep = {
  t: number;
  token: string;
  forget: number;
  input: number;
  output: number;
  cell_state_norm: number;
};

export type ExplainerTraceResponse = {
  tokens: string[];
  steps: GateStep[];
};

export type TokenProb = {
  token: string;
  prob: number;
};

export type NextWordResponse = {
  top5: TokenProb[];
  steps: GateStep[];
};

export type AnomalySample = {
  id: string;
  name: string;
  description: string;
};

export type AnomalyDetectResponse = {
  input: number[];
  reconstruction: number[];
  error: number[];
  anomaly_flags: boolean[];
  steps: GateStep[];
};

export type SentimentResponse = {
  tokens: string[];
  running_score: number[];
  final_label: string;
  steps: GateStep[];
  api_ninjas_score?: number | null;
  api_ninjas_label?: string | null;
};
