// In-Browser High-Performance LSTM Neural Engine
// Computes real ExposedLSTMCell gate activations, cell state trajectories, and predictions in-browser

export type GateStep = {
  t: number;
  token: string;
  forget: number;
  input: number;
  output: number;
  cell_state_norm: number;
};

export function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-Math.max(Math.min(x, 50.0), -50.0)));
}

export function tanh(x: number): number {
  const e2x = Math.exp(2 * Math.max(Math.min(x, 50.0), -50.0));
  return (e2x - 1) / (e2x + 1);
}

export function dotProduct(v1: number[], v2: number[]): number {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += v1[i] * (v2[i] || 0);
  }
  return sum;
}

export function matrixVectorMul(matrix: number[][], vector: number[]): number[] {
  const res = new Array(matrix.length);
  for (let i = 0; i < matrix.length; i++) {
    res[i] = dotProduct(matrix[i], vector);
  }
  return res;
}

export class ExposedLSTMCellEngine {
  weight_ih: number[][];
  weight_hh: number[][];
  bias_ih: number[];
  bias_hh: number[];
  hidden_size: number;

  constructor(
    weight_ih: number[][],
    weight_hh: number[][],
    bias_ih: number[],
    bias_hh: number[]
  ) {
    this.weight_ih = weight_ih;
    this.weight_hh = weight_hh;
    this.bias_ih = bias_ih;
    this.bias_hh = bias_hh;
    this.hidden_size = weight_hh[0].length;
  }

  forward(
    x: number[],
    hx: number[],
    cx: number[]
  ): [[number[], number[]], [number, number, number, number]] {
    const H = this.hidden_size;
    const gates_ih = matrixVectorMul(this.weight_ih, x);
    const gates_hh = matrixVectorMul(this.weight_hh, hx);

    const i_gate = new Array(H);
    const f_gate = new Array(H);
    const g_gate = new Array(H);
    const o_gate = new Array(H);

    let f_sum = 0;
    let i_sum = 0;
    let o_sum = 0;

    for (let j = 0; j < H; j++) {
      const g_i = gates_ih[j] + this.bias_ih[j] + gates_hh[j] + this.bias_hh[j];
      const g_f = gates_ih[j + H] + this.bias_ih[j + H] + gates_hh[j + H] + this.bias_hh[j + H];
      const g_c = gates_ih[j + 2 * H] + this.bias_ih[j + 2 * H] + gates_hh[j + 2 * H] + this.bias_hh[j + 2 * H];
      const g_o = gates_ih[j + 3 * H] + this.bias_ih[j + 3 * H] + gates_hh[j + 3 * H] + this.bias_hh[j + 3 * H];

      const val_i = sigmoid(g_i);
      const val_f = sigmoid(g_f);
      const val_g = tanh(g_c);
      const val_o = sigmoid(g_o);

      i_gate[j] = val_i;
      f_gate[j] = val_f;
      g_gate[j] = val_g;
      o_gate[j] = val_o;

      i_sum += val_i;
      f_sum += val_f;
      o_sum += val_o;
    }

    const c_next = new Array(H);
    const h_next = new Array(H);
    let c_sq_sum = 0;

    for (let j = 0; j < H; j++) {
      const c_val = f_gate[j] * cx[j] + i_gate[j] * g_gate[j];
      c_next[j] = c_val;
      h_next[j] = o_gate[j] * tanh(c_val);
      c_sq_sum += c_val * c_val;
    }

    const c_norm = Math.sqrt(c_sq_sum);
    return [[h_next, c_next], [i_sum / H, f_sum / H, o_sum / H, c_norm]];
  }
}
