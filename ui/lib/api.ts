const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { next: { revalidate: 60 } });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface PairSummary {
  pair: string;
  total: number;
  counts: { adversarial: number; cooperative: number; neutral: number };
  percent: { adversarial: number; cooperative: number; neutral: number };
}

export interface SummaryAll {
  pairs: Record<string, PairSummary>;
  model: string;
}

export interface Alert {
  pair: string;
  severity: "high" | "medium" | "low";
  delta_adversarial_pp: number;
  this_window_n: number;
  prev_window_n: number;
  message: string;
}

export interface AlertsResponse {
  alerts: Alert[];
}

export interface TrendPoint {
  date: string;
  adversarial: number;  // fraction 0-1
  cooperative: number;
  neutral: number;
  n: number;
}

export interface TrendsResponse {
  pair: string;
  model: string;
  series: TrendPoint[];
  rolling_adversarial?: { date: string; adversarial_rolling_avg: number; n_points: number }[];
}

export interface HeadlineItem {
  id: string;
  headline: string;
  label: string;
  confidence: number;
  published_at: string;
  source: string;
}

export interface HeadlinesResponse {
  pair: string;
  items: HeadlineItem[];
}

export interface CompareResponse {
  pair1: PairSummary;
  pair2: PairSummary;
  model: string;
}

export interface ClassifyResponse {
  text_used: string;
  pair?: string;
  baseline?: { label: string; confidence: number; probabilities: Record<string, number> };
  transformer?: { label: string; confidence: number; probabilities: Record<string, number> };
}

// ── API calls ─────────────────────────────────────────────────────────────────

export const fetchSummaryAll = (model = "baseline", startDate?: string, endDate?: string) => {
  const p = new URLSearchParams({ model });
  if (startDate) p.set("start_date", startDate);
  if (endDate) p.set("end_date", endDate);
  return get<SummaryAll>(`/summary/all?${p}`);
};

export const fetchAlerts = (model = "baseline", days = 7, threshold = 15) =>
  get<AlertsResponse>(`/alerts?model=${model}&days=${days}&threshold_pp=${threshold}`);

export const fetchTrends = (pair: string, model = "baseline", rolling = 7, startDate?: string, endDate?: string) => {
  const p = new URLSearchParams({ pair, model, rolling: String(rolling) });
  if (startDate) p.set("start_date", startDate);
  if (endDate) p.set("end_date", endDate);
  return get<TrendsResponse>(`/trends?${p}`);
};

export const fetchHeadlines = (
  pair: string,
  model = "baseline",
  label?: string,
  limit = 100,
  startDate?: string,
  endDate?: string,
) => {
  const params = new URLSearchParams({ pair, model, limit: String(limit) });
  if (label) params.set("label", label);
  if (startDate) params.set("start_date", startDate);
  if (endDate) params.set("end_date", endDate);
  return get<HeadlinesResponse>(`/headlines?${params}`);
};

export const fetchCompare = (pair1: string, pair2: string, model = "baseline", startDate?: string, endDate?: string) => {
  const p = new URLSearchParams({ pair1, pair2, model });
  if (startDate) p.set("start_date", startDate);
  if (endDate) p.set("end_date", endDate);
  return get<CompareResponse>(`/compare?${p}`);
};

export async function classify(
  text: string,
  pair: string,
  model: "baseline" | "transformer" | "both"
): Promise<ClassifyResponse> {
  const res = await fetch(`${BASE}/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, pair, model }),
  });
  if (!res.ok) throw new Error(`classify → ${res.status}`);
  return res.json();
}

// ── Causality types ───────────────────────────────────────────────────────────

export interface CausalNode {
  id: string;
  label: "adversarial" | "cooperative" | "neutral";
  window_start: string;
  window_end: string;
  total: number;
  avg_confidence: number;
  top_headlines: string[];
  color: string;
}

export interface CausalEdge {
  source: string;
  target: string;
  source_label: string;
  target_label: string;
  transition: string;
  same: boolean;
}

export interface SpikeWindow {
  total: number;
  pct: { adversarial: number; cooperative: number; neutral: number };
  top_headlines: { headline: string; confidence: number; label: string }[];
}

export interface SpikeAnalysis {
  pair: string;
  narrative: string;
  deltas: { adversarial: number; cooperative: number; neutral: number };
  this_window: SpikeWindow;
  prev_window: SpikeWindow;
}

export interface CausalPattern {
  pattern: string;
  steps: string[];
  count: number;
  ends_adversarial: boolean;
}

export interface CausalGraph {
  nodes: CausalNode[];
  edges: CausalEdge[];
  transitions: { from: string; to: string; count: number; transition: string }[];
  adversarial_precursors: Record<string, number>;
  total_predictions: number;
  date_range?: string;
  message?: string;
}

export interface CausalityResponse {
  pair: string;
  graph: CausalGraph;
  spike: SpikeAnalysis;
  patterns: CausalPattern[];
}

export const fetchCausality = (
  pair: string,
  model = "baseline",
  days = 30,
  minConfidence = 0.65,
) =>
  get<CausalityResponse>(
    `/causality?pair=${pair}&model=${model}&days=${days}&min_confidence=${minConfidence}`,
  );
