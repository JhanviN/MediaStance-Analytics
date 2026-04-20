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

export const fetchSummaryAll = (model = "baseline") =>
  get<SummaryAll>(`/summary/all?model=${model}`);

export const fetchAlerts = (model = "baseline") =>
  get<AlertsResponse>(`/alerts?model=${model}`);

export const fetchTrends = (pair: string, model = "baseline", rolling = 7) =>
  get<TrendsResponse>(`/trends?pair=${pair}&model=${model}&rolling=${rolling}`);

export const fetchHeadlines = (
  pair: string,
  model = "baseline",
  label?: string,
  limit = 100
) => {
  const params = new URLSearchParams({ pair, model, limit: String(limit) });
  if (label) params.set("label", label);
  return get<HeadlinesResponse>(`/headlines?${params}`);
};

export const fetchCompare = (pair1: string, pair2: string, model = "baseline") =>
  get<CompareResponse>(`/compare?pair1=${pair1}&pair2=${pair2}&model=${model}`);

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
