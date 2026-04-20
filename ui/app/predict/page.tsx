"use client";

import { useState } from "react";
import { classify, type ClassifyResponse } from "@/lib/api";
import { ADV_COLOR, COOP_COLOR, NEUT_COLOR, ALL_PAIRS, LABEL_COLORS } from "@/lib/constants";
import ErrorBanner from "@/components/ErrorBanner";

function ProbBar({ label, value }: { label: string; value: number }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
        <span style={{ color: "var(--muted)" }}>{label}</span>
        <span className="mono" style={{ color: LABEL_COLORS[label] }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: "#eef0f5", borderRadius: 3, overflow: "hidden" }}>
        <div
          style={{
            height: "100%",
            width: `${value * 100}%`,
            background: LABEL_COLORS[label] ?? "var(--muted)",
            borderRadius: 3,
            transition: "width 0.4s ease",
          }}
        />
      </div>
    </div>
  );
}

function ResultBlock({ title, result }: { title: string; result: ClassifyResponse["baseline"] }) {
  if (!result) return null;
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "16px 20px" }}>
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>{title}</div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <span
          style={{
            padding: "4px 12px",
            borderRadius: 4,
            background: `${LABEL_COLORS[result.label]}22`,
            color: LABEL_COLORS[result.label],
            fontWeight: 500,
            fontSize: 14,
          }}
        >
          {result.label}
        </span>
        <span className="mono" style={{ color: "var(--muted)", fontSize: 13 }}>
          {(result.confidence * 100).toFixed(1)}% confidence
        </span>
      </div>
      {Object.entries(result.probabilities).map(([l, v]) => (
        <ProbBar key={l} label={l} value={v} />
      ))}
    </div>
  );
}

export default function PredictPage() {
  const [text, setText] = useState("");
  const [pair, setPair] = useState("IN-US");
  const [model, setModel] = useState<"baseline" | "transformer" | "both">("both");
  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await classify(text, pair, model);
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 680 }}>
      <h1 style={{ fontWeight: 500, fontSize: 20, marginBottom: 24 }}>Live Predict</h1>

      {error && <ErrorBanner message={error} />}

      <form onSubmit={handleSubmit} style={{ marginBottom: 24 }}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a headline or article snippet..."
          rows={4}
          style={{
            width: "100%",
            background: "#f9fafb",
            border: "1px solid var(--border)",
            borderRadius: 8,
            color: "var(--text)",
            padding: "12px 14px",
            fontSize: 14,
            resize: "vertical",
            marginBottom: 12,
            outline: "none",
          }}
        />
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <select
            value={pair}
            onChange={(e) => setPair(e.target.value)}
            style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", padding: "7px 10px", fontSize: 13, fontFamily: "monospace" }}
          >
            {ALL_PAIRS.map((p) => <option key={p}>{p}</option>)}
          </select>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as typeof model)}
            style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", padding: "7px 10px", fontSize: 13 }}
          >
            <option value="both">both models</option>
            <option value="baseline">baseline</option>
            <option value="transformer">transformer</option>
          </select>
          <button
            type="submit"
            disabled={loading || !text.trim()}
            style={{
              background: loading ? "#e2e6ef" : ADV_COLOR,
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "7px 20px",
              fontSize: 13,
              fontWeight: 500,
              cursor: loading ? "not-allowed" : "pointer",
              transition: "background 0.15s",
            }}
          >
            {loading ? "Classifying..." : "Classify"}
          </button>
        </div>
      </form>

      {loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div className="skeleton" style={{ height: 140, borderRadius: 8 }} />
          <div className="skeleton" style={{ height: 140, borderRadius: 8 }} />
        </div>
      )}

      {result && !loading && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <ResultBlock title="Baseline (TF-IDF + LR)" result={result.baseline} />
          <ResultBlock title="Transformer (fine-tuned BERT)" result={result.transformer} />
        </div>
      )}
    </div>
  );
}
