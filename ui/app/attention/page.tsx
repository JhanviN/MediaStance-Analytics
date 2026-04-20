"use client";

import { useState } from "react";
import { ADV_COLOR, ALL_PAIRS } from "@/lib/constants";
import ErrorBanner from "@/components/ErrorBanner";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

interface AttentionResponse {
  tokens: string[];
  weights: number[];
  label: string;
  confidence: number;
}

function TokenHeatmap({ tokens, weights }: { tokens: string[]; weights: number[] }) {
  const max = Math.max(...weights, 0.001);
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, lineHeight: 2 }}>
      {tokens.map((tok, i) => {
        const norm = weights[i] / max;
        return (
          <span
            key={i}
            title={`weight: ${weights[i].toFixed(4)}`}
            style={{
              padding: "2px 6px",
              borderRadius: 4,
              background: `rgba(226, 75, 74, ${norm * 0.85})`,
              color: norm > 0.5 ? "#fff" : "var(--text)",
              fontSize: 14,
              fontFamily: "monospace",
              cursor: "default",
              transition: "background 0.2s",
            }}
          >
            {tok}
          </span>
        );
      })}
    </div>
  );
}

export default function AttentionPage() {
  const [text, setText] = useState("");
  const [pair, setPair] = useState("IN-US");
  const [result, setResult] = useState<AttentionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${BASE}/attention`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, pair }),
      });
      if (!res.ok) throw new Error(`API → ${res.status}`);
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 720 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontWeight: 500, fontSize: 20, marginBottom: 6 }}>Attention</h1>
        <p style={{ color: "var(--muted)", fontSize: 13, margin: 0 }}>
          Token-level attention weights from the transformer — red intensity = attention weight.
        </p>
      </div>

      {error && <ErrorBanner message={error} />}

      <form onSubmit={handleSubmit} style={{ marginBottom: 28 }}>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a headline to visualize attention weights..."
          rows={3}
          style={{
            width: "100%",
            background: "var(--surface)",
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
          <button
            type="submit"
            disabled={loading || !text.trim()}
            style={{
              background: loading ? "#2a2f3d" : ADV_COLOR,
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "7px 20px",
              fontSize: 13,
              fontWeight: 500,
              cursor: loading ? "not-allowed" : "pointer",
            }}
          >
            {loading ? "Analyzing..." : "Visualize"}
          </button>
        </div>
      </form>

      {loading && (
        <div className="skeleton" style={{ height: 120, borderRadius: 8 }} />
      )}

      {result && !loading && (
        <div>
          <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 16 }}>
            <span
              style={{
                padding: "4px 12px",
                borderRadius: 4,
                background: `${ADV_COLOR}22`,
                color: ADV_COLOR,
                fontWeight: 500,
                fontSize: 13,
              }}
            >
              {result.label}
            </span>
            <span className="mono" style={{ color: "var(--muted)", fontSize: 13 }}>
              {(result.confidence * 100).toFixed(1)}% confidence
            </span>
          </div>

          <div
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "20px",
              marginBottom: 16,
            }}
          >
            <TokenHeatmap tokens={result.tokens} weights={result.weights} />
          </div>

          {/* Scale legend */}
          <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 12, color: "var(--muted)" }}>
            <span>low attention</span>
            <div
              style={{
                width: 120,
                height: 8,
                borderRadius: 4,
                background: `linear-gradient(to right, rgba(226,75,74,0.05), rgba(226,75,74,0.85))`,
              }}
            />
            <span>high attention</span>
          </div>
        </div>
      )}

      {!result && !loading && (
        <div
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "32px 20px",
            textAlign: "center",
            color: "var(--muted)",
            fontSize: 13,
          }}
        >
          Enter a headline above to see which tokens the transformer attends to most.
        </div>
      )}
    </div>
  );
}
