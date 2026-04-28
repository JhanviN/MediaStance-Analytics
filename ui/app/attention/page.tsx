"use client";

import { useState } from "react";
import { ADV_COLOR, ALL_PAIRS, pairLabel } from "@/lib/constants";
import ErrorBanner from "@/components/ErrorBanner";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

interface AttentionResponse {
  tokens: string[];
  weights: number[];
  label: string;
  confidence: number;
}

function TokenHeatmap({ tokens, weights }: { tokens: string[]; weights: number[] }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, lineHeight: 2.2 }}>
      {tokens.map((tok, i) => {
        const w = weights[i] ?? 0;
        // Use a stepped color scale for better visibility
        const bg = w > 0.75
          ? `rgba(224, 92, 92, 0.85)`
          : w > 0.5
          ? `rgba(224, 92, 92, 0.55)`
          : w > 0.25
          ? `rgba(224, 92, 92, 0.25)`
          : `rgba(224, 92, 92, 0.06)`;
        const textColor = w > 0.6 ? "#fff" : "var(--text)";
        return (
          <span
            key={i}
            title={`weight: ${w.toFixed(3)}`}
            style={{
              padding: "3px 8px",
              borderRadius: 5,
              background: bg,
              color: textColor,
              fontSize: 14,
              fontFamily: "monospace",
              cursor: "default",
              border: `1px solid rgba(224, 92, 92, ${w * 0.4 + 0.05})`,
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
            {ALL_PAIRS.map((p) => <option key={p} value={p}>{pairLabel(p)}</option>)}
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
