"use client";

import { Suspense } from "react";
import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { fetchHeadlines, type HeadlineItem } from "@/lib/api";
import { ADV_COLOR, COOP_COLOR, NEUT_COLOR, ALL_PAIRS, LABEL_COLORS, pairLabel } from "@/lib/constants";
import ModelToggle from "@/components/ModelToggle";
import ErrorBanner from "@/components/ErrorBanner";
import DateRangeFilter, { type DateRange } from "@/components/DateRangeFilter";

const LABELS = ["", "adversarial", "cooperative", "neutral"];

export default function HeadlinesPage() {
  return (
    <Suspense>
      <HeadlinesInner />
    </Suspense>
  );
}

function HeadlinesInner() {
  const router = useRouter();
  const params = useSearchParams();
  const [pair, setPair] = useState(params.get("pair") ?? "IN-US");
  const [model, setModel] = useState<"baseline" | "transformer">("baseline");
  const [label, setLabel] = useState(params.get("label") ?? "");
  const [dateRange, setDateRange] = useState<DateRange>({ startDate: undefined, endDate: undefined });
  const [items, setItems] = useState<HeadlineItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    router.replace(
      `/headlines?pair=${pair}&model=${model}${label ? `&label=${label}` : ""}`,
      { scroll: false }
    );
  }, [pair, model, label]);

  useEffect(() => {
    setItems(null);
    setError(null);
    fetchHeadlines(pair, model, label || undefined, 100, dateRange.startDate, dateRange.endDate)
      .then((r) => setItems(r.items))
      .catch((e) => setError(e.message));
  }, [pair, model, label, dateRange]);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
        <h1 style={{ fontWeight: 500, fontSize: 20, margin: 0 }}>Headlines</h1>
        <select
          value={pair}
          onChange={(e) => setPair(e.target.value)}
          style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", padding: "5px 10px", fontSize: 13, fontFamily: "monospace" }}
        >
          {ALL_PAIRS.map((p) => <option key={p} value={p}>{pairLabel(p)}</option>)}
        </select>
        <select
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", padding: "5px 10px", fontSize: 13 }}
        >
          {LABELS.map((l) => <option key={l} value={l}>{l || "all labels"}</option>)}
        </select>
        <ModelToggle value={model} onChange={setModel} />
        <DateRangeFilter value={dateRange} onChange={setDateRange} />
      </div>

      {error && <ErrorBanner message={error} />}

      {items === null ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 52, borderRadius: 6 }} />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div style={{ color: "var(--muted)", padding: "40px 0" }}>
          No headlines found for {pair}{label ? ` with label ${label}` : ""}.
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {items.map((item) => (
            <div
              key={item.id}
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: "10px 16px",
                display: "flex",
                alignItems: "center",
                gap: 16,
              }}
            >
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: LABEL_COLORS[item.label] ?? "var(--muted)",
                  flexShrink: 0,
                }}
              />
              <span style={{ flex: 1, fontSize: 13 }}>{item.headline}</span>
              <span className="mono" style={{ color: "var(--muted)", fontSize: 12, flexShrink: 0 }}>
                {(item.confidence * 100).toFixed(0)}%
              </span>
              <span style={{ color: "var(--muted)", fontSize: 12, flexShrink: 0 }}>
                {item.published_at?.slice(0, 10)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
