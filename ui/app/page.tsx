"use client";

import { useEffect, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { fetchSummaryAll, fetchAlerts, type SummaryAll, type Alert } from "@/lib/api";
import { ADV_COLOR, COOP_COLOR, pairLabel } from "@/lib/constants";
import MetricCard from "@/components/MetricCard";
import SegmentBar from "@/components/SegmentBar";
import ModelToggle from "@/components/ModelToggle";
import ErrorBanner from "@/components/ErrorBanner";
import DateRangeFilter, { type DateRange } from "@/components/DateRangeFilter";

function PairCardSkeleton() {
  return (
    <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: 16 }}>
      <div className="skeleton" style={{ height: 16, width: 60, marginBottom: 12 }} />
      <div className="skeleton" style={{ height: 5, marginBottom: 10 }} />
      <div style={{ display: "flex", gap: 8 }}>
        <div className="skeleton" style={{ height: 12, width: 40 }} />
        <div className="skeleton" style={{ height: 12, width: 40 }} />
        <div className="skeleton" style={{ height: 12, width: 40 }} />
      </div>
    </div>
  );
}

export default function OverviewPage() {
  const router = useRouter();
  const [model, setModel] = useState<"baseline" | "transformer">("baseline");
  const [dateRange, setDateRange] = useState<DateRange>({ startDate: undefined, endDate: undefined });
  const [data, setData] = useState<SummaryAll | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [, startTransition] = useTransition();

  useEffect(() => {
    setData(null);
    setError(null);
    startTransition(async () => {
      try {
        const [summary, alertsRes] = await Promise.all([
          fetchSummaryAll(model, dateRange.startDate, dateRange.endDate),
          fetchAlerts(model),
        ]);
        setData(summary);
        setAlerts(alertsRes.alerts ?? []);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    });
  }, [model, dateRange]);

  // Derive flat array from dict
  const pairsArray = data ? Object.values(data.pairs) : [];
  const alertPairs = new Set(alerts.map((a) => a.pair));

  const totalPredictions = pairsArray.reduce((s, p) => s + p.total, 0);
  const globalAdv = pairsArray.length
    ? pairsArray.reduce((s, p) => s + p.percent.adversarial, 0) / pairsArray.length
    : 0;
  const globalCoop = pairsArray.length
    ? pairsArray.reduce((s, p) => s + p.percent.cooperative, 0) / pairsArray.length
    : 0;
  const mostAdv = pairsArray.length
    ? pairsArray.reduce((a, b) => b.percent.adversarial > a.percent.adversarial ? b : a)
    : null;

  return (
    <div>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap", gap: 12 }}>
        <h1 style={{ fontWeight: 500, fontSize: 20, margin: 0 }}>Overview</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <DateRangeFilter value={dateRange} onChange={setDateRange} />
          <ModelToggle value={model} onChange={setModel} />
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {/* Alert banner */}
      {alerts.length > 0 ? (
        <div
          style={{
            background: "#fff5f5",
            border: "1px solid #fecaca",
            borderRadius: 6,
            padding: "10px 16px",
            marginBottom: 20,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span style={{ color: "#b91c1c", fontSize: 13 }}>
            {alerts.length} active alert{alerts.length > 1 ? "s" : ""} —{" "}
            {alerts.map((a) => a.pair).join(", ")}
          </span>
          <button
            onClick={() => router.push("/alerts")}
            style={{ background: "none", border: "none", color: ADV_COLOR, cursor: "pointer", fontSize: 13 }}
          >
            Investigate →
          </button>
        </div>
      ) : (
        <div
          style={{
            background: "#f0fdf4",
            border: "1px solid #bbf7d0",
            borderRadius: 6,
            padding: "8px 16px",
            marginBottom: 20,
            color: "#15803d",
            fontSize: 13,
          }}
        >
          ✓ No active alerts
        </div>
      )}

      {/* Metric cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 28 }}>
        {data ? (
          <>
            <MetricCard
              label="Total Predictions"
              value={totalPredictions.toLocaleString()}
              sub={`across ${pairsArray.length} pairs`}
            />
            <MetricCard
              label="Global Adversarial Avg"
              value={`${globalAdv.toFixed(1)}%`}
              valueColor={ADV_COLOR}
            />
            <MetricCard
              label="Global Cooperative Avg"
              value={`${globalCoop.toFixed(1)}%`}
              valueColor={COOP_COLOR}
            />
            <MetricCard
              label="Most Adversarial Pair"
              value={mostAdv ? pairLabel(mostAdv.pair) : "—"}
              sub={mostAdv ? `${mostAdv.percent.adversarial.toFixed(1)}% adversarial` : undefined}
              valueColor={ADV_COLOR}
            />
          </>
        ) : (
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "18px 20px" }}>
              <div className="skeleton" style={{ height: 12, width: 120, marginBottom: 10 }} />
              <div className="skeleton" style={{ height: 28, width: 80 }} />
            </div>
          ))
        )}
      </div>

      {/* Pair grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
        {data
          ? pairsArray.map((p) => (
              <div
                key={p.pair}
                onClick={() => router.push(`/trends?pair=${p.pair}&model=${model}`)}
                style={{
                  background: "var(--surface)",
                  border: `1px solid var(--border)`,
                  borderLeft: alertPairs.has(p.pair) ? `3px solid ${ADV_COLOR}` : "1px solid var(--border)",
                  borderRadius: 8,
                  padding: "14px 16px",
                  cursor: "pointer",
                  transition: "border-color 0.15s",
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLDivElement).style.borderColor = "#c7cdd9";
                  (e.currentTarget as HTMLDivElement).style.boxShadow = "0 2px 8px rgba(0,0,0,0.06)";
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLDivElement).style.borderColor =
                    alertPairs.has(p.pair) ? ADV_COLOR : "var(--border)";
                  (e.currentTarget as HTMLDivElement).style.boxShadow = "none";
                }}
              >
                <div className="mono" style={{ fontWeight: 500, fontSize: 15, marginBottom: 10 }}>
                  {pairLabel(p.pair)}
                </div>
                <SegmentBar
                  adv={p.percent.adversarial}
                  coop={p.percent.cooperative}
                  neut={p.percent.neutral}
                />
                <div style={{ display: "flex", gap: 12, marginTop: 8, fontSize: 12 }}>
                  <span style={{ color: ADV_COLOR }}>{p.percent.adversarial.toFixed(1)}%</span>
                  <span style={{ color: COOP_COLOR }}>{p.percent.cooperative.toFixed(1)}%</span>
                  <span style={{ color: "#378ADD" }}>{p.percent.neutral.toFixed(1)}%</span>
                </div>
              </div>
            ))
          : Array.from({ length: 15 }).map((_, i) => <PairCardSkeleton key={i} />)}
      </div>
    </div>
  );
}
