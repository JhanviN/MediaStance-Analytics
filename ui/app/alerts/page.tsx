"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { fetchAlerts, type Alert } from "@/lib/api";
import { ADV_COLOR, pairLabel } from "@/lib/constants";
import ModelToggle from "@/components/ModelToggle";
import ErrorBanner from "@/components/ErrorBanner";

const SEVERITY_COLOR: Record<string, string> = {
  high: "#E24B4A",
  medium: "#f59e0b",
  low: "#64748b",
};

export default function AlertsPage() {
  const router = useRouter();
  const [model, setModel] = useState<"baseline" | "transformer">("baseline");
  const [days, setDays] = useState(7);
  const [threshold, setThreshold] = useState(15);
  const [alerts, setAlerts] = useState<Alert[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setAlerts(null);
    setError(null);
    fetchAlerts(model, days, threshold)
      .then((r) => setAlerts(r.alerts ?? []))
      .catch((e) => setError(e.message));
  }, [model, days, threshold]);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24, flexWrap: "wrap", gap: 12 }}>
        <h1 style={{ fontWeight: 500, fontSize: 20, margin: 0 }}>Alerts</h1>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>Window</span>
            <select
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
              style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", padding: "4px 8px", fontSize: 12 }}
            >
              {[3, 7, 14, 30].map((d) => <option key={d} value={d}>{d}d</option>)}
            </select>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 12, color: "var(--muted)" }}>Threshold</span>
            <select
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
              style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", padding: "4px 8px", fontSize: 12 }}
            >
              {[5, 10, 15, 20, 25].map((t) => <option key={t} value={t}>{t}pp</option>)}
            </select>
          </div>
          <ModelToggle value={model} onChange={setModel} />
        </div>
      </div>

      {error && <ErrorBanner message={error} />}

      {alerts === null ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="skeleton" style={{ height: 64, borderRadius: 8 }} />
          ))}
        </div>
      ) : alerts.length === 0 ? (
        <div
          style={{
            background: "#f0fdf4",
            border: "1px solid #bbf7d0",
            borderRadius: 8,
            padding: "20px 24px",
            color: "#15803d",
          }}
        >
          ✓ No active alerts for {model} model
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {alerts.map((a) => (
            <div
              key={a.pair}
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderLeft: `3px solid ${SEVERITY_COLOR[a.severity] ?? ADV_COLOR}`,
                borderRadius: 8,
                padding: "14px 20px",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
                <span className="mono" style={{ fontWeight: 500, fontSize: 15 }}>{pairLabel(a.pair)}</span>
                <span
                  style={{
                    fontSize: 11,
                    padding: "2px 8px",
                    borderRadius: 4,
                    background: `${SEVERITY_COLOR[a.severity]}22`,
                    color: SEVERITY_COLOR[a.severity],
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                  }}
                >
                  {a.severity}
                </span>
                <span style={{ color: "var(--muted)", fontSize: 13 }}>
                  adversarial rose{" "}
                  <span className="mono" style={{ color: ADV_COLOR }}>
                    +{a.delta_adversarial_pp.toFixed(1)}pp
                  </span>{" "}
                  vs prior window
                </span>
              </div>
              <button
                onClick={() => router.push(`/trends?pair=${a.pair}&model=${model}`)}
                style={{
                  background: "none",
                  border: "none",
                  color: ADV_COLOR,
                  cursor: "pointer",
                  fontSize: 13,
                }}
              >
                Investigate →
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
