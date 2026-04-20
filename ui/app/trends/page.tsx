"use client";

import { Suspense } from "react";
import { useEffect, useState, useTransition } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";
import { fetchTrends, type TrendsResponse } from "@/lib/api";
import { ADV_COLOR, COOP_COLOR, NEUT_COLOR, ALL_PAIRS } from "@/lib/constants";
import ModelToggle from "@/components/ModelToggle";
import ErrorBanner from "@/components/ErrorBanner";

export default function TrendsPage() {
  return (
    <Suspense>
      <TrendsInner />
    </Suspense>
  );
}

function TrendsInner() {
  const router = useRouter();
  const params = useSearchParams();
  const [pair, setPair] = useState(params.get("pair") ?? "IN-US");
  const [model, setModel] = useState<"baseline" | "transformer">(
    (params.get("model") as "baseline" | "transformer") ?? "baseline"
  );
  const [data, setData] = useState<TrendsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  // Sync URL
  useEffect(() => {
    router.replace(`/trends?pair=${pair}&model=${model}`, { scroll: false });
  }, [pair, model]);

  useEffect(() => {
    setData(null);
    setError(null);
    startTransition(async () => {
      try {
        const res = await fetchTrends(pair, model, 7);
        setData(res);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    });
  }, [pair, model]);

  // Convert fractions (0-1) to percentages for display
  const chartSeries = data?.series.map((p) => ({
    ...p,
    adversarial: +(p.adversarial * 100).toFixed(1),
    cooperative: +(p.cooperative * 100).toFixed(1),
    neutral: +(p.neutral * 100).toFixed(1),
  }));

  const rollingData = data?.rolling_adversarial?.map((p) => ({
    date: p.date,
    value: +((p.adversarial_rolling_avg ?? 0) * 100).toFixed(1),
  }));

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24, flexWrap: "wrap" }}>
        <h1 style={{ fontWeight: 500, fontSize: 20, margin: 0 }}>Trends</h1>
        <select
          value={pair}
          onChange={(e) => setPair(e.target.value)}
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            color: "var(--text)",
            padding: "5px 10px",
            fontSize: 13,
            fontFamily: "monospace",
          }}
        >
          {ALL_PAIRS.map((p) => <option key={p}>{p}</option>)}
        </select>
        <ModelToggle value={model} onChange={setModel} />
      </div>

      {error && <ErrorBanner message={error} />}

      {isPending || !data ? (
        <div>
          <div className="skeleton" style={{ height: 300, borderRadius: 8 }} />
        </div>
      ) : !chartSeries || chartSeries.length === 0 ? (
        <div style={{ color: "var(--muted)", padding: "40px 0" }}>
          No trend data found for {pair} with model {model}.
        </div>
      ) : (
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "20px 16px" }}>
          <div style={{ marginBottom: 12, fontSize: 13, color: "var(--muted)" }}>
            Daily label distribution — <span className="mono">{pair}</span>
            {chartSeries.length === 1 && (
              <span style={{ marginLeft: 12, color: "#f59e0b", fontSize: 12 }}>
                only 1 day of data — more days will appear as predictions accumulate
              </span>
            )}
          </div>
          <ResponsiveContainer width="100%" height={300}>
            {chartSeries.length === 1 ? (
              <BarChart data={chartSeries} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e6ef" />
                <XAxis dataKey="date" tick={{ fill: "#8a93a8", fontSize: 11 }} tickLine={false} />
                <YAxis tick={{ fill: "#8a93a8", fontSize: 11 }} tickLine={false} axisLine={false} unit="%" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: "#ffffff", border: "1px solid #e2e6ef", borderRadius: 6, fontSize: 12 }}
                  formatter={(v: number) => `${v}%`}
                />
                <Bar dataKey="adversarial" fill={ADV_COLOR} name="Adversarial" radius={[3,3,0,0]} />
                <Bar dataKey="cooperative" fill={COOP_COLOR} name="Cooperative" radius={[3,3,0,0]} />
                <Bar dataKey="neutral" fill={NEUT_COLOR} name="Neutral" radius={[3,3,0,0]} />
              </BarChart>
            ) : (
              <LineChart data={chartSeries} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e6ef" />
                <XAxis dataKey="date" tick={{ fill: "#8a93a8", fontSize: 11 }} tickLine={false} />
                <YAxis tick={{ fill: "#8a93a8", fontSize: 11 }} tickLine={false} axisLine={false} unit="%" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ background: "#ffffff", border: "1px solid #e2e6ef", borderRadius: 6, fontSize: 12 }}
                  labelStyle={{ color: "#8a93a8" }}
                  formatter={(v: number) => `${v}%`}
                />
                <Line type="monotone" dataKey="adversarial" stroke={ADV_COLOR} dot={false} strokeWidth={2} name="Adversarial" />
                <Line type="monotone" dataKey="cooperative" stroke={COOP_COLOR} dot={false} strokeWidth={2} name="Cooperative" />
                <Line type="monotone" dataKey="neutral" stroke={NEUT_COLOR} dot={false} strokeWidth={2} name="Neutral" />
                {rollingData && rollingData.length > 0 && (
                  <Line
                    type="monotone"
                    data={rollingData}
                    dataKey="value"
                    stroke={ADV_COLOR}
                    dot={false}
                    strokeWidth={1.5}
                    strokeDasharray="4 2"
                    name="Adv 7d avg"
                  />
                )}
              </LineChart>
            )}
          </ResponsiveContainer>
          {/* Legend */}
          <div style={{ display: "flex", gap: 20, marginTop: 12, fontSize: 12 }}>
            {[["Adversarial", ADV_COLOR], ["Cooperative", COOP_COLOR], ["Neutral", NEUT_COLOR]].map(([l, c]) => (
              <span key={l} style={{ display: "flex", alignItems: "center", gap: 6, color: "var(--muted)" }}>
                <span style={{ width: 16, height: 2, background: c, display: "inline-block" }} />
                {l}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
