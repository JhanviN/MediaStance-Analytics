"use client";

import { Suspense } from "react";
import { useEffect, useState, useTransition } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
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
      ) : data.series.length === 0 ? (
        <div style={{ color: "var(--muted)", padding: "40px 0" }}>
          No trend data found for {pair} with model {model}.
        </div>
      ) : (
        <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "20px 16px" }}>
          <div style={{ marginBottom: 12, fontSize: 13, color: "var(--muted)" }}>
            Daily label distribution — <span className="mono">{pair}</span>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data.series} margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3d" />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} tickLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickLine={false} axisLine={false} unit="%" />
              <Tooltip
                contentStyle={{ background: "#181c24", border: "1px solid #2a2f3d", borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Line type="monotone" dataKey="adversarial" stroke={ADV_COLOR} dot={false} strokeWidth={2} name="Adversarial" />
              <Line type="monotone" dataKey="cooperative" stroke={COOP_COLOR} dot={false} strokeWidth={2} name="Cooperative" />
              <Line type="monotone" dataKey="neutral" stroke={NEUT_COLOR} dot={false} strokeWidth={2} name="Neutral" />
              {data.rolling_adversarial && (
                <Line
                  type="monotone"
                  data={data.rolling_adversarial}
                  dataKey="value"
                  stroke={ADV_COLOR}
                  dot={false}
                  strokeWidth={1.5}
                  strokeDasharray="4 2"
                  name="Adv 7d avg"
                />
              )}
            </LineChart>
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
