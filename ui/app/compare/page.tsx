"use client";

import { Suspense } from "react";
import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { fetchCompare, type PairSummary } from "@/lib/api";
import { ADV_COLOR, COOP_COLOR, NEUT_COLOR, ALL_PAIRS } from "@/lib/constants";
import SegmentBar from "@/components/SegmentBar";
import ModelToggle from "@/components/ModelToggle";
import ErrorBanner from "@/components/ErrorBanner";

function StatRow({ label, v1, v2, color }: { label: string; v1: number; v2: number; color: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "8px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ flex: 1, color: "var(--muted)", fontSize: 13 }}>{label}</span>
      <span className="mono" style={{ width: 60, textAlign: "right", color }}>{v1.toFixed(1)}%</span>
      <span className="mono" style={{ width: 60, textAlign: "right", color }}>{v2.toFixed(1)}%</span>
    </div>
  );
}
export default function ComparePage() {
  return (
    <Suspense>
      <CompareInner />
    </Suspense>
  );
}

function CompareInner() {
  const router = useRouter();
  const params = useSearchParams();
  const [pair1, setPair1] = useState(params.get("pair1") ?? "IN-US");
  const [pair2, setPair2] = useState(params.get("pair2") ?? "CN-US");
  const [model, setModel] = useState<"baseline" | "transformer">("baseline");
  const [data, setData] = useState<{ pair1: PairSummary; pair2: PairSummary } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    router.replace(`/compare?pair1=${pair1}&pair2=${pair2}&model=${model}`, { scroll: false });
  }, [pair1, pair2, model]);

  useEffect(() => {
    setData(null);
    setError(null);
    fetchCompare(pair1, pair2, model)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [pair1, pair2, model]);

  const PairSelect = ({ value, onChange }: { value: string; onChange: (v: string) => void }) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, color: "var(--text)", padding: "5px 10px", fontSize: 13, fontFamily: "monospace" }}
    >
      {ALL_PAIRS.map((p) => <option key={p}>{p}</option>)}
    </select>
  );

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
        <h1 style={{ fontWeight: 500, fontSize: 20, margin: 0 }}>Compare Pairs</h1>
        <PairSelect value={pair1} onChange={setPair1} />
        <span style={{ color: "var(--muted)" }}>vs</span>
        <PairSelect value={pair2} onChange={setPair2} />
        <ModelToggle value={model} onChange={setModel} />
      </div>

      {error && <ErrorBanner message={error} />}

      {data === null ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          {[0, 1].map((i) => <div key={i} className="skeleton" style={{ height: 200, borderRadius: 8 }} />)}
        </div>
      ) : (
        <div>
          {/* Header row */}
          <div style={{ display: "flex", gap: 12, marginBottom: 4 }}>
            <div style={{ flex: 1 }} />
            {[data.pair1, data.pair2].map((p) => (
              <span key={p.pair} className="mono" style={{ width: 60, textAlign: "right", fontWeight: 500 }}>{p.pair}</span>
            ))}
          </div>

          <div style={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 8, padding: "16px 20px" }}>
            <StatRow label="Adversarial" v1={data.pair1.percent.adversarial} v2={data.pair2.percent.adversarial} color={ADV_COLOR} />
            <StatRow label="Cooperative" v1={data.pair1.percent.cooperative} v2={data.pair2.percent.cooperative} color={COOP_COLOR} />
            <StatRow label="Neutral" v1={data.pair1.percent.neutral} v2={data.pair2.percent.neutral} color={NEUT_COLOR} />
            <div style={{ padding: "12px 0 4px" }}>
              <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 6 }}>Distribution</div>
              <div style={{ display: "flex", gap: 12 }}>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>{data.pair1.pair}</div>
                  <SegmentBar adv={data.pair1.percent.adversarial} coop={data.pair1.percent.cooperative} neut={data.pair1.percent.neutral} height={8} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>{data.pair2.pair}</div>
                  <SegmentBar adv={data.pair2.percent.adversarial} coop={data.pair2.percent.cooperative} neut={data.pair2.percent.neutral} height={8} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
