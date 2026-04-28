"use client";

import { useEffect, useState, useTransition } from "react";
import {
  fetchCausality,
  type CausalityResponse,
  type CausalNode,
  type CausalEdge,
} from "@/lib/api";
import { ADV_COLOR, COOP_COLOR, NEUT_COLOR, LABEL_COLORS, ALL_PAIRS } from "@/lib/constants";
import ModelToggle from "@/components/ModelToggle";
import ErrorBanner from "@/components/ErrorBanner";

// ── Helpers ───────────────────────────────────────────────────────────────────

function labelColor(label: string) {
  return LABEL_COLORS[label] ?? "#8a93a8";
}

// ── Spike summary card ────────────────────────────────────────────────────────

function SpikeSummary({ spike }: { spike: CausalityResponse["spike"] }) {
  const { narrative, deltas, this_window, prev_window } = spike;
  const advDelta = deltas.adversarial;

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "16px 20px",
        marginBottom: 20,
      }}
    >
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 8 }}>
        What changed — last 7 days vs prior 7 days
      </div>
      <p style={{ margin: "0 0 14px", fontSize: 14, color: "var(--text)", lineHeight: 1.5 }}>
        {narrative}
      </p>
      <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
        {(["adversarial", "cooperative", "neutral"] as const).map((lab) => {
          const delta = deltas[lab];
          const color = labelColor(lab);
          return (
            <div key={lab}>
              <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 2 }}>
                {lab}
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                <span className="mono" style={{ fontSize: 15, color }}>
                  {this_window.pct[lab].toFixed(1)}%
                </span>
                <span
                  className="mono"
                  style={{
                    fontSize: 11,
                    color: delta > 0 ? ADV_COLOR : COOP_COLOR,
                  }}
                >
                  {delta > 0 ? "+" : ""}
                  {delta.toFixed(1)}pp
                </span>
              </div>
            </div>
          );
        })}
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 2 }}>
            this window
          </div>
          <span className="mono" style={{ fontSize: 15 }}>{this_window.total}</span>
        </div>
        <div>
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 2 }}>
            prior window
          </div>
          <span className="mono" style={{ fontSize: 15 }}>{prev_window.total}</span>
        </div>
      </div>

      {/* Top headlines this window */}
      {this_window.top_headlines.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 6 }}>
            Top headlines this window
          </div>
          {this_window.top_headlines.slice(0, 3).map((h, i) => (
            <div
              key={i}
              style={{
                borderLeft: `3px solid ${labelColor(h.label)}`,
                paddingLeft: 10,
                marginBottom: 6,
                fontSize: 13,
                color: "var(--text)",
              }}
            >
              {h.headline}
              <span
                className="mono"
                style={{ marginLeft: 8, fontSize: 11, color: "var(--muted)" }}
              >
                {(h.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── State transition graph (SVG) ──────────────────────────────────────────────

function CausalGraphViz({
  nodes,
  edges,
}: {
  nodes: CausalNode[];
  edges: CausalEdge[];
}) {
  const [hovered, setHovered] = useState<string | null>(null);

  if (nodes.length === 0) {
    return (
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
        Not enough data to build a state graph for this pair and time range.
      </div>
    );
  }

  // Layout: x = time index, y = label row
  const LABEL_Y: Record<string, number> = { adversarial: 60, neutral: 140, cooperative: 220 };
  const X_STEP = Math.max(48, Math.min(80, 900 / nodes.length));
  const W = X_STEP * nodes.length + 80;
  const H = 300;

  const nodePos: Record<string, { x: number; y: number }> = {};
  nodes.forEach((n, i) => {
    nodePos[n.id] = {
      x: 40 + i * X_STEP,
      y: LABEL_Y[n.label] ?? 140,
    };
  });

  const maxTotal = Math.max(...nodes.map((n) => n.total), 1);

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "16px",
        overflowX: "auto",
      }}
    >
      {/* Y-axis labels */}
      <div style={{ display: "flex" }}>
        <div
          style={{
            width: 90,
            minWidth: 90,
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-around",
            paddingBottom: 24,
          }}
        >
          {(["adversarial", "neutral", "cooperative"] as const).map((lab) => (
            <span
              key={lab}
              style={{ fontSize: 11, color: labelColor(lab), fontWeight: 500 }}
            >
              {lab}
            </span>
          ))}
        </div>

        <svg width={W} height={H} style={{ overflow: "visible", minWidth: W }}>
          {/* Grid lines */}
          {Object.values(LABEL_Y).map((y) => (
            <line
              key={y}
              x1={0}
              y1={y}
              x2={W}
              y2={y}
              stroke="#e2e6ef"
              strokeDasharray="4 3"
              strokeWidth={1}
            />
          ))}

          {/* Edges */}
          {edges.map((e, i) => {
            const s = nodePos[e.source];
            const t = nodePos[e.target];
            if (!s || !t) return null;
            const color = e.same ? "#c8cdd9" : ADV_COLOR;
            const mx = (s.x + t.x) / 2;
            const my = Math.min(s.y, t.y) - 20;
            const d =
              s.y === t.y
                ? `M ${s.x} ${s.y} L ${t.x} ${t.y}`
                : `M ${s.x} ${s.y} Q ${mx} ${my} ${t.x} ${t.y}`;
            return (
              <path
                key={i}
                d={d}
                fill="none"
                stroke={color}
                strokeWidth={1.5}
                strokeOpacity={0.6}
                markerEnd="url(#arrow)"
              />
            );
          })}

          {/* Arrow marker */}
          <defs>
            <marker
              id="arrow"
              markerWidth="6"
              markerHeight="6"
              refX="5"
              refY="3"
              orient="auto"
            >
              <path d="M0,0 L0,6 L6,3 z" fill="#c8cdd9" />
            </marker>
          </defs>

          {/* Nodes */}
          {nodes.map((n) => {
            const { x, y } = nodePos[n.id];
            const r = 6 + (n.total / maxTotal) * 10;
            const isHovered = hovered === n.id;
            return (
              <g key={n.id}>
                <circle
                  cx={x}
                  cy={y}
                  r={r}
                  fill={labelColor(n.label)}
                  fillOpacity={isHovered ? 1 : 0.75}
                  stroke={isHovered ? "#1e2330" : "white"}
                  strokeWidth={isHovered ? 2 : 1}
                  style={{ cursor: "pointer", transition: "r 0.15s" }}
                  onMouseEnter={() => setHovered(n.id)}
                  onMouseLeave={() => setHovered(null)}
                />
                {/* Date label on hover */}
                {isHovered && (
                  <text
                    x={x}
                    y={y - r - 6}
                    textAnchor="middle"
                    fontSize={10}
                    fill="var(--text)"
                  >
                    {n.window_start}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Hovered node detail */}
      {hovered && (() => {
        const n = nodes.find((x) => x.id === hovered);
        if (!n) return null;
        return (
          <div
            style={{
              marginTop: 12,
              padding: "10px 14px",
              background: "#f9fafb",
              borderRadius: 6,
              fontSize: 13,
              borderLeft: `3px solid ${labelColor(n.label)}`,
            }}
          >
            <strong>{n.window_start}</strong> — {n.total} predictions,{" "}
            {(n.avg_confidence * 100).toFixed(0)}% avg confidence
            {n.top_headlines.slice(0, 2).map((h, i) => (
              <div key={i} style={{ color: "var(--muted)", marginTop: 4, fontSize: 12 }}>
                • {h}
              </div>
            ))}
          </div>
        );
      })()}

      <div style={{ marginTop: 10, fontSize: 11, color: "var(--muted)" }}>
        Each node = a {3}-day window. Size = number of predictions. Hover for details.
        Curved edges = state transitions.
      </div>
    </div>
  );
}

// ── Recurring patterns ────────────────────────────────────────────────────────

function Patterns({ patterns }: { patterns: CausalityResponse["patterns"] }) {
  if (patterns.length === 0) return null;
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "16px 20px",
        marginTop: 20,
      }}
    >
      <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 12 }}>
        Recurring 3-step patterns
      </div>
      {patterns.slice(0, 6).map((p, i) => (
        <div
          key={i}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            marginBottom: 8,
            fontSize: 13,
          }}
        >
          <span className="mono" style={{ color: "var(--muted)", minWidth: 20 }}>
            ×{p.count}
          </span>
          <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
            {p.steps.map((step, j) => (
              <span key={j} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span
                  style={{
                    padding: "2px 8px",
                    borderRadius: 4,
                    background: `${labelColor(step)}18`,
                    color: labelColor(step),
                    fontSize: 12,
                    fontWeight: 500,
                  }}
                >
                  {step}
                </span>
                {j < p.steps.length - 1 && (
                  <span style={{ color: "var(--muted)", fontSize: 11 }}>→</span>
                )}
              </span>
            ))}
          </div>
          {p.ends_adversarial && (
            <span
              style={{
                fontSize: 11,
                color: ADV_COLOR,
                background: `${ADV_COLOR}12`,
                padding: "1px 6px",
                borderRadius: 3,
              }}
            >
              escalation risk
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function CausalityPage() {
  const [pair, setPair] = useState("CN-US");
  const [model, setModel] = useState<"baseline" | "transformer">("baseline");
  const [days, setDays] = useState(30);
  const [data, setData] = useState<CausalityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();

  useEffect(() => {
    setData(null);
    setError(null);
    startTransition(async () => {
      try {
        const res = await fetchCausality(pair, model, days);
        setData(res);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "Unknown error");
      }
    });
  }, [pair, model, days]);

  return (
    <div>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 16,
          marginBottom: 24,
          flexWrap: "wrap",
        }}
      >
        <h1 style={{ fontWeight: 500, fontSize: 20, margin: 0 }}>Causality</h1>
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
          {ALL_PAIRS.map((p) => (
            <option key={p}>{p}</option>
          ))}
        </select>
        <select
          value={days}
          onChange={(e) => setDays(Number(e.target.value))}
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            color: "var(--text)",
            padding: "5px 10px",
            fontSize: 13,
          }}
        >
          {[14, 30, 60, 90].map((d) => (
            <option key={d} value={d}>
              {d} days
            </option>
          ))}
        </select>
        <ModelToggle value={model} onChange={setModel} />
      </div>

      {error && <ErrorBanner message={error} />}

      {isPending || !data ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div className="skeleton" style={{ height: 140, borderRadius: 8 }} />
          <div className="skeleton" style={{ height: 280, borderRadius: 8 }} />
        </div>
      ) : (
        <>
          {/* Spike summary */}
          <SpikeSummary spike={data.spike} />

          {/* Graph */}
          <div style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>
              State Transition Graph —{" "}
              <span className="mono">{pair}</span>
              {data.graph.date_range && (
                <span
                  style={{ fontSize: 12, color: "var(--muted)", marginLeft: 8, fontWeight: 400 }}
                >
                  {data.graph.date_range}
                </span>
              )}
            </div>
            <CausalGraphViz nodes={data.graph.nodes} edges={data.graph.edges} />
          </div>

          {/* Transition summary */}
          {data.graph.transitions && data.graph.transitions.length > 0 && (
            <div
              style={{
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                padding: "14px 20px",
                marginTop: 16,
              }}
            >
              <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 10 }}>
                Most common transitions
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {data.graph.transitions.slice(0, 6).map((t, i) => (
                  <span
                    key={i}
                    style={{
                      fontSize: 12,
                      padding: "3px 10px",
                      borderRadius: 4,
                      background: "#f3f4f6",
                      color: "var(--text)",
                    }}
                  >
                    <span style={{ color: labelColor(t.from) }}>{t.from}</span>
                    {" → "}
                    <span style={{ color: labelColor(t.to) }}>{t.to}</span>
                    <span className="mono" style={{ color: "var(--muted)", marginLeft: 6 }}>
                      ×{t.count}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Recurring patterns */}
          <Patterns patterns={data.patterns} />
        </>
      )}
    </div>
  );
}
