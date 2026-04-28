"use client";

import { useState } from "react";

export interface DateRange {
  startDate: string | undefined; // YYYY-MM-DD
  endDate: string | undefined;
}

interface Props {
  value: DateRange;
  onChange: (range: DateRange) => void;
}

const PRESETS = [
  { label: "7d",  days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
  { label: "All", days: 0 },
];

function daysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString().slice(0, 10);
}

function activePreset(range: DateRange): number {
  if (!range.startDate && !range.endDate) return 0; // "All"
  if (!range.startDate) return -1;
  const diffMs = Date.now() - new Date(range.startDate).getTime();
  const diffDays = Math.round(diffMs / 86400000);
  const match = PRESETS.find((p) => p.days === diffDays);
  return match ? match.days : -1;
}

export default function DateRangeFilter({ value, onChange }: Props) {
  const [showCustom, setShowCustom] = useState(false);
  const active = activePreset(value);

  const applyPreset = (days: number) => {
    setShowCustom(false);
    if (days === 0) {
      onChange({ startDate: undefined, endDate: undefined });
    } else {
      onChange({ startDate: daysAgo(days), endDate: undefined });
    }
  };

  const selectStyle = (isActive: boolean): React.CSSProperties => ({
    padding: "4px 10px",
    fontSize: 12,
    borderRadius: 5,
    border: isActive ? "1px solid #c7cdd9" : "1px solid var(--border)",
    background: isActive ? "#1e2330" : "var(--surface)",
    color: isActive ? "#fff" : "var(--muted)",
    cursor: "pointer",
    fontWeight: isActive ? 500 : 400,
    transition: "all 0.12s",
  });

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
      {PRESETS.map((p) => (
        <button
          key={p.label}
          onClick={() => applyPreset(p.days)}
          style={selectStyle(active === p.days)}
        >
          {p.label}
        </button>
      ))}

      {/* Custom toggle */}
      <button
        onClick={() => setShowCustom((v) => !v)}
        style={selectStyle(showCustom || active === -1)}
      >
        Custom
      </button>

      {/* Custom date inputs */}
      {showCustom && (
        <>
          <input
            type="date"
            value={value.startDate ?? ""}
            onChange={(e) =>
              onChange({ ...value, startDate: e.target.value || undefined })
            }
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 5,
              color: "var(--text)",
              padding: "3px 8px",
              fontSize: 12,
            }}
          />
          <span style={{ color: "var(--muted)", fontSize: 12 }}>→</span>
          <input
            type="date"
            value={value.endDate ?? ""}
            onChange={(e) =>
              onChange({ ...value, endDate: e.target.value || undefined })
            }
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 5,
              color: "var(--text)",
              padding: "3px 8px",
              fontSize: 12,
            }}
          />
        </>
      )}
    </div>
  );
}
