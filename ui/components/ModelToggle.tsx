"use client";

interface Props {
  value: "baseline" | "transformer";
  onChange: (v: "baseline" | "transformer") => void;
}

export default function ModelToggle({ value, onChange }: Props) {
  return (
    <div
      style={{
        display: "inline-flex",
        background: "var(--bg)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        overflow: "hidden",
      }}
    >
      {(["baseline", "transformer"] as const).map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          style={{
            padding: "5px 14px",
            fontSize: 13,
            fontWeight: value === m ? 500 : 400,
            background: value === m ? "#ffffff" : "transparent",
            color: value === m ? "var(--text)" : "var(--muted)",
            border: "none",
            cursor: "pointer",
            transition: "background 0.15s",
            boxShadow: value === m ? "0 1px 3px rgba(0,0,0,0.08)" : "none",
          }}
        >
          {m}
        </button>
      ))}
    </div>
  );
}
