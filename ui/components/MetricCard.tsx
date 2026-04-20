interface Props {
  label: string;
  value: string | number;
  sub?: string;
  valueColor?: string;
}

export default function MetricCard({ label, value, sub, valueColor }: Props) {
  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "18px 20px",
      }}
    >
      <div style={{ color: "var(--muted)", fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div
        style={{
          fontSize: 28,
          fontWeight: 500,
          color: valueColor ?? "var(--text)",
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ color: "var(--muted)", fontSize: 12, marginTop: 6 }}>{sub}</div>
      )}
    </div>
  );
}
