export default function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      style={{
        background: "#2d1f0a",
        border: "1px solid #92400e",
        borderRadius: 6,
        padding: "10px 16px",
        color: "#fbbf24",
        fontSize: 13,
        marginBottom: 20,
      }}
    >
      ⚠ API unavailable — {message}
    </div>
  );
}
