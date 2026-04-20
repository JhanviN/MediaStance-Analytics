export default function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      style={{
        background: "#fffbeb",
        border: "1px solid #fcd34d",
        borderRadius: 6,
        padding: "10px 16px",
        color: "#92400e",
        fontSize: 13,
        marginBottom: 20,
      }}
    >
      ⚠ API unavailable — {message}
    </div>
  );
}
