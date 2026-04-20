"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/trends", label: "Trends" },
  { href: "/alerts", label: "Alerts" },
  { href: "/headlines", label: "Headlines" },
  { href: "/compare", label: "Compare Pairs" },
  { href: "/predict", label: "Live Predict" },
  { href: "/attention", label: "Attention" },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <nav
      style={{
        width: "var(--sidebar-w)",
        minWidth: "var(--sidebar-w)",
        background: "var(--surface)",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        padding: "24px 0",
        position: "sticky",
        top: 0,
        height: "100vh",
      }}
    >
      {/* Logo */}
      <div style={{ padding: "0 20px 24px", borderBottom: "1px solid var(--border)" }}>
        <span style={{ fontWeight: 500, fontSize: 16, letterSpacing: "-0.3px" }}>
          Media<span style={{ color: "var(--adv)" }}>Stance</span>
        </span>
      </div>

      {/* Nav items */}
      <div style={{ flex: 1, padding: "12px 0" }}>
        {NAV.map(({ href, label }) => {
          const active = path === href || (href !== "/" && path.startsWith(href));
          return (
            <Link
              key={href}
              href={href}
              style={{
                display: "block",
                padding: "8px 20px",
                color: active ? "#f1f5f9" : "var(--muted)",
                textDecoration: "none",
                fontWeight: active ? 500 : 400,
                borderLeft: active ? "3px solid var(--adv)" : "3px solid transparent",
                transition: "color 0.15s",
              }}
            >
              {label}
            </Link>
          );
        })}
      </div>

      {/* Footer */}
      <div
        style={{
          padding: "16px 20px",
          borderTop: "1px solid var(--border)",
          color: "var(--muted)",
          fontSize: 12,
          lineHeight: 1.6,
        }}
      >
        <div className="mono" style={{ color: "#94a3b8" }}>human gold F1: 78%</div>
        <div style={{ marginTop: 2 }}>mixed score: 87%</div>
      </div>
    </nav>
  );
}
