"use client";

import Image from "next/image";
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
  { href: "/causality", label: "Causality" },
];

export default function Sidebar() {
  const path = usePathname();

  return (
    <nav
      style={{
        width: "var(--sidebar-w)",
        minWidth: "var(--sidebar-w)",
        background: "#ffffff",
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        padding: "24px 0",
        position: "sticky",
        top: 0,
        height: "100vh",
        boxShadow: "1px 0 0 var(--border)",
      }}
    >
      {/* Logo */}
      <div style={{ padding: "0 20px 24px", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", gap: 10 }}>
        <Image src="/msa.svg" alt="MediaStance" width={32} height={32} style={{ borderRadius: 8 }} />
        <span style={{ fontWeight: 500, fontSize: 15, letterSpacing: "-0.3px" }}>
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
                color: active ? "#1e2330" : "var(--muted)",
                textDecoration: "none",
                fontWeight: active ? 500 : 400,
                borderLeft: active ? "3px solid var(--adv)" : "3px solid transparent",
                background: active ? "#fef2f2" : "transparent",
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
        <div className="mono" style={{ color: "var(--muted)" }}>human gold F1: 78%</div>
        <div style={{ marginTop: 2 }}>mixed score: 87%</div>
      </div>
    </nav>
  );
}
