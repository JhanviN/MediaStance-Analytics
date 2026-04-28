import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "MediaStance Analytics",
  description: "Real-time bilateral geopolitical sentiment classification from news headlines",
  icons: {
    icon: "/msa.svg",
    apple: "/msa.svg",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ display: "flex", minHeight: "100vh" }}>
        <Sidebar />
        <main style={{ flex: 1, padding: "28px 32px", overflowY: "auto" }}>
          {children}
        </main>
      </body>
    </html>
  );
}
