import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  output: "export",   // static export for Netlify — all pages are client-side
  trailingSlash: true, // Netlify needs this for clean URLs
};

export default nextConfig;
