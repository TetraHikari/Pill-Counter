import type { NextConfig } from "next";
import { networkInterfaces } from "node:os";

// The phone opens the dev server via the Mac's Wi-Fi IP, which Next.js blocks unless listed.
const lanIps = Object.values(networkInterfaces())
  .flat()
  .filter((n) => n && n.family === "IPv4" && !n.internal)
  .map((n) => n!.address);

const API_URL = process.env.PILLTALLY_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  allowedDevOrigins: lanIps,
  // The browser only talks to Next.js; /api/* is forwarded to the Python model server.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/:path*` }];
  },
};

export default nextConfig;
