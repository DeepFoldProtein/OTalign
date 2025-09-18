import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Optimize for Cloudflare Workers
  output: "standalone",
  // Disable image optimization that might cause issues
  images: {
    unoptimized: true,
  },
  // Ensure proper bundling for Cloudflare
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals = [...(config.externals || []), "fs", "path"];
    }
    // Disable dynamic requires for Cloudflare compatibility
    config.resolve = config.resolve || {};
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      os: false,
      crypto: false,
    };
    return config;
  },
  // External packages for server components
  serverExternalPackages: [],
};

export default nextConfig;
