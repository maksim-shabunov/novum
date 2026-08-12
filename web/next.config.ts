import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /**
   * Standalone output: the Docker runtime stage copies a self-contained server
   * with only the modules actually reached, so the shipped image carries no
   * node_modules tree and no build toolchain.
   */
  output: "standalone",

  /**
   * The console is a static page over precomputed JSON. No image optimiser is
   * needed — the only image is one sprite atlas served straight from public/,
   * and running an optimiser over it would add a server dependency for nothing.
   */
  images: { unoptimized: true },
};

export default nextConfig;
