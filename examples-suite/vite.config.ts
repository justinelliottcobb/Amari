import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

// Updated for v0.6.0 - uses latest published amari-core

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react(), tsconfigPaths()],
    build: {
      outDir: "dist",
    },
    server: {
      host: '0.0.0.0',
      port: parseInt(env.VITE_PORT || '5173', 10),
      strictPort: true,
      allowedHosts: env.VITE_ALLOWED_HOST ? [env.VITE_ALLOWED_HOST, 'localhost'] : undefined,
    },
    assetsInclude: ['**/*.wasm'],
    optimizeDeps: {
      exclude: ['@justinelliottcobb/amari-core']
    },
  };
});
