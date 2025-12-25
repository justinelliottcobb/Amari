import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tsconfigPaths from 'vite-tsconfig-paths';
import path from 'path';

export default defineConfig({
  plugins: [react(), tsconfigPaths()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/test/',
        '**/*.d.ts',
        '**/*.config.*',
        '**/types/',
      ],
    },
  },
  resolve: {
    alias: {
      // Mock WASM module since it's not available in Node environment
      '@justinelliottcobb/amari-wasm/amari_wasm.js': path.resolve(__dirname, 'src/test/mocks/amari-wasm.ts'),
      '@justinelliottcobb/amari-wasm/amari_wasm_bg.wasm?url': path.resolve(__dirname, 'src/test/mocks/wasm-url.ts'),
      '@justinelliottcobb/amari-wasm': path.resolve(__dirname, 'src/test/mocks/amari-wasm.ts'),
    },
  },
});
