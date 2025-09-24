import { useState, useEffect } from 'react';
import init, * as amari from '@justinelliottcobb/amari-core/pkg/amari_wasm.js';

interface AmariModule {
  ready: boolean;
  error: string | null;
  amari: typeof amari | null;
}

export function useAmariWasm(): AmariModule {
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [amariModule, setAmariModule] = useState<typeof amari | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadWasm() {
      try {
        // Initialize the WASM module
        await init();

        if (!cancelled) {
          setAmariModule(amari);
          setReady(true);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load WASM module');
        }
      }
    }

    loadWasm();

    return () => {
      cancelled = true;
    };
  }, []);

  return { ready, error, amari: amariModule };
}