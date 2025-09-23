import { useState, useEffect, useCallback } from "react";
import { safeWasmExecute, type ExecutionResult } from "~/utils/safeExecution";

interface WasmLoadState {
  loading: boolean;
  loaded: boolean;
  error: string | null;
  module: any;
}

interface UseAmariWasmResult extends WasmLoadState {
  execute: <T>(
    operation: (module: any) => T,
    fallback?: () => T
  ) => Promise<ExecutionResult<T>>;
  retry: () => void;
}

/**
 * Hook for safely loading and using the Amari WASM module
 */
export function useAmariWasm(): UseAmariWasmResult {
  const [state, setState] = useState<WasmLoadState>({
    loading: false,
    loaded: false,
    error: null,
    module: null,
  });

  const loadWasm = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      // In a real implementation, this would load the actual WASM module
      // For now, we'll simulate the loading and create a mock module
      await new Promise(resolve => setTimeout(resolve, 100));

      const mockModule = {
        WasmMultivector: {
          basisVector: (index: number) => ({
            getCoefficients: () => {
              const coeffs = [0, 0, 0, 0, 0, 0, 0, 0];
              if (index < 3) coeffs[index + 1] = 1;
              return coeffs;
            },
            getCoefficient: (i: number) => {
              const coeffs = [0, 0, 0, 0, 0, 0, 0, 0];
              if (index < 3) coeffs[index + 1] = 1;
              return coeffs[i];
            },
            geometricProduct: function(other: any) {
              // Simple e1 * e2 = e12 simulation
              const result = [0, 0, 0, 0, 1, 0, 0, 0];
              return {
                getCoefficients: () => result,
                getCoefficient: (i: number) => result[i]
              };
            },
            innerProduct: function(other: any) {
              // Simple inner product simulation
              return {
                getCoefficients: () => [1, 0, 0, 0, 0, 0, 0, 0],
                getCoefficient: (i: number) => i === 0 ? 1 : 0
              };
            }
          }),
          fromCoefficients: (coeffs: number[]) => ({
            getCoefficients: () => [...coeffs],
            getCoefficient: (i: number) => coeffs[i] || 0
          })
        }
      };

      setState({
        loading: false,
        loaded: true,
        error: null,
        module: mockModule,
      });
    } catch (error) {
      setState({
        loading: false,
        loaded: false,
        error: error instanceof Error ? error.message : String(error),
        module: null,
      });
    }
  }, []);

  const execute = useCallback(
    async <T>(
      operation: (module: any) => T,
      fallback?: () => T
    ): Promise<ExecutionResult<T>> => {
      return safeWasmExecute(state.module, operation, fallback);
    },
    [state.module]
  );

  const retry = useCallback(() => {
    loadWasm();
  }, [loadWasm]);

  useEffect(() => {
    if (!state.loaded && !state.loading) {
      loadWasm();
    }
  }, [loadWasm, state.loaded, state.loading]);

  return {
    ...state,
    execute,
    retry,
  };
}

/**
 * Hook for checking WebGPU availability with fallbacks
 */
export function useWebGPU() {
  const [state, setState] = useState<{
    available: boolean;
    loading: boolean;
    device: GPUDevice | null;
    error: string | null;
  }>({
    available: false,
    loading: true,
    device: null,
    error: null,
  });

  useEffect(() => {
    async function checkWebGPU() {
      try {
        if (!navigator.gpu) {
          setState({
            available: false,
            loading: false,
            device: null,
            error: "WebGPU not supported in this browser",
          });
          return;
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          setState({
            available: false,
            loading: false,
            device: null,
            error: "WebGPU adapter not available",
          });
          return;
        }

        const device = await adapter.requestDevice();
        setState({
          available: true,
          loading: false,
          device,
          error: null,
        });
      } catch (error) {
        setState({
          available: false,
          loading: false,
          device: null,
          error: error instanceof Error ? error.message : String(error),
        });
      }
    }

    checkWebGPU();
  }, []);

  return state;
}

/**
 * Hook for performance monitoring with automatic cleanup
 */
export function usePerformanceMonitor(name: string) {
  const [stats, setStats] = useState<{
    avg: number;
    median: number;
    min: number;
    max: number;
    count: number;
  } | null>(null);

  const measure = useCallback(
    async <T>(operation: () => Promise<T> | T): Promise<T> => {
      const start = performance.now();
      const result = await Promise.resolve(operation());
      const duration = performance.now() - start;

      // Update stats (simplified version)
      setStats(prev => {
        const measurements = prev ? [prev.avg] : [];
        measurements.push(duration);

        const sorted = measurements.sort((a, b) => a - b);
        const avg = measurements.reduce((a, b) => a + b, 0) / measurements.length;
        const median = sorted[Math.floor(sorted.length / 2)];
        const min = sorted[0];
        const max = sorted[sorted.length - 1];

        return { avg, median, min, max, count: measurements.length };
      });

      return result;
    },
    []
  );

  const clear = useCallback(() => {
    setStats(null);
  }, []);

  return { stats, measure, clear };
}