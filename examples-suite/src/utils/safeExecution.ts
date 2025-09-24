/**
 * Safe execution utilities for handling WASM operations and mathematical computations
 */

export interface ExecutionResult<T> {
  success: boolean;
  data?: T;
  error?: string;
  fallbackUsed?: boolean;
}

export interface SafeExecutionOptions {
  timeout?: number;
  fallback?: () => any;
  retries?: number;
  retryDelay?: number;
}

/**
 * Safely execute a function with error handling and optional fallback
 */
export async function safeExecute<T>(
  fn: () => Promise<T> | T,
  options: SafeExecutionOptions = {}
): Promise<ExecutionResult<T>> {
  const { timeout = 5000, fallback, retries = 0, retryDelay = 1000 } = options;

  let lastError: Error | null = null;
  let attempts = 0;

  while (attempts <= retries) {
    try {
      const timeoutPromise = new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('Operation timeout')), timeout);
      });

      const result = await Promise.race([
        Promise.resolve(fn()),
        timeoutPromise
      ]);

      return { success: true, data: result };
    } catch (error) {
      lastError = error instanceof Error ? error : new Error(String(error));
      attempts++;

      if (attempts <= retries) {
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }

  // If we have a fallback, try it
  if (fallback) {
    try {
      const fallbackResult = await Promise.resolve(fallback());
      return {
        success: true,
        data: fallbackResult,
        fallbackUsed: true
      };
    } catch (fallbackError) {
      return {
        success: false,
        error: `Primary execution failed: ${lastError?.message}. Fallback also failed: ${fallbackError}`,
      };
    }
  }

  return {
    success: false,
    error: lastError?.message || 'Unknown error occurred',
  };
}

/**
 * WASM-specific safe execution with module validation
 */
export async function safeWasmExecute<T>(
  wasmModule: any,
  operation: (module: any) => T,
  fallback?: () => T
): Promise<ExecutionResult<T>> {
  if (!wasmModule) {
    if (fallback) {
      try {
        return { success: true, data: fallback(), fallbackUsed: true };
      } catch (error) {
        return { success: false, error: `WASM module not loaded and fallback failed: ${error}` };
      }
    }
    return { success: false, error: 'WASM module not loaded and no fallback provided' };
  }

  return safeExecute(() => operation(wasmModule), { fallback });
}

/**
 * Validate numerical inputs
 */
export function validateNumbers(...values: unknown[]): { valid: boolean; error?: string } {
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (typeof value !== 'number') {
      return { valid: false, error: `Argument ${i + 1} is not a number: ${typeof value}` };
    }
    if (!isFinite(value)) {
      return { valid: false, error: `Argument ${i + 1} is not finite: ${value}` };
    }
  }
  return { valid: true };
}

/**
 * Validate array inputs
 */
export function validateArray(
  value: unknown,
  expectedLength?: number,
  elementValidator?: (element: unknown) => boolean
): { valid: boolean; error?: string } {
  if (!Array.isArray(value)) {
    return { valid: false, error: `Expected array, got ${typeof value}` };
  }

  if (expectedLength !== undefined && value.length !== expectedLength) {
    return { valid: false, error: `Expected array of length ${expectedLength}, got ${value.length}` };
  }

  if (elementValidator) {
    for (let i = 0; i < value.length; i++) {
      if (!elementValidator(value[i])) {
        return { valid: false, error: `Invalid element at index ${i}: ${value[i]}` };
      }
    }
  }

  return { valid: true };
}

/**
 * Safe mathematical operations with overflow protection
 */
export const safeMath = {
  add: (a: number, b: number): number => {
    const result = a + b;
    if (!isFinite(result)) {
      throw new Error(`Addition overflow: ${a} + ${b} = ${result}`);
    }
    return result;
  },

  multiply: (a: number, b: number): number => {
    const result = a * b;
    if (!isFinite(result)) {
      throw new Error(`Multiplication overflow: ${a} * ${b} = ${result}`);
    }
    return result;
  },

  divide: (a: number, b: number): number => {
    if (b === 0) {
      throw new Error('Division by zero');
    }
    const result = a / b;
    if (!isFinite(result)) {
      throw new Error(`Division overflow: ${a} / ${b} = ${result}`);
    }
    return result;
  },

  sqrt: (x: number): number => {
    if (x < 0) {
      throw new Error(`Square root of negative number: ${x}`);
    }
    return Math.sqrt(x);
  },

  log: (x: number): number => {
    if (x <= 0) {
      throw new Error(`Logarithm of non-positive number: ${x}`);
    }
    return Math.log(x);
  }
};

/**
 * Performance monitoring for operations
 */
export class PerformanceMonitor {
  private measurements: Map<string, number[]> = new Map();

  async measure<T>(
    name: string,
    operation: () => Promise<T> | T
  ): Promise<{ result: T; duration: number }> {
    const start = performance.now();
    const result = await Promise.resolve(operation());
    const duration = performance.now() - start;

    if (!this.measurements.has(name)) {
      this.measurements.set(name, []);
    }
    this.measurements.get(name)!.push(duration);

    return { result, duration };
  }

  getStats(name: string) {
    const measurements = this.measurements.get(name) || [];
    if (measurements.length === 0) {
      return null;
    }

    const sorted = [...measurements].sort((a, b) => a - b);
    const avg = measurements.reduce((a, b) => a + b, 0) / measurements.length;
    const median = sorted[Math.floor(sorted.length / 2)];
    const min = sorted[0];
    const max = sorted[sorted.length - 1];

    return { avg, median, min, max, count: measurements.length };
  }

  clear(name?: string) {
    if (name) {
      this.measurements.delete(name);
    } else {
      this.measurements.clear();
    }
  }
}

export const globalPerformanceMonitor = new PerformanceMonitor();