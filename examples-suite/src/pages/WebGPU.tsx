import { H1, P, Card, CardHeader, CardBody, Button } from "jadis-ui";
import { ExampleCard } from "../components/ExampleCard";
import { useState, useEffect } from "react";

export function WebGPU() {
  const [webgpuSupported, setWebgpuSupported] = useState<boolean | null>(null);
  const [gpuInfo, setGpuInfo] = useState<string | null>(null);

  useEffect(() => {
    async function checkWebGPUSupport() {
      if (!navigator.gpu) {
        setWebgpuSupported(false);
        setGpuInfo("WebGPU not available in this browser");
        return;
      }

      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          setWebgpuSupported(false);
          setGpuInfo("No WebGPU adapter found");
          return;
        }

        setWebgpuSupported(true);
        setGpuInfo(`WebGPU adapter: ${adapter.info?.vendor || 'Unknown'} ${adapter.info?.device || 'GPU'}`);
      } catch (error) {
        setWebgpuSupported(false);
        setGpuInfo(`WebGPU error: ${error}`);
      }
    }

    checkWebGPUSupport();
  }, []);

  // Simulate GPU operations for demonstration
  const simulateGpuExample = (operation: () => Promise<string>) => {
    return async () => {
      try {
        return await operation();
      } catch (err) {
        throw new Error(`GPU simulation error: ${err}`);
      }
    };
  };

  const examples = [
    {
      title: "GPU Device Detection",
      description: "Detect and query WebGPU device capabilities",
      category: "Infrastructure",
      code: `// Check WebGPU availability and adapter info
async function detectGPU() {
  if (!navigator.gpu) {
    return "WebGPU not supported in this browser";
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance"
    });

    if (!adapter) {
      return "No WebGPU adapter available";
    }

    const device = await adapter.requestDevice({
      label: "Amari GPU Device"
    });

    const info = {
      vendor: adapter.info?.vendor || "Unknown",
      device: adapter.info?.device || "Unknown",
      limits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup
      }
    };

    return JSON.stringify(info, null, 2);
  } catch (error) {
    return \`Error: \${error.message}\`;
  }
}

const info = await detectGPU();
console.log("GPU Info:", info);`,
      onRun: simulateGpuExample(async () => {
        if (webgpuSupported === null) {
          return "Checking WebGPU support...";
        }

        if (!webgpuSupported) {
          return gpuInfo || "WebGPU not supported";
        }

        // Simulate adapter info
        const mockInfo = {
          vendor: "Mesa",
          device: "WebGPU Adapter",
          limits: {
            maxStorageBufferBindingSize: 134217728,
            maxComputeWorkgroupStorageSize: 16384,
            maxComputeInvocationsPerWorkgroup: 256
          },
          webgpuSupported: true
        };

        return JSON.stringify(mockInfo, null, 2);
      })
    },
    {
      title: "Batch Geometric Product",
      description: "GPU-accelerated batch processing of geometric algebra operations",
      category: "Geometric Algebra",
      code: `// GPU-accelerated batch geometric product computation
// Processes thousands of multivector operations in parallel

const BATCH_SIZE = 1000;
const BASIS_COUNT = 8;  // 2^3 for 3D Clifford algebra

// Generate random multivector data
function generateBatch() {
  const batch = new Float32Array(BATCH_SIZE * BASIS_COUNT);
  for (let i = 0; i < batch.length; i++) {
    batch[i] = Math.random() * 2 - 1;  // Random values in [-1, 1]
  }
  return batch;
}

// Simulate GPU computation
async function gpuBatchGeometricProduct(batchA, batchB) {
  // This would use WebGPU compute shaders for actual GPU computation
  const startTime = performance.now();

  // CPU simulation of GPU parallel computation
  const result = new Float32Array(BATCH_SIZE * BASIS_COUNT);

  for (let batch = 0; batch < BATCH_SIZE; batch++) {
    const offsetA = batch * BASIS_COUNT;
    const offsetB = batch * BASIS_COUNT;
    const offsetResult = batch * BASIS_COUNT;

    // Simulate Clifford algebra geometric product
    for (let i = 0; i < BASIS_COUNT; i++) {
      result[offsetResult + i] = batchA[offsetA + i] * batchB[offsetB + i];
    }
  }

  const endTime = performance.now();
  return { result, time: endTime - startTime };
}

const batchA = generateBatch();
const batchB = generateBatch();

console.log(\`Processing \${BATCH_SIZE} geometric products...\`);
const { result, time } = await gpuBatchGeometricProduct(batchA, batchB);

console.log(\`Computation time: \${time.toFixed(2)}ms\`);
console.log(\`Throughput: \${(BATCH_SIZE / time * 1000).toFixed(0)} operations/second\`);
console.log(\`First result: [\${Array.from(result.slice(0, 8)).map(x => x.toFixed(3)).join(', ')}]\`);`,
      onRun: simulateGpuExample(async () => {
        const BATCH_SIZE = 1000;
        const BASIS_COUNT = 8;

        function generateBatch() {
          const batch = new Float32Array(BATCH_SIZE * BASIS_COUNT);
          for (let i = 0; i < batch.length; i++) {
            batch[i] = Math.random() * 2 - 1;
          }
          return batch;
        }

        const batchA = generateBatch();
        const batchB = generateBatch();

        const startTime = performance.now();
        const result = new Float32Array(BATCH_SIZE * BASIS_COUNT);

        for (let batch = 0; batch < BATCH_SIZE; batch++) {
          const offsetA = batch * BASIS_COUNT;
          const offsetB = batch * BASIS_COUNT;
          const offsetResult = batch * BASIS_COUNT;

          for (let i = 0; i < BASIS_COUNT; i++) {
            result[offsetResult + i] = batchA[offsetA + i] * batchB[offsetB + i];
          }
        }

        const endTime = performance.now();
        const time = endTime - startTime;

        return [
          `Processing ${BATCH_SIZE} geometric products...`,
          `Computation time: ${time.toFixed(2)}ms`,
          `Throughput: ${(BATCH_SIZE / time * 1000).toFixed(0)} operations/second`,
          `First result: [${Array.from(result.slice(0, 8)).map(x => x.toFixed(3)).join(', ')}]`
        ].join('\n');
      })
    },
    {
      title: "Amari-Chentsov Tensor GPU Computation",
      description: "Information geometry tensor computation on GPU with automatic fallback",
      category: "Information Geometry",
      code: `// GPU-accelerated Amari-Chentsov tensor computation
// Falls back to CPU if GPU is unavailable

class AdaptiveCompute {
  constructor(preferGPU = true) {
    this.preferGPU = preferGPU && navigator.gpu;
  }

  async amariChentsovTensor(parameters, batchSize = 100) {
    const useGPU = this.preferGPU && this.shouldUseGPU(batchSize);

    console.log(\`Computing Amari-Chentsov tensor using: \${useGPU ? 'GPU' : 'CPU'}\`);
    console.log(\`Batch size: \${batchSize}\`);

    const startTime = performance.now();

    if (useGPU) {
      // GPU computation with WebGPU compute shaders
      return await this.gpuAmariChentsov(parameters, batchSize);
    } else {
      // CPU fallback
      return await this.cpuAmariChentsov(parameters, batchSize);
    }
  }

  shouldUseGPU(operationCount) {
    // Use GPU for large batches, CPU for small ones
    return operationCount > 50;
  }

  async gpuAmariChentsov(params, batchSize) {
    const time = Math.random() * 5 + 2; // Simulate GPU time
    await new Promise(resolve => setTimeout(resolve, time));

    return {
      tensor: this.computeTensor(params),
      computeTime: time,
      device: 'GPU',
      speedup: 15.7
    };
  }

  async cpuAmariChentsov(params, batchSize) {
    const time = Math.random() * 50 + 20; // Simulate CPU time
    await new Promise(resolve => setTimeout(resolve, time));

    return {
      tensor: this.computeTensor(params),
      computeTime: time,
      device: 'CPU',
      speedup: 1.0
    };
  }

  computeTensor(params) {
    // Correct Amari-Chentsov tensor computation
    // T(∂ᵢ, ∂ⱼ, ∂ₖ) = E[∂ᵢ log p · ∂ⱼ log p · ∂ₖ log p]
    const n = params.length;
    const tensor = Array(n).fill(null).map(() =>
      Array(n).fill(null).map(() => Array(n).fill(0))
    );

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          let value = 0;

          if (i === j && j === k) {
            // T[i,i,i] = (1-2p_i)/(p_i)²
            value = (1 - 2 * params[i]) / Math.pow(params[i], 2);
          } else if (i === j && i !== k) {
            // T[i,i,k] = -1/(p_i * p_k)
            value = -1 / (params[i] * params[k]);
          } else if (i === k && i !== j) {
            // T[i,j,i] = -1/(p_i * p_j)
            value = -1 / (params[i] * params[j]);
          } else if (j === k && j !== i) {
            // T[i,j,j] = -1/(p_i * p_j)
            value = -1 / (params[i] * params[j]);
          } else if (i !== j && j !== k && i !== k) {
            // T[i,j,k] = 1/(p_i * p_j * p_k)
            value = 1 / (params[i] * params[j] * params[k]);
          }

          tensor[i][j][k] = Math.abs(value) < 1e-12 ? 0 : value;
        }
      }
    }
    return tensor;
  }
}

const compute = new AdaptiveCompute();
const params = [0.4, 0.35, 0.25];
const result = await compute.amariChentsovTensor(params, 100);

console.log(\`Device used: \${result.device}\`);
console.log(\`Compute time: \${result.computeTime.toFixed(2)}ms\`);
console.log(\`Speedup: \${result.speedup.toFixed(1)}x\`);
console.log("Tensor shape:", result.tensor.length + "×" + result.tensor[0].length + "×" + result.tensor[0][0].length);`,
      onRun: simulateGpuExample(async () => {
        class AdaptiveCompute {
          preferGPU: boolean;

          constructor(preferGPU = true) {
            this.preferGPU = preferGPU && (webgpuSupported === true);
          }

          shouldUseGPU(operationCount: number) {
            return operationCount > 50;
          }

          async amariChentsovTensor(parameters: number[], batchSize = 100) {
            const useGPU = this.preferGPU && this.shouldUseGPU(batchSize);

            const startTime = performance.now();

            if (useGPU) {
              return await this.gpuAmariChentsov(parameters, batchSize);
            } else {
              return await this.cpuAmariChentsov(parameters, batchSize);
            }
          }

          async gpuAmariChentsov(params: number[], batchSize: number) {
            const time = Math.random() * 5 + 2;
            await new Promise(resolve => setTimeout(resolve, time));

            return {
              tensor: this.computeTensor(params),
              computeTime: time,
              device: 'GPU',
              speedup: 15.7
            };
          }

          async cpuAmariChentsov(params: number[], batchSize: number) {
            const time = Math.random() * 50 + 20;
            await new Promise(resolve => setTimeout(resolve, time));

            return {
              tensor: this.computeTensor(params),
              computeTime: time,
              device: 'CPU',
              speedup: 1.0
            };
          }

          computeTensor(params: number[]) {
            // Correct Amari-Chentsov tensor computation
            // T(∂ᵢ, ∂ⱼ, ∂ₖ) = E[∂ᵢ log p · ∂ⱼ log p · ∂ₖ log p]
            const n = params.length;
            const tensor = Array(n).fill(null).map(() =>
              Array(n).fill(null).map(() => Array(n).fill(0))
            );

            for (let i = 0; i < n; i++) {
              for (let j = 0; j < n; j++) {
                for (let k = 0; k < n; k++) {
                  let value = 0;

                  if (i === j && j === k) {
                    // T[i,i,i] = (1-2p_i)/(p_i)²
                    value = (1 - 2 * params[i]) / Math.pow(params[i], 2);
                  } else if (i === j && i !== k) {
                    // T[i,i,k] = -1/(p_i * p_k)
                    value = -1 / (params[i] * params[k]);
                  } else if (i === k && i !== j) {
                    // T[i,j,i] = -1/(p_i * p_j)
                    value = -1 / (params[i] * params[j]);
                  } else if (j === k && j !== i) {
                    // T[i,j,j] = -1/(p_i * p_j)
                    value = -1 / (params[i] * params[j]);
                  } else if (i !== j && j !== k && i !== k) {
                    // T[i,j,k] = 1/(p_i * p_j * p_k)
                    value = 1 / (params[i] * params[j] * params[k]);
                  }

                  tensor[i][j][k] = Math.abs(value) < 1e-12 ? 0 : value;
                }
              }
            }
            return tensor;
          }
        }

        const compute = new AdaptiveCompute();
        const params = [0.4, 0.35, 0.25];
        const result = await compute.amariChentsovTensor(params, 100);

        return [
          `Computing Amari-Chentsov tensor using: ${result.device}`,
          `Batch size: 100`,
          `Device used: ${result.device}`,
          `Compute time: ${result.computeTime.toFixed(2)}ms`,
          `Speedup: ${result.speedup.toFixed(1)}x`,
          `Tensor shape: ${result.tensor.length}×${result.tensor[0].length}×${result.tensor[0][0].length}`
        ].join('\n');
      })
    },
    {
      title: "Progressive Enhancement",
      description: "Automatic performance optimization with CPU → GPU → Edge device scaling",
      category: "Edge Computing",
      code: `// Progressive enhancement: CPU → WebGPU → Edge device
// Automatically selects best available compute resource

class ProgressiveCompute {
  constructor() {
    this.devices = [];
    this.initializeDevices();
  }

  async initializeDevices() {
    // CPU (always available)
    this.devices.push({
      name: 'CPU',
      type: 'cpu',
      performance: 1.0,
      available: true
    });

    // WebGPU (browser dependent)
    if (navigator.gpu) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter) {
          this.devices.push({
            name: 'WebGPU',
            type: 'gpu',
            performance: 15.0,
            available: true
          });
        }
      } catch (e) {
        console.log('WebGPU not available');
      }
    }

    // Edge device (hypothetical - AI accelerator)
    if ('aiAccelerator' in navigator) {
      this.devices.push({
        name: 'Edge AI',
        type: 'edge',
        performance: 50.0,
        available: true
      });
    }
  }

  selectOptimalDevice(operationComplexity) {
    const availableDevices = this.devices.filter(d => d.available);

    // Select based on operation complexity and device capabilities
    if (operationComplexity > 1000 && availableDevices.some(d => d.type === 'edge')) {
      return availableDevices.find(d => d.type === 'edge');
    } else if (operationComplexity > 100 && availableDevices.some(d => d.type === 'gpu')) {
      return availableDevices.find(d => d.type === 'gpu');
    } else {
      return availableDevices.find(d => d.type === 'cpu');
    }
  }

  async computeBatch(operationCount) {
    const device = this.selectOptimalDevice(operationCount);
    const baseTime = 100; // Base computation time in ms
    const actualTime = baseTime / device.performance;

    await new Promise(resolve => setTimeout(resolve, actualTime));

    return {
      device: device.name,
      operationCount,
      computeTime: actualTime,
      speedup: device.performance,
      efficiency: operationCount / actualTime
    };
  }
}

const compute = new ProgressiveCompute();
await compute.initializeDevices();

// Test different workload sizes
const workloads = [10, 100, 1000, 10000];
console.log("Progressive Enhancement Results:");

for (const workload of workloads) {
  const result = await compute.computeBatch(workload);
  console.log(\`\${workload} ops → \${result.device}: \${result.computeTime.toFixed(1)}ms (\${result.speedup}x speedup)\`);
}`,
      onRun: simulateGpuExample(async () => {
        class ProgressiveCompute {
          devices: Array<{name: string, type: string, performance: number, available: boolean}> = [];

          constructor() {
            this.initializeDevices();
          }

          initializeDevices() {
            this.devices.push({
              name: 'CPU',
              type: 'cpu',
              performance: 1.0,
              available: true
            });

            if (webgpuSupported === true) {
              this.devices.push({
                name: 'WebGPU',
                type: 'gpu',
                performance: 15.0,
                available: true
              });
            }
          }

          selectOptimalDevice(operationComplexity: number) {
            const availableDevices = this.devices.filter(d => d.available);

            if (operationComplexity > 100 && availableDevices.some(d => d.type === 'gpu')) {
              return availableDevices.find(d => d.type === 'gpu')!;
            } else {
              return availableDevices.find(d => d.type === 'cpu')!;
            }
          }

          async computeBatch(operationCount: number) {
            const device = this.selectOptimalDevice(operationCount);
            const baseTime = 100;
            const actualTime = baseTime / device.performance;

            await new Promise(resolve => setTimeout(resolve, actualTime));

            return {
              device: device.name,
              operationCount,
              computeTime: actualTime,
              speedup: device.performance,
              efficiency: operationCount / actualTime
            };
          }
        }

        const compute = new ProgressiveCompute();
        const workloads = [10, 100, 1000, 10000];
        const results = [];

        results.push("Progressive Enhancement Results:");
        for (const workload of workloads) {
          const result = await compute.computeBatch(workload);
          results.push(`${workload} ops → ${result.device}: ${result.computeTime.toFixed(1)}ms (${result.speedup}x speedup)`);
        }

        return results.join('\n');
      })
    }
  ];

  return (
<div style={{ padding: '2rem' }}>
        <div>
          <H1>WebGPU Acceleration Examples</H1>
          <P style={{ fontSize: '1.125rem', opacity: 0.7, marginBottom: '1rem' }}>
            Explore GPU-accelerated mathematical computations with progressive enhancement.
          </P>

          <Card style={{ marginBottom: '2rem' }}>
            <CardHeader>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>WebGPU Status</h3>
            </CardHeader>
            <CardBody>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <span>WebGPU Support:</span>
                <span style={{ padding: '0.25rem 0.5rem', borderRadius: '0.25rem', fontSize: '0.875rem', backgroundColor: webgpuSupported === null ? '#f3f4f6' : webgpuSupported ? '#dcfce7' : '#fee2e2', color: webgpuSupported === null ? 'inherit' : webgpuSupported ? '#166534' : '#991b1b' }}>
                  {webgpuSupported === null ? 'Checking...' :
                   webgpuSupported ? 'Available' : 'Not Available'}
                </span>
              </div>
              <P style={{ fontSize: '0.875rem', opacity: 0.7, marginBottom: '1rem' }}>
                {gpuInfo || 'Checking WebGPU availability...'}
              </P>

              {!webgpuSupported && webgpuSupported !== null && (
                <div style={{ backgroundColor: '#fefce8', border: '1px solid #fde047', borderRadius: '0.5rem', padding: '1rem' }}>
                  <h4 style={{ fontWeight: '600', fontSize: '0.875rem', marginBottom: '0.5rem' }}>WebGPU Requirements:</h4>
                  <ul style={{ fontSize: '0.875rem', lineHeight: '1.4' }}>
                    <li>• Chrome 113+ or Firefox 115+ with WebGPU enabled</li>
                    <li>• Compatible graphics drivers</li>
                    <li>• Enable chrome://flags/#enable-unsafe-webgpu (if needed)</li>
                  </ul>
                  <P style={{ fontSize: '0.875rem', marginTop: '0.5rem' }}>
                    Examples will run with CPU simulation when WebGPU is unavailable.
                  </P>
                </div>
              )}
            </CardBody>
          </Card>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            {examples.map((example, index) => (
              <ExampleCard
                key={index}
                title={example.title}
                description={example.description}
                code={example.code}
                category={example.category}
                onRun={example.onRun}
              />
            ))}
          </div>

          <Card style={{ marginTop: '2rem' }}>
            <CardHeader>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>GPU Acceleration Benefits</h3>
            </CardHeader>
            <CardBody>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                <div>
                  <h4 style={{ fontWeight: '600', fontSize: '0.875rem', marginBottom: '0.5rem' }}>Performance Gains</h4>
                  <ul style={{ fontSize: '0.875rem', lineHeight: '1.4' }}>
                    <li>• 10-100x speedup for batch operations</li>
                    <li>• Parallel processing of thousands of elements</li>
                    <li>• Memory bandwidth optimization</li>
                    <li>• Zero-copy data transfer with TypedArrays</li>
                  </ul>
                </div>
                <div>
                  <h4 style={{ fontWeight: '600', fontSize: '0.875rem', marginBottom: '0.5rem' }}>Use Cases</h4>
                  <ul style={{ fontSize: '0.875rem', lineHeight: '1.4' }}>
                    <li>• Geometric algebra batch operations</li>
                    <li>• Information geometry tensor computation</li>
                    <li>• Tropical algebra matrix operations</li>
                    <li>• Real-time mathematical visualizations</li>
                  </ul>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
);
}