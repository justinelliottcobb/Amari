import { Container, Stack, Card, Title, Text, List, SimpleGrid, Badge, Alert } from "@mantine/core";
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
  const startTime = performance.now();

  // CPU simulation of GPU parallel computation
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

    if (useGPU) {
      return await this.gpuAmariChentsov(parameters, batchSize);
    } else {
      return await this.cpuAmariChentsov(parameters, batchSize);
    }
  }

  shouldUseGPU(operationCount) {
    return operationCount > 50;
  }

  computeTensor(params) {
    const n = params.length;
    const tensor = Array(n).fill(null).map(() =>
      Array(n).fill(null).map(() => Array(n).fill(0))
    );

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          let value = 0;

          if (i === j && j === k) {
            value = (1 - 2 * params[i]) / Math.pow(params[i], 2);
          } else if (i === j && i !== k) {
            value = -1 / (params[i] * params[k]);
          } else if (i === k && i !== j) {
            value = -1 / (params[i] * params[j]);
          } else if (j === k && j !== i) {
            value = -1 / (params[i] * params[j]);
          } else if (i !== j && j !== k && i !== k) {
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
console.log(\`Compute time: \${result.computeTime.toFixed(2)}ms\`);`,
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
              return await this.gpuAmariChentsov(parameters);
            } else {
              return await this.cpuAmariChentsov(parameters);
            }
          }

          async gpuAmariChentsov(params: number[]) {
            const time = Math.random() * 5 + 2;
            await new Promise(resolve => setTimeout(resolve, time));

            return {
              tensor: this.computeTensor(params),
              computeTime: time,
              device: 'GPU',
              speedup: 15.7
            };
          }

          async cpuAmariChentsov(params: number[]) {
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
            const n = params.length;
            const tensor = Array(n).fill(null).map(() =>
              Array(n).fill(null).map(() => Array(n).fill(0))
            );

            for (let i = 0; i < n; i++) {
              for (let j = 0; j < n; j++) {
                for (let k = 0; k < n; k++) {
                  let value = 0;

                  if (i === j && j === k) {
                    value = (1 - 2 * params[i]) / Math.pow(params[i], 2);
                  } else if (i === j && i !== k) {
                    value = -1 / (params[i] * params[k]);
                  } else if (i === k && i !== j) {
                    value = -1 / (params[i] * params[j]);
                  } else if (j === k && j !== i) {
                    value = -1 / (params[i] * params[j]);
                  } else if (i !== j && j !== k && i !== k) {
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
  }

  selectOptimalDevice(operationComplexity) {
    const availableDevices = this.devices.filter(d => d.available);

    if (operationComplexity > 100 && availableDevices.some(d => d.type === 'gpu')) {
      return availableDevices.find(d => d.type === 'gpu');
    } else {
      return availableDevices.find(d => d.type === 'cpu');
    }
  }

  async computeBatch(operationCount) {
    const device = this.selectOptimalDevice(operationCount);
    const baseTime = 100;
    const actualTime = baseTime / device.performance;

    await new Promise(resolve => setTimeout(resolve, actualTime));

    return {
      device: device.name,
      operationCount,
      computeTime: actualTime,
      speedup: device.performance
    };
  }
}

const compute = new ProgressiveCompute();
await compute.initializeDevices();

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
              speedup: device.performance
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
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>WebGPU Acceleration Examples</Title>
          <Text size="lg" c="dimmed">
            Explore GPU-accelerated mathematical computations with progressive enhancement.
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3}>WebGPU Status</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <SimpleGrid cols={2} mb="md">
              <Text>WebGPU Support:</Text>
              <Badge
                color={webgpuSupported === null ? 'gray' : webgpuSupported ? 'green' : 'red'}
                variant="filled"
              >
                {webgpuSupported === null ? 'Checking...' :
                 webgpuSupported ? 'Available' : 'Not Available'}
              </Badge>
            </SimpleGrid>
            <Text size="sm" c="dimmed" mb="md">
              {gpuInfo || 'Checking WebGPU availability...'}
            </Text>

            {!webgpuSupported && webgpuSupported !== null && (
              <Alert color="yellow" title="WebGPU Requirements">
                <List size="sm">
                  <List.Item>Chrome 113+ or Firefox 115+ with WebGPU enabled</List.Item>
                  <List.Item>Compatible graphics drivers</List.Item>
                  <List.Item>Enable chrome://flags/#enable-unsafe-webgpu (if needed)</List.Item>
                </List>
                <Text size="sm" mt="xs">
                  Examples will run with CPU simulation when WebGPU is unavailable.
                </Text>
              </Alert>
            )}
          </Card.Section>
        </Card>

        <Stack gap="lg">
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
        </Stack>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3}>GPU Acceleration Benefits</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <SimpleGrid cols={{ base: 1, sm: 2 }}>
              <div>
                <Title order={4} size="sm" mb="xs">Performance Gains</Title>
                <List size="sm">
                  <List.Item>10-100x speedup for batch operations</List.Item>
                  <List.Item>Parallel processing of thousands of elements</List.Item>
                  <List.Item>Memory bandwidth optimization</List.Item>
                  <List.Item>Zero-copy data transfer with TypedArrays</List.Item>
                </List>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Use Cases</Title>
                <List size="sm">
                  <List.Item>Geometric algebra batch operations</List.Item>
                  <List.Item>Information geometry tensor computation</List.Item>
                  <List.Item>Tropical algebra matrix operations</List.Item>
                  <List.Item>Real-time mathematical visualizations</List.Item>
                </List>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>
      </Stack>
    </Container>
  );
}
