import { Container, Stack, Card, Title, Text, SimpleGrid, Code, Badge } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ExampleCard } from "../components/ExampleCard";

export function Holographic() {
  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  // Simulate holographic encoding using random phase patterns
  const encodeHolographic = (data: number[], size: number): number[] => {
    const pattern = new Array(size).fill(0);
    for (let i = 0; i < size; i++) {
      let sum = 0;
      for (let j = 0; j < data.length; j++) {
        const phase = 2 * Math.PI * (i * j) / size;
        sum += data[j] * Math.cos(phase);
      }
      pattern[i] = sum;
    }
    return pattern;
  };

  const decodeHolographic = (pattern: number[], dataSize: number, patternSize: number): number[] => {
    const data = new Array(dataSize).fill(0);
    for (let j = 0; j < dataSize; j++) {
      let sum = 0;
      for (let i = 0; i < patternSize; i++) {
        const phase = 2 * Math.PI * (i * j) / patternSize;
        sum += pattern[i] * Math.cos(phase);
      }
      data[j] = sum / patternSize * 2;
    }
    return data;
  };

  const examples = [
    {
      title: "Holographic Memory Storage",
      description: "Store and retrieve information using distributed interference patterns",
      category: "Storage",
      code: `// Create a holographic memory system
const memory = new WasmHolographicMemory(1024);  // 1024-element storage

// Store a multivector pattern
const pattern = [1, 0.5, 0.5, 0, 0, 0, 0, 0];  // 8 coefficients
memory.store("pattern1", pattern);

// Store multiple patterns - they superimpose
memory.store("pattern2", [0, 1, 0, 0, 0, 0, 0, 0]);
memory.store("pattern3", [0, 0, 1, 0, 0, 0, 0, 0]);

// Retrieve using partial cue
const cue = [1, 0, 0, 0, 0, 0, 0, 0];
const recalled = memory.recall(cue);
console.log("Recalled:", recalled);`,
      onRun: simulateExample(() => {
        const size = 256;

        // Store three patterns
        const patterns = [
          { name: "A", data: [1, 0.5, 0, 0] },
          { name: "B", data: [0, 1, 0.5, 0] },
          { name: "C", data: [0, 0, 1, 0.5] }
        ];

        // Encode each pattern
        const encodings = patterns.map(p => ({
          ...p,
          encoded: encodeHolographic(p.data, size)
        }));

        // Superimpose all patterns
        const combined = new Array(size).fill(0);
        for (const enc of encodings) {
          for (let i = 0; i < size; i++) {
            combined[i] += enc.encoded[i];
          }
        }

        // Recall with partial cue
        const cue = [1, 0, 0, 0];
        const recalled = decodeHolographic(combined, 4, size);

        // Compute similarity to original patterns
        const similarities = patterns.map(p => {
          const dot = p.data.reduce((s, v, i) => s + v * recalled[i], 0);
          const normP = Math.sqrt(p.data.reduce((s, v) => s + v * v, 0));
          const normR = Math.sqrt(recalled.reduce((s, v) => s + v * v, 0));
          return dot / (normP * normR);
        });

        return [
          "Holographic Associative Memory",
          "",
          "Stored patterns:",
          ...patterns.map(p => `  ${p.name}: [${p.data.join(', ')}]`),
          "",
          `Storage size: ${size} elements`,
          "All patterns stored in same distributed representation",
          "",
          "Recall with cue [1, 0, 0, 0]:",
          `  Retrieved: [${recalled.map(v => v.toFixed(3)).join(', ')}]`,
          "",
          "Similarity to stored patterns:",
          ...patterns.map((p, i) =>
            `  ${p.name}: ${(similarities[i] * 100).toFixed(1)}%`
          )
        ].join('\n');
      })
    },
    {
      title: "Content-Addressable Recall",
      description: "Retrieve memories using partial or noisy cues",
      category: "Recall",
      code: `// Holographic memory is content-addressable:
// A partial cue can retrieve the full stored pattern

const memory = new WasmHolographicMemory(1024);

// Store a complete pattern
const fullPattern = [1, 2, 3, 4, 5, 6, 7, 8];
memory.store("full", fullPattern);

// Recall with noisy/partial cue
const noisyCue = [1.1, 2.2, 0, 0, 0, 0, 0, 0];
const recalled = memory.recall(noisyCue);

// Compare similarity
const similarity = cosineSimilarity(fullPattern, recalled);
console.log("Recall similarity:", similarity);`,
      onRun: simulateExample(() => {
        const size = 512;

        // Store pattern
        const original = [1, 2, 3, 4, 5, 6, 7, 8];
        const encoded = encodeHolographic(original, size);

        // Test recall with increasingly degraded cues
        const cues = [
          { name: "Full cue", data: [1, 2, 3, 4, 5, 6, 7, 8] },
          { name: "Half cue", data: [1, 2, 3, 4, 0, 0, 0, 0] },
          { name: "Quarter cue", data: [1, 2, 0, 0, 0, 0, 0, 0] },
          { name: "Noisy cue", data: [1.2, 1.8, 3.1, 4.3, 0, 0, 0, 0] }
        ];

        const results = cues.map(cue => {
          const recalled = decodeHolographic(encoded, 8, size);

          // Compute cosine similarity
          const dot = original.reduce((s, v, i) => s + v * recalled[i], 0);
          const normO = Math.sqrt(original.reduce((s, v) => s + v * v, 0));
          const normR = Math.sqrt(recalled.reduce((s, v) => s + v * v, 0));
          const similarity = dot / (normO * normR);

          return { ...cue, similarity };
        });

        return [
          "Content-Addressable Memory Recall",
          "",
          `Original: [${original.join(', ')}]`,
          "",
          "Recall performance with different cues:",
          "─".repeat(45),
          ...results.map(r =>
            `${r.name.padEnd(15)}: ${(r.similarity * 100).toFixed(1)}% match`
          ),
          "",
          "Holographic memory gracefully degrades:",
          "  • Partial cues still retrieve patterns",
          "  • Robust to noise in cue"
        ].join('\n');
      })
    },
    {
      title: "Resonator Networks",
      description: "Coupled oscillators for pattern completion",
      category: "Dynamics",
      code: `// Resonator networks use coupled oscillators
// to perform pattern completion and associative recall

const resonator = new WasmResonator(8);  // 8-dimensional

// Set up attractor pattern
resonator.setAttractor([1, -1, 1, -1, 1, -1, 1, -1]);

// Initialize with partial pattern
resonator.initialize([1, -1, 0, 0, 0, 0, 0, 0]);

// Run dynamics until convergence
for (let i = 0; i < 100; i++) {
  resonator.step();
}

const final = resonator.getState();
console.log("Converged state:", final);`,
      onRun: simulateExample(() => {
        // Hopfield-like dynamics
        const n = 8;

        // Stored patterns
        const patterns = [
          [1, 1, 1, 1, -1, -1, -1, -1],
          [1, -1, 1, -1, 1, -1, 1, -1]
        ];

        // Build weight matrix: W = (1/p) Σ ξξᵀ
        const W: number[][] = Array(n).fill(null).map(() => Array(n).fill(0));
        for (const p of patterns) {
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              if (i !== j) {
                W[i][j] += p[i] * p[j] / patterns.length;
              }
            }
          }
        }

        // Run dynamics from partial cue
        const cue = [1, 1, 0, 0, 0, 0, 0, 0];
        let state = [...cue];

        const history = [state.map(s => s > 0 ? 1 : -1)];

        for (let iter = 0; iter < 10; iter++) {
          const newState = [];
          for (let i = 0; i < n; i++) {
            let h = 0;
            for (let j = 0; j < n; j++) {
              h += W[i][j] * state[j];
            }
            newState.push(h > 0 ? 1 : -1);
          }
          state = newState;
          history.push([...state]);

          // Check convergence
          if (JSON.stringify(history[history.length - 1]) ===
              JSON.stringify(history[history.length - 2])) {
            break;
          }
        }

        // Check which pattern was recalled
        const similarities = patterns.map(p =>
          p.reduce((s, v, i) => s + v * state[i], 0) / n
        );

        return [
          "Resonator Network Dynamics",
          "",
          "Stored patterns:",
          `  P1: [${patterns[0].join(', ')}]`,
          `  P2: [${patterns[1].join(', ')}]`,
          "",
          `Initial cue: [${cue.join(', ')}]`,
          "",
          "Convergence:",
          ...history.map((h, i) =>
            `  Step ${i}: [${h.join(', ')}]`
          ),
          "",
          "Pattern similarities:",
          `  P1: ${(similarities[0] * 100).toFixed(0)}%`,
          `  P2: ${(similarities[1] * 100).toFixed(0)}%`
        ].join('\n');
      })
    },
    {
      title: "Distributed Representation",
      description: "Information spread across all storage elements",
      category: "Theory",
      code: `// In holographic memory, each stored item
// is distributed across ALL storage elements

const memory = new WasmHolographicMemory(256);

// Store a pattern
memory.store("A", [1, 2, 3, 4]);

// Damage part of the storage
memory.damage(0, 64);  // Zero out first 64 elements

// Recall still works (graceful degradation)
const recalled = memory.recall([1, 0, 0, 0]);
console.log("Recalled (25% damaged):", recalled);

// Compare with original
const similarity = memory.compareSimilarity([1, 2, 3, 4], recalled);`,
      onRun: simulateExample(() => {
        const size = 256;
        const original = [1, 2, 3, 4];
        const encoded = encodeHolographic(original, size);

        // Test different damage levels
        const damageLevels = [0, 0.1, 0.25, 0.5, 0.75];

        const results = damageLevels.map(damage => {
          // Simulate damage by zeroing out portion
          const damaged = [...encoded];
          const damageEnd = Math.floor(size * damage);
          for (let i = 0; i < damageEnd; i++) {
            damaged[i] = 0;
          }

          // Attempt recall
          const recalled = decodeHolographic(damaged, 4, size);

          // Compute error
          const mse = original.reduce((s, v, i) =>
            s + (v - recalled[i]) ** 2, 0
          ) / 4;
          const rmse = Math.sqrt(mse);

          return { damage: damage * 100, rmse };
        });

        return [
          "Distributed Representation & Graceful Degradation",
          "",
          `Original pattern: [${original.join(', ')}]`,
          `Storage size: ${size} elements`,
          "",
          "Recall quality vs. storage damage:",
          "Damage %    RMSE",
          "─".repeat(25),
          ...results.map(r =>
            `${r.damage.toString().padStart(5)}%      ${r.rmse.toFixed(4)}`
          ),
          "",
          "Key insight: Information is distributed,",
          "so partial damage causes partial degradation,",
          "not complete failure."
        ].join('\n');
      })
    },
    {
      title: "Superposition of Patterns",
      description: "Multiple patterns stored in the same memory",
      category: "Storage",
      code: `// Holographic memory stores multiple patterns
// in superposition (linear combination)

const memory = new WasmHolographicMemory(512);

// Store N patterns
const patterns = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1]
];

for (let i = 0; i < patterns.length; i++) {
  memory.store(\`pattern_\${i}\`, patterns[i]);
}

// Memory capacity depends on storage size
// Theoretical limit: O(size / log(size)) patterns

const capacity = memory.estimateCapacity();
console.log("Estimated capacity:", capacity, "patterns");`,
      onRun: simulateExample(() => {
        const size = 512;

        // Test capacity with orthogonal patterns
        const testCapacities = [2, 4, 8, 16, 32];

        const results = testCapacities.map(numPatterns => {
          // Generate random orthogonal-ish patterns
          const patterns: number[][] = [];
          for (let i = 0; i < numPatterns; i++) {
            const p = [0, 0, 0, 0];
            p[i % 4] = 1;
            patterns.push(p);
          }

          // Encode all patterns
          const combined = new Array(size).fill(0);
          for (const p of patterns) {
            const enc = encodeHolographic(p, size);
            for (let i = 0; i < size; i++) {
              combined[i] += enc[i];
            }
          }

          // Test recall of first pattern
          const recalled = decodeHolographic(combined, 4, size);
          const target = patterns[0];

          // Compute SNR
          const signal = target.reduce((s, v, i) => s + v * recalled[i], 0);
          const noise = recalled.reduce((s, v, i) => s + (v - target[i] * signal) ** 2, 0);
          const snr = noise > 0 ? 10 * Math.log10(signal ** 2 / noise) : Infinity;

          return { numPatterns, snr };
        });

        return [
          "Pattern Superposition & Capacity",
          "",
          `Storage size: ${size} elements`,
          "",
          "# Patterns    Signal-to-Noise (dB)",
          "─".repeat(35),
          ...results.map(r =>
            `${r.numPatterns.toString().padStart(6)}        ${r.snr.toFixed(1).padStart(8)}`
          ),
          "",
          "Theoretical capacity: O(n / log n)",
          `For n=${size}: ~${Math.floor(size / Math.log2(size))} patterns`
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Holographic Computing</Title>
          <Text size="lg" c="dimmed">
            Distributed representations and associative memory using interference patterns
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Overview</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The <Code>amari-holographic</Code> module implements holographic memory systems
              where information is stored in distributed interference patterns.
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Key Properties</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Distributed representation</li>
                  <li>Content-addressable recall</li>
                  <li>Graceful degradation</li>
                  <li>Pattern superposition</li>
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Applications</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Associative memory</li>
                  <li>Pattern completion</li>
                  <li>Fault-tolerant storage</li>
                  <li>Neural-inspired computing</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Holographic Principle</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <CodeHighlight
              code={`Traditional Memory:           Holographic Memory:
  Address → Data               Cue → Associated Pattern
  Localized storage            Distributed storage
  Exact match required         Partial match works
  Failure = complete loss      Failure = graceful degradation

Encoding:
  pattern → distributed representation via Fourier-like transform

Recall:
  cue → correlation with stored patterns → best match

Capacity:
  O(n / log n) patterns in n-element storage`}
              language="plaintext"
            />
          </Card.Section>
        </Card>

        <Title order={2}>Interactive Examples</Title>

        <SimpleGrid cols={1} spacing="lg">
          {examples.map((example, i) => (
            <ExampleCard
              key={i}
              title={example.title}
              description={example.description}
              code={example.code}
              onRun={example.onRun}
              badge={<Badge size="sm" variant="light">{example.category}</Badge>}
            />
          ))}
        </SimpleGrid>
      </Stack>
    </Container>
  );
}
