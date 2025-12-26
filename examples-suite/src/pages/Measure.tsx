import { Container, Stack, Card, Title, Text, SimpleGrid, Code, Badge } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ExampleCard } from "../components/ExampleCard";

export function Measure() {
  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  // Helper functions
  const entropy = (probs: number[]): number => {
    return -probs.reduce((sum, p) => {
      if (p > 0) return sum + p * Math.log(p);
      return sum;
    }, 0);
  };

  const klDivergence = (p: number[], q: number[]): number => {
    return p.reduce((sum, pi, i) => {
      if (pi > 0 && q[i] > 0) return sum + pi * Math.log(pi / q[i]);
      return sum;
    }, 0);
  };

  const examples = [
    {
      title: "Lebesgue Measure",
      description: "The standard measure for integration on ℝⁿ",
      category: "Measures",
      code: `// Lebesgue measure assigns "length/area/volume" to sets
const lebesgue = new WasmLebesgueMeasure(3);  // 3 dimensions

// Measure of a box [0,1] × [0,2] × [0,3]
const lower = [0, 0, 0];
const upper = [1, 2, 3];
const volume = lebesgue.measureBox(lower, upper);
console.log("Volume:", volume);  // 1 × 2 × 3 = 6

// Integration with respect to Lebesgue measure
// ∫f dμ = usual Riemann integral
const integral = lebesgue.integrate(
  (x, y, z) => x * y * z,
  lower, upper
);`,
      onRun: simulateExample(() => {
        const dims = [1, 2, 3];
        const results = dims.map(d => {
          // Volume of unit hypercube
          const unitVol = 1;
          // Volume of [0,1]×[0,2]×...×[0,d]
          let vol = 1;
          for (let i = 1; i <= d; i++) vol *= i;
          return { dim: d, unitCubeVol: unitVol, boxVol: vol };
        });

        // Example integration: ∫∫ xy dA over [0,1]²
        const integral2D = 0.25; // (1/2)(1/2) = 1/4

        return [
          "Lebesgue Measure in ℝⁿ",
          "",
          "Dimension  Unit Cube   [0,1]×[0,2]×...",
          "─".repeat(40),
          ...results.map(r =>
            `    ${r.dim}         ${r.unitCubeVol}           ${r.boxVol}`
          ),
          "",
          "Integration example:",
          `  ∫∫[0,1]² xy dA = ${integral2D}`,
          "",
          "Lebesgue measure is translation-invariant:",
          "  μ(A + x) = μ(A) for all x ∈ ℝⁿ"
        ].join('\n');
      })
    },
    {
      title: "Probability Measure",
      description: "Measures that integrate to 1 (probability distributions)",
      category: "Measures",
      code: `// A probability measure satisfies:
// 1. μ(Ω) = 1 (total probability is 1)
// 2. μ(A) ≥ 0 for all sets A
// 3. Countable additivity

// Create a probability measure from a density
const density = new WasmParametricDensity("gaussian", [0, 1]);
const probMeasure = new WasmProbabilityMeasure(density);

// Compute probability of an interval
const prob = probMeasure.probability(-1, 1);
console.log("P(-1 ≤ X ≤ 1):", prob);  // ≈ 0.6827

// Expectation
const mean = probMeasure.expectation(x => x);
const variance = probMeasure.expectation(x => x * x);`,
      onRun: simulateExample(() => {
        // Standard normal distribution
        const normalPDF = (x: number) => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);

        // Numerical integration for probabilities
        const integrate = (f: (x: number) => number, a: number, b: number, n = 1000): number => {
          const h = (b - a) / n;
          let sum = 0;
          for (let i = 0; i < n; i++) {
            const x = a + (i + 0.5) * h;
            sum += f(x) * h;
          }
          return sum;
        };

        const intervals = [
          { a: -1, b: 1, desc: "±1σ" },
          { a: -2, b: 2, desc: "±2σ" },
          { a: -3, b: 3, desc: "±3σ" }
        ];

        const probs = intervals.map(int => ({
          ...int,
          prob: integrate(normalPDF, int.a, int.b)
        }));

        const totalProb = integrate(normalPDF, -10, 10);

        return [
          "Probability Measure (Standard Normal)",
          "",
          "Density: f(x) = exp(-x²/2) / √(2π)",
          "",
          "Interval Probabilities:",
          ...probs.map(p =>
            `  P(${p.a} ≤ X ≤ ${p.b}) = ${(p.prob * 100).toFixed(2)}%  (${p.desc})`
          ),
          "",
          `Total probability: ${totalProb.toFixed(6)} ≈ 1`,
          "",
          "68-95-99.7 rule confirmed!"
        ].join('\n');
      })
    },
    {
      title: "Counting Measure",
      description: "Discrete measure that counts elements",
      category: "Measures",
      code: `// Counting measure: μ(A) = |A| (cardinality)
const counting = new WasmCountingMeasure();

// Measure of a finite set
const set = [1, 2, 3, 4, 5];
const measure = counting.measure(set);
console.log("Count:", measure);  // 5

// Integration = summation
// ∫f dμ = Σ f(x)
const sum = counting.integrate(x => x * x, set);
console.log("Sum of squares:", sum);  // 1 + 4 + 9 + 16 + 25 = 55`,
      onRun: simulateExample(() => {
        const sets = [
          { name: "A", elements: [1, 2, 3] },
          { name: "B", elements: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] },
          { name: "C", elements: [2, 4, 6, 8] }
        ];

        const results = sets.map(s => ({
          ...s,
          count: s.elements.length,
          sum: s.elements.reduce((a, b) => a + b, 0),
          sumSquares: s.elements.reduce((a, b) => a + b * b, 0)
        }));

        return [
          "Counting Measure",
          "",
          "For discrete sets, μ(A) = |A|",
          "",
          "Set       Elements              |A|   Σx    Σx²",
          "─".repeat(55),
          ...results.map(r =>
            `${r.name.padEnd(8)}  [${r.elements.join(',')}]`.padEnd(30) +
            `${r.count.toString().padStart(3)}  ${r.sum.toString().padStart(4)}  ${r.sumSquares.toString().padStart(5)}`
          ),
          "",
          "Integration with counting measure:",
          "  ∫f dμ = Σᵢ f(xᵢ)"
        ].join('\n');
      })
    },
    {
      title: "KL Divergence",
      description: "Measure the difference between probability distributions",
      category: "Information Theory",
      code: `// KL divergence: D_KL(P||Q) = Σ p(x) log(p(x)/q(x))
// Measures how much P differs from Q

const p = new WasmParametricDensity("gaussian", [0, 1]);
const q = new WasmParametricDensity("gaussian", [0.5, 1]);

const kl = klDivergence(p, q, pParams, qParams, samplePoints);
console.log("D_KL(P||Q):", kl);

// Properties:
// - D_KL(P||Q) ≥ 0 (Gibbs' inequality)
// - D_KL(P||Q) = 0 iff P = Q
// - NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)`,
      onRun: simulateExample(() => {
        // Discrete distributions
        const P = [0.25, 0.25, 0.25, 0.25];  // Uniform
        const Q1 = [0.1, 0.2, 0.3, 0.4];     // Skewed
        const Q2 = [0.25, 0.25, 0.25, 0.25]; // Same as P
        const Q3 = [0.4, 0.3, 0.2, 0.1];     // Opposite skew

        const results = [
          { name: "Uniform vs Skewed", p: P, q: Q1 },
          { name: "Uniform vs Uniform", p: P, q: Q2 },
          { name: "Uniform vs Opposite", p: P, q: Q3 }
        ].map(r => ({
          ...r,
          klPQ: klDivergence(r.p, r.q),
          klQP: klDivergence(r.q, r.p)
        }));

        return [
          "KL Divergence: D_KL(P||Q) = Σ p(x) log(p(x)/q(x))",
          "",
          "Distributions:",
          `  P (uniform): [${P.join(', ')}]`,
          `  Q1 (skewed): [${Q1.join(', ')}]`,
          `  Q3 (opposite): [${Q3.join(', ')}]`,
          "",
          "Results:",
          ...results.map(r =>
            `  ${r.name.padEnd(22)}: D_KL(P||Q) = ${r.klPQ.toFixed(4)}, D_KL(Q||P) = ${r.klQP.toFixed(4)}`
          ),
          "",
          "Note: KL divergence is NOT symmetric!"
        ].join('\n');
      })
    },
    {
      title: "Entropy and Information",
      description: "Measure uncertainty in probability distributions",
      category: "Information Theory",
      code: `// Shannon entropy: H(P) = -Σ p(x) log p(x)
// Measures uncertainty/information content

const uniform = [0.25, 0.25, 0.25, 0.25];
const peaked = [0.97, 0.01, 0.01, 0.01];

const H_uniform = entropy(uniform);
const H_peaked = entropy(peaked);

console.log("Entropy (uniform):", H_uniform);  // log(4) ≈ 1.39
console.log("Entropy (peaked):", H_peaked);    // ≈ 0.24

// Maximum entropy = log(n) for n outcomes
// Minimum entropy = 0 for deterministic distribution`,
      onRun: simulateExample(() => {
        const distributions = [
          { name: "Uniform (4)", probs: [0.25, 0.25, 0.25, 0.25] },
          { name: "Peaked", probs: [0.97, 0.01, 0.01, 0.01] },
          { name: "Binary", probs: [0.5, 0.5] },
          { name: "Deterministic", probs: [1.0, 0.0, 0.0, 0.0] },
          { name: "Bimodal", probs: [0.4, 0.1, 0.1, 0.4] }
        ];

        const results = distributions.map(d => ({
          ...d,
          entropy: entropy(d.probs),
          maxEntropy: Math.log(d.probs.length)
        }));

        return [
          "Shannon Entropy: H(P) = -Σ p(x) log p(x)",
          "",
          "Distribution        Entropy    Max Entropy    Ratio",
          "─".repeat(55),
          ...results.map(r =>
            `${r.name.padEnd(18)}  ${r.entropy.toFixed(4).padStart(8)}    ${r.maxEntropy.toFixed(4).padStart(8)}    ${(r.entropy / r.maxEntropy * 100).toFixed(1).padStart(5)}%`
          ),
          "",
          "Interpretation:",
          "  • High entropy = high uncertainty",
          "  • Maximum when uniform (all outcomes equally likely)",
          "  • Zero when deterministic (one outcome certain)"
        ].join('\n');
      })
    },
    {
      title: "Fisher Measure",
      description: "Information geometry metric on parameter space",
      category: "Information Geometry",
      code: `// Fisher information matrix defines a Riemannian metric
// on the space of probability distributions

const fisherMeasure = new WasmFisherMeasure("gaussian");

// Get Fisher information for Gaussian at θ = (μ, σ)
const params = [0, 1];  // μ = 0, σ = 1
const fisher = fisherMeasure.getMatrix(params);

console.log("Fisher information matrix:");
console.log(fisher);
// For Gaussian: I = [[1/σ², 0], [0, 2/σ²]]

// Natural gradient uses Fisher as metric
// Δθ = I⁻¹ ∇L`,
      onRun: simulateExample(() => {
        // Fisher information for Gaussian N(μ, σ²)
        // I_μμ = 1/σ², I_σσ = 2/σ², I_μσ = 0

        const sigmas = [0.5, 1.0, 2.0];
        const results = sigmas.map(sigma => {
          const I_mu = 1 / (sigma * sigma);
          const I_sigma = 2 / (sigma * sigma);
          return { sigma, I_mu, I_sigma };
        });

        // Jeffreys prior √det(I) ∝ 1/σ²
        const jeffreys = sigmas.map(sigma => 1 / (sigma * sigma));

        return [
          "Fisher Information Matrix for Gaussian",
          "",
          "For N(μ, σ²):",
          "  I = | 1/σ²    0   |",
          "      |  0    2/σ²  |",
          "",
          "σ      I_μμ      I_σσ      √det(I)",
          "─".repeat(40),
          ...results.map((r, i) =>
            `${r.sigma.toFixed(1)}    ${r.I_mu.toFixed(3).padStart(6)}    ${r.I_sigma.toFixed(3).padStart(6)}    ${jeffreys[i].toFixed(3).padStart(6)}`
          ),
          "",
          "Fisher metric enables natural gradient descent:",
          "  Δθ = -η I⁻¹(θ) ∇L(θ)"
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Measure Theory</Title>
          <Text size="lg" c="dimmed">
            Rigorous foundation for integration, probability, and information theory
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Overview</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The <Code>amari-measure</Code> module provides measure-theoretic foundations
              for integration, probability theory, and information geometry.
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Measure Types</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Lebesgue measure (integration)</li>
                  <li>Counting measure (discrete)</li>
                  <li>Probability measure</li>
                  <li>Fisher measure (information)</li>
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Information Theory</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Shannon entropy</li>
                  <li>KL divergence</li>
                  <li>Mutual information</li>
                  <li>Fisher information</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Key Concepts</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <CodeHighlight
              code={`A measure μ on a σ-algebra Σ satisfies:
  1. μ(∅) = 0
  2. μ(A) ≥ 0 for all A ∈ Σ
  3. μ(∪Aᵢ) = Σμ(Aᵢ) for disjoint sets (countable additivity)

Probability measure: μ(Ω) = 1

Integration:
  Lebesgue: ∫f dμ = usual integral
  Counting: ∫f dμ = Σf(xᵢ)

Information measures:
  Entropy:      H(P) = -Σ p log p
  KL Divergence: D(P||Q) = Σ p log(p/q)
  Fisher Info:  I(θ) = E[(∂log p/∂θ)²]`}
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
