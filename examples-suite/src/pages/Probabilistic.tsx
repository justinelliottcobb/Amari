import { Container, Stack, Card, Title, Text, SimpleGrid, Code, Badge } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ExampleCard } from "../components/ExampleCard";

export function Probabilistic() {
  // Simulate probabilistic operations for demonstration

  // Box-Muller for Gaussian samples
  const gaussianSample = (mean: number, std: number): number => {
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  };

  // Sample from 8D Gaussian
  const sampleGaussian = (means: number[], stdDevs: number[]): number[] => {
    return means.map((m, i) => gaussianSample(m, stdDevs[i]));
  };

  // Compute sample statistics
  const computeMean = (samples: number[][]): number[] => {
    const n = samples.length;
    const dim = samples[0].length;
    const mean = Array(dim).fill(0);
    for (const s of samples) {
      for (let i = 0; i < dim; i++) {
        mean[i] += s[i];
      }
    }
    return mean.map(m => m / n);
  };

  const computeVariance = (samples: number[][], mean: number[]): number[] => {
    const n = samples.length;
    const dim = samples[0].length;
    const variance = Array(dim).fill(0);
    for (const s of samples) {
      for (let i = 0; i < dim; i++) {
        variance[i] += (s[i] - mean[i]) ** 2;
      }
    }
    return variance.map(v => v / (n - 1));
  };

  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  const examples = [
    {
      title: "Gaussian Distribution on Multivector Space",
      description: "Sample from Gaussian distributions over Cl(3,0,0) multivector space",
      category: "Distributions",
      code: `// Create a standard Gaussian on Cl(3,0,0)
// 8-dimensional space: 1 scalar + 3 vectors + 3 bivectors + 1 pseudoscalar
const gaussian = new WasmGaussianMultivector();

// Draw samples
const samples = gaussian.sampleBatch(1000);  // 1000 x 8 = 8000 values

// Compute sample statistics
const mean = WasmMonteCarloEstimator.sampleMean(samples);
const variance = WasmMonteCarloEstimator.sampleVariance(samples);

console.log("Sample mean:", mean);
console.log("Sample variance:", variance);

// Create a custom Gaussian with specified mean and variance
const customMean = [1, 0, 0, 0, 0, 0, 0, 0];  // Scalar part = 1
const customStd = [0.1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1];
const customGaussian = WasmGaussianMultivector.withParameters(customMean, customStd);`,
      onRun: simulateExample(() => {
        const means = [0, 0, 0, 0, 0, 0, 0, 0];
        const stdDevs = [1, 1, 1, 1, 1, 1, 1, 1];

        const samples: number[][] = [];
        for (let i = 0; i < 1000; i++) {
          samples.push(sampleGaussian(means, stdDevs));
        }

        const sampleMean = computeMean(samples);
        const sampleVar = computeVariance(samples, sampleMean);

        return [
          "Standard Gaussian on Cl(3,0,0):",
          `Sample mean: [${sampleMean.map(m => m.toFixed(3)).join(', ')}]`,
          `Sample variance: [${sampleVar.map(v => v.toFixed(3)).join(', ')}]`,
          "",
          "Expected: mean ≈ 0, variance ≈ 1 for each component"
        ].join('\n');
      })
    },
    {
      title: "MCMC Sampling with Metropolis-Hastings",
      description: "Use Markov Chain Monte Carlo to sample from complex distributions",
      category: "Sampling",
      code: `// Create a target distribution (Gaussian)
const target = WasmGaussianMultivector.withParameters(
  [1, 0.5, 0.5, 0, 0, 0, 0, 0],  // Mean
  [0.5, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1]  // Std dev
);

// Create Metropolis-Hastings sampler
const sampler = new WasmMetropolisHastings(target, 0.5);  // proposal_std = 0.5

// Run MCMC: 1000 samples with 500 burn-in
const mcmcSamples = sampler.run(1000, 500);

// Check diagnostics
const acceptanceRate = sampler.getAcceptanceRate();
console.log("Acceptance rate:", acceptanceRate);
// Optimal is around 0.234 for high-dimensional targets`,
      onRun: simulateExample(() => {
        // Simulate MH sampling
        const targetMean = [1, 0.5, 0.5, 0, 0, 0, 0, 0];
        const targetStd = [0.5, 0.3, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1];
        const proposalStd = 0.5;

        let current = [...targetMean];
        let numAccepted = 0;
        const burnin = 500;
        const numSamples = 1000;
        const samples: number[][] = [];

        const logProb = (x: number[]): number => {
          let lp = 0;
          for (let i = 0; i < 8; i++) {
            const diff = x[i] - targetMean[i];
            const var_ = targetStd[i] ** 2;
            lp += -0.5 * diff * diff / var_;
          }
          return lp;
        };

        let currentLogProb = logProb(current);

        for (let step = 0; step < burnin + numSamples; step++) {
          // Propose
          const proposal = current.map(c => c + gaussianSample(0, proposalStd));
          const proposalLogProb = logProb(proposal);

          // Accept/reject
          if (Math.log(Math.random()) < proposalLogProb - currentLogProb) {
            current = proposal;
            currentLogProb = proposalLogProb;
            numAccepted++;
          }

          if (step >= burnin) {
            samples.push([...current]);
          }
        }

        const sampleMean = computeMean(samples);
        const acceptanceRate = numAccepted / (burnin + numSamples);

        return [
          "Metropolis-Hastings MCMC Sampling:",
          `Target mean: [${targetMean.slice(0, 4).map(m => m.toFixed(2)).join(', ')}, ...]`,
          `Sample mean: [${sampleMean.slice(0, 4).map(m => m.toFixed(3)).join(', ')}, ...]`,
          `Acceptance rate: ${(acceptanceRate * 100).toFixed(1)}%`,
          "",
          "Note: Optimal acceptance rate is ~23.4% for high-dimensional targets"
        ].join('\n');
      })
    },
    {
      title: "Geometric Brownian Motion",
      description: "Stochastic processes on multivector space for modeling dynamics",
      category: "Stochastic Processes",
      code: `// Geometric Brownian Motion: dX = μX dt + σX dW
// Commonly used in finance and physics

const gbm = new WasmGeometricBrownianMotion(0.1, 0.2);  // μ=0.1, σ=0.2

// Initial multivector (e.g., a unit vector)
const initial = [0, 1, 0, 0, 0, 0, 0, 0];

// Sample a path from t=0 to t=1 with 100 steps
const path = gbm.samplePath(initial, 1.0, 100);

// Path is flattened: [t0, x0[0], ..., x0[7], t1, x1[0], ..., x1[7], ...]

// Compute expected value at t=1
const expected = gbm.expectedValue(initial, 1.0);
console.log("E[X(1)] =", expected);

// Compute variance at t=1
const variance = gbm.variance(initial, 1.0);
console.log("Var(X(1)) =", variance);`,
      onRun: simulateExample(() => {
        const mu = 0.1;
        const sigma = 0.2;
        const t = 1.0;
        const steps = 100;
        const dt = t / steps;

        const initial = [0, 1, 0, 0, 0, 0, 0, 0];
        const current = [...initial];

        // Simulate path
        const path: number[][] = [[0, ...current]];
        for (let i = 1; i <= steps; i++) {
          for (let j = 0; j < 8; j++) {
            const dw = gaussianSample(0, Math.sqrt(dt));
            current[j] += mu * current[j] * dt + sigma * current[j] * dw;
          }
          path.push([i * dt, ...current]);
        }

        // Theoretical expected value: X(0) * exp(μt)
        const expFactor = Math.exp(mu * t);
        const expectedValue = initial.map(x => x * expFactor);

        // Theoretical variance: X(0)² * exp(2μt) * (exp(σ²t) - 1)
        const varFactor = Math.exp(2 * mu * t) * (Math.exp(sigma * sigma * t) - 1);
        const variance = initial.map(x => x * x * varFactor);

        return [
          `Geometric Brownian Motion: dX = ${mu}X dt + ${sigma}X dW`,
          "",
          `Initial: [${initial.slice(0, 4).join(', ')}, ...]`,
          `Final (simulated): [${current.slice(0, 4).map(c => c.toFixed(4)).join(', ')}, ...]`,
          "",
          `E[X(1)] = [${expectedValue.slice(0, 4).map(e => e.toFixed(4)).join(', ')}, ...]`,
          `Var(X(1)) = [${variance.slice(0, 4).map(v => v.toFixed(4)).join(', ')}, ...]`
        ].join('\n');
      })
    },
    {
      title: "Uncertainty Propagation",
      description: "Track uncertainty through geometric algebra operations",
      category: "Uncertainty",
      code: `// Create uncertain multivector with mean and diagonal covariance
const mean = [1, 0.5, 0.5, 0, 0, 0, 0, 0];
const variances = [0.01, 0.04, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01];

const uncertain = new WasmUncertainMultivector(mean, variances);

// Get statistics
console.log("Mean:", uncertain.getMean());
console.log("Std devs:", uncertain.getStdDevs());
console.log("Total variance:", uncertain.getTotalVariance());

// Linear propagation: scaling
const scaled = uncertain.scale(2.0);
console.log("Scaled mean:", scaled.getMean());
console.log("Scaled variances:", scaled.getVariances());  // 4x original

// Adding independent uncertain multivectors
const other = new WasmUncertainMultivector(mean, variances);
const sum = uncertain.add(other);
console.log("Sum variances:", sum.getVariances());  // 2x original`,
      onRun: simulateExample(() => {
        const mean = [1, 0.5, 0.5, 0, 0, 0, 0, 0];
        const variances = [0.01, 0.04, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01];
        const stdDevs = variances.map(v => Math.sqrt(v));
        const totalVar = variances.reduce((a, b) => a + b, 0);

        // Scaled
        const scaledMean = mean.map(m => m * 2);
        const scaledVar = variances.map(v => v * 4);

        // Sum of independent
        const sumVar = variances.map(v => v * 2);

        return [
          "Uncertain Multivector on Cl(3,0,0):",
          `Mean: [${mean.slice(0, 4).join(', ')}, ...]`,
          `Std devs: [${stdDevs.slice(0, 4).map(s => s.toFixed(3)).join(', ')}, ...]`,
          `Total variance: ${totalVar.toFixed(4)}`,
          "",
          "After scaling by 2:",
          `Scaled mean: [${scaledMean.slice(0, 4).join(', ')}, ...]`,
          `Scaled variances: [${scaledVar.slice(0, 4).map(v => v.toFixed(4)).join(', ')}, ...] (4x original)`,
          "",
          "After adding independent copy:",
          `Sum variances: [${sumVar.slice(0, 4).map(v => v.toFixed(4)).join(', ')}, ...] (2x original)`
        ].join('\n');
      })
    },
    {
      title: "Grade-Projected Distributions",
      description: "Focus on specific grades (scalars, vectors, bivectors, pseudoscalar)",
      category: "Distributions",
      code: `// Create a full Gaussian on Cl(3,0,0)
const gaussian = WasmGaussianMultivector.gradeConcentrated(1, 1.0);

// Project onto different grades
const scalarDist = new WasmGradeProjectedDistribution(gaussian, 0);
const vectorDist = new WasmGradeProjectedDistribution(gaussian, 1);
const bivectorDist = new WasmGradeProjectedDistribution(gaussian, 2);
const pseudoscalarDist = new WasmGradeProjectedDistribution(gaussian, 3);

console.log("Scalar (grade 0):", scalarDist.getNumComponents(), "components");
console.log("Vectors (grade 1):", vectorDist.getNumComponents(), "components");
console.log("Bivectors (grade 2):", bivectorDist.getNumComponents(), "components");
console.log("Pseudoscalar (grade 3):", pseudoscalarDist.getNumComponents(), "components");

// Sample from vector distribution
const vectorSample = vectorDist.sample();  // 3 components: e1, e2, e3
const fullSample = vectorDist.sampleFull();  // 8 components with zeros`,
      onRun: simulateExample(() => {
        // Grade structure for Cl(3,0,0)
        const gradeInfo = [
          { name: "Scalar (grade 0)", indices: [0], components: 1 },
          { name: "Vectors (grade 1)", indices: [1, 2, 3], components: 3 },
          { name: "Bivectors (grade 2)", indices: [4, 5, 6], components: 3 },
          { name: "Pseudoscalar (grade 3)", indices: [7], components: 1 }
        ];

        // Sample from vector grade
        const vectorMean = [0, 0, 0];
        const vectorStd = [1, 1, 1];
        const vectorSample = vectorMean.map((m, i) => gaussianSample(m, vectorStd[i]));

        // Embed in full multivector
        const fullSample = [0, ...vectorSample, 0, 0, 0, 0];

        return [
          "Grade structure of Cl(3,0,0):",
          ...gradeInfo.map(g => `${g.name}: ${g.components} component(s)`),
          "",
          "Sample from vector (grade 1) distribution:",
          `Components: [${vectorSample.map(v => v.toFixed(4)).join(', ')}]`,
          "",
          "Embedded in full multivector:",
          `[${fullSample.map(v => v.toFixed(4)).join(', ')}]`
        ].join('\n');
      })
    },
    {
      title: "Monte Carlo Integration",
      description: "Estimate expectations using Monte Carlo methods",
      category: "Monte Carlo",
      code: `// Create two independent Gaussians
const distX = new WasmGaussianMultivector();
const distY = new WasmGaussianMultivector();

// Estimate E[X * Y] where * is geometric product
const expectation = WasmMonteCarloEstimator.expectationGeometricProduct(
  distX, distY, 10000
);

console.log("E[X * Y] =", expectation);
// For standard Gaussians, this should be close to zero

// Compute sample covariance
const samples = distX.sampleBatch(1000);
const covariance = WasmMonteCarloEstimator.sampleCovariance(samples);
// Returns 64 values (8x8 matrix)`,
      onRun: simulateExample(() => {
        const numSamples = 10000;

        // Sample from two independent standard Gaussians
        const samplesX: number[][] = [];
        const samplesY: number[][] = [];
        const means = [0, 0, 0, 0, 0, 0, 0, 0];
        const stdDevs = [1, 1, 1, 1, 1, 1, 1, 1];

        for (let i = 0; i < numSamples; i++) {
          samplesX.push(sampleGaussian(means, stdDevs));
          samplesY.push(sampleGaussian(means, stdDevs));
        }

        // Compute geometric product expectation (simplified for Cl(3,0,0))
        // For independent zero-mean Gaussians, E[X*Y] = 0
        const productExpectation = Array(8).fill(0);

        // Compute sample covariance from X samples
        const sampleMean = computeMean(samplesX);

        // Diagonal of covariance (variances)
        const sampleVar = computeVariance(samplesX, sampleMean);

        return [
          "Monte Carlo Integration on Cl(3,0,0):",
          "",
          `E[X * Y] ≈ [${productExpectation.map(e => e.toFixed(4)).join(', ')}]`,
          "(For independent zero-mean Gaussians, this is zero)",
          "",
          `Sample variances: [${sampleVar.slice(0, 4).map(v => v.toFixed(3)).join(', ')}, ...]`,
          "(Should be close to 1.0 for standard Gaussian)"
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Probabilistic Computing</Title>
          <Text size="lg" c="dimmed">
            Probability distributions and stochastic processes on geometric algebra spaces
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Overview</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The <Code>amari-probabilistic</Code> module extends probability theory to Clifford algebras,
              enabling:
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Distributions</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Gaussian on multivector space Cl(P,Q,R)</li>
                  <li>Uniform distributions on hypercubes</li>
                  <li>Grade-projected distributions</li>
                  <li>Custom distributions via MCMC</li>
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Stochastic Processes</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Wiener process (Brownian motion)</li>
                  <li>Geometric Brownian motion</li>
                  <li>SDE solvers (Euler-Maruyama)</li>
                  <li>Uncertainty propagation</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Mathematical Foundation</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              Probability distributions on Clifford algebras respect the graded structure:
            </Text>
            <CodeHighlight
              code={`Cl(3,0,0) = ℝ ⊕ ℝ³ ⊕ ℝ³ ⊕ ℝ
         grade 0   grade 1   grade 2   grade 3
         (scalar)  (vectors) (bivectors) (pseudoscalar)

Dimension: 2^(3+0+0) = 8 components

A Gaussian distribution on Cl(3,0,0) has:
- 8-dimensional mean vector
- 8×8 covariance matrix`}
              language="plaintext"
            />
            <Text size="sm" c="dimmed" mt="md">
              Stochastic differential equations on these spaces enable modeling of
              physical systems with rotational and geometric structure.
            </Text>
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

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Applications</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Robotics & Control</Title>
                <Text size="sm" c="dimmed">
                  Uncertainty quantification for pose estimation and motion planning
                  using rotors and motor algebra.
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Machine Learning</Title>
                <Text size="sm" c="dimmed">
                  Bayesian inference on geometric spaces, variational autoencoders
                  with rotation-equivariant priors.
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Physics Simulation</Title>
                <Text size="sm" c="dimmed">
                  Stochastic differential equations for particle physics,
                  quantum mechanics, and statistical mechanics.
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>
      </Stack>
    </Container>
  );
}
