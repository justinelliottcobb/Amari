import { Container, Stack, Card, Title, Text, List, Code } from "@mantine/core";
import { ExampleCard } from "../components/ExampleCard";

export function InformationGeometry() {
  // Simulate information geometry operations for demonstration
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
      title: "Fisher Information Matrix",
      description: "Compute the Fisher information metric for probability distributions",
      category: "Fundamentals",
      code: `// Fisher Information Matrix for a probability distribution
// G_ij = E[∂²(-log L)/∂θᵢ∂θⱼ] where L is the likelihood

function fisherMatrix(probabilities) {
  const n = probabilities.length;
  const matrix = Array(n).fill().map(() => Array(n).fill(0));

  // For multinomial distribution: G_ii = 1/p_i, G_ij = 0 (i≠j)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = probabilities[i] > 1e-12 ? 1.0 / probabilities[i] : 1e12;
      } else {
        matrix[i][j] = 0.0;
      }
    }
  }

  return matrix;
}

// Example: 3-sided die probabilities
const probs = [0.4, 0.35, 0.25];
const fisherMat = fisherMatrix(probs);

console.log("Probabilities:", probs);
console.log("Fisher Matrix:");
fisherMat.forEach((row, i) => {
  console.log(\`  Row \${i}: [\${row.map(x => x.toFixed(3)).join(', ')}]\`);
});

// Eigenvalues (diagonal elements for diagonal matrix)
const eigenvalues = fisherMat.map((row, i) => row[i]);
console.log("Eigenvalues:", eigenvalues.map(x => x.toFixed(3)));
console.log("Positive definite:", eigenvalues.every(x => x > 0));`,
      onRun: simulateExample(() => {
        function fisherMatrix(probabilities: number[]) {
          const n = probabilities.length;
          const matrix = Array(n).fill(null).map(() => Array(n).fill(0));

          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              if (i === j) {
                matrix[i][j] = probabilities[i] > 1e-12 ? 1.0 / probabilities[i] : 1e12;
              } else {
                matrix[i][j] = 0.0;
              }
            }
          }
          return matrix;
        }

        const probs = [0.4, 0.35, 0.25];
        const fisherMat = fisherMatrix(probs);
        const eigenvalues = fisherMat.map((row, i) => row[i]);

        return [
          `Probabilities: [${probs.join(', ')}]`,
          `Fisher Matrix:`,
          ...fisherMat.map((row, i) => `  Row ${i}: [${row.map(x => x.toFixed(3)).join(', ')}]`),
          `Eigenvalues: [${eigenvalues.map(x => x.toFixed(3)).join(', ')}]`,
          `Positive definite: ${eigenvalues.every(x => x > 0)}`
        ].join('\n');
      })
    },
    {
      title: "Bregman Divergence (KL Divergence)",
      description: "Compute the Kullback-Leibler divergence between probability distributions",
      category: "Divergences",
      code: `// Bregman divergence for probability distributions
// KL divergence: D_KL(p||q) = Σ p_i log(p_i/q_i)

function klDivergence(p, q) {
  let divergence = 0.0;

  for (let i = 0; i < p.length && i < q.length; i++) {
    if (p[i] > 1e-12 && q[i] > 1e-12) {
      divergence += p[i] * Math.log(p[i] / q[i]);
    }
  }

  return divergence;
}

// Example distributions
const p1 = [0.5, 0.3, 0.2];    // Distribution 1
const p2 = [0.4, 0.4, 0.2];    // Distribution 2
const uniform = [1/3, 1/3, 1/3]; // Uniform distribution

// Compute various divergences
const d_p1_p2 = klDivergence(p1, p2);
const d_p2_p1 = klDivergence(p2, p1);
const d_p1_uniform = klDivergence(p1, uniform);
const d_self = klDivergence(p1, p1);

console.log("Distribution p1:", p1);
console.log("Distribution p2:", p2);
console.log("Uniform distribution:", uniform);
console.log();
console.log("D_KL(p1||p2) =", d_p1_p2.toFixed(6));
console.log("D_KL(p2||p1) =", d_p2_p1.toFixed(6));
console.log("D_KL(p1||uniform) =", d_p1_uniform.toFixed(6));
console.log("D_KL(p1||p1) =", d_self.toFixed(6), "(should be 0)");
console.log();
console.log("KL divergence is asymmetric:", (d_p1_p2 !== d_p2_p1).toString());`,
      onRun: simulateExample(() => {
        function klDivergence(p: number[], q: number[]): number {
          let divergence = 0.0;
          for (let i = 0; i < p.length && i < q.length; i++) {
            if (p[i] > 1e-12 && q[i] > 1e-12) {
              divergence += p[i] * Math.log(p[i] / q[i]);
            }
          }
          return divergence;
        }

        const p1 = [0.5, 0.3, 0.2];
        const p2 = [0.4, 0.4, 0.2];
        const uniform = [1/3, 1/3, 1/3];

        const d_p1_p2 = klDivergence(p1, p2);
        const d_p2_p1 = klDivergence(p2, p1);
        const d_p1_uniform = klDivergence(p1, uniform);
        const d_self = klDivergence(p1, p1);

        return [
          `Distribution p1: [${p1.join(', ')}]`,
          `Distribution p2: [${p2.join(', ')}]`,
          `Uniform distribution: [${uniform.map(x => x.toFixed(3)).join(', ')}]`,
          ``,
          `D_KL(p1||p2) = ${d_p1_p2.toFixed(6)}`,
          `D_KL(p2||p1) = ${d_p2_p1.toFixed(6)}`,
          `D_KL(p1||uniform) = ${d_p1_uniform.toFixed(6)}`,
          `D_KL(p1||p1) = ${d_self.toFixed(6)} (should be 0)`,
          ``,
          `KL divergence is asymmetric: ${d_p1_p2 !== d_p2_p1}`
        ].join('\n');
      })
    },
    {
      title: "α-Connection Family",
      description: "Explore the family of α-connections on statistical manifolds",
      category: "Connections",
      code: `// α-connection family: interpolates between e-connection (α=-1) and m-connection (α=1)
// The Fisher metric provides a Riemannian structure

class AlphaConnection {
  constructor(alpha) {
    this.alpha = alpha;
  }

  // Get connection name based on α value
  getName() {
    if (Math.abs(this.alpha - (-1)) < 1e-10) return "e-connection (exponential)";
    if (Math.abs(this.alpha - 1) < 1e-10) return "m-connection (mixture)";
    if (Math.abs(this.alpha) < 1e-10) return "Levi-Civita connection";
    return \`α-connection (α = \${this.alpha})\`;
  }

  // Simplified curvature tensor component
  getCurvatureInfo() {
    // For exponential families, connections have specific geometric properties
    if (this.alpha === -1) {
      return "Flat in exponential coordinates (η-coordinates)";
    } else if (this.alpha === 1) {
      return "Flat in mixture coordinates (ξ-coordinates)";
    } else {
      return "Generally curved, not flat in any coordinate system";
    }
  }
}

// Create different connections
const connections = [-1, -0.5, 0, 0.5, 1].map(α => new AlphaConnection(α));

console.log("α-Connection Family:");
connections.forEach(conn => {
  console.log(\`α = \${conn.alpha.toString().padStart(4)}: \${conn.getName()}\`);
  console.log(\`           \${conn.getCurvatureInfo()}\`);
  console.log();
});

console.log("Dually flat manifold:");
console.log("• e-connection and m-connection are dual");
console.log("• Fisher metric is the only metric making both connections flat");
console.log("• Enables efficient computation via coordinate duality");`,
      onRun: simulateExample(() => {
        class AlphaConnection {
          alpha: number;

          constructor(alpha: number) {
            this.alpha = alpha;
          }

          getName() {
            if (Math.abs(this.alpha - (-1)) < 1e-10) return "e-connection (exponential)";
            if (Math.abs(this.alpha - 1) < 1e-10) return "m-connection (mixture)";
            if (Math.abs(this.alpha) < 1e-10) return "Levi-Civita connection";
            return `α-connection (α = ${this.alpha})`;
          }

          getCurvatureInfo() {
            if (this.alpha === -1) {
              return "Flat in exponential coordinates (η-coordinates)";
            } else if (this.alpha === 1) {
              return "Flat in mixture coordinates (ξ-coordinates)";
            } else {
              return "Generally curved, not flat in any coordinate system";
            }
          }
        }

        const connections = [-1, -0.5, 0, 0.5, 1].map(α => new AlphaConnection(α));

        const result = [
          "α-Connection Family:",
          ...connections.flatMap(conn => [
            `α = ${conn.alpha.toString().padStart(4)}: ${conn.getName()}`,
            `           ${conn.getCurvatureInfo()}`,
            ""
          ]),
          "Dually flat manifold:",
          "• e-connection and m-connection are dual",
          "• Fisher metric is the only metric making both connections flat",
          "• Enables efficient computation via coordinate duality"
        ];

        return result.join('\n');
      })
    },
    {
      title: "Amari-Chentsov Tensor Computation",
      description: "Compute the fundamental tensor that defines α-connections in information geometry",
      category: "Advanced Theory",
      code: `// Amari-Chentsov Tensor: T(∂ᵢ, ∂ⱼ, ∂ₖ) = E[∂ᵢ log p · ∂ⱼ log p · ∂ₖ log p]
// This 3rd-order tensor defines the α-connection family and geometric structure

function amariChentsovTensor(probabilities) {
  const n = probabilities.length;

  // Initialize 3rd-order tensor T[i][j][k]
  const tensor = Array(n).fill().map(() =>
    Array(n).fill().map(() => Array(n).fill(0))
  );

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        let tensorValue = 0;

        if (i === j && j === k) {
          tensorValue = (1 - 2 * probabilities[i]) / Math.pow(probabilities[i], 2);
        } else if (i === j && i !== k) {
          tensorValue = -1 / (probabilities[i] * probabilities[k]);
        } else if (i === k && i !== j) {
          tensorValue = -1 / (probabilities[i] * probabilities[j]);
        } else if (j === k && j !== i) {
          tensorValue = -1 / (probabilities[i] * probabilities[j]);
        } else if (i !== j && j !== k && i !== k) {
          tensorValue = 1 / (probabilities[i] * probabilities[j] * probabilities[k]);
        }

        tensor[i][j][k] = Math.abs(tensorValue) < 1e-12 ? 0 : tensorValue;
      }
    }
  }

  return tensor;
}

const probs = [0.5, 0.3, 0.2];
const tensor = amariChentsovTensor(probs);

console.log("Probability distribution:", probs);
console.log("\\nDiagonal components T[i,i,i]:");
for (let i = 0; i < probs.length; i++) {
  console.log(\`  T[\${i},\${i},\${i}] = \${tensor[i][i][i].toFixed(4)}\`);
}`,
      onRun: simulateExample(() => {
        function amariChentsovTensor(probabilities: number[]) {
          const n = probabilities.length;
          const tensor = Array(n).fill(null).map(() =>
            Array(n).fill(null).map(() => Array(n).fill(0))
          );

          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              for (let k = 0; k < n; k++) {
                let tensorValue = 0;

                if (i === j && j === k) {
                  tensorValue = (1 - 2 * probabilities[i]) / Math.pow(probabilities[i], 2);
                } else if (i === j && i !== k) {
                  tensorValue = -1 / (probabilities[i] * probabilities[k]);
                } else if (i === k && i !== j) {
                  tensorValue = -1 / (probabilities[i] * probabilities[j]);
                } else if (j === k && j !== i) {
                  tensorValue = -1 / (probabilities[i] * probabilities[j]);
                } else if (i !== j && j !== k && i !== k) {
                  tensorValue = 1 / (probabilities[i] * probabilities[j] * probabilities[k]);
                }

                tensor[i][j][k] = Math.abs(tensorValue) < 1e-12 ? 0 : tensorValue;
              }
            }
          }
          return tensor;
        }

        const probs = [0.5, 0.3, 0.2];
        const tensor = amariChentsovTensor(probs);

        const results = [
          `Probability distribution: [${probs.join(', ')}]`,
          ``,
          `Amari-Chentsov Tensor T[i,j,k]:`,
          `Diagonal components T[i,i,i]:`,
          ...probs.map((_, i) => `  T[${i},${i},${i}] = ${tensor[i][i][i].toFixed(4)}`),
          ``,
          `Off-diagonal components T[0,1,2]:`,
          `  T[0,1,2] = ${tensor[0][1][2].toFixed(4)}`,
          ``,
          `Geometric interpretation:`,
          `• Tensor encodes all statistical curvature information`,
          `• Defines the unique geometric structure of statistical manifolds`,
          `• Foundation for natural gradient descent and efficient optimization`
        ];

        return results.join('\n');
      })
    },
    {
      title: "Statistical Learning Applications",
      description: "Information geometry in machine learning and optimization",
      category: "Applications",
      code: `// Natural gradient descent using Fisher information metric
// Gradients are transformed by inverse Fisher matrix for better convergence

function naturalGradient(parameters, gradient, learningRate = 0.01) {
  // Compute Fisher information matrix
  const fisher = fisherInformationMatrix(parameters);

  // Compute inverse Fisher matrix (simplified for diagonal case)
  const invFisher = fisher.map((row, i) =>
    row.map((val, j) => i === j ? (val > 1e-12 ? 1.0 / val : 0) : 0)
  );

  // Natural gradient = inverse Fisher × regular gradient
  const naturalGrad = gradient.map((g, i) => invFisher[i][i] * g);

  // Update parameters
  return parameters.map((p, i) => p - learningRate * naturalGrad[i]);
}

function fisherInformationMatrix(params) {
  // Simplified Fisher matrix for probability distribution
  return params.map((p, i) =>
    params.map((_, j) => i === j ? (p > 1e-12 ? 1.0 / p : 1e12) : 0)
  );
}

// Example: optimizing a probability distribution
let params = [0.2, 0.3, 0.5];  // Initial distribution
const target = [0.4, 0.35, 0.25];  // Target distribution
const gradient = params.map((p, i) => 2 * (p - target[i]));  // L2 loss gradient

console.log("Initial parameters:", params.map(x => x.toFixed(3)));
console.log("Target parameters:", target.map(x => x.toFixed(3)));
console.log("Gradient:", gradient.map(x => x.toFixed(3)));

// Apply natural gradient update
const updated = naturalGradient(params, gradient, 0.1);
console.log("After natural gradient:", updated.map(x => x.toFixed(3)));

// Compare with regular gradient update
const regularUpdate = params.map((p, i) => p - 0.1 * gradient[i]);
console.log("After regular gradient:", regularUpdate.map(x => x.toFixed(3)));

console.log("\\nNatural gradient benefits:");
console.log("• Invariant to reparameterization");
console.log("• Faster convergence for statistical models");
console.log("• Respects geometry of parameter space");`,
      onRun: simulateExample(() => {
        function naturalGradient(parameters: number[], gradient: number[], learningRate = 0.01) {
          const fisher = fisherInformationMatrix(parameters);
          const invFisher = fisher.map((row, i) =>
            row.map((val, j) => i === j ? (val > 1e-12 ? 1.0 / val : 0) : 0)
          );
          const naturalGrad = gradient.map((g, i) => invFisher[i][i] * g);
          return parameters.map((p, i) => p - learningRate * naturalGrad[i]);
        }

        function fisherInformationMatrix(params: number[]) {
          return params.map((p, i) =>
            params.map((_, j) => i === j ? (p > 1e-12 ? 1.0 / p : 1e12) : 0)
          );
        }

        let params = [0.2, 0.3, 0.5];
        const target = [0.4, 0.35, 0.25];
        const gradient = params.map((p, i) => 2 * (p - target[i]));

        const updated = naturalGradient(params, gradient, 0.1);
        const regularUpdate = params.map((p, i) => p - 0.1 * gradient[i]);

        return [
          `Initial parameters: [${params.map(x => x.toFixed(3)).join(', ')}]`,
          `Target parameters: [${target.map(x => x.toFixed(3)).join(', ')}]`,
          `Gradient: [${gradient.map(x => x.toFixed(3)).join(', ')}]`,
          `After natural gradient: [${updated.map(x => x.toFixed(3)).join(', ')}]`,
          `After regular gradient: [${regularUpdate.map(x => x.toFixed(3)).join(', ')}]`,
          ``,
          `Natural gradient benefits:`,
          `• Invariant to reparameterization`,
          `• Faster convergence for statistical models`,
          `• Respects geometry of parameter space`
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Information Geometry Examples</Title>
          <Text size="lg" c="dimmed">
            Explore Fisher metrics, Bregman divergences, and α-connections on statistical manifolds.
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3}>What is Information Geometry?</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              Information geometry studies statistical models as Riemannian manifolds, providing geometric insights into:
            </Text>
            <List size="sm" mb="md">
              <List.Item><Text fw={600} span>Fisher Information Metric</Text>: Riemannian metric on parameter space</List.Item>
              <List.Item><Text fw={600} span>α-Connections</Text>: Family of affine connections (-1 ≤ α ≤ 1)</List.Item>
              <List.Item><Text fw={600} span>Bregman Divergences</Text>: Generalization of squared distance</List.Item>
              <List.Item><Text fw={600} span>Dually Flat Manifolds</Text>: Special structure with dual coordinate systems</List.Item>
            </List>
            <Card withBorder bg="dark.7" p="md">
              <Title order={4} size="sm" mb="xs">Key Applications:</Title>
              <List size="sm">
                <List.Item>Natural gradient descent in machine learning</List.Item>
                <List.Item>Optimal transport and Wasserstein geometry</List.Item>
                <List.Item>Statistical inference and hypothesis testing</List.Item>
                <List.Item>Neural network optimization</List.Item>
              </List>
            </Card>
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
            <Title order={3}>Amari-Chentsov Tensor</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The Amari-Chentsov tensor is a fundamental object in information geometry that captures
              the intrinsic geometric structure of statistical manifolds.
            </Text>
            <Code block mb="md">
              T(∂ᵢ, ∂ⱼ, ∂ₖ) = E[∂ᵢ log p · ∂ⱼ log p · ∂ₖ log p]
            </Code>
            <Text size="sm" c="dimmed">
              This tensor defines the α-connections and provides the unique geometric structure
              that makes information geometry so powerful for statistical applications.
            </Text>
          </Card.Section>
        </Card>
      </Stack>
    </Container>
  );
}
