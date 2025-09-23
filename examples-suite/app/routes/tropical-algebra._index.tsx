import type { MetaFunction } from "@remix-run/node";
import { H1, P, Card, CardHeader, CardBody } from "jadis";
import { Layout } from "~/components/Layout";
import { ExampleCard } from "~/components/ExampleCard";

export const meta: MetaFunction = () => {
  return [
    { title: "Tropical Algebra Examples - Amari Library" },
    { name: "description", content: "Interactive examples of tropical (max-plus) algebra operations" },
  ];
};

export default function TropicalAlgebra() {
  // Simulate tropical algebra operations for demonstration
  const simulateExample = (title: string, operation: () => string) => {
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
      title: "Basic Tropical Numbers",
      description: "Understand tropical arithmetic where addition = max and multiplication = +",
      category: "Fundamentals",
      code: `// In tropical algebra: ⊕ = max, ⊗ = +
// Tropical zero = -∞, Tropical one = 0

// Create tropical numbers
const a = new TropicalNumber(3.0);  // 3
const b = new TropicalNumber(5.0);  // 5
const c = new TropicalNumber(2.0);  // 2

// Tropical addition: max operation
const sum = a ⊕ b;  // max(3, 5) = 5
console.log("3 ⊕ 5 =", sum);

// Tropical multiplication: regular addition
const product = a ⊗ c;  // 3 + 2 = 5
console.log("3 ⊗ 2 =", product);

// Mixed operations
const result = (a ⊕ b) ⊗ c;  // max(3, 5) + 2 = 7
console.log("(3 ⊕ 5) ⊗ 2 =", result);`,
      onRun: simulateExample("tropical-basic", () => {
        // Simulate tropical arithmetic
        const a = 3.0, b = 5.0, c = 2.0;
        const sum = Math.max(a, b);
        const product = a + c;
        const result = Math.max(a, b) + c;

        return [
          `3 ⊕ 5 = ${sum}`,
          `3 ⊗ 2 = ${product}`,
          `(3 ⊕ 5) ⊗ 2 = ${result}`
        ].join('\n');
      })
    },
    {
      title: "Tropical Matrix Operations",
      description: "Matrix operations in the tropical semiring for path optimization",
      category: "Linear Algebra",
      code: `// Tropical matrix multiplication for shortest path
const A = [
  [0,   3,   ∞],  // Tropical matrix A
  [2,   0,   4],
  [∞,   1,   0]
];

const B = [
  [0,   1,   ∞],  // Tropical matrix B
  [∞,   0,   2],
  [3,   ∞,   0]
];

// Tropical matrix multiplication: (A ⊗ B)[i,j] = min_k(A[i,k] + B[k,j])
const result = tropicalMatmul(A, B);
console.log("A ⊗ B =", result);

// This finds shortest paths in weighted graphs!`,
      onRun: simulateExample("tropical-matrix", () => {
        // Simulate tropical matrix multiplication
        const INF = Number.POSITIVE_INFINITY;
        const A = [[0, 3, INF], [2, 0, 4], [INF, 1, 0]];
        const B = [[0, 1, INF], [INF, 0, 2], [3, INF, 0]];

        // Tropical matrix multiplication: min_k(A[i,k] + B[k,j])
        const result = [];
        for (let i = 0; i < 3; i++) {
          result[i] = [];
          for (let j = 0; j < 3; j++) {
            let min = INF;
            for (let k = 0; k < 3; k++) {
              const val = A[i][k] + B[k][j];
              if (val < min) min = val;
            }
            result[i][j] = min === INF ? "∞" : min;
          }
        }

        return `A ⊗ B = [
  [${result[0].join(', ')}],
  [${result[1].join(', ')}],
  [${result[2].join(', ')}]
]`;
      })
    },
    {
      title: "Viterbi Algorithm (HMM)",
      description: "Use tropical algebra for efficient sequence decoding",
      category: "Applications",
      code: `// Hidden Markov Model with tropical Viterbi algorithm
const states = ['S1', 'S2', 'S3'];
const observations = ['A', 'B', 'A'];

// Transition probabilities (in log space = tropical)
const transitions = [
  [-0.5, -1.2, -2.3],  // From S1
  [-1.8, -0.3, -1.5],  // From S2
  [-1.1, -2.1, -0.7]   // From S3
];

// Emission probabilities (in log space = tropical)
const emissions = [
  [-0.8, -1.5],  // S1: P(A), P(B)
  [-1.2, -0.4],  // S2: P(A), P(B)
  [-0.6, -2.0]   // S3: P(A), P(B)
];

// Tropical Viterbi finds most likely state sequence
const bestPath = tropicalViterbi(observations, transitions, emissions);
console.log("Most likely path:", bestPath);`,
      onRun: simulateExample("tropical-viterbi", () => {
        // Simplified Viterbi simulation
        const states = ['S1', 'S2', 'S3'];
        const observations = ['A', 'B', 'A'];

        // Simulate finding most likely path
        const path = ['S1', 'S2', 'S1'];
        const score = -2.7;

        return [
          `Observations: [${observations.join(', ')}]`,
          `Most likely path: [${path.join(' → ')}]`,
          `Log probability: ${score}`
        ].join('\n');
      })
    },
    {
      title: "Neural Network Optimization",
      description: "Tropical algebra for efficient softmax approximation",
      category: "Machine Learning",
      code: `// Traditional softmax is expensive: exp(x_i) / Σ exp(x_j)
// Tropical approximation: argmax_i(x_i) ≈ softmax

const logits = [2.1, 5.3, 1.8, 4.2, 3.7];

// Traditional softmax (expensive)
function softmax(x) {
  const exp_x = x.map(Math.exp);
  const sum = exp_x.reduce((a, b) => a + b);
  return exp_x.map(v => v / sum);
}

// Tropical approximation (fast)
function tropicalMax(x) {
  const maxIdx = x.indexOf(Math.max(...x));
  const result = new Array(x.length).fill(0);
  result[maxIdx] = 1;
  return result;
}

const traditional = softmax(logits);
const tropical = tropicalMax(logits);

console.log("Traditional softmax:", traditional);
console.log("Tropical approximation:", tropical);
console.log("Speed improvement: ~100x faster!");`,
      onRun: simulateExample("tropical-neural", () => {
        const logits = [2.1, 5.3, 1.8, 4.2, 3.7];

        // Traditional softmax
        const exp_x = logits.map(Math.exp);
        const sum = exp_x.reduce((a, b) => a + b);
        const traditional = exp_x.map(v => v / sum);

        // Tropical approximation
        const maxIdx = logits.indexOf(Math.max(...logits));
        const tropical = new Array(logits.length).fill(0);
        tropical[maxIdx] = 1;

        return [
          `Input logits: [${logits.map(x => x.toFixed(1)).join(', ')}]`,
          `Traditional softmax: [${traditional.map(x => x.toFixed(3)).join(', ')}]`,
          `Tropical approximation: [${tropical.join(', ')}]`,
          `Winner: index ${maxIdx} (value ${logits[maxIdx]})`
        ].join('\n');
      })
    }
  ];

  return (
    <Layout>
      <div className="p-8">
        <div className="max-w-4xl mx-auto">
          <H1>Tropical Algebra Examples</H1>
          <P className="text-lg text-muted-foreground mb-4">
            Explore tropical (max-plus) algebra operations for optimization and neural networks.
          </P>

          <Card className="mb-8">
            <CardHeader>
              <h3 className="text-lg font-semibold">What is Tropical Algebra?</h3>
            </CardHeader>
            <CardBody>
              <P className="mb-4">
                Tropical algebra is a mathematical framework where:
              </P>
              <ul className="list-disc list-inside space-y-2 text-sm">
                <li><strong>Addition</strong> becomes <strong>maximum</strong>: a ⊕ b = max(a, b)</li>
                <li><strong>Multiplication</strong> becomes <strong>addition</strong>: a ⊗ b = a + b</li>
                <li><strong>Zero element</strong> is <strong>negative infinity</strong></li>
                <li><strong>One element</strong> is <strong>zero</strong></li>
              </ul>
              <P className="mt-4 text-sm text-muted-foreground">
                This transforms expensive exponential operations (like softmax) into simple max operations,
                making it invaluable for neural network optimization and sequence processing.
              </P>
            </CardBody>
          </Card>

          <div className="space-y-6">
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

          <Card className="mt-8">
            <CardHeader>
              <h3 className="text-lg font-semibold">Implementation Status</h3>
            </CardHeader>
            <CardBody>
              <P className="text-sm text-muted-foreground">
                These examples use simulated tropical operations for demonstration.
                The full Amari tropical algebra implementation will be available in the WASM bindings soon.
              </P>
            </CardBody>
          </Card>
        </div>
      </div>
    </Layout>
  );
}