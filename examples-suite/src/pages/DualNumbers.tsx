import { H1, P, Card, CardHeader, CardBody } from "jadis-ui";
import { ExampleCard } from "../components/ExampleCard";

export function DualNumbers() {
  // Simulate dual number operations for demonstration
  class DualNumber {
    constructor(public real: number, public dual: number) {}

    static variable(value: number): DualNumber {
      return new DualNumber(value, 1.0);
    }

    static constant(value: number): DualNumber {
      return new DualNumber(value, 0.0);
    }

    add(other: DualNumber): DualNumber {
      return new DualNumber(this.real + other.real, this.dual + other.dual);
    }

    multiply(other: DualNumber): DualNumber {
      return new DualNumber(
        this.real * other.real,
        this.real * other.dual + this.dual * other.real
      );
    }

    sin(): DualNumber {
      return new DualNumber(Math.sin(this.real), Math.cos(this.real) * this.dual);
    }

    exp(): DualNumber {
      const expVal = Math.exp(this.real);
      return new DualNumber(expVal, expVal * this.dual);
    }

    toString(): string {
      return `${this.real.toFixed(3)} + ${this.dual.toFixed(3)}ε`;
    }
  }

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
      title: "Basic Dual Number Arithmetic",
      description: "Understand dual numbers and how they compute derivatives automatically",
      category: "Fundamentals",
      code: `// Dual numbers: a + bε where ε² = 0
// Real part = function value, Dual part = derivative

// Create a variable (derivative = 1) and a constant (derivative = 0)
const x = DualNumber.variable(2.0);  // 2 + 1ε
const c = DualNumber.constant(3.0);  // 3 + 0ε

console.log("x =", x.toString());
console.log("c =", c.toString());

// Addition: (a + bε) + (c + dε) = (a + c) + (b + d)ε
const sum = x.add(c);
console.log("x + c =", sum.toString());

// Multiplication: (a + bε)(c + dε) = ac + (ad + bc)ε
const product = x.multiply(c);
console.log("x * c =", product.toString());

// The dual part gives us the derivative!
console.log("d/dx(x * 3) =", product.dual, "(should be 3)");`,
      onRun: simulateExample(() => {
        const x = DualNumber.variable(2.0);
        const c = DualNumber.constant(3.0);
        const sum = x.add(c);
        const product = x.multiply(c);

        return [
          `x = ${x.toString()}`,
          `c = ${c.toString()}`,
          `x + c = ${sum.toString()}`,
          `x * c = ${product.toString()}`,
          `d/dx(x * 3) = ${product.dual} (should be 3)`
        ].join('\n');
      })
    },
    {
      title: "Transcendental Functions",
      description: "Automatic differentiation for sin, cos, exp, and other functions",
      category: "Functions",
      code: `// Let's compute f(x) = sin(x) and its derivative at x = π/4
const x = DualNumber.variable(Math.PI / 4);

// sin(x) with automatic differentiation
const sinX = x.sin();
console.log("x =", x.real.toFixed(3));
console.log("sin(x) =", sinX.real.toFixed(3));
console.log("d/dx sin(x) = cos(x) =", sinX.dual.toFixed(3));
console.log("Expected cos(π/4) =", Math.cos(Math.PI / 4).toFixed(3));

// Exponential function
const expX = x.exp();
console.log("\\nexp(x) =", expX.real.toFixed(3));
console.log("d/dx exp(x) = exp(x) =", expX.dual.toFixed(3));
console.log("Values match:", Math.abs(expX.real - expX.dual) < 1e-10);`,
      onRun: simulateExample(() => {
        const x = DualNumber.variable(Math.PI / 4);
        const sinX = x.sin();
        const expX = x.exp();

        return [
          `x = ${x.real.toFixed(3)}`,
          `sin(x) = ${sinX.real.toFixed(3)}`,
          `d/dx sin(x) = cos(x) = ${sinX.dual.toFixed(3)}`,
          `Expected cos(π/4) = ${Math.cos(Math.PI / 4).toFixed(3)}`,
          ``,
          `exp(x) = ${expX.real.toFixed(3)}`,
          `d/dx exp(x) = exp(x) = ${expX.dual.toFixed(3)}`,
          `Values match: ${Math.abs(expX.real - expX.dual) < 1e-10}`
        ].join('\n');
      })
    },
    {
      title: "Chain Rule Automation",
      description: "Complex function composition with automatic chain rule application",
      category: "Composition",
      code: `// Compute f(x) = sin(exp(x²)) and its derivative
// This involves multiple chain rule applications

function complexFunction(x) {
  // x²
  const xSquared = x.multiply(x);

  // exp(x²)
  const expXSquared = xSquared.exp();

  // sin(exp(x²))
  const result = expXSquared.sin();

  return result;
}

const x = DualNumber.variable(0.5);
const result = complexFunction(x);

console.log("x =", x.real);
console.log("f(x) = sin(exp(x²)) =", result.real.toFixed(6));
console.log("f'(x) =", result.dual.toFixed(6));

// Manual verification: f'(x) = cos(exp(x²)) * exp(x²) * 2x
const manual = Math.cos(Math.exp(x.real ** 2)) * Math.exp(x.real ** 2) * 2 * x.real;
console.log("Manual calculation =", manual.toFixed(6));
console.log("Match:", Math.abs(result.dual - manual) < 1e-10);`,
      onRun: simulateExample(() => {
        function complexFunction(x: DualNumber) {
          const xSquared = x.multiply(x);
          const expXSquared = xSquared.exp();
          const result = expXSquared.sin();
          return result;
        }

        const x = DualNumber.variable(0.5);
        const result = complexFunction(x);

        // Manual verification
        const manual = Math.cos(Math.exp(x.real ** 2)) * Math.exp(x.real ** 2) * 2 * x.real;

        return [
          `x = ${x.real}`,
          `f(x) = sin(exp(x²)) = ${result.real.toFixed(6)}`,
          `f'(x) = ${result.dual.toFixed(6)}`,
          `Manual calculation = ${manual.toFixed(6)}`,
          `Match: ${Math.abs(result.dual - manual) < 1e-10}`
        ].join('\n');
      })
    },
    {
      title: "Neural Network Gradient",
      description: "Compute gradients for a simple neural network layer",
      category: "Machine Learning",
      code: `// Simple linear layer: y = W*x + b
// Compute ∂loss/∂W for gradient descent

function linearLayer(x, w, b) {
  return w.multiply(x).add(b);
}

function squaredLoss(prediction, target) {
  const diff = prediction.add(target.multiply(DualNumber.constant(-1)));
  return diff.multiply(diff);
}

// Training example: x=2, target=5, initial w=1, b=0
const x = DualNumber.constant(2.0);
const target = DualNumber.constant(5.0);
const w = DualNumber.variable(1.0);  // We want gradient w.r.t. w
const b = DualNumber.constant(0.0);

// Forward pass
const prediction = linearLayer(x, w, b);
const loss = squaredLoss(prediction, target);

console.log("Input x =", x.real);
console.log("Weight w =", w.real);
console.log("Bias b =", b.real);
console.log("Prediction =", prediction.real);
console.log("Target =", target.real);
console.log("Loss =", loss.real);
console.log("∂loss/∂w =", loss.dual);

// Update rule: w_new = w - learning_rate * gradient
const learningRate = 0.1;
const newW = w.real - learningRate * loss.dual;
console.log("Updated weight =", newW.toFixed(3));`,
      onRun: simulateExample(() => {
        function linearLayer(x: DualNumber, w: DualNumber, b: DualNumber) {
          return w.multiply(x).add(b);
        }

        function squaredLoss(prediction: DualNumber, target: DualNumber) {
          const diff = prediction.add(target.multiply(DualNumber.constant(-1)));
          return diff.multiply(diff);
        }

        const x = DualNumber.constant(2.0);
        const target = DualNumber.constant(5.0);
        const w = DualNumber.variable(1.0);
        const b = DualNumber.constant(0.0);

        const prediction = linearLayer(x, w, b);
        const loss = squaredLoss(prediction, target);

        const learningRate = 0.1;
        const newW = w.real - learningRate * loss.dual;

        return [
          `Input x = ${x.real}`,
          `Weight w = ${w.real}`,
          `Bias b = ${b.real}`,
          `Prediction = ${prediction.real}`,
          `Target = ${target.real}`,
          `Loss = ${loss.real}`,
          `∂loss/∂w = ${loss.dual}`,
          `Updated weight = ${newW.toFixed(3)}`
        ].join('\n');
      })
    }
  ];

  return (
<div style={{ padding: '2rem' }}>
        <div>
          <H1>Dual Number Automatic Differentiation</H1>
          <P style={{ fontSize: '1.125rem', opacity: 0.7, marginBottom: '1rem' }}>
            Explore forward-mode automatic differentiation with dual numbers for exact gradient computation.
          </P>

          <Card style={{ marginBottom: '2rem' }}>
            <CardHeader>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>What are Dual Numbers?</h3>
            </CardHeader>
            <CardBody>
              <P style={{ marginBottom: '1rem' }}>
                Dual numbers extend real numbers with an infinitesimal unit ε where ε² = 0:
              </P>
              <div style={{ backgroundColor: 'var(--muted)', padding: '1rem', borderRadius: '0.5rem', marginBottom: '1rem' }}>
                <code style={{ fontSize: '0.875rem' }}>
                  x = a + bε
                  <br />
                  where a = function value, b = derivative
                </code>
              </div>
              <ul style={{ listStyleType: 'disc', listStylePosition: 'inside', fontSize: '0.875rem', lineHeight: '1.5' }}>
                <li><strong>Addition</strong>: (a + bε) + (c + dε) = (a + c) + (b + d)ε</li>
                <li><strong>Multiplication</strong>: (a + bε)(c + dε) = ac + (ad + bc)ε</li>
                <li><strong>Chain Rule</strong>: Automatically applied through operations</li>
                <li><strong>No Approximation</strong>: Exact derivatives, not finite differences</li>
              </ul>
              <P style={{ marginTop: '1rem', fontSize: '0.875rem', opacity: 0.7 }}>
                This enables efficient forward-mode automatic differentiation without computational graphs,
                perfect for gradients in neural networks and optimization.
              </P>
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
              <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>Advantages of Dual Numbers</h3>
            </CardHeader>
            <CardBody>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                <div>
                  <h4 style={{ fontWeight: '600', fontSize: '0.875rem', marginBottom: '0.5rem' }}>vs. Numerical Differentiation</h4>
                  <ul style={{ fontSize: '0.875rem', lineHeight: '1.4' }}>
                    <li>✅ Exact (no approximation error)</li>
                    <li>✅ No step size tuning</li>
                    <li>✅ Numerically stable</li>
                  </ul>
                </div>
                <div>
                  <h4 style={{ fontWeight: '600', fontSize: '0.875rem', marginBottom: '0.5rem' }}>vs. Symbolic Differentiation</h4>
                  <ul style={{ fontSize: '0.875rem', lineHeight: '1.4' }}>
                    <li>✅ No expression explosion</li>
                    <li>✅ Works with any code structure</li>
                    <li>✅ Efficient for many variables</li>
                  </ul>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
);
}