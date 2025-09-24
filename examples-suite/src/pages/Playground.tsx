import { H1, P, Card, CardHeader, CardBody, Button, Code } from "jadis-ui";
import { useState, useCallback, useEffect } from "react";

// Predefined code templates
const codeTemplates = {
  geometric: `// Geometric Algebra Example
const mv1 = amari.WasmMultivector.basisVector(0); // e1
const mv2 = amari.WasmMultivector.basisVector(1); // e2

// Geometric product
const product = mv1.geometricProduct(mv2);

// Display results
console.log('e1 * e2 =', product.getCoefficients());
console.log('e12 component:', product.getCoefficient(4));

return { product: product.getCoefficients() };`,

  tropical: `// Tropical Algebra Example
function tropicalAdd(a, b) {
  return Math.max(a, b);
}

function tropicalMultiply(a, b) {
  return a + b;
}

const a = 3.5, b = 2.1, c = 4.7;

// Tropical operations
const sum = tropicalAdd(tropicalAdd(a, b), c);
const product = tropicalMultiply(a, tropicalMultiply(b, c));

console.log('Tropical sum (max):', sum);
console.log('Tropical product (addition):', product);

// Tropical matrix multiplication for shortest path
const matrix = [
  [0, 3, Infinity],
  [2, 0, 1],
  [Infinity, 4, 0]
];

function tropicalMatrixMultiply(A, B) {
  const n = A.length;
  const result = Array(n).fill(null).map(() => Array(n).fill(Infinity));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        result[i][j] = Math.min(result[i][j], A[i][k] + B[k][j]);
      }
    }
  }
  return result;
}

const result = tropicalMatrixMultiply(matrix, matrix);
console.log('Shortest paths (2 hops):', result);

return { sum, product, shortestPaths: result };`,

  dual: `// Dual Number Automatic Differentiation
class DualNumber {
  constructor(real, dual) {
    this.real = real;
    this.dual = dual;
  }

  static variable(x) {
    return new DualNumber(x, 1);
  }

  static constant(x) {
    return new DualNumber(x, 0);
  }

  add(other) {
    return new DualNumber(
      this.real + other.real,
      this.dual + other.dual
    );
  }

  multiply(other) {
    return new DualNumber(
      this.real * other.real,
      this.real * other.dual + this.dual * other.real
    );
  }

  sin() {
    return new DualNumber(
      Math.sin(this.real),
      Math.cos(this.real) * this.dual
    );
  }

  toString() {
    return \`\${this.real.toFixed(3)} + \${this.dual.toFixed(3)}ε\`;
  }
}

// Compute f(x) = x² + sin(x) and its derivative at x = π/4
const x = DualNumber.variable(Math.PI / 4);

// f(x) = x² + sin(x)
const x_squared = x.multiply(x);
const sin_x = x.sin();
const f_x = x_squared.add(sin_x);

console.log('x =', x.toString());
console.log('f(x) = x² + sin(x) =', f_x.toString());
console.log('Value at x:', f_x.real.toFixed(6));
console.log("f'(x) =", f_x.dual.toFixed(6));

// Verify manually: f'(x) = 2x + cos(x)
const manual = 2 * (Math.PI / 4) + Math.cos(Math.PI / 4);
console.log('Manual derivative:', manual.toFixed(6));

return {
  value: f_x.real,
  derivative: f_x.dual,
  manual: manual,
  match: Math.abs(f_x.dual - manual) < 1e-10
};`,

  fusion: `// TropicalDualClifford Fusion System
class TDC {
  constructor(tropical, dual, clifford) {
    this.tropical = tropical;
    this.dual = dual;
    this.clifford = clifford;
  }

  static fromLogits(logits) {
    const maxLogit = Math.max(...logits);
    const tropical = logits.map(x => x - maxLogit);

    const dual = logits.map((x, i) => ({
      real: x,
      dual: i === 0 ? 1 : 0
    }));

    const norm = Math.sqrt(logits.reduce((s, x) => s + x*x, 0));
    const clifford = [norm, ...logits.slice(0, 3), 0, 0, 0, 0];

    return new TDC(tropical, dual, clifford);
  }

  tropicalAttention(keys) {
    const scores = this.tropical.map((q, i) => q + (keys[i] || 0));
    const maxScore = Math.max(...scores);
    return scores.map(s => s - maxScore);
  }

  gradient() {
    return this.dual.map(d => d.dual);
  }

  geometricNorm() {
    return Math.sqrt(this.clifford.slice(0, 4).reduce((s, x) => s + x*x, 0));
  }
}

// Example: Attention mechanism with TDC
const query = [1.2, 0.8, -0.5, 0.3];
const keys = [0.5, 1.0, 0.2, -0.1];

const tdc = TDC.fromLogits(query);

const attention = tdc.tropicalAttention(keys);
const gradients = tdc.gradient();
const geoNorm = tdc.geometricNorm();

console.log('Query logits:', query);
console.log('Tropical attention:', attention.map(x => x.toFixed(3)));
console.log('Gradients:', gradients);
console.log('Geometric norm:', geoNorm.toFixed(3));

// Performance comparison
const iterations = 1000;
const start = performance.now();

for (let i = 0; i < iterations; i++) {
  const scores = query.map((q, j) => q + keys[j]);
  const maxScore = Math.max(...scores);
  const result = scores.map(s => s - maxScore);
}

const tropicalTime = performance.now() - start;

const startSoftmax = performance.now();
for (let i = 0; i < iterations; i++) {
  const scores = query.map((q, j) => q + keys[j]);
  const expScores = scores.map(s => Math.exp(s));
  const sum = expScores.reduce((a, b) => a + b, 0);
  const result = expScores.map(e => e / sum);
}
const softmaxTime = performance.now() - startSoftmax;

const speedup = softmaxTime / tropicalTime;

console.log(\`Tropical: \${tropicalTime.toFixed(2)}ms, Softmax: \${softmaxTime.toFixed(2)}ms\`);
console.log(\`Speedup: \${speedup.toFixed(2)}x\`);

return {
  attention,
  gradients,
  geoNorm,
  speedup
};`
};

export function Playground() {
  const [code, setCode] = useState(codeTemplates.geometric);
  const [output, setOutput] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [amariLoaded, setAmariLoaded] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState("geometric");

  useEffect(() => {
    // Simulate Amari module loading
    setTimeout(() => setAmariLoaded(true), 100);
  }, []);

  const runCode = useCallback(async () => {
    setIsRunning(true);
    setError(null);
    setOutput("");

    try {
      // Create a console proxy to capture output
      const logs: string[] = [];
      const consoleProxy = {
        log: (...args: any[]) => {
          logs.push(args.map(arg =>
            typeof arg === 'object' ? JSON.stringify(arg, null, 2) : String(arg)
          ).join(' '));
        }
      };

      // Create mock Amari module for demonstration
      const amari = {
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
              const result = [0, 0, 0, 0, 1, 0, 0, 0]; // e12 for e1 * e2
              return {
                getCoefficients: () => result,
                getCoefficient: (i: number) => result[i]
              };
            }
          })
        }
      };

      // Execute user code in a sandboxed function
      const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
      const userFunction = new AsyncFunction('console', 'amari', 'performance', code);

      const result = await userFunction(consoleProxy, amari, performance);

      // Set output
      if (logs.length > 0) {
        setOutput(logs.join('\n'));
      }

      if (result !== undefined) {
        setOutput(prev => prev + (prev ? '\n\n' : '') + 'Returned: ' + JSON.stringify(result, null, 2));
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsRunning(false);
    }
  }, [code]);

  const loadTemplate = useCallback((templateName: string) => {
    setSelectedTemplate(templateName);
    setCode(codeTemplates[templateName as keyof typeof codeTemplates]);
    setOutput("");
    setError(null);
  }, []);

  return (
<div className="p-8">
        <div className="max-w-6xl mx-auto">
          <H1>Interactive Playground</H1>
          <P className="text-lg text-muted-foreground mb-6">
            Experiment with Amari mathematical operations in real-time
          </P>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Code Editor Section */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Code Editor</h3>
                    <div className="flex gap-2">
                      <Button
                        onClick={runCode}
                        disabled={isRunning || !amariLoaded}
                        size="sm"
                      >
                        {isRunning ? 'Running...' : 'Run Code'}
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardBody>
                  <textarea
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    className="w-full h-96 p-4 font-mono text-sm bg-muted rounded-lg border border-border focus:ring-2 focus:ring-primary focus:border-primary"
                    spellCheck={false}
                  />
                </CardBody>
              </Card>

              {/* Output Section */}
              <Card className="mt-6">
                <CardHeader>
                  <h3 className="text-lg font-semibold">Output</h3>
                </CardHeader>
                <CardBody>
                  {error ? (
                    <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
                      <Code className="text-sm text-destructive whitespace-pre-wrap">
                        Error: {error}
                      </Code>
                    </div>
                  ) : output ? (
                    <div className="bg-muted rounded-lg p-4">
                      <Code className="text-sm whitespace-pre-wrap">
                        {output}
                      </Code>
                    </div>
                  ) : (
                    <div className="text-muted-foreground text-sm">
                      No output yet. Click "Run Code" to execute.
                    </div>
                  )}
                </CardBody>
              </Card>
            </div>

            {/* Templates Section */}
            <div>
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">Templates</h3>
                </CardHeader>
                <CardBody>
                  <div className="space-y-3">
                    <Button
                      onClick={() => loadTemplate('geometric')}
                      variant={selectedTemplate === 'geometric' ? 'default' : 'outline'}
                      className="w-full justify-start"
                      size="sm"
                    >
                      Geometric Algebra
                    </Button>
                    <Button
                      onClick={() => loadTemplate('tropical')}
                      variant={selectedTemplate === 'tropical' ? 'default' : 'outline'}
                      className="w-full justify-start"
                      size="sm"
                    >
                      Tropical Algebra
                    </Button>
                    <Button
                      onClick={() => loadTemplate('dual')}
                      variant={selectedTemplate === 'dual' ? 'default' : 'outline'}
                      className="w-full justify-start"
                      size="sm"
                    >
                      Dual Numbers AD
                    </Button>
                    <Button
                      onClick={() => loadTemplate('fusion')}
                      variant={selectedTemplate === 'fusion' ? 'default' : 'outline'}
                      className="w-full justify-start"
                      size="sm"
                    >
                      TDC Fusion System
                    </Button>
                  </div>
                </CardBody>
              </Card>

              <Card className="mt-6">
                <CardHeader>
                  <h3 className="text-lg font-semibold">Quick Reference</h3>
                </CardHeader>
                <CardBody>
                  <div className="space-y-3 text-sm">
                    <div>
                      <h4 className="font-semibold mb-1">Available Objects</h4>
                      <ul className="space-y-1 text-xs">
                        <li><Code>amari</Code> - WASM module</li>
                        <li><Code>console.log()</Code> - Output</li>
                        <li><Code>performance</Code> - Timing</li>
                      </ul>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-1">Return Values</h4>
                      <p className="text-xs text-muted-foreground">
                        Return an object to display structured data in the output
                      </p>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-1">Tips</h4>
                      <ul className="space-y-1 text-xs text-muted-foreground">
                        <li>• Use templates as starting points</li>
                        <li>• Console output appears immediately</li>
                        <li>• Return values show as JSON</li>
                        <li>• Errors are safely caught</li>
                      </ul>
                    </div>
                  </div>
                </CardBody>
              </Card>

              <Card className="mt-6">
                <CardHeader>
                  <h3 className="text-lg font-semibold">Status</h3>
                </CardHeader>
                <CardBody>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Amari Module</span>
                      <span className={`px-2 py-0.5 rounded text-xs ${
                        amariLoaded ? 'bg-green-100 text-green-800' : 'bg-gray-100'
                      }`}>
                        {amariLoaded ? 'Loaded' : 'Loading...'}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>WebAssembly</span>
                      <span className="px-2 py-0.5 bg-green-100 text-green-800 rounded text-xs">
                        Available
                      </span>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </div>
          </div>
        </div>
      </div>
);
}