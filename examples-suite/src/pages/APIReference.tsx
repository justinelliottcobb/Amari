import { H1, P, Card, CardHeader, CardBody, Button, CodeBlock, H2, H3, H4, StatusBadge } from "jadis-ui";
import { RealTimeVisualization } from "../components/RealTimeDisplay";
import { useState } from "react";

interface ApiSection {
  id: string;
  title: string;
  description: string;
  methods: ApiMethod[];
}

interface ApiMethod {
  name: string;
  signature: string;
  description: string;
  example: string;
  parameters?: { name: string; type: string; description: string; }[];
  returns?: string;
}

const apiSections: ApiSection[] = [
  {
    id: "geometric",
    title: "Geometric Algebra",
    description: "Multivector operations and geometric product computations",
    methods: [
      {
        name: "WasmMultivector.basisVector",
        signature: "static basisVector(index: number): WasmMultivector",
        description: "Create a basis vector (e1, e2, e3, etc.)",
        parameters: [
          { name: "index", type: "number", description: "Basis vector index (0-2 for 3D)" }
        ],
        returns: "WasmMultivector representing the basis vector",
        example: `const e1 = amari.WasmMultivector.basisVector(0);
const e2 = amari.WasmMultivector.basisVector(1);

// Get coefficients [scalar, e1, e2, e3, e12, e13, e23, e123]
console.log(e1.getCoefficients()); // [0, 1, 0, 0, 0, 0, 0, 0]`
      },
      {
        name: "geometricProduct",
        signature: "geometricProduct(other: WasmMultivector): WasmMultivector",
        description: "Compute the geometric product of two multivectors",
        parameters: [
          { name: "other", type: "WasmMultivector", description: "The multivector to multiply with" }
        ],
        returns: "WasmMultivector result of the geometric product",
        example: `const e1 = amari.WasmMultivector.basisVector(0);
const e2 = amari.WasmMultivector.basisVector(1);

// e1 * e2 = e12 (bivector)
const e12 = e1.geometricProduct(e2);
console.log(e12.getCoefficient(4)); // 1.0 (e12 component)`
      },
      {
        name: "innerProduct",
        signature: "innerProduct(other: WasmMultivector): WasmMultivector",
        description: "Compute the inner product (contraction)",
        parameters: [
          { name: "other", type: "WasmMultivector", description: "The multivector to contract with" }
        ],
        returns: "WasmMultivector result of the inner product",
        example: `const v1 = amari.WasmMultivector.fromCoefficients([0, 1, 2, 3, 0, 0, 0, 0]);
const v2 = amari.WasmMultivector.fromCoefficients([0, 4, 5, 6, 0, 0, 0, 0]);

const inner = v1.innerProduct(v2);
console.log(inner.getCoefficient(0)); // Scalar result: 1*4 + 2*5 + 3*6 = 32`
      }
    ]
  },
  {
    id: "tropical",
    title: "Tropical Algebra",
    description: "Max-plus semiring operations for efficient computation",
    methods: [
      {
        name: "tropicalAdd",
        signature: "function tropicalAdd(a: number, b: number): number",
        description: "Tropical addition: maximum of two values",
        parameters: [
          { name: "a", type: "number", description: "First value" },
          { name: "b", type: "number", description: "Second value" }
        ],
        returns: "number - the maximum of a and b",
        example: `function tropicalAdd(a, b) {
  return Math.max(a, b);
}

console.log(tropicalAdd(3.5, 2.1)); // 3.5
console.log(tropicalAdd(-1, 0));    // 0`
      },
      {
        name: "tropicalMultiply",
        signature: "function tropicalMultiply(a: number, b: number): number",
        description: "Tropical multiplication: sum of two values",
        parameters: [
          { name: "a", type: "number", description: "First value" },
          { name: "b", type: "number", description: "Second value" }
        ],
        returns: "number - the sum of a and b",
        example: `function tropicalMultiply(a, b) {
  return a + b;
}

console.log(tropicalMultiply(3.5, 2.1)); // 5.6
console.log(tropicalMultiply(0, -1));    // -1`
      }
    ]
  },
  {
    id: "dual",
    title: "Dual Numbers",
    description: "Automatic differentiation with dual number arithmetic",
    methods: [
      {
        name: "DualNumber.variable",
        signature: "static variable(value: number): DualNumber",
        description: "Create a variable for differentiation (dual part = 1)",
        parameters: [
          { name: "value", type: "number", description: "The real part value" }
        ],
        returns: "DualNumber with dual part set to 1",
        example: `const x = DualNumber.variable(2.0); // 2 + 1ε
console.log(x.real); // 2.0
console.log(x.dual); // 1.0`
      },
      {
        name: "multiply",
        signature: "multiply(other: DualNumber): DualNumber",
        description: "Multiply two dual numbers with automatic differentiation",
        parameters: [
          { name: "other", type: "DualNumber", description: "The dual number to multiply with" }
        ],
        returns: "DualNumber with computed derivative",
        example: `const x = DualNumber.variable(3.0);
const c = DualNumber.constant(2.0);

// d/dx(x * 2) = 2
const result = x.multiply(c);
console.log(result.real); // 6.0
console.log(result.dual); // 2.0`
      }
    ]
  },
  {
    id: "fusion",
    title: "TropicalDualClifford",
    description: "Unified framework combining all three algebraic systems",
    methods: [
      {
        name: "TDC.fromLogits",
        signature: "static fromLogits(logits: number[]): TropicalDualClifford",
        description: "Create TDC object from neural network logits",
        parameters: [
          { name: "logits", type: "number[]", description: "Array of logit values" }
        ],
        returns: "TropicalDualClifford with unified representation",
        example: `const logits = [1.2, 0.8, -0.5, 0.3];
const tdc = TropicalDualClifford.fromLogits(logits);

console.log(tdc.tropical);  // Max-plus normalized
console.log(tdc.dual);      // Gradient-ready
console.log(tdc.clifford);  // Geometric structure`
      }
    ]
  }
];

export function APIReference() {
  const [activeSection, setActiveSection] = useState("geometric");
  const [visualizations, setVisualizations] = useState({
    rotor: false,
    tropical: false,
    dual: false,
    fisher: false
  });

  const toggleVisualization = (type: keyof typeof visualizations) => {
    setVisualizations(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  const currentSection = apiSections.find(section => section.id === activeSection);

  return (
    <div style={{ padding: '2rem' }}>
      <div style={{ maxWidth: '1536px', margin: '0 auto' }}>
        <H1>API Reference</H1>
        <P style={{ fontSize: '1.125rem', marginBottom: '1.5rem', opacity: 0.7 }}>
          Complete documentation with real-time mathematical visualizations
        </P>

        <div style={{ display: 'flex', flexDirection: 'row', gap: '1.5rem' }}>
          {/* Navigation Sidebar */}
          <div style={{ minWidth: '280px', maxWidth: '320px' }}>
            <Card>
              <CardHeader>
                <H3>API Sections</H3>
              </CardHeader>
              <CardBody>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {apiSections.map((section) => (
                    <Button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      variant={activeSection === section.id ? 'default' : 'outline'}
                      style={{ width: '100%', justifyContent: 'flex-start', textAlign: 'left' }}
                    >
                      <div>
                        <div style={{ fontWeight: '500' }}>{section.title}</div>
                        <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                          {section.methods.length} methods
                        </div>
                      </div>
                    </Button>
                  ))}
                </div>
              </CardBody>
            </Card>

            <Card style={{ marginTop: '1.5rem' }}>
              <CardHeader>
                <H3>Live Visualizations</H3>
              </CardHeader>
              <CardBody>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <Button
                    onClick={() => toggleVisualization('rotor')}
                    variant={visualizations.rotor ? 'default' : 'outline'}
                    style={{ width: '100%' }}
                  >
                    Rotor Evolution
                  </Button>
                  <Button
                    onClick={() => toggleVisualization('tropical')}
                    variant={visualizations.tropical ? 'default' : 'outline'}
                    style={{ width: '100%' }}
                  >
                    Tropical Convergence
                  </Button>
                  <Button
                    onClick={() => toggleVisualization('dual')}
                    variant={visualizations.dual ? 'default' : 'outline'}
                    style={{ width: '100%' }}
                  >
                    Dual Number AD
                  </Button>
                  <Button
                    onClick={() => toggleVisualization('fisher')}
                    variant={visualizations.fisher ? 'default' : 'outline'}
                    style={{ width: '100%' }}
                  >
                    Fisher Information
                  </Button>
                </div>
              </CardBody>
            </Card>
          </div>

          {/* Main Content */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', flex: 1 }}>
            {/* Real-time Visualizations */}
            {visualizations.rotor && (
              <RealTimeVisualization
                title="Rotor Evolution"
                description="Real-time visualization of geometric rotor rotations"
                type="rotor"
                isRunning={visualizations.rotor}
                onToggle={() => toggleVisualization('rotor')}
              />
            )}

            {visualizations.tropical && (
              <RealTimeVisualization
                title="Tropical Convergence"
                description="Live tropical algebra convergence to maximum value"
                type="tropical"
                isRunning={visualizations.tropical}
                onToggle={() => toggleVisualization('tropical')}
              />
            )}

            {visualizations.dual && (
              <RealTimeVisualization
                title="Dual Number Automatic Differentiation"
                description="Real-time function and derivative computation"
                type="dual"
                isRunning={visualizations.dual}
                onToggle={() => toggleVisualization('dual')}
              />
            )}

            {visualizations.fisher && (
              <RealTimeVisualization
                title="Fisher Information Matrix"
                description="Live Fisher information evolution on probability simplex"
                type="fisher"
                isRunning={visualizations.fisher}
                onToggle={() => toggleVisualization('fisher')}
              />
            )}

            {/* API Documentation */}
            {currentSection && (
              <Card>
                <CardHeader>
                  <H2>{currentSection.title}</H2>
                  <P style={{ opacity: 0.7 }}>{currentSection.description}</P>
                </CardHeader>
                <CardBody>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
                    {currentSection.methods.map((method, index) => (
                      <div key={index} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div>
                          <H3 style={{ marginBottom: '0.5rem' }}>{method.name}</H3>
                          <CodeBlock
                            language="typescript"
                            showCopyButton={true}
                            style={{ marginBottom: '0.75rem', width: '100%' }}
                          >
                            {method.signature}
                          </CodeBlock>
                          <P style={{ fontSize: '0.875rem', marginBottom: '0.75rem', opacity: 0.7 }}>
                            {method.description}
                          </P>
                        </div>

                        {method.parameters && (
                          <div>
                            <H4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Parameters:</H4>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                              {method.parameters.map((param, i) => (
                                <div key={i} style={{ display: 'flex', gap: '1rem', fontSize: '0.875rem' }}>
                                  <StatusBadge variant="muted">
                                    {param.name}: {param.type}
                                  </StatusBadge>
                                  <span style={{ opacity: 0.7 }}>
                                    {param.description}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {method.returns && (
                          <div>
                            <H4 style={{ fontSize: '0.875rem', marginBottom: '0.25rem' }}>Returns:</H4>
                            <P style={{ fontSize: '0.875rem', opacity: 0.7 }}>{method.returns}</P>
                          </div>
                        )}

                        <div>
                          <H4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Example:</H4>
                          <CodeBlock
                            language="javascript"
                            showLineNumbers={true}
                            showCopyButton={true}
                            style={{ width: '100%' }}
                          >
                            {method.example}
                          </CodeBlock>
                        </div>

                        {index < currentSection.methods.length - 1 && (
                          <hr style={{ border: 'none', borderTop: '1px solid var(--border)' }} />
                        )}
                      </div>
                    ))}
                  </div>
                </CardBody>
              </Card>
            )}

            {/* Quick Reference */}
            <Card>
              <CardHeader>
                <H3>Quick Reference</H3>
              </CardHeader>
              <CardBody>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <H4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Common Patterns</H4>
                    <ul style={{ fontSize: '0.875rem', opacity: 0.7, listStyle: 'none', padding: 0 }}>
                      <li>• Create basis vectors for geometric operations</li>
                      <li>• Use tropical operations for efficient approximations</li>
                      <li>• Apply dual numbers for automatic gradients</li>
                      <li>• Combine systems with TDC for neural networks</li>
                    </ul>
                  </div>
                  <div>
                    <H4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>Performance Tips</H4>
                    <ul style={{ fontSize: '0.875rem', opacity: 0.7, listStyle: 'none', padding: 0 }}>
                      <li>• Batch operations when possible</li>
                      <li>• Use WebGPU for large computations</li>
                      <li>• Prefer tropical approximations for softmax</li>
                      <li>• Cache multivector calculations</li>
                    </ul>
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