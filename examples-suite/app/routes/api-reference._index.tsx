import type { MetaFunction } from "@remix-run/node";
import { H1, P, Card, CardHeader, CardBody, Button, Code } from "jadis-ui";
import { Layout } from "~/components/Layout";
import { RealTimeVisualization } from "~/components/RealTimeDisplay";
import { useState } from "react";

export const meta: MetaFunction = () => {
  return [
    { title: "API Reference - Amari Library" },
    { name: "description", content: "Complete API documentation with real-time mathematical visualizations" },
  ];
};

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

export default function ApiReference() {
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
    <Layout>
      <div className="p-8">
        <div className="max-w-6xl mx-auto">
          <H1>API Reference</H1>
          <P className="text-lg text-muted-foreground mb-6">
            Complete documentation with real-time mathematical visualizations
          </P>

          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Navigation Sidebar */}
            <div className="lg:col-span-1">
              <Card>
                <CardHeader>
                  <h3 className="text-lg font-semibold">API Sections</h3>
                </CardHeader>
                <CardBody>
                  <div className="space-y-2">
                    {apiSections.map((section) => (
                      <Button
                        key={section.id}
                        onClick={() => setActiveSection(section.id)}
                        variant={activeSection === section.id ? 'default' : 'outline'}
                        className="w-full justify-start text-left"
                        size="sm"
                      >
                        <div>
                          <div className="font-medium">{section.title}</div>
                          <div className="text-xs opacity-70">
                            {section.methods.length} methods
                          </div>
                        </div>
                      </Button>
                    ))}
                  </div>
                </CardBody>
              </Card>

              <Card className="mt-6">
                <CardHeader>
                  <h3 className="text-lg font-semibold">Live Visualizations</h3>
                </CardHeader>
                <CardBody>
                  <div className="space-y-2">
                    <Button
                      onClick={() => toggleVisualization('rotor')}
                      variant={visualizations.rotor ? 'default' : 'outline'}
                      size="sm"
                      className="w-full"
                    >
                      Rotor Evolution
                    </Button>
                    <Button
                      onClick={() => toggleVisualization('tropical')}
                      variant={visualizations.tropical ? 'default' : 'outline'}
                      size="sm"
                      className="w-full"
                    >
                      Tropical Convergence
                    </Button>
                    <Button
                      onClick={() => toggleVisualization('dual')}
                      variant={visualizations.dual ? 'default' : 'outline'}
                      size="sm"
                      className="w-full"
                    >
                      Dual Number AD
                    </Button>
                    <Button
                      onClick={() => toggleVisualization('fisher')}
                      variant={visualizations.fisher ? 'default' : 'outline'}
                      size="sm"
                      className="w-full"
                    >
                      Fisher Information
                    </Button>
                  </div>
                </CardBody>
              </Card>
            </div>

            {/* Main Content */}
            <div className="lg:col-span-3 space-y-6">
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
                    <h2 className="text-xl font-semibold">{currentSection.title}</h2>
                    <p className="text-muted-foreground">{currentSection.description}</p>
                  </CardHeader>
                  <CardBody>
                    <div className="space-y-8">
                      {currentSection.methods.map((method, index) => (
                        <div key={index} className="space-y-4">
                          <div>
                            <h3 className="text-lg font-semibold mb-2">{method.name}</h3>
                            <div className="bg-muted p-3 rounded-lg mb-3">
                              <Code className="text-sm">{method.signature}</Code>
                            </div>
                            <p className="text-sm text-muted-foreground mb-3">
                              {method.description}
                            </p>
                          </div>

                          {method.parameters && (
                            <div>
                              <h4 className="font-medium text-sm mb-2">Parameters:</h4>
                              <div className="space-y-2">
                                {method.parameters.map((param, i) => (
                                  <div key={i} className="flex gap-4 text-sm">
                                    <Code className="text-xs bg-muted px-2 py-1 rounded">
                                      {param.name}: {param.type}
                                    </Code>
                                    <span className="text-muted-foreground">
                                      {param.description}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {method.returns && (
                            <div>
                              <h4 className="font-medium text-sm mb-1">Returns:</h4>
                              <p className="text-sm text-muted-foreground">{method.returns}</p>
                            </div>
                          )}

                          <div>
                            <h4 className="font-medium text-sm mb-2">Example:</h4>
                            <div className="bg-muted p-4 rounded-lg overflow-x-auto">
                              <Code className="text-sm whitespace-pre">{method.example}</Code>
                            </div>
                          </div>

                          {index < currentSection.methods.length - 1 && (
                            <hr className="border-border" />
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
                  <h3 className="text-lg font-semibold">Quick Reference</h3>
                </CardHeader>
                <CardBody>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium text-sm mb-2">Common Patterns</h4>
                      <ul className="text-sm space-y-1 text-muted-foreground">
                        <li>• Create basis vectors for geometric operations</li>
                        <li>• Use tropical operations for efficient approximations</li>
                        <li>• Apply dual numbers for automatic gradients</li>
                        <li>• Combine systems with TDC for neural networks</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium text-sm mb-2">Performance Tips</h4>
                      <ul className="text-sm space-y-1 text-muted-foreground">
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
    </Layout>
  );
}