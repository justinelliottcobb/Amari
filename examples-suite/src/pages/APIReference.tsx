import { Container, Stack, Card, Title, Text, Button, Badge, SimpleGrid, Code } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
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
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>API Reference</Title>
          <Text size="lg" c="dimmed">
            Complete documentation with real-time mathematical visualizations
          </Text>
        </div>

        <div style={{ display: 'flex', flexDirection: 'row', gap: '1.5rem' }}>
          {/* Navigation Sidebar */}
          <div style={{ minWidth: '280px', maxWidth: '320px' }}>
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={3} size="h4">API Sections</Title>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <Stack gap="xs">
                  {apiSections.map((section) => (
                    <Button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      variant={activeSection === section.id ? 'filled' : 'outline'}
                      fullWidth
                      justify="flex-start"
                      styles={{ inner: { justifyContent: 'flex-start' } }}
                    >
                      <div style={{ textAlign: 'left' }}>
                        <div style={{ fontWeight: 500 }}>{section.title}</div>
                        <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                          {section.methods.length} methods
                        </div>
                      </div>
                    </Button>
                  ))}
                </Stack>
              </Card.Section>
            </Card>

            <Card withBorder mt="lg">
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={3} size="h4">Live Visualizations</Title>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <Stack gap="xs">
                  <Button
                    onClick={() => toggleVisualization('rotor')}
                    variant={visualizations.rotor ? 'filled' : 'outline'}
                    fullWidth
                  >
                    Rotor Evolution
                  </Button>
                  <Button
                    onClick={() => toggleVisualization('tropical')}
                    variant={visualizations.tropical ? 'filled' : 'outline'}
                    fullWidth
                  >
                    Tropical Convergence
                  </Button>
                  <Button
                    onClick={() => toggleVisualization('dual')}
                    variant={visualizations.dual ? 'filled' : 'outline'}
                    fullWidth
                  >
                    Dual Number AD
                  </Button>
                  <Button
                    onClick={() => toggleVisualization('fisher')}
                    variant={visualizations.fisher ? 'filled' : 'outline'}
                    fullWidth
                  >
                    Fisher Information
                  </Button>
                </Stack>
              </Card.Section>
            </Card>
          </div>

          {/* Main Content */}
          <Stack gap="lg" style={{ flex: 1 }}>
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
              <Card withBorder>
                <Card.Section inheritPadding py="xs" bg="dark.6">
                  <Title order={2} size="h3">{currentSection.title}</Title>
                  <Text size="sm" c="dimmed">{currentSection.description}</Text>
                </Card.Section>
                <Card.Section inheritPadding py="md">
                  <Stack gap="xl">
                    {currentSection.methods.map((method, index) => (
                      <div key={index}>
                        <Title order={3} size="h4" mb="xs">{method.name}</Title>
                        <CodeHighlight
                          code={method.signature}
                          language="typescript"
                          mb="sm"
                        />
                        <Text size="sm" c="dimmed" mb="md">
                          {method.description}
                        </Text>

                        {method.parameters && (
                          <div style={{ marginBottom: '1rem' }}>
                            <Title order={4} size="sm" mb="xs">Parameters:</Title>
                            <Stack gap="xs">
                              {method.parameters.map((param, i) => (
                                <div key={i} style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                                  <Badge variant="light" size="sm">
                                    {param.name}: {param.type}
                                  </Badge>
                                  <Text size="sm" c="dimmed">
                                    {param.description}
                                  </Text>
                                </div>
                              ))}
                            </Stack>
                          </div>
                        )}

                        {method.returns && (
                          <div style={{ marginBottom: '1rem' }}>
                            <Title order={4} size="sm" mb="xs">Returns:</Title>
                            <Text size="sm" c="dimmed">{method.returns}</Text>
                          </div>
                        )}

                        <div>
                          <Title order={4} size="sm" mb="xs">Example:</Title>
                          <CodeHighlight
                            code={method.example}
                            language="javascript"
                            withCopyButton
                          />
                        </div>

                        {index < currentSection.methods.length - 1 && (
                          <hr style={{ border: 'none', borderTop: '1px solid var(--mantine-color-dark-4)', marginTop: '1.5rem' }} />
                        )}
                      </div>
                    ))}
                  </Stack>
                </Card.Section>
              </Card>
            )}

            {/* Quick Reference */}
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={3} size="h4">Quick Reference</Title>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
                  <div>
                    <Title order={4} size="sm" mb="xs">Common Patterns</Title>
                    <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                      <li>Create basis vectors for geometric operations</li>
                      <li>Use tropical operations for efficient approximations</li>
                      <li>Apply dual numbers for automatic gradients</li>
                      <li>Combine systems with TDC for neural networks</li>
                    </Text>
                  </div>
                  <div>
                    <Title order={4} size="sm" mb="xs">Performance Tips</Title>
                    <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                      <li>Batch operations when possible</li>
                      <li>Use WebGPU for large computations</li>
                      <li>Prefer tropical approximations for softmax</li>
                      <li>Cache multivector calculations</li>
                    </Text>
                  </div>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </Stack>
        </div>
      </Stack>
    </Container>
  );
}
