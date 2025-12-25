import { Container, Stack, Card, Title, Text, SimpleGrid, Code, Badge } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ExampleCard } from "../components/ExampleCard";

export function Calculus() {
  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  // Helper for numerical derivatives
  const numericalGradient = (f: (x: number[]) => number, point: number[], h = 1e-5): number[] => {
    const n = point.length;
    const grad = [];
    for (let i = 0; i < n; i++) {
      const xPlus = [...point];
      const xMinus = [...point];
      xPlus[i] += h;
      xMinus[i] -= h;
      grad.push((f(xPlus) - f(xMinus)) / (2 * h));
    }
    return grad;
  };

  // Simpson's rule integration
  const simpson = (f: (x: number) => number, a: number, b: number, n: number): number => {
    if (n % 2 !== 0) n++;
    const h = (b - a) / n;
    let sum = f(a) + f(b);
    for (let i = 1; i < n; i++) {
      const x = a + i * h;
      sum += (i % 2 === 0 ? 2 : 4) * f(x);
    }
    return (h / 3) * sum;
  };

  const examples = [
    {
      title: "Scalar Field Gradient",
      description: "Compute the gradient of a scalar field using numerical derivatives",
      category: "Differential",
      code: `// Define a scalar field f(x, y, z) = x² + y² + z²
const field = ScalarField.fromFunction3D((x, y, z) => x*x + y*y + z*z);

// Create numerical derivative calculator
const derivative = new NumericalDerivative(1e-5);  // step size

// Compute gradient at point (1, 2, 3)
const point = [1, 2, 3];
const gradient = derivative.gradient(field, point);

console.log("Gradient at (1,2,3):", gradient);
// Expected: [2, 4, 6] = [2x, 2y, 2z]

// The gradient points in the direction of steepest ascent
const magnitude = Math.sqrt(gradient.reduce((s, g) => s + g*g, 0));
console.log("Gradient magnitude:", magnitude);`,
      onRun: simulateExample(() => {
        const f = (p: number[]) => p[0]**2 + p[1]**2 + p[2]**2;
        const point = [1, 2, 3];
        const grad = numericalGradient(f, point);
        const magnitude = Math.sqrt(grad.reduce((s, g) => s + g*g, 0));

        return [
          "Scalar field: f(x,y,z) = x² + y² + z²",
          "",
          `Point: (${point.join(', ')})`,
          `Gradient: [${grad.map(g => g.toFixed(4)).join(', ')}]`,
          `Expected: [2, 4, 6]`,
          "",
          `Gradient magnitude: ${magnitude.toFixed(4)}`,
          "The gradient points toward increasing function values"
        ].join('\n');
      })
    },
    {
      title: "Vector Field Divergence",
      description: "Compute the divergence of a vector field: ∇·F",
      category: "Differential",
      code: `// Define a vector field F(x,y,z) = [x, y, z]
const field = VectorField.fromComponents3D(
  (x, y, z) => x,  // Fx
  (x, y, z) => y,  // Fy
  (x, y, z) => z   // Fz
);

// Compute divergence: ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z
const derivative = new NumericalDerivative();
const divergence = derivative.divergence(field, [1, 2, 3]);

console.log("Divergence:", divergence);
// Expected: 3 (since ∂x/∂x + ∂y/∂y + ∂z/∂z = 1 + 1 + 1)`,
      onRun: simulateExample(() => {
        // F = [x, y, z]
        // div F = ∂x/∂x + ∂y/∂y + ∂z/∂z = 1 + 1 + 1 = 3
        const h = 1e-5;

        // Compute each partial derivative numerically
        const Fx = (p: number[]) => p[0];
        const Fy = (p: number[]) => p[1];
        const Fz = (p: number[]) => p[2];

        const point = [1, 2, 3];

        const dFx_dx = (Fx([point[0]+h, point[1], point[2]]) - Fx([point[0]-h, point[1], point[2]])) / (2*h);
        const dFy_dy = (Fy([point[0], point[1]+h, point[2]]) - Fy([point[0], point[1]-h, point[2]])) / (2*h);
        const dFz_dz = (Fz([point[0], point[1], point[2]+h]) - Fz([point[0], point[1], point[2]-h])) / (2*h);

        const divergence = dFx_dx + dFy_dy + dFz_dz;

        return [
          "Vector field: F(x,y,z) = [x, y, z]",
          "",
          `Point: (${point.join(', ')})`,
          `∂Fx/∂x = ${dFx_dx.toFixed(4)}`,
          `∂Fy/∂y = ${dFy_dy.toFixed(4)}`,
          `∂Fz/∂z = ${dFz_dz.toFixed(4)}`,
          "",
          `Divergence (∇·F) = ${divergence.toFixed(4)}`,
          "Expected: 3.0"
        ].join('\n');
      })
    },
    {
      title: "Vector Field Curl",
      description: "Compute the curl of a vector field: ∇×F",
      category: "Differential",
      code: `// Define a rotating vector field F(x,y,z) = [-y, x, 0]
const field = VectorField.fromComponents3D(
  (x, y, z) => -y,  // Fx
  (x, y, z) => x,   // Fy
  (x, y, z) => 0    // Fz
);

// Compute curl at a point
const derivative = new NumericalDerivative();
const curl = derivative.curl(field, [1, 2, 3]);

console.log("Curl:", curl);
// Expected: [0, 0, 2] (rotation around z-axis)
// curl = [∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y]`,
      onRun: simulateExample(() => {
        // F = [-y, x, 0]
        // curl F = [∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y]
        //        = [0 - 0, 0 - 0, 1 - (-1)] = [0, 0, 2]

        const h = 1e-5;
        const point = [1, 2, 3];

        const Fx = (x: number, y: number, _z: number) => -y;
        const Fy = (x: number, _y: number, _z: number) => x;
        const Fz = (_x: number, _y: number, _z: number) => 0;

        // ∂Fy/∂x
        const dFy_dx = (Fy(point[0]+h, point[1], point[2]) - Fy(point[0]-h, point[1], point[2])) / (2*h);
        // ∂Fx/∂y
        const dFx_dy = (Fx(point[0], point[1]+h, point[2]) - Fx(point[0], point[1]-h, point[2])) / (2*h);

        const curlZ = dFy_dx - dFx_dy;

        return [
          "Vector field: F(x,y,z) = [-y, x, 0]",
          "(Describes rotation around z-axis)",
          "",
          `Point: (${point.join(', ')})`,
          `∂Fy/∂x = ${dFy_dx.toFixed(4)}`,
          `∂Fx/∂y = ${dFx_dy.toFixed(4)}`,
          "",
          `Curl (∇×F) = [0, 0, ${curlZ.toFixed(4)}]`,
          "Expected: [0, 0, 2]",
          "",
          "The curl points along the axis of rotation"
        ].join('\n');
      })
    },
    {
      title: "1D Integration (Simpson's Rule)",
      description: "Compute definite integrals using numerical quadrature",
      category: "Integration",
      code: `// Integrate sin(x) from 0 to π
const result = Integration.integrate1D(
  x => Math.sin(x),
  0,          // lower bound
  Math.PI,    // upper bound
  100         // subdivisions
);

console.log("∫sin(x)dx from 0 to π:", result);
// Expected: 2.0 (since -cos(π) - (-cos(0)) = 1 - (-1) = 2)

// Compare with different methods
const riemann = integrate(x => Math.sin(x), 0, Math.PI, 1000, WasmIntegrationMethod.Riemann);
const simpson = integrate(x => Math.sin(x), 0, Math.PI, 100, WasmIntegrationMethod.Simpson);
const adaptive = integrate(x => Math.sin(x), 0, Math.PI, 50, WasmIntegrationMethod.Adaptive);`,
      onRun: simulateExample(() => {
        const f = (x: number) => Math.sin(x);
        const a = 0;
        const b = Math.PI;

        const result = simpson(f, a, b, 100);
        const exact = 2.0;  // -cos(π) - (-cos(0)) = 2

        return [
          "Integral: ∫₀^π sin(x) dx",
          "",
          `Simpson's rule (n=100): ${result.toFixed(10)}`,
          `Exact value: ${exact.toFixed(10)}`,
          `Error: ${Math.abs(result - exact).toExponential(4)}`,
          "",
          "Simpson's rule achieves high accuracy with few points"
        ].join('\n');
      })
    },
    {
      title: "2D Integration",
      description: "Compute double integrals over rectangular regions",
      category: "Integration",
      code: `// Integrate f(x,y) = x*y over [0,1] × [0,1]
const result = Integration.integrate2D(
  (x, y) => x * y,
  0, 1,    // x bounds
  0, 1,    // y bounds
  50, 50   // subdivisions
);

console.log("∬xy dA over [0,1]²:", result);
// Expected: 1/4 = 0.25

// Integrate x² + y² over unit square
const result2 = Integration.integrate2D(
  (x, y) => x*x + y*y,
  0, 1,
  0, 1,
  50, 50
);
console.log("∬(x²+y²) dA:", result2);
// Expected: 2/3 ≈ 0.667`,
      onRun: simulateExample(() => {
        // 2D Simpson's rule
        const simpson2D = (
          f: (x: number, y: number) => number,
          ax: number, bx: number,
          ay: number, by: number,
          nx: number, ny: number
        ): number => {
          const hx = (bx - ax) / nx;
          const hy = (by - ay) / ny;
          let sum = 0;

          for (let i = 0; i <= nx; i++) {
            const x = ax + i * hx;
            const wx = i === 0 || i === nx ? 1 : (i % 2 === 0 ? 2 : 4);

            for (let j = 0; j <= ny; j++) {
              const y = ay + j * hy;
              const wy = j === 0 || j === ny ? 1 : (j % 2 === 0 ? 2 : 4);
              sum += wx * wy * f(x, y);
            }
          }

          return (hx * hy / 9) * sum;
        };

        const result1 = simpson2D((x, y) => x * y, 0, 1, 0, 1, 50, 50);
        const exact1 = 0.25;

        const result2 = simpson2D((x, y) => x*x + y*y, 0, 1, 0, 1, 50, 50);
        const exact2 = 2/3;

        return [
          "2D Integration over [0,1] × [0,1]:",
          "",
          "∬ xy dA:",
          `  Computed: ${result1.toFixed(6)}`,
          `  Exact: ${exact1.toFixed(6)}`,
          "",
          "∬ (x² + y²) dA:",
          `  Computed: ${result2.toFixed(6)}`,
          `  Exact: ${exact2.toFixed(6)}`
        ].join('\n');
      })
    },
    {
      title: "Laplacian Operator",
      description: "Compute the Laplacian ∇²f of a scalar field",
      category: "Differential",
      code: `// Define f(x,y,z) = x² + y² + z²
const field = ScalarField.fromFunction3D((x, y, z) => x*x + y*y + z*z);

// Compute Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
const derivative = new NumericalDerivative(1e-4);
const laplacian = derivative.laplacian(field, [1, 2, 3]);

console.log("Laplacian:", laplacian);
// Expected: 6 (since ∂²(x²)/∂x² = 2 for each variable)

// Laplacian is key in:
// - Heat equation: ∂u/∂t = α∇²u
// - Wave equation: ∂²u/∂t² = c²∇²u
// - Laplace equation: ∇²φ = 0`,
      onRun: simulateExample(() => {
        const f = (p: number[]) => p[0]**2 + p[1]**2 + p[2]**2;
        const point = [1, 2, 3];
        const h = 1e-4;

        // Compute second derivatives
        const d2f_dx2 = (f([point[0]+h, point[1], point[2]]) - 2*f(point) + f([point[0]-h, point[1], point[2]])) / (h*h);
        const d2f_dy2 = (f([point[0], point[1]+h, point[2]]) - 2*f(point) + f([point[0], point[1]-h, point[2]])) / (h*h);
        const d2f_dz2 = (f([point[0], point[1], point[2]+h]) - 2*f(point) + f([point[0], point[1], point[2]-h])) / (h*h);

        const laplacian = d2f_dx2 + d2f_dy2 + d2f_dz2;

        return [
          "Scalar field: f(x,y,z) = x² + y² + z²",
          "",
          `Point: (${point.join(', ')})`,
          `∂²f/∂x² = ${d2f_dx2.toFixed(4)}`,
          `∂²f/∂y² = ${d2f_dy2.toFixed(4)}`,
          `∂²f/∂z² = ${d2f_dz2.toFixed(4)}`,
          "",
          `Laplacian (∇²f) = ${laplacian.toFixed(4)}`,
          "Expected: 6.0",
          "",
          "∇² measures how a function differs from its local average"
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Differential Calculus</Title>
          <Text size="lg" c="dimmed">
            Scalar and vector field operations with numerical derivatives and integration
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Overview</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The <Code>amari-calculus</Code> module provides numerical methods for differential
              and integral calculus on scalar and vector fields.
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Differential Operators</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Gradient: ∇f (direction of steepest ascent)</li>
                  <li>Divergence: ∇·F (scalar from vector field)</li>
                  <li>Curl: ∇×F (rotation of vector field)</li>
                  <li>Laplacian: ∇²f (second derivatives)</li>
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Integration Methods</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Riemann sums</li>
                  <li>Trapezoidal rule</li>
                  <li>Simpson's rule</li>
                  <li>Adaptive quadrature</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Vector Calculus Identities</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <CodeHighlight
              code={`Gradient:     ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]

Divergence:   ∇·F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z

Curl:         ∇×F = [∂Fz/∂y - ∂Fy/∂z,
                     ∂Fx/∂z - ∂Fz/∂x,
                     ∂Fy/∂x - ∂Fx/∂y]

Laplacian:    ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²

Key identities:
  ∇×(∇f) = 0        (curl of gradient is zero)
  ∇·(∇×F) = 0       (divergence of curl is zero)
  ∇²f = ∇·(∇f)      (Laplacian is divergence of gradient)`}
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

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Applications</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <SimpleGrid cols={{ base: 1, sm: 3 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Physics</Title>
                <Text size="sm" c="dimmed">
                  Maxwell's equations, fluid dynamics, heat transfer, and wave propagation
                  all use differential operators.
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Optimization</Title>
                <Text size="sm" c="dimmed">
                  Gradient descent follows ∇f to find minima. Hessian (matrix of ∂²f)
                  enables Newton's method.
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Computer Graphics</Title>
                <Text size="sm" c="dimmed">
                  Surface normals, lighting calculations, and fluid simulation rely on
                  gradient and curl computations.
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>
      </Stack>
    </Container>
  );
}
