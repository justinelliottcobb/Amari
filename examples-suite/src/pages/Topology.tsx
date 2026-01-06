import { Container, Stack, Card, Title, Text, SimpleGrid, Code } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ExampleCard } from "../components/ExampleCard";

export function Topology() {
  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  // Helper to compute boundary of a simplex
  const computeBoundary = (vertices: number[]): { face: number[], sign: number }[] => {
    const boundary: { face: number[], sign: number }[] = [];
    for (let i = 0; i < vertices.length; i++) {
      const face = vertices.filter((_, j) => j !== i);
      const sign = (i % 2 === 0) ? 1 : -1;
      boundary.push({ face, sign });
    }
    return boundary;
  };

  // Helper for Euler characteristic
  const eulerCharacteristic = (vertices: number, edges: number, faces: number): number => {
    return vertices - edges + faces;
  };

  // Helper for Betti numbers from simplices
  const computeBettiNumbers = (vertices: number, edges: number, faces: number): number[] => {
    // For a connected surface: β₀=1 (components), χ = β₀ - β₁ + β₂
    const chi = eulerCharacteristic(vertices, edges, faces);
    // Simple case: assuming closed surface with β₂=1
    const beta0 = 1;
    const beta2 = 1;
    const beta1 = beta0 + beta2 - chi;
    return [beta0, beta1, beta2];
  };

  const examples = [
    {
      title: "Simplicial Complex Construction",
      description: "Build a simplicial complex from vertices, edges, and triangles",
      category: "Basics",
      code: `// Create a simplicial complex for a triangle
import { WasmSimplicialComplex, WasmSimplex } from '@justinelliottcobb/amari-wasm';

// Create complex and add simplices
const complex = new WasmSimplicialComplex();

// Add vertices (0-simplices)
complex.addSimplex(new WasmSimplex(new Uint32Array([0])));
complex.addSimplex(new WasmSimplex(new Uint32Array([1])));
complex.addSimplex(new WasmSimplex(new Uint32Array([2])));

// Add edges (1-simplices)
complex.addSimplex(new WasmSimplex(new Uint32Array([0, 1])));
complex.addSimplex(new WasmSimplex(new Uint32Array([1, 2])));
complex.addSimplex(new WasmSimplex(new Uint32Array([0, 2])));

// Add face (2-simplex)
complex.addSimplex(new WasmSimplex(new Uint32Array([0, 1, 2])));

// Compute topological invariants
console.log("Euler characteristic:", complex.eulerCharacteristic());
console.log("Dimension:", complex.dimension());
console.log("Betti numbers:", complex.bettiNumbers());`,
      onRun: simulateExample(() => {
        // Simulate triangle complex
        const vertices = 3;
        const edges = 3;
        const faces = 1;
        const euler = eulerCharacteristic(vertices, edges, faces);

        return [
          "Simplicial Complex: Filled Triangle",
          "",
          "Structure:",
          "  Vertices (0-simplices): 3",
          "  Edges (1-simplices): 3",
          "  Faces (2-simplices): 1",
          "",
          `Euler characteristic: χ = V - E + F = ${vertices} - ${edges} + ${faces} = ${euler}`,
          "Dimension: 2",
          "Betti numbers: [1, 0, 0]",
          "  β₀ = 1 (one connected component)",
          "  β₁ = 0 (no holes/tunnels)",
          "  β₂ = 0 (no voids)"
        ].join('\n');
      })
    },
    {
      title: "Boundary Operator",
      description: "Compute the boundary of simplices with proper orientation",
      category: "Homology",
      code: `// The boundary operator maps n-simplices to (n-1)-chains
import { WasmSimplex } from '@justinelliottcobb/amari-wasm';

// Create a 2-simplex (triangle) [0, 1, 2]
const triangle = new WasmSimplex(new Uint32Array([0, 1, 2]));

// Get boundary faces with orientations
const boundary = triangle.boundaryFaces();

// Boundary of [0,1,2] = [1,2] - [0,2] + [0,1]
for (const face of boundary) {
  console.log(\`\${face.sign > 0 ? '+' : '-'}[\${face.face.join(',')}]\`);
}

// Key property: ∂² = 0 (boundary of boundary is zero)
// ∂[0,1,2] = [1,2] - [0,2] + [0,1]
// ∂(∂[0,1,2]) = ([2]-[1]) - ([2]-[0]) + ([1]-[0])
//             = [2]-[1]-[2]+[0]+[1]-[0] = 0`,
      onRun: simulateExample(() => {
        const triangle = [0, 1, 2];
        const boundary = computeBoundary(triangle);

        // Compute boundary of boundary
        let boundaryOfBoundary: Map<string, number> = new Map();
        for (const { face, sign } of boundary) {
          const subBoundary = computeBoundary(face);
          for (const { face: subFace, sign: subSign } of subBoundary) {
            const key = subFace.join(',');
            const current = boundaryOfBoundary.get(key) || 0;
            boundaryOfBoundary.set(key, current + sign * subSign);
          }
        }

        // Filter out zeros
        const nonZero = Array.from(boundaryOfBoundary.entries())
          .filter(([_, v]) => v !== 0);

        return [
          "Simplex: [0, 1, 2] (triangle)",
          "",
          "Boundary ∂[0,1,2]:",
          ...boundary.map(({ face, sign }) =>
            `  ${sign > 0 ? '+' : '-'}[${face.join(', ')}]`
          ),
          "",
          "= [1,2] - [0,2] + [0,1]",
          "",
          "Verifying ∂² = 0:",
          "  ∂([1,2]) = [2] - [1]",
          "  ∂(-[0,2]) = -[2] + [0]",
          "  ∂([0,1]) = [1] - [0]",
          "",
          "  Sum: [2]-[1]-[2]+[0]+[1]-[0] = 0 ✓",
          "",
          `∂² is ${nonZero.length === 0 ? 'zero' : 'non-zero'}: ∂∂ = 0 ✓`
        ].join('\n');
      })
    },
    {
      title: "Persistent Homology",
      description: "Track topological features across a filtration",
      category: "Persistence",
      code: `// Compute persistent homology from a point cloud
import { WasmFiltration, WasmPersistentHomology, ripsFromDistances }
  from '@justinelliottcobb/amari-wasm';

// Points: 3 points forming a triangle
const points = [[0, 0], [1, 0], [0.5, 0.866]];

// Compute distance matrix
const distances = new Float64Array([
  0, 1, 1,     // distances from point 0
  1, 0, 1,     // distances from point 1
  1, 1, 0      // distances from point 2
]);

// Build Rips filtration up to max distance
const filtration = ripsFromDistances(distances, 3, 1.5);

// Compute persistent homology
const ph = new WasmPersistentHomology();
const diagram = ph.compute(filtration);

// Get birth-death pairs
const intervals = diagram.getIntervals();
for (const interval of intervals) {
  console.log(\`H\${interval.dimension}: born=\${interval.birth}, died=\${interval.death}\`);
}`,
      onRun: simulateExample(() => {
        // Simulate Rips filtration on 3 points (equilateral triangle)
        // ε=0: 3 components, ε=1: edges appear, ε=1+: triangle fills

        return [
          "Point cloud: 3 points (equilateral triangle)",
          "  P₀ = (0, 0)",
          "  P₁ = (1, 0)",
          "  P₂ = (0.5, 0.866)",
          "",
          "Distance matrix:",
          "  [0.00, 1.00, 1.00]",
          "  [1.00, 0.00, 1.00]",
          "  [1.00, 1.00, 0.00]",
          "",
          "Rips Filtration:",
          "  ε = 0.0: {v₀}, {v₁}, {v₂}  (3 components)",
          "  ε = 1.0: edges appear, single component",
          "  ε = 1.0: triangle [0,1,2] appears",
          "",
          "Persistence Diagram:",
          "  H₀: (0.0, 1.0) - component merged",
          "  H₀: (0.0, 1.0) - component merged",
          "  H₀: (0.0, ∞)   - surviving component",
          "  H₁: (1.0, 1.0) - cycle immediately filled",
          "",
          "Betti numbers at ε = 0.5: β₀=3, β₁=0",
          "Betti numbers at ε = 1.5: β₀=1, β₁=0"
        ].join('\n');
      })
    },
    {
      title: "Morse Theory",
      description: "Find critical points of a function and build the Morse complex",
      category: "Morse",
      code: `// Find critical points of a 2D height function
import { findCriticalPoints2D, WasmMorseComplex }
  from '@justinelliottcobb/amari-wasm';

// Height function on a grid (8x8 for this example)
// f(x,y) = sin(πx)·sin(πy) has critical points
const gridSize = 8;
const values = new Float64Array(gridSize * gridSize);
for (let i = 0; i < gridSize; i++) {
  for (let j = 0; j < gridSize; j++) {
    const x = i / (gridSize - 1);
    const y = j / (gridSize - 1);
    values[i * gridSize + j] = Math.sin(Math.PI * x) * Math.sin(Math.PI * y);
  }
}

// Find critical points
const criticalPoints = findCriticalPoints2D(values, gridSize, gridSize);

for (const cp of criticalPoints) {
  console.log(\`\${cp.type} at (\${cp.x}, \${cp.y}), value=\${cp.value}\`);
}

// Morse inequality: #minima - #saddles + #maxima = χ`,
      onRun: simulateExample(() => {
        // For f(x,y) = sin(πx)sin(πy) on [0,1]²:
        // Critical points: corners (minima), center (maximum), edge midpoints (saddles)

        return [
          "Height function: f(x,y) = sin(πx)·sin(πy)",
          "",
          "Critical Points:",
          "  Minimum at (0, 0), value = 0.00",
          "  Minimum at (0, 1), value = 0.00",
          "  Minimum at (1, 0), value = 0.00",
          "  Minimum at (1, 1), value = 0.00",
          "  Maximum at (0.5, 0.5), value = 1.00",
          "  Saddle at (0.5, 0), value = 0.00",
          "  Saddle at (0.5, 1), value = 0.00",
          "  Saddle at (0, 0.5), value = 0.00",
          "  Saddle at (1, 0.5), value = 0.00",
          "",
          "Morse numbers:",
          "  m₀ (minima) = 4",
          "  m₁ (saddles) = 4",
          "  m₂ (maxima) = 1",
          "",
          "Weak Morse inequality: mₖ ≥ βₖ",
          "Strong Morse inequality: χ = Σ(-1)ᵏmₖ",
          "  χ = 4 - 4 + 1 = 1 ✓ (for torus boundary)"
        ].join('\n');
      })
    },
    {
      title: "Euler Characteristic",
      description: "Compute the Euler characteristic from different complexes",
      category: "Invariants",
      code: `// Euler characteristic: χ = V - E + F = Σ(-1)^k·cₖ
import { WasmSimplicialComplex, WasmSimplex } from '@justinelliottcobb/amari-wasm';

// Build a tetrahedron (simplest 3D triangulation of S²)
const tetrahedron = new WasmSimplicialComplex();

// 4 vertices, 6 edges, 4 faces (surface only, no interior)
const vertices = [[0], [1], [2], [3]];
const edges = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]];
const faces = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]];

for (const v of vertices) tetrahedron.addSimplex(new WasmSimplex(new Uint32Array(v)));
for (const e of edges) tetrahedron.addSimplex(new WasmSimplex(new Uint32Array(e)));
for (const f of faces) tetrahedron.addSimplex(new WasmSimplex(new Uint32Array(f)));

const chi = tetrahedron.eulerCharacteristic();
console.log(\`Tetrahedron χ = \${chi}\`);  // χ = 4 - 6 + 4 = 2 (sphere)

// χ = 2 for sphere, 0 for torus, 2-2g for genus-g surface`,
      onRun: simulateExample(() => {
        // Various surfaces
        const surfaces = [
          { name: "Tetrahedron (S²)", V: 4, E: 6, F: 4, expected: 2 },
          { name: "Cube surface (S²)", V: 8, E: 12, F: 6, expected: 2 },
          { name: "Octahedron (S²)", V: 6, E: 12, F: 8, expected: 2 },
          { name: "Torus", V: 9, E: 27, F: 18, expected: 0 },
        ];

        const lines = [
          "Euler Characteristic: χ = V - E + F",
          "",
          "Surface triangulations:",
          ""
        ];

        for (const s of surfaces) {
          const chi = s.V - s.E + s.F;
          lines.push(`${s.name}:`);
          lines.push(`  V=${s.V}, E=${s.E}, F=${s.F}`);
          lines.push(`  χ = ${s.V} - ${s.E} + ${s.F} = ${chi}`);
          lines.push("");
        }

        lines.push("Topological meaning:");
        lines.push("  χ = 2  → Sphere (S²)");
        lines.push("  χ = 0  → Torus (T²)");
        lines.push("  χ = 2-2g for genus-g surface");

        return lines.join('\n');
      })
    },
    {
      title: "Betti Numbers",
      description: "Compute the rank of homology groups",
      category: "Homology",
      code: `// Betti numbers count topological features
import { WasmSimplicialComplex } from '@justinelliottcobb/amari-wasm';

// Build a hollow torus triangulation
const torus = buildTorusComplex();

// Compute Betti numbers
const betti = torus.bettiNumbers();
console.log("β₀ =", betti[0], "(connected components)");
console.log("β₁ =", betti[1], "(independent cycles/tunnels)");
console.log("β₂ =", betti[2], "(voids/cavities)");

// For a torus: β₀=1, β₁=2, β₂=1
// The two 1-cycles are the meridian and longitude
// The 2-cycle encloses the hollow interior`,
      onRun: simulateExample(() => {
        return [
          "Betti Numbers: βₖ = rank(Hₖ)",
          "",
          "Sphere S²:",
          "  β₀ = 1 (one component)",
          "  β₁ = 0 (no tunnels)",
          "  β₂ = 1 (one void)",
          "",
          "Torus T²:",
          "  β₀ = 1 (one component)",
          "  β₁ = 2 (meridian + longitude)",
          "  β₂ = 1 (hollow interior)",
          "",
          "Circle S¹:",
          "  β₀ = 1 (one component)",
          "  β₁ = 1 (the loop itself)",
          "",
          "Two disjoint circles:",
          "  β₀ = 2 (two components)",
          "  β₁ = 2 (two loops)",
          "",
          "Relationship to Euler characteristic:",
          "  χ = β₀ - β₁ + β₂ - β₃ + ..."
        ].join('\n');
      })
    },
    {
      title: "Vietoris-Rips Complex",
      description: "Build a Rips complex from a point cloud at a given scale",
      category: "Persistence",
      code: `// Rips complex: connect points within distance ε
import { ripsFromDistances } from '@justinelliottcobb/amari-wasm';

// 4 points forming a square with diagonal 1.4
const n = 4;
const distances = new Float64Array([
  0.0, 1.0, 1.4, 1.0,  // point 0
  1.0, 0.0, 1.0, 1.4,  // point 1
  1.4, 1.0, 0.0, 1.0,  // point 2
  1.0, 1.4, 1.0, 0.0   // point 3
]);

// At ε=1.0: only edges of the square
const rips1 = ripsFromDistances(distances, n, 1.0);
console.log("ε=1.0: 4 vertices, 4 edges, forms a cycle (β₁=1)");

// At ε=1.5: diagonals appear, triangles fill in
const rips2 = ripsFromDistances(distances, n, 1.5);
console.log("ε=1.5: diagonals added, cycle filled (β₁=0)");`,
      onRun: simulateExample(() => {
        return [
          "Vietoris-Rips Complex from Point Cloud",
          "",
          "Points: Square with side 1.0, diagonal ≈ 1.41",
          "  P₀ = (0, 0)",
          "  P₁ = (1, 0)",
          "  P₂ = (1, 1)",
          "  P₃ = (0, 1)",
          "",
          "At ε = 1.0:",
          "  Edges: [0,1], [1,2], [2,3], [0,3]",
          "  Forms a 4-cycle (square boundary)",
          "  Betti: β₀=1, β₁=1 (one hole)",
          "",
          "At ε = 1.5 (> diagonal):",
          "  + Diagonals: [0,2], [1,3]",
          "  + Triangles: [0,1,2], [0,2,3], [0,1,3], [1,2,3]",
          "  + Tetrahedron: [0,1,2,3]",
          "  Betti: β₀=1, β₁=0 (hole filled)",
          "",
          "The hole (1-cycle) is born at ε=1.0 and dies at ε≈1.41"
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Computational Topology</Title>
          <Text size="lg" c="dimmed">
            Simplicial complexes, homology, persistent homology, and Morse theory
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Overview</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The <Code>amari-topology</Code> module provides tools for computational algebraic
              topology, enabling analysis of shape and structure through homology theory.
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Core Concepts</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Simplicial complexes (vertices, edges, faces...)</li>
                  <li>Chain groups and boundary operators</li>
                  <li>Homology groups and Betti numbers</li>
                  <li>Euler characteristic</li>
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Advanced Features</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Persistent homology (TDA)</li>
                  <li>Vietoris-Rips filtrations</li>
                  <li>Morse theory and critical points</li>
                  <li>Parallel computation via Rayon</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Key Formulas</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <CodeHighlight
              code={`Boundary operator:
  ∂[v₀, v₁, ..., vₙ] = Σᵢ (-1)ⁱ [v₀, ..., v̂ᵢ, ..., vₙ]
  where v̂ᵢ means vertex vᵢ is omitted

Fundamental property:
  ∂² = 0  (boundary of a boundary is zero)

Homology groups:
  Hₖ = Zₖ / Bₖ = ker(∂ₖ) / im(∂ₖ₊₁)

Betti numbers:
  βₖ = rank(Hₖ) = dim(Zₖ) - dim(Bₖ)

Euler characteristic:
  χ = Σₖ (-1)ᵏ cₖ = Σₖ (-1)ᵏ βₖ
  where cₖ = number of k-simplices

Euler-Poincaré formula:
  χ = V - E + F  (for 2D complexes)`}
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
              category={example.category}
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
                <Title order={4} size="sm" mb="xs">Data Analysis (TDA)</Title>
                <Text size="sm" c="dimmed">
                  Persistent homology reveals multi-scale structure in point clouds,
                  time series, and high-dimensional data.
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Shape Recognition</Title>
                <Text size="sm" c="dimmed">
                  Betti numbers and persistence diagrams provide robust shape
                  descriptors invariant to deformation.
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Network Analysis</Title>
                <Text size="sm" c="dimmed">
                  Clique complexes and persistent homology detect community structure
                  and higher-order interactions in networks.
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>
      </Stack>
    </Container>
  );
}
