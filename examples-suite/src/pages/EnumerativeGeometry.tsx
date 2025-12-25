import { useState, useCallback } from 'react';
import {
  Container, Stack, Card, Title, Text, Button, SimpleGrid,
  Code, Table, Badge, Tabs, NumberInput, TextInput
} from '@mantine/core';
import { ExampleCard } from '../components/ExampleCard';

// Mock implementations for demonstration (these would normally come from compiled WASM)
const mockEnumerativeGeometry = {
  ProjectiveSpace: {
    new: (dimension: number) => ({
      dimension,
      intersect: (class1: any, class2: any) => ({
        multiplicity: () => class1.degree * class2.degree
      })
    })
  },
  ChowClass: {
    hypersurface: (degree: number) => ({ degree, dimension: 1 }),
    point: () => ({ degree: 1, dimension: 0 }),
    line: () => ({ degree: 1, dimension: 1 })
  },
  SchubertClass: {
    new: (partition: number[], grassmannian: [number, number]) => ({
      partition,
      grassmannian,
      degree: () => partition.reduce((a, b) => a + b, 0)
    })
  },
  Grassmannian: {
    new: (k: number, n: number) => ({
      dimension: k * (n - k),
      schubertCycle: (partition: number[]) => ({ partition, degree: partition.reduce((a, b) => a + b, 0) })
    })
  },
  TropicalCurve: {
    new: (degree: number, constraints: number) => ({
      degree,
      constraints,
      count: () => Math.pow(degree, constraints - degree + 1) // Simplified tropical counting
    })
  },
  HigherGenusCurve: {
    new: (genus: number, _degree: number) => ({
      genus,
      _degree,
      canonicalDegree: 2 * genus - 2,
      riemannRochDimension: (d: number) => Math.max(0, d - genus + 1)
    })
  }
};

interface ComputationResult {
  input: string;
  output: any;
  time: number;
  error?: string;
}

export function EnumerativeGeometry() {
  const [computationHistory, setComputationHistory] = useState<ComputationResult[]>([]);
  const [isComputing, setIsComputing] = useState(false);

  // Intersection Theory Demo
  const [projDimension, setProjDimension] = useState<number | string>(2);
  const [degree1, setDegree1] = useState<number | string>(3);
  const [degree2, setDegree2] = useState<number | string>(4);
  const [intersectionResult, setIntersectionResult] = useState<number | null>(null);

  // Schubert Calculus Demo
  const [grassmannianK, setGrassmannianK] = useState<number | string>(2);
  const [grassmannianN, setGrassmannianN] = useState<number | string>(5);
  const [partition1, setPartition1] = useState('1');
  const [partition2, setPartition2] = useState('1');
  const [schubertResult, setSchubertResult] = useState<any>(null);

  // Tropical Geometry Demo
  const [tropicalDegree, setTropicalDegree] = useState<number | string>(3);
  const [tropicalConstraints, setTropicalConstraints] = useState<number | string>(8);
  const [tropicalResult, setTropicalResult] = useState<number | null>(null);

  // Higher Genus Demo
  const [genus, setGenus] = useState<number | string>(2);
  const [curveDegree, setCurveDegree] = useState<number | string>(4);
  const [rrDegree, setRrDegree] = useState<number | string>(5);
  const [higherGenusResult, setHigherGenusResult] = useState<any>(null);

  const addToHistory = useCallback((input: string, output: any, time: number, error?: string) => {
    setComputationHistory(prev => [{ input, output, time, error }, ...prev.slice(0, 9)]);
  }, []);

  const computeIntersection = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const space = mockEnumerativeGeometry.ProjectiveSpace.new(Number(projDimension));
      const class1 = mockEnumerativeGeometry.ChowClass.hypersurface(Number(degree1));
      const class2 = mockEnumerativeGeometry.ChowClass.hypersurface(Number(degree2));
      const intersection = space.intersect(class1, class2);
      const result = intersection.multiplicity();

      setIntersectionResult(result);
      addToHistory(
        `P${projDimension}: deg ${degree1} ∩ deg ${degree2}`,
        `${result} (Bézout's theorem)`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`Intersection computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [projDimension, degree1, degree2, addToHistory]);

  const computeSchubert = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const gr = mockEnumerativeGeometry.Grassmannian.new(Number(grassmannianK), Number(grassmannianN));
      const p1 = partition1.split(',').map(x => parseInt(x.trim()));
      const p2 = partition2.split(',').map(x => parseInt(x.trim()));

      const cycle1 = gr.schubertCycle(p1);
      const cycle2 = gr.schubertCycle(p2);

      const result = {
        grassmannian: `Gr(${grassmannianK}, ${grassmannianN})`,
        dimension: gr.dimension,
        partitions: [p1, p2],
        intersection: cycle1.degree + cycle2.degree // Simplified
      };

      setSchubertResult(result);
      addToHistory(
        `Gr(${grassmannianK},${grassmannianN}): σ${p1} ∩ σ${p2}`,
        `${result.intersection}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`Schubert computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [grassmannianK, grassmannianN, partition1, partition2, addToHistory]);

  const computeTropical = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const curve = mockEnumerativeGeometry.TropicalCurve.new(Number(tropicalDegree), Number(tropicalConstraints));
      const result = curve.count();

      setTropicalResult(result);
      addToHistory(
        `Tropical curves deg ${tropicalDegree}, ${tropicalConstraints} constraints`,
        `${result} curves`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`Tropical computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [tropicalDegree, tropicalConstraints, addToHistory]);

  const computeHigherGenus = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const curve = mockEnumerativeGeometry.HigherGenusCurve.new(Number(genus), Number(curveDegree));
      const rrDim = curve.riemannRochDimension(Number(rrDegree));

      const result = {
        genus: Number(genus),
        degree: Number(curveDegree),
        canonicalDegree: curve.canonicalDegree,
        riemannRochDim: rrDim
      };

      setHigherGenusResult(result);
      addToHistory(
        `Genus ${genus} curve, deg ${curveDegree}, H^0(L_${rrDegree})`,
        `dim = ${rrDim}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`Higher genus computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [genus, curveDegree, rrDegree, addToHistory]);

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Enumerative Geometry</Title>
          <Text size="lg" c="dimmed">
            Interactive examples of intersection theory, Schubert calculus, tropical geometry, and curve counting
          </Text>
        </div>

        {/* Overview Section */}
        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={2} size="h3">Mathematical Framework</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
              <div>
                <Title order={3} size="h4" mb="sm">Core Concepts</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem' }}>
                  <li><Text span fw={600}>Intersection Theory:</Text> Chow rings and Bézout's theorem</li>
                  <li><Text span fw={600}>Schubert Calculus:</Text> Grassmannians and flag varieties</li>
                  <li><Text span fw={600}>Gromov-Witten Theory:</Text> Curve counting and quantum cohomology</li>
                  <li><Text span fw={600}>Tropical Geometry:</Text> Piecewise-linear structures</li>
                </Text>
              </div>
              <div>
                <Title order={3} size="h4" mb="sm">Performance Features</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem' }}>
                  <li><Text span fw={600}>WASM-First:</Text> Optimized for web deployment</li>
                  <li><Text span fw={600}>GPU Acceleration:</Text> WGPU compute shaders</li>
                  <li><Text span fw={600}>Parallel Computing:</Text> Multi-threaded algorithms</li>
                  <li><Text span fw={600}>Memory Efficient:</Text> Sparse matrix operations</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        {/* Tabs */}
        <Tabs defaultValue="intersection">
          <Tabs.List>
            <Tabs.Tab value="intersection">Intersection Theory</Tabs.Tab>
            <Tabs.Tab value="schubert">Schubert Calculus</Tabs.Tab>
            <Tabs.Tab value="tropical">Tropical Geometry</Tabs.Tab>
            <Tabs.Tab value="higher-genus">Higher Genus</Tabs.Tab>
            <Tabs.Tab value="performance">Performance</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="intersection" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Intersection Theory</Title>
                <Text size="sm" c="dimmed">Compute intersection numbers using Bézout's theorem in projective space</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <NumberInput
                      label="Projective Space Dimension"
                      min={1}
                      max={5}
                      value={projDimension}
                      onChange={setProjDimension}
                    />
                    <NumberInput
                      label="First Hypersurface Degree"
                      min={1}
                      max={10}
                      value={degree1}
                      onChange={setDegree1}
                    />
                    <NumberInput
                      label="Second Hypersurface Degree"
                      min={1}
                      max={10}
                      value={degree2}
                      onChange={setDegree2}
                    />
                    <Button onClick={computeIntersection} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Compute Intersection'}
                    </Button>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Result</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {intersectionResult !== null ? (
                        <Stack gap="sm">
                          <Text><Text span fw={600}>Intersection Number:</Text> {intersectionResult}</Text>
                          <Text size="sm" c="dimmed">
                            By Bézout's theorem, two hypersurfaces of degrees {String(degree1)} and {String(degree2)}
                            in P<sup>{String(projDimension)}</sup> intersect in exactly {String(degree1)} × {String(degree2)} = {intersectionResult} points
                            (counting multiplicities).
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute Intersection" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="schubert" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Schubert Calculus</Title>
                <Text size="sm" c="dimmed">Intersection theory on Grassmannians using Schubert cycles</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <NumberInput
                      label="Grassmannian k (subspace dimension)"
                      min={1}
                      max={5}
                      value={grassmannianK}
                      onChange={setGrassmannianK}
                    />
                    <NumberInput
                      label="Grassmannian n (ambient dimension)"
                      min={Number(grassmannianK) + 1}
                      max={8}
                      value={grassmannianN}
                      onChange={setGrassmannianN}
                    />
                    <TextInput
                      label="First Partition (comma-separated)"
                      value={partition1}
                      onChange={(e) => setPartition1(e.target.value)}
                      placeholder="e.g., 1,0"
                    />
                    <TextInput
                      label="Second Partition (comma-separated)"
                      value={partition2}
                      onChange={(e) => setPartition2(e.target.value)}
                      placeholder="e.g., 1,0"
                    />
                    <Button onClick={computeSchubert} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Compute Schubert Intersection'}
                    </Button>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Result</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {schubertResult ? (
                        <Stack gap="sm">
                          <Text><Text span fw={600}>Grassmannian:</Text> {schubertResult.grassmannian}</Text>
                          <Text><Text span fw={600}>Dimension:</Text> {schubertResult.dimension}</Text>
                          <Text><Text span fw={600}>Intersection Number:</Text> {schubertResult.intersection}</Text>
                          <Text size="sm" c="dimmed">
                            Schubert cycles σ<sub>{schubertResult.partitions[0].join(',')}</sub> and
                            σ<sub>{schubertResult.partitions[1].join(',')}</sub> on Gr({String(grassmannianK)},{String(grassmannianN)})
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute Schubert Intersection" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="tropical" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Tropical Geometry</Title>
                <Text size="sm" c="dimmed">Count tropical curves using Mikhalkin correspondence</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <NumberInput
                      label="Curve Degree"
                      min={1}
                      max={6}
                      value={tropicalDegree}
                      onChange={setTropicalDegree}
                    />
                    <NumberInput
                      label="Number of Constraints"
                      min={Number(tropicalDegree)}
                      max={12}
                      value={tropicalConstraints}
                      onChange={setTropicalConstraints}
                    />
                    <Button onClick={computeTropical} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Count Tropical Curves'}
                    </Button>
                    <Card withBorder p="sm">
                      <Text size="sm">
                        <Text span fw={600}>Note:</Text> For degree {String(tropicalDegree)} curves, the expected dimension is {3 * Number(tropicalDegree) - 1}.
                        Adjust constraints accordingly.
                      </Text>
                    </Card>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Result</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {tropicalResult !== null ? (
                        <Stack gap="sm">
                          <Text><Text span fw={600}>Curve Count:</Text> {tropicalResult}</Text>
                          <Text size="sm" c="dimmed">
                            Number of degree-{String(tropicalDegree)} tropical curves satisfying {String(tropicalConstraints)} generic constraints.
                            This matches the classical count by Mikhalkin correspondence.
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Count Tropical Curves" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="higher-genus" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Higher Genus Curves</Title>
                <Text size="sm" c="dimmed">Riemann-Roch theorem and moduli space computations</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <NumberInput
                      label="Genus"
                      min={0}
                      max={5}
                      value={genus}
                      onChange={setGenus}
                    />
                    <NumberInput
                      label="Curve Degree"
                      min={1}
                      max={8}
                      value={curveDegree}
                      onChange={setCurveDegree}
                    />
                    <NumberInput
                      label="Line Bundle Degree (for Riemann-Roch)"
                      min={0}
                      max={10}
                      value={rrDegree}
                      onChange={setRrDegree}
                    />
                    <Button onClick={computeHigherGenus} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Compute Riemann-Roch'}
                    </Button>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Result</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {higherGenusResult ? (
                        <Stack gap="sm">
                          <Text><Text span fw={600}>Genus:</Text> {higherGenusResult.genus}</Text>
                          <Text><Text span fw={600}>Degree:</Text> {higherGenusResult.degree}</Text>
                          <Text><Text span fw={600}>Canonical Degree:</Text> {higherGenusResult.canonicalDegree}</Text>
                          <Text><Text span fw={600}>h⁰(L<sub>{String(rrDegree)}</sub>):</Text> {higherGenusResult.riemannRochDim}</Text>
                          <Text size="sm" c="dimmed">
                            By Riemann-Roch: h⁰(L) - h¹(L) = deg(L) + 1 - g = {String(rrDegree)} + 1 - {String(genus)} = {Number(rrDegree) + 1 - Number(genus)}
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute Riemann-Roch" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="performance" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Performance & Optimization</Title>
                <Text size="sm" c="dimmed">WASM-first architecture with GPU acceleration</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <div>
                      <Title order={3} size="h4" mb="xs">WASM Optimization</Title>
                      <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem' }}>
                        <li>Memory-efficient sparse matrix operations</li>
                        <li>Custom memory pooling for large computations</li>
                        <li>SIMD optimizations where available</li>
                        <li>Configurable batch processing</li>
                      </Text>
                    </div>
                    <div>
                      <Title order={3} size="h4" mb="xs">GPU Acceleration</Title>
                      <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem' }}>
                        <li>WGPU compute shaders for intersection numbers</li>
                        <li>Parallel Schubert calculus kernels</li>
                        <li>Gromov-Witten invariant computation</li>
                        <li>Tropical curve counting acceleration</li>
                      </Text>
                    </div>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Feature Flags</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      <Table>
                        <Table.Thead>
                          <Table.Tr>
                            <Table.Th>Feature</Table.Th>
                            <Table.Th>Status</Table.Th>
                          </Table.Tr>
                        </Table.Thead>
                        <Table.Tbody>
                          <Table.Tr>
                            <Table.Td><Code>wgpu</Code></Table.Td>
                            <Table.Td><Badge color="green">Available</Badge></Table.Td>
                          </Table.Tr>
                          <Table.Tr>
                            <Table.Td><Code>wasm</Code></Table.Td>
                            <Table.Td><Badge color="green">Available</Badge></Table.Td>
                          </Table.Tr>
                          <Table.Tr>
                            <Table.Td><Code>parallel</Code></Table.Td>
                            <Table.Td><Badge color="green">Available</Badge></Table.Td>
                          </Table.Tr>
                          <Table.Tr>
                            <Table.Td><Code>performance</Code></Table.Td>
                            <Table.Td><Badge color="green">Available</Badge></Table.Td>
                          </Table.Tr>
                        </Table.Tbody>
                      </Table>
                    </Card.Section>
                  </Card>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </Tabs.Panel>
        </Tabs>

        {/* Computation History */}
        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={2} size="h3">Computation History</Title>
            <Text size="sm" c="dimmed">Recent calculations and timing information</Text>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            {computationHistory.length > 0 ? (
              <Table>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Input</Table.Th>
                    <Table.Th>Output</Table.Th>
                    <Table.Th>Time (ms)</Table.Th>
                    <Table.Th>Status</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {computationHistory.map((result, index) => (
                    <Table.Tr key={index}>
                      <Table.Td><Code>{result.input}</Code></Table.Td>
                      <Table.Td>{result.output}</Table.Td>
                      <Table.Td>{result.time}</Table.Td>
                      <Table.Td>
                        <Badge color={result.error ? "red" : "green"}>
                          {result.error ? "Error" : "Success"}
                        </Badge>
                      </Table.Td>
                    </Table.Tr>
                  ))}
                </Table.Tbody>
              </Table>
            ) : (
              <Text c="dimmed">No computations yet. Try the examples above!</Text>
            )}
          </Card.Section>
        </Card>

        {/* Code Examples */}
        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={2} size="h3">Code Examples</Title>
            <Text size="sm" c="dimmed">Learn how to use the enumerative geometry API</Text>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Stack gap="lg">
              <ExampleCard
                title="Basic Intersection Theory"
                description="Compute intersection numbers in projective space"
                code={`// Import the enumerative geometry library
import { ProjectiveSpace, ChowClass } from 'amari-enumerative';

// Create projective 2-space
const p2 = ProjectiveSpace.new(2);

// Define two curves
const cubic = ChowClass.hypersurface(3);
const quartic = ChowClass.hypersurface(4);

// Compute intersection number (Bézout's theorem)
const intersection = p2.intersect(cubic, quartic);
console.log(intersection.multiplicity()); // 12`}
              />

              <ExampleCard
                title="Schubert Calculus"
                description="Work with Grassmannians and Schubert cycles"
                code={`// Import Schubert calculus components
import { Grassmannian, SchubertClass } from 'amari-enumerative';

// Create Grassmannian Gr(2,5)
const gr = Grassmannian.new(2, 5);

// Define Schubert cycles
const sigma1 = SchubertClass.new([1, 0], [2, 5]);
const sigma2 = SchubertClass.new([0, 1], [2, 5]);

// Compute intersection
const result = gr.intersect(sigma1, sigma2);
console.log(\`Intersection number: \${result.multiplicity()}\`);`}
              />

              <ExampleCard
                title="Performance Optimization"
                description="Configure WASM performance settings"
                code={`// Import performance components
import { WasmPerformanceConfig, FastIntersectionComputer } from 'amari-enumerative';

// Configure for high performance
const config = WasmPerformanceConfig.default();
config.enable_gpu = true;
config.cache_size = 50000;
config.max_workers = 8;

// Create optimized computer
const computer = FastIntersectionComputer.new(config);

// Perform fast computations
const result = computer.fast_intersect(p2, cubic, quartic);`}
              />
            </Stack>
          </Card.Section>
        </Card>
      </Stack>
    </Container>
  );
}
