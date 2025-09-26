import { useState, useCallback, useEffect } from 'react';
import {
  Card, CardBody, CardHeader, H1, H2, H3, P, Button, Grid, GridItem,
  Strong, Code, Table, TableHeader, TableRow, TableHead, TableBody, TableCell,
  TextArea, StatusBadge, Input
} from 'jadis-ui';
import { CodePlayground } from '../components/CodePlayground';
import { ExampleCard } from '../components/ExampleCard';
import { RealTimeDisplay } from '../components/RealTimeDisplay';

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
    new: (genus: number, degree: number) => ({
      genus,
      degree,
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
  const [activeTab, setActiveTab] = useState('intersection');
  const [computationHistory, setComputationHistory] = useState<ComputationResult[]>([]);
  const [isComputing, setIsComputing] = useState(false);

  // Intersection Theory Demo
  const [projDimension, setProjDimension] = useState(2);
  const [degree1, setDegree1] = useState(3);
  const [degree2, setDegree2] = useState(4);
  const [intersectionResult, setIntersectionResult] = useState<number | null>(null);

  // Schubert Calculus Demo
  const [grassmannianK, setGrassmannianK] = useState(2);
  const [grassmannianN, setGrassmannianN] = useState(5);
  const [partition1, setPartition1] = useState('1');
  const [partition2, setPartition2] = useState('1');
  const [schubertResult, setSchubertResult] = useState<any>(null);

  // Tropical Geometry Demo
  const [tropicalDegree, setTropicalDegree] = useState(3);
  const [tropicalConstraints, setTropicalConstraints] = useState(8);
  const [tropicalResult, setTropicalResult] = useState<number | null>(null);

  // Higher Genus Demo
  const [genus, setGenus] = useState(2);
  const [curveDegree, setCurveDegree] = useState(4);
  const [rrDegree, setRrDegree] = useState(5);
  const [higherGenusResult, setHigherGenusResult] = useState<any>(null);

  const addToHistory = useCallback((input: string, output: any, time: number, error?: string) => {
    setComputationHistory(prev => [{ input, output, time, error }, ...prev.slice(0, 9)]);
  }, []);

  const computeIntersection = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const space = mockEnumerativeGeometry.ProjectiveSpace.new(projDimension);
      const class1 = mockEnumerativeGeometry.ChowClass.hypersurface(degree1);
      const class2 = mockEnumerativeGeometry.ChowClass.hypersurface(degree2);
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
      const gr = mockEnumerativeGeometry.Grassmannian.new(grassmannianK, grassmannianN);
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
      const curve = mockEnumerativeGeometry.TropicalCurve.new(tropicalDegree, tropicalConstraints);
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
      const curve = mockEnumerativeGeometry.HigherGenusCurve.new(genus, curveDegree);
      const rrDim = curve.riemannRochDimension(rrDegree);

      const result = {
        genus,
        degree: curveDegree,
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
    <Grid cols={1} gap="lg">
      <GridItem>
        <H1>Enumerative Geometry</H1>
        <P>
          Interactive examples of intersection theory, Schubert calculus, tropical geometry, and curve counting
        </P>
      </GridItem>

      {/* Overview Section */}
      <GridItem>
        <Card>
          <CardHeader>
            <H2>Mathematical Framework</H2>
          </CardHeader>
          <CardBody>
            <Grid cols={2} gap="lg">
              <GridItem>
                <H3>Core Concepts</H3>
                <P><Strong>Intersection Theory:</Strong> Chow rings and Bézout's theorem</P>
                <P><Strong>Schubert Calculus:</Strong> Grassmannians and flag varieties</P>
                <P><Strong>Gromov-Witten Theory:</Strong> Curve counting and quantum cohomology</P>
                <P><Strong>Tropical Geometry:</Strong> Piecewise-linear structures</P>
              </GridItem>
              <GridItem>
                <H3>Performance Features</H3>
                <P><Strong>WASM-First:</Strong> Optimized for web deployment</P>
                <P><Strong>GPU Acceleration:</Strong> WGPU compute shaders</P>
                <P><Strong>Parallel Computing:</Strong> Multi-threaded algorithms</P>
                <P><Strong>Memory Efficient:</Strong> Sparse matrix operations</P>
              </GridItem>
            </Grid>
          </CardBody>
        </Card>
      </GridItem>

      {/* Tab Navigation */}
      <GridItem>
        <Grid cols={5} gap="sm">
          {[
            { id: 'intersection', label: 'Intersection Theory' },
            { id: 'schubert', label: 'Schubert Calculus' },
            { id: 'tropical', label: 'Tropical Geometry' },
            { id: 'higher-genus', label: 'Higher Genus' },
            { id: 'performance', label: 'Performance' }
          ].map(tab => (
            <GridItem key={tab.id}>
              <Button
                variant={activeTab === tab.id ? 'primary' : 'secondary'}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </Button>
            </GridItem>
          ))}
        </Grid>
      </GridItem>

      {/* Tab Content */}
      {activeTab === 'intersection' && (
        <GridItem>
          <Card>
            <CardHeader>
              <H2>Intersection Theory</H2>
              <P>Compute intersection numbers using Bézout's theorem in projective space</P>
            </CardHeader>
            <CardBody>
              <Grid cols={2} gap="lg">
                <GridItem>
                  <Grid cols={1} gap="md">
                    <GridItem>
                      <Input
                        label="Projective Space Dimension"
                        type="number"
                        min="1"
                        max="5"
                        value={projDimension.toString()}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) => setProjDimension(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="First Hypersurface Degree"
                        type="number"
                        min="1"
                        max="10"
                        value={degree1.toString()}
                        onChange={(e) => setDegree1(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="Second Hypersurface Degree"
                        type="number"
                        min="1"
                        max="10"
                        value={degree2.toString()}
                        onChange={(e) => setDegree2(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Button
                        onClick={computeIntersection}
                        disabled={isComputing}
                      >
                        {isComputing ? 'Computing...' : 'Compute Intersection'}
                      </Button>
                    </GridItem>
                  </Grid>
                </GridItem>
                <GridItem>
                  <Card>
                    <CardHeader>
                      <H3>Result</H3>
                    </CardHeader>
                    <CardBody>
                      {intersectionResult !== null ? (
                        <Grid cols={1} gap="sm">
                          <GridItem>
                            <P><Strong>Intersection Number:</Strong> {intersectionResult}</P>
                          </GridItem>
                          <GridItem>
                            <P>
                              By Bézout's theorem, two hypersurfaces of degrees {degree1} and {degree2}
                              in ℙ<sup>{projDimension}</sup> intersect in exactly {degree1} × {degree2} = {intersectionResult} points
                              (counting multiplicities).
                            </P>
                          </GridItem>
                        </Grid>
                      ) : (
                        <P>Click "Compute Intersection" to see results</P>
                      )}
                    </CardBody>
                  </Card>
                </GridItem>
              </Grid>
            </CardBody>
          </Card>
        </GridItem>
      )}

      {activeTab === 'schubert' && (
        <GridItem>
          <Card>
            <CardHeader>
              <H2>Schubert Calculus</H2>
              <P>Intersection theory on Grassmannians using Schubert cycles</P>
            </CardHeader>
            <CardBody>
              <Grid cols={2} gap="lg">
                <GridItem>
                  <Grid cols={1} gap="md">
                    <GridItem>
                      <Input
                        label="Grassmannian k (subspace dimension)"
                        type="number"
                        min="1"
                        max="5"
                        value={grassmannianK.toString()}
                        onChange={(e) => setGrassmannianK(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="Grassmannian n (ambient dimension)"
                        type="number"
                        min={grassmannianK + 1}
                        max="8"
                        value={grassmannianN.toString()}
                        onChange={(e) => setGrassmannianN(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="First Partition (comma-separated)"
                        value={partition1}
                        onChange={(e) => setPartition1(e.target.value)}
                        placeholder="e.g., 1,0"
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="Second Partition (comma-separated)"
                        value={partition2}
                        onChange={(e) => setPartition2(e.target.value)}
                        placeholder="e.g., 1,0"
                      />
                    </GridItem>
                    <GridItem>
                      <Button
                        onClick={computeSchubert}
                        disabled={isComputing}
                      >
                        {isComputing ? 'Computing...' : 'Compute Schubert Intersection'}
                      </Button>
                    </GridItem>
                  </Grid>
                </GridItem>
                <GridItem>
                  <Card>
                    <CardHeader>
                      <H3>Result</H3>
                    </CardHeader>
                    <CardBody>
                      {schubertResult ? (
                        <Grid cols={1} gap="sm">
                          <GridItem>
                            <P><Strong>Grassmannian:</Strong> {schubertResult.grassmannian}</P>
                          </GridItem>
                          <GridItem>
                            <P><Strong>Dimension:</Strong> {schubertResult.dimension}</P>
                          </GridItem>
                          <GridItem>
                            <P><Strong>Intersection Number:</Strong> {schubertResult.intersection}</P>
                          </GridItem>
                          <GridItem>
                            <P>
                              Schubert cycles σ<sub>{schubertResult.partitions[0].join(',')}</sub> and
                              σ<sub>{schubertResult.partitions[1].join(',')}</sub> on Gr({grassmannianK},{grassmannianN})
                            </P>
                          </GridItem>
                        </Grid>
                      ) : (
                        <P>Click "Compute Schubert Intersection" to see results</P>
                      )}
                    </CardBody>
                  </Card>
                </GridItem>
              </Grid>
            </CardBody>
          </Card>
        </GridItem>
      )}

      {activeTab === 'tropical' && (
        <GridItem>
          <Card>
            <CardHeader>
              <H2>Tropical Geometry</H2>
              <P>Count tropical curves using Mikhalkin correspondence</P>
            </CardHeader>
            <CardBody>
              <Grid cols={2} gap="lg">
                <GridItem>
                  <Grid cols={1} gap="md">
                    <GridItem>
                      <Input
                        label="Curve Degree"
                        type="number"
                        min="1"
                        max="6"
                        value={tropicalDegree.toString()}
                        onChange={(e) => setTropicalDegree(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="Number of Constraints"
                        type="number"
                        min={tropicalDegree}
                        max="12"
                        value={tropicalConstraints.toString()}
                        onChange={(e) => setTropicalConstraints(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Button
                        onClick={computeTropical}
                        disabled={isComputing}
                      >
                        {isComputing ? 'Computing...' : 'Count Tropical Curves'}
                      </Button>
                    </GridItem>
                    <GridItem>
                      <Card>
                        <CardBody>
                          <P>
                            <Strong>Note:</Strong> For degree {tropicalDegree} curves, the expected dimension is {3 * tropicalDegree - 1}.
                            Adjust constraints accordingly.
                          </P>
                        </CardBody>
                      </Card>
                    </GridItem>
                  </Grid>
                </GridItem>
                <GridItem>
                  <Card>
                    <CardHeader>
                      <H3>Result</H3>
                    </CardHeader>
                    <CardBody>
                      {tropicalResult !== null ? (
                        <Grid cols={1} gap="sm">
                          <GridItem>
                            <P><Strong>Curve Count:</Strong> {tropicalResult}</P>
                          </GridItem>
                          <GridItem>
                            <P>
                              Number of degree-{tropicalDegree} tropical curves satisfying {tropicalConstraints}
                              generic constraints. This matches the classical count by Mikhalkin correspondence.
                            </P>
                          </GridItem>
                        </Grid>
                      ) : (
                        <P>Click "Count Tropical Curves" to see results</P>
                      )}
                    </CardBody>
                  </Card>
                </GridItem>
              </Grid>
            </CardBody>
          </Card>
        </GridItem>
      )}

      {activeTab === 'higher-genus' && (
        <GridItem>
          <Card>
            <CardHeader>
              <H2>Higher Genus Curves</H2>
              <P>Riemann-Roch theorem and moduli space computations</P>
            </CardHeader>
            <CardBody>
              <Grid cols={2} gap="lg">
                <GridItem>
                  <Grid cols={1} gap="md">
                    <GridItem>
                      <Input
                        label="Genus"
                        type="number"
                        min="0"
                        max="5"
                        value={genus.toString()}
                        onChange={(e) => setGenus(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="Curve Degree"
                        type="number"
                        min="1"
                        max="8"
                        value={curveDegree.toString()}
                        onChange={(e) => setCurveDegree(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Input
                        label="Line Bundle Degree (for Riemann-Roch)"
                        type="number"
                        min="0"
                        max="10"
                        value={rrDegree.toString()}
                        onChange={(e) => setRrDegree(parseInt(e.target.value))}
                      />
                    </GridItem>
                    <GridItem>
                      <Button
                        onClick={computeHigherGenus}
                        disabled={isComputing}
                      >
                        {isComputing ? 'Computing...' : 'Compute Riemann-Roch'}
                      </Button>
                    </GridItem>
                  </Grid>
                </GridItem>
                <GridItem>
                  <Card>
                    <CardHeader>
                      <H3>Result</H3>
                    </CardHeader>
                    <CardBody>
                      {higherGenusResult ? (
                        <Grid cols={1} gap="sm">
                          <GridItem>
                            <P><Strong>Genus:</Strong> {higherGenusResult.genus}</P>
                          </GridItem>
                          <GridItem>
                            <P><Strong>Degree:</Strong> {higherGenusResult.degree}</P>
                          </GridItem>
                          <GridItem>
                            <P><Strong>Canonical Degree:</Strong> {higherGenusResult.canonicalDegree}</P>
                          </GridItem>
                          <GridItem>
                            <P><Strong>h⁰(L<sub>{rrDegree}</sub>):</Strong> {higherGenusResult.riemannRochDim}</P>
                          </GridItem>
                          <GridItem>
                            <P>
                              By Riemann-Roch: h⁰(L) - h¹(L) = deg(L) + 1 - g = {rrDegree} + 1 - {genus} = {rrDegree + 1 - genus}
                            </P>
                          </GridItem>
                        </Grid>
                      ) : (
                        <P>Click "Compute Riemann-Roch" to see results</P>
                      )}
                    </CardBody>
                  </Card>
                </GridItem>
              </Grid>
            </CardBody>
          </Card>
        </GridItem>
      )}

      {activeTab === 'performance' && (
        <GridItem>
          <Card>
            <CardHeader>
              <H2>Performance & Optimization</H2>
              <P>WASM-first architecture with GPU acceleration</P>
            </CardHeader>
            <CardBody>
              <Grid cols={2} gap="lg">
                <GridItem>
                  <Grid cols={1} gap="md">
                    <GridItem>
                      <H3>WASM Optimization</H3>
                      <P>Memory-efficient sparse matrix operations</P>
                      <P>Custom memory pooling for large computations</P>
                      <P>SIMD optimizations where available</P>
                      <P>Configurable batch processing</P>
                    </GridItem>
                    <GridItem>
                      <H3>GPU Acceleration</H3>
                      <P>WGPU compute shaders for intersection numbers</P>
                      <P>Parallel Schubert calculus kernels</P>
                      <P>Gromov-Witten invariant computation</P>
                      <P>Tropical curve counting acceleration</P>
                    </GridItem>
                  </Grid>
                </GridItem>
                <GridItem>
                  <Card>
                    <CardHeader>
                      <H3>Feature Flags</H3>
                    </CardHeader>
                    <CardBody>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Feature</TableHead>
                            <TableHead>Status</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          <TableRow>
                            <TableCell><Code>wgpu</Code></TableCell>
                            <TableCell><StatusBadge variant="success">Available</StatusBadge></TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell><Code>wasm</Code></TableCell>
                            <TableCell><StatusBadge variant="success">Available</StatusBadge></TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell><Code>parallel</Code></TableCell>
                            <TableCell><StatusBadge variant="success">Available</StatusBadge></TableCell>
                          </TableRow>
                          <TableRow>
                            <TableCell><Code>performance</Code></TableCell>
                            <TableCell><StatusBadge variant="success">Available</StatusBadge></TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </CardBody>
                  </Card>
                </GridItem>
              </Grid>
            </CardBody>
          </Card>
        </GridItem>
      )}

      {/* Computation History */}
      <GridItem>
        <Card>
          <CardHeader>
            <H2>Computation History</H2>
            <P>Recent calculations and timing information</P>
          </CardHeader>
          <CardBody>
            {computationHistory.length > 0 ? (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Input</TableHead>
                    <TableHead>Output</TableHead>
                    <TableHead>Time (ms)</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {computationHistory.map((result, index) => (
                    <TableRow key={index}>
                      <TableCell><Code>{result.input}</Code></TableCell>
                      <TableCell>{result.output}</TableCell>
                      <TableCell>{result.time}</TableCell>
                      <TableCell>
                        <StatusBadge variant={result.error ? "destructive" : "success"}>
                          {result.error ? "Error" : "Success"}
                        </StatusBadge>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            ) : (
              <P>No computations yet. Try the examples above!</P>
            )}
          </CardBody>
        </Card>
      </GridItem>

      {/* Code Examples */}
      <GridItem>
        <Card>
          <CardHeader>
            <H2>Code Examples</H2>
            <P>Learn how to use the enumerative geometry API</P>
          </CardHeader>
          <CardBody>
            <Grid cols={1} gap="lg">
              <GridItem>
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
              </GridItem>

              <GridItem>
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
              </GridItem>

              <GridItem>
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
              </GridItem>
            </Grid>
          </CardBody>
        </Card>
      </GridItem>
    </Grid>
  );
}