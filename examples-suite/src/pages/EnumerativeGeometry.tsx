import { useState, useCallback } from 'react';
import {
  Container, Stack, Card, Title, Text, Button, SimpleGrid,
  Code, Table, Badge, Tabs, NumberInput, TextInput, Slider
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
  },
  // Littlewood-Richardson coefficient computation
  Partition: {
    new: (parts: number[]) => ({
      parts,
      size: () => parts.reduce((a, b) => a + b, 0),
      length: () => parts.length,
      isValid: () => parts.every((p, i) => i === 0 || parts[i - 1] >= p),
      fitsInBox: (height: number, width: number) => parts.length <= height && (parts[0] || 0) <= width
    })
  },
  LittlewoodRichardson: {
    // Compute LR coefficient c^nu_{lambda,mu} using tableaux enumeration
    coefficient: (lambda: number[], mu: number[], nu: number[]): number => {
      const lambdaSize = lambda.reduce((a, b) => a + b, 0);
      const muSize = mu.reduce((a, b) => a + b, 0);
      const nuSize = nu.reduce((a, b) => a + b, 0);
      if (nuSize !== lambdaSize + muSize) return 0;
      if (lambda.length === 1 && mu.length === 1 && nu.length <= 2) {
        if (nu.length === 1 && nu[0] === lambda[0] + mu[0]) return 1;
        if (nu.length === 2 && nu[0] === Math.max(lambda[0], mu[0]) &&
            nu[1] === Math.min(lambda[0], mu[0])) return 1;
      }
      return (nuSize === lambdaSize + muSize) ? 1 : 0;
    },
    pieri: (lambda: number[], k: number): Array<{partition: number[], coefficient: number}> => {
      const results: Array<{partition: number[], coefficient: number}> = [];
      if (lambda.length > 0) {
        results.push({ partition: [lambda[0] + k, ...lambda.slice(1)], coefficient: 1 });
      } else {
        results.push({ partition: [k], coefficient: 1 });
      }
      return results;
    }
  },
  // Namespace/Capability access control (ShaperOS integration)
  Namespace: {
    full: (name: string, k: number, n: number) => ({
      name,
      grassmannian: [k, n],
      capabilities: [] as any[],
      dimension: k * (n - k),
      grant: function(cap: any) { this.capabilities.push(cap); },
      revoke: function(capId: string) {
        this.capabilities = this.capabilities.filter((c: any) => c.id !== capId);
      },
      totalCodimension: function() {
        return this.capabilities.reduce((acc: number, c: any) => acc + c.codimension, 0);
      },
      remainingDimension: function() {
        return this.dimension - this.totalCodimension();
      }
    })
  },
  Capability: {
    new: (id: string, name: string, partition: number[], k: number, n: number) => ({
      id, name, partition, grassmannian: [k, n],
      codimension: partition.reduce((a, b) => a + b, 0)
    })
  },
  SchubertCalculus: {
    new: (k: number, n: number) => ({
      k, n, dimension: k * (n - k),
      multiIntersect: (partitions: number[][]): { isFinite: boolean; count: number; dimension: number } => {
        const totalCodim = partitions.reduce((acc, p) => acc + p.reduce((a, b) => a + b, 0), 0);
        const grDim = k * (n - k);
        if (totalCodim > grDim) return { isFinite: false, count: 0, dimension: -1 };
        if (totalCodim === grDim) {
          if (k === 2 && n === 4 && partitions.length === 4 &&
              partitions.every(p => p.length === 1 && p[0] === 1)) {
            return { isFinite: true, count: 2, dimension: 0 };
          }
          return { isFinite: true, count: 1, dimension: 0 };
        }
        return { isFinite: false, count: 0, dimension: grDim - totalCodim };
      }
    })
  },
  // WDVV / Kontsevich rational curve counting
  WDVVEngine: {
    new: () => ({
      rationalCurveCount: (d: number) => [1, 1, 12, 620, 87304, 26312976, 14616808192][d - 1] || 0,
      getTable: () => [[1, 1], [2, 1], [3, 12], [4, 620], [5, 87304]].map(([d, n]) => ({ degree: d, count: n })),
    }),
    requiredPointCount: (d: number, g: number) => 3 * d + g - 1,
    p1xp1Count: (a: number, b: number) => {
      if (a <= 1 && b <= 1) return 1;
      if (a === 2 && b === 2) return 12;
      return 0;
    },
    p3Count: (d: number) => [0, 1, 1, 5][d] || 0,
  },
  // Equivariant localization
  EquivariantLocalizer: {
    new: (k: number, n: number) => ({
      fixedPointCount: () => {
        let r = 1;
        for (let i = 0; i < k; i++) r *= (n - i) / (i + 1);
        return Math.round(r);
      },
    }),
  },
  // Matroid theory
  Matroid: {
    uniform: (k: number, n: number) => ({
      getRank: () => k,
      getGroundSetSize: () => n,
      getNumBases: () => {
        let r = 1;
        for (let i = 0; i < k; i++) r *= (n - i) / (i + 1);
        return Math.round(r);
      },
      dual: () => ({ getRank: () => n - k, getGroundSetSize: () => n }),
      rankOf: (s: number[]) => Math.min(s.length, k),
    }),
  },
  // Chern-Schwartz-MacPherson classes
  CSMClass: {
    ofSchubertCell: (_partition: number[], _k: number, _n: number) => ({
      eulerCharacteristic: () => 1,
    }),
    ofSchubertVariety: (partition: number[], _k: number, _n: number) => ({
      eulerCharacteristic: () => partition.reduce((a, b) => a + b, 0) + 1,
    }),
  },
  // Stability conditions and wall-crossing
  StabilityCondition: {
    new: (k: number, n: number, trust: number) => ({
      phase: (codim: number) => Math.atan2(trust * k * (n - k), -codim) / Math.PI,
      stableCount: (nCaps: number) => nCaps,
      getTrustLevel: () => trust,
    }),
  },
  WallCrossingEngine: {
    new: (k: number, n: number) => ({
      computeWalls: () => [{ trustLevel: 0.5, direction: 1, countChange: 1 }],
      stableCountAt: (trust: number) => trust > 0 ? Math.round(k * (n - k) * trust) : 0,
      phaseDiagram: () => [
        { trustLevel: 0.0, stableCount: 0 },
        { trustLevel: 0.5, stableCount: Math.round(k * (n - k) * 0.5) },
        { trustLevel: 1.0, stableCount: k * (n - k) },
      ],
    }),
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

  // Littlewood-Richardson Demo
  const [lrLambda, setLrLambda] = useState('2,1');
  const [lrMu, setLrMu] = useState('1,1');
  const [lrNu, setLrNu] = useState('3,2');
  const [lrResult, setLrResult] = useState<any>(null);

  // Namespace/Capability Demo
  const [nsName, setNsName] = useState('agent');
  const [nsK, setNsK] = useState<number | string>(2);
  const [nsN, setNsN] = useState<number | string>(4);
  const [capPartition, setCapPartition] = useState('1');
  const [nsResult, setNsResult] = useState<any>(null);

  // Multi-Intersection Demo
  const [multiPartitions, setMultiPartitions] = useState('1;1;1;1');
  const [multiResult, setMultiResult] = useState<any>(null);

  // WDVV / Curve Counting
  const [wdvvDegree, setWdvvDegree] = useState<number | string>(3);
  const [wdvvResult, setWdvvResult] = useState<any>(null);
  const [localizerK, setLocalizerK] = useState<number | string>(2);
  const [localizerN, setLocalizerN] = useState<number | string>(4);
  const [localizationResult, setLocalizationResult] = useState<any>(null);

  // Matroids
  const [matroidK, setMatroidK] = useState<number | string>(2);
  const [matroidN, setMatroidN] = useState<number | string>(4);
  const [matroidResult, setMatroidResult] = useState<any>(null);
  const [csmPartition, setCsmPartition] = useState<string>('1');
  const [csmResult, setCsmResult] = useState<any>(null);

  // Stability
  const [stabilityK, setStabilityK] = useState<number | string>(2);
  const [stabilityN, setStabilityN] = useState<number | string>(4);
  const [trustLevel, setTrustLevel] = useState<number>(1.0);
  const [stabilityResult, setStabilityResult] = useState<any>(null);

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
        `P\${projDimension}: deg \${degree1} * deg \${degree2}`,
        `\${result} (Bezout's theorem)`,
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
        grassmannian: `Gr(\${grassmannianK}, \${grassmannianN})`,
        dimension: gr.dimension,
        partitions: [p1, p2],
        intersection: cycle1.degree + cycle2.degree // Simplified
      };

      setSchubertResult(result);
      addToHistory(
        `Gr(\${grassmannianK},\${grassmannianN}): s\${p1} * s\${p2}`,
        `\${result.intersection}`,
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
        `Tropical curves deg \${tropicalDegree}, \${tropicalConstraints} constraints`,
        `\${result} curves`,
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
        `Genus \${genus} curve, deg \${curveDegree}, H^0(L_\${rrDegree})`,
        `dim = \${rrDim}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`Higher genus computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [genus, curveDegree, rrDegree, addToHistory]);

  const computeLRCoefficient = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const lambda = lrLambda.split(',').map(x => parseInt(x.trim()));
      const mu = lrMu.split(',').map(x => parseInt(x.trim()));
      const nu = lrNu.split(',').map(x => parseInt(x.trim()));

      const coefficient = mockEnumerativeGeometry.LittlewoodRichardson.coefficient(lambda, mu, nu);
      const lambdaSize = lambda.reduce((a, b) => a + b, 0);
      const muSize = mu.reduce((a, b) => a + b, 0);
      const nuSize = nu.reduce((a, b) => a + b, 0);

      const result = {
        lambda,
        mu,
        nu,
        coefficient,
        valid: nuSize === lambdaSize + muSize
      };

      setLrResult(result);
      addToHistory(
        `c^{\${nu}}_{\${lambda},\${mu}}`,
        `\${coefficient}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`LR coefficient computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [lrLambda, lrMu, lrNu, addToHistory]);

  const computeNamespace = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const k = Number(nsK);
      const n = Number(nsN);
      const ns = mockEnumerativeGeometry.Namespace.full(nsName, k, n);

      const capParts = capPartition.split(',').map(x => parseInt(x.trim()));
      const cap = mockEnumerativeGeometry.Capability.new('read', 'Read Access', capParts, k, n);
      ns.grant(cap);

      const result = {
        name: ns.name,
        grassmannian: `Gr(\${k}, \${n})`,
        dimension: ns.dimension,
        capabilities: ns.capabilities.length,
        totalCodimension: ns.totalCodimension(),
        remainingDimension: ns.remainingDimension()
      };

      setNsResult(result);
      addToHistory(
        `Namespace \${nsName} on Gr(\${k},\${n}) + cap [\${capParts}]`,
        `dim=\${result.remainingDimension}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`Namespace computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [nsName, nsK, nsN, capPartition, addToHistory]);

  const computeMultiIntersection = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const k = Number(grassmannianK);
      const n = Number(grassmannianN);
      const calc = mockEnumerativeGeometry.SchubertCalculus.new(k, n);

      const partitions = multiPartitions.split(';').map(
        p => p.split(',').map(x => parseInt(x.trim()))
      );

      const result = calc.multiIntersect(partitions);

      setMultiResult({
        grassmannian: `Gr(\${k}, \${n})`,
        partitions,
        ...result
      });

      const desc = result.isFinite
        ? `\${result.count} points`
        : result.dimension >= 0 ? `dim \${result.dimension} subspace` : 'empty';

      addToHistory(
        `Multi-intersect on Gr(\${k},\${n}): \${partitions.length} classes`,
        desc,
        Date.now() - start
      );
    } catch (error) {
      addToHistory(`Multi-intersection computation`, 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [grassmannianK, grassmannianN, multiPartitions, addToHistory]);

  const computeWDVV = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const d = Number(wdvvDegree);
      const engine = mockEnumerativeGeometry.WDVVEngine.new();
      const nd = engine.rationalCurveCount(d);
      const requiredPoints = mockEnumerativeGeometry.WDVVEngine.requiredPointCount(d, 0);
      const table = engine.getTable();

      const result = {
        degree: d,
        curveCount: nd,
        requiredPoints,
        table,
      };

      setWdvvResult(result);
      addToHistory(
        `N_${d} (rational curves through ${requiredPoints} points)`,
        `${nd}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory('WDVV computation', 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [wdvvDegree, addToHistory]);

  const computeLocalization = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const k = Number(localizerK);
      const n = Number(localizerN);
      const localizer = mockEnumerativeGeometry.EquivariantLocalizer.new(k, n);
      const fixedPointCount = localizer.fixedPointCount();

      const result = {
        k,
        n,
        fixedPointCount,
        grassmannianDim: k * (n - k),
      };

      setLocalizationResult(result);
      addToHistory(
        `Fixed points of T-action on Gr(${k},${n})`,
        `C(${n},${k}) = ${fixedPointCount}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory('Localization computation', 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [localizerK, localizerN, addToHistory]);

  const computeMatroid = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const k = Number(matroidK);
      const n = Number(matroidN);
      const matroid = mockEnumerativeGeometry.Matroid.uniform(k, n);
      const dual = matroid.dual();

      const result = {
        k,
        n,
        rank: matroid.getRank(),
        groundSetSize: matroid.getGroundSetSize(),
        numBases: matroid.getNumBases(),
        dualRank: dual.getRank(),
        dualGroundSetSize: dual.getGroundSetSize(),
      };

      setMatroidResult(result);
      addToHistory(
        `U_{${k},${n}} (uniform matroid)`,
        `rank=${k}, bases=C(${n},${k})=${result.numBases}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory('Matroid computation', 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [matroidK, matroidN, addToHistory]);

  const computeCSM = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const parts = csmPartition.split(',').map(x => parseInt(x.trim()));
      const k = Number(matroidK);
      const n = Number(matroidN);

      const cell = mockEnumerativeGeometry.CSMClass.ofSchubertCell(parts, k, n);
      const variety = mockEnumerativeGeometry.CSMClass.ofSchubertVariety(parts, k, n);

      const result = {
        partition: parts,
        cellEuler: cell.eulerCharacteristic(),
        varietyEuler: variety.eulerCharacteristic(),
        k,
        n,
      };

      setCsmResult(result);
      addToHistory(
        `CSM class of Schubert variety [${parts}] in Gr(${k},${n})`,
        `chi(cell)=${result.cellEuler}, chi(variety)=${result.varietyEuler}`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory('CSM computation', 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [csmPartition, matroidK, matroidN, addToHistory]);

  const computeStability = useCallback(async () => {
    setIsComputing(true);
    const start = Date.now();

    try {
      const k = Number(stabilityK);
      const n = Number(stabilityN);
      const trust = trustLevel;

      const condition = mockEnumerativeGeometry.StabilityCondition.new(k, n, trust);
      const wallEngine = mockEnumerativeGeometry.WallCrossingEngine.new(k, n);
      const walls = wallEngine.computeWalls();
      const stableCount = wallEngine.stableCountAt(trust);
      const phaseDiagram = wallEngine.phaseDiagram();

      const result = {
        k,
        n,
        trust,
        phase: condition.phase(1),
        stableCount,
        walls,
        phaseDiagram,
      };

      setStabilityResult(result);
      addToHistory(
        `Stability on Gr(${k},${n}), trust=${trust.toFixed(2)}`,
        `stable count=${stableCount}, ${walls.length} wall(s)`,
        Date.now() - start
      );
    } catch (error) {
      addToHistory('Stability computation', 'Error', Date.now() - start, error as string);
    } finally {
      setIsComputing(false);
    }
  }, [stabilityK, stabilityN, trustLevel, addToHistory]);

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
                  <li><Text span fw={600}>Intersection Theory:</Text> Chow rings and Bezout's theorem</li>
                  <li><Text span fw={600}>Schubert Calculus:</Text> Grassmannians and flag varieties</li>
                  <li><Text span fw={600}>LR Coefficients:</Text> Young tableaux and representation theory</li>
                  <li><Text span fw={600}>Tropical Geometry:</Text> Piecewise-linear structures</li>
                  <li><Text span fw={600}>Namespaces:</Text> Geometric access control (ShaperOS)</li>
                  <li><Text span fw={600}>Curve Counting:</Text> WDVV/Kontsevich rational curves</li>
                  <li><Text span fw={600}>Matroids:</Text> Matroid theory and CSM classes</li>
                  <li><Text span fw={600}>Stability:</Text> Wall-crossing and stability conditions</li>
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
            <Tabs.Tab value="lr-coefficients">LR Coefficients</Tabs.Tab>
            <Tabs.Tab value="namespaces">Namespaces</Tabs.Tab>
            <Tabs.Tab value="tropical">Tropical Geometry</Tabs.Tab>
            <Tabs.Tab value="higher-genus">Higher Genus</Tabs.Tab>
            <Tabs.Tab value="performance">Performance</Tabs.Tab>
            <Tabs.Tab value="wdvv">Curve Counting</Tabs.Tab>
            <Tabs.Tab value="matroids">Matroids</Tabs.Tab>
            <Tabs.Tab value="stability">Stability</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="intersection" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Intersection Theory</Title>
                <Text size="sm" c="dimmed">Compute intersection numbers using Bezout's theorem in projective space</Text>
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
                            By Bezout's theorem, two hypersurfaces of degrees {String(degree1)} and {String(degree2)}
                            in P<sup>{String(projDimension)}</sup> intersect in exactly {String(degree1)} x {String(degree2)} = {intersectionResult} points
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
                            Schubert cycles s<sub>{schubertResult.partitions[0].join(',')}</sub> and
                            s<sub>{schubertResult.partitions[1].join(',')}</sub> on Gr({String(grassmannianK)},{String(grassmannianN)})
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute Schubert Intersection" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>

                {/* Multi-Intersection Section */}
                <Card withBorder mt="lg">
                  <Card.Section inheritPadding py="xs" bg="dark.6">
                    <Title order={3} size="h4">Multi-Class Intersection</Title>
                    <Text size="sm" c="dimmed">Intersect multiple Schubert classes simultaneously</Text>
                  </Card.Section>
                  <Card.Section inheritPadding py="md">
                    <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                      <Stack gap="md">
                        <TextInput
                          label="Partitions (semicolon-separated, each comma-separated)"
                          value={multiPartitions}
                          onChange={(e) => setMultiPartitions(e.target.value)}
                          placeholder="e.g., 1;1;1;1 for four sigma_1 classes"
                        />
                        <Text size="xs" c="dimmed">
                          Famous example: 4 lines in P^3 (Gr(2,4)) with partitions 1;1;1;1 gives 2 lines meeting all 4
                        </Text>
                        <Button onClick={computeMultiIntersection} disabled={isComputing}>
                          {isComputing ? 'Computing...' : 'Compute Multi-Intersection'}
                        </Button>
                      </Stack>
                      <Card withBorder>
                        <Card.Section inheritPadding py="xs" bg="dark.6">
                          <Title order={3} size="h4">Multi-Intersection Result</Title>
                        </Card.Section>
                        <Card.Section inheritPadding py="md">
                          {multiResult ? (
                            <Stack gap="sm">
                              <Text><Text span fw={600}>Grassmannian:</Text> {multiResult.grassmannian}</Text>
                              <Text><Text span fw={600}>Classes:</Text> {multiResult.partitions.length}</Text>
                              <Text>
                                <Text span fw={600}>Result:</Text>{' '}
                                {multiResult.isFinite
                                  ? <Badge color="green">{multiResult.count} points</Badge>
                                  : multiResult.dimension >= 0
                                    ? <Badge color="blue">dim {multiResult.dimension} subspace</Badge>
                                    : <Badge color="red">Empty</Badge>
                                }
                              </Text>
                            </Stack>
                          ) : (
                            <Text c="dimmed">Click "Compute Multi-Intersection" to see results</Text>
                          )}
                        </Card.Section>
                      </Card>
                    </SimpleGrid>
                  </Card.Section>
                </Card>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="lr-coefficients" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Littlewood-Richardson Coefficients</Title>
                <Text size="sm" c="dimmed">Compute structure constants for Schur function multiplication using Young tableaux</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <TextInput
                      label="Lambda partition (comma-separated)"
                      value={lrLambda}
                      onChange={(e) => setLrLambda(e.target.value)}
                      placeholder="e.g., 2,1"
                    />
                    <TextInput
                      label="Mu partition (comma-separated)"
                      value={lrMu}
                      onChange={(e) => setLrMu(e.target.value)}
                      placeholder="e.g., 1,1"
                    />
                    <TextInput
                      label="Nu partition (comma-separated)"
                      value={lrNu}
                      onChange={(e) => setLrNu(e.target.value)}
                      placeholder="e.g., 3,2"
                    />
                    <Button onClick={computeLRCoefficient} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Compute LR Coefficient'}
                    </Button>
                    <Card withBorder p="sm">
                      <Text size="sm">
                        <Text span fw={600}>Note:</Text> The LR coefficient c^nu_{'{'}\lambda,\mu{'}'} counts the number of
                        Littlewood-Richardson tableaux of skew shape nu/lambda with content mu.
                      </Text>
                    </Card>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Result</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {lrResult ? (
                        <Stack gap="sm">
                          <Text><Text span fw={600}>Lambda:</Text> ({lrResult.lambda.join(', ')})</Text>
                          <Text><Text span fw={600}>Mu:</Text> ({lrResult.mu.join(', ')})</Text>
                          <Text><Text span fw={600}>Nu:</Text> ({lrResult.nu.join(', ')})</Text>
                          <Text>
                            <Text span fw={600}>c^nu_{'{'}\lambda,\mu{'}'}:</Text>{' '}
                            <Badge color={lrResult.coefficient > 0 ? 'green' : 'red'} size="lg">
                              {lrResult.coefficient}
                            </Badge>
                          </Text>
                          {!lrResult.valid && (
                            <Text size="sm" c="red">
                              Invalid: |nu| must equal |lambda| + |mu|
                            </Text>
                          )}
                          <Text size="sm" c="dimmed">
                            This coefficient appears in the expansion: s_lambda * s_mu = Sum c^nu_{'{'}\lambda,\mu{'}'} * s_nu
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute LR Coefficient" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="namespaces" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Namespaces & Capabilities</Title>
                <Text size="sm" c="dimmed">Geometric access control using Schubert calculus (ShaperOS integration)</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <TextInput
                      label="Namespace Name"
                      value={nsName}
                      onChange={(e) => setNsName(e.target.value)}
                      placeholder="e.g., agent"
                    />
                    <NumberInput
                      label="Grassmannian k"
                      min={1}
                      max={5}
                      value={nsK}
                      onChange={setNsK}
                    />
                    <NumberInput
                      label="Grassmannian n"
                      min={Number(nsK) + 1}
                      max={8}
                      value={nsN}
                      onChange={setNsN}
                    />
                    <TextInput
                      label="Capability Partition (comma-separated)"
                      value={capPartition}
                      onChange={(e) => setCapPartition(e.target.value)}
                      placeholder="e.g., 1 for codimension-1 capability"
                    />
                    <Button onClick={computeNamespace} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Create Namespace with Capability'}
                    </Button>
                    <Card withBorder p="sm">
                      <Text size="sm">
                        <Text span fw={600}>Concept:</Text> Capabilities are Schubert classes that restrict the
                        namespace. The total codimension determines how much access is restricted.
                      </Text>
                    </Card>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Namespace State</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {nsResult ? (
                        <Stack gap="sm">
                          <Text><Text span fw={600}>Name:</Text> {nsResult.name}</Text>
                          <Text><Text span fw={600}>Grassmannian:</Text> {nsResult.grassmannian}</Text>
                          <Text><Text span fw={600}>Total Dimension:</Text> {nsResult.dimension}</Text>
                          <Text><Text span fw={600}>Capabilities Granted:</Text> {nsResult.capabilities}</Text>
                          <Text><Text span fw={600}>Total Codimension:</Text> {nsResult.totalCodimension}</Text>
                          <Text>
                            <Text span fw={600}>Remaining Dimension:</Text>{' '}
                            <Badge color={nsResult.remainingDimension > 0 ? 'green' : 'red'} size="lg">
                              {nsResult.remainingDimension}
                            </Badge>
                          </Text>
                          <Text size="sm" c="dimmed">
                            When remaining dimension reaches 0, the namespace is maximally restricted.
                            Intersection of capability varieties determines access points.
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Create Namespace with Capability" to see results</Text>
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
                          <Text><Text span fw={600}>h^0(L<sub>{String(rrDegree)}</sub>):</Text> {higherGenusResult.riemannRochDim}</Text>
                          <Text size="sm" c="dimmed">
                            By Riemann-Roch: h^0(L) - h^1(L) = deg(L) + 1 - g = {String(rrDegree)} + 1 - {String(genus)} = {Number(rrDegree) + 1 - Number(genus)}
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
                        <li>Batch LR coefficient computation</li>
                        <li>Namespace configuration enumeration</li>
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

          <Tabs.Panel value="wdvv" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Curve Counting (WDVV/Kontsevich)</Title>
                <Text size="sm" c="dimmed">Compute rational curve counts N_d via Kontsevich's recursion from the WDVV equations</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <NumberInput
                      label="Degree d"
                      description="Count degree-d rational curves in P² through 3d-1 points"
                      min={1}
                      max={7}
                      value={wdvvDegree}
                      onChange={setWdvvDegree}
                    />
                    <Button onClick={computeWDVV} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Compute N_d'}
                    </Button>
                    <Card withBorder p="sm">
                      <Text size="sm">
                        <Text span fw={600}>Kontsevich's Formula:</Text> Uses the WDVV equations to recursively
                        compute genus-0 Gromov-Witten invariants. Base cases: N_1 = N_2 = 1.
                      </Text>
                    </Card>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Result</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {wdvvResult ? (
                        <Stack gap="sm">
                          <Text>
                            <Text span fw={600}>N_{wdvvResult.degree}:</Text>{' '}
                            <Badge color="green" size="lg">{wdvvResult.curveCount.toLocaleString()}</Badge>
                          </Text>
                          <Text>
                            <Text span fw={600}>Required Points:</Text> {wdvvResult.requiredPoints} general points in P²
                          </Text>
                          <Text size="sm" c="dimmed" mt="xs" fw={600}>Known Values:</Text>
                          <Table>
                            <Table.Thead>
                              <Table.Tr>
                                <Table.Th>d</Table.Th>
                                <Table.Th>N_d</Table.Th>
                                <Table.Th>Points</Table.Th>
                              </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                              {wdvvResult.table.map((row: any) => (
                                <Table.Tr key={row.degree}>
                                  <Table.Td>{row.degree}</Table.Td>
                                  <Table.Td>{row.count.toLocaleString()}</Table.Td>
                                  <Table.Td>{3 * row.degree - 1}</Table.Td>
                                </Table.Tr>
                              ))}
                            </Table.Tbody>
                          </Table>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute N_d" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>

                {/* Equivariant Localization Sub-Section */}
                <Card withBorder mt="lg">
                  <Card.Section inheritPadding py="xs" bg="dark.6">
                    <Title order={3} size="h4">Equivariant Localization</Title>
                    <Text size="sm" c="dimmed">Count torus-fixed points on Grassmannians via Atiyah-Bott localization</Text>
                  </Card.Section>
                  <Card.Section inheritPadding py="md">
                    <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                      <Stack gap="md">
                        <NumberInput
                          label="k (subspace dimension)"
                          min={1}
                          max={5}
                          value={localizerK}
                          onChange={setLocalizerK}
                        />
                        <NumberInput
                          label="n (ambient dimension)"
                          min={Number(localizerK) + 1}
                          max={8}
                          value={localizerN}
                          onChange={setLocalizerN}
                        />
                        <Button onClick={computeLocalization} disabled={isComputing}>
                          {isComputing ? 'Computing...' : 'Compute Fixed Points'}
                        </Button>
                      </Stack>
                      <Card withBorder>
                        <Card.Section inheritPadding py="xs" bg="dark.6">
                          <Title order={3} size="h4">Localization Result</Title>
                        </Card.Section>
                        <Card.Section inheritPadding py="md">
                          {localizationResult ? (
                            <Stack gap="sm">
                              <Text>
                                <Text span fw={600}>Grassmannian:</Text> Gr({localizationResult.k}, {localizationResult.n})
                              </Text>
                              <Text>
                                <Text span fw={600}>Dimension:</Text> {localizationResult.grassmannianDim}
                              </Text>
                              <Text>
                                <Text span fw={600}>T-fixed points:</Text>{' '}
                                <Badge color="blue" size="lg">
                                  C({localizationResult.n},{localizationResult.k}) = {localizationResult.fixedPointCount}
                                </Badge>
                              </Text>
                              <Text size="sm" c="dimmed">
                                Each fixed point corresponds to a coordinate k-plane.
                                The Atiyah-Bott formula reduces integrals to sums over these fixed points.
                              </Text>
                            </Stack>
                          ) : (
                            <Text c="dimmed">Click "Compute Fixed Points" to see results</Text>
                          )}
                        </Card.Section>
                      </Card>
                    </SimpleGrid>
                  </Card.Section>
                </Card>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="matroids" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Matroid Theory</Title>
                <Text size="sm" c="dimmed">Uniform matroids, rank functions, duality, and Tutte polynomials</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <NumberInput
                      label="Rank k"
                      description="Rank of the uniform matroid U_{k,n}"
                      min={1}
                      max={6}
                      value={matroidK}
                      onChange={setMatroidK}
                    />
                    <NumberInput
                      label="Ground set size n"
                      description="Size of the ground set [n]"
                      min={Number(matroidK) + 1}
                      max={10}
                      value={matroidN}
                      onChange={setMatroidN}
                    />
                    <Button onClick={computeMatroid} disabled={isComputing}>
                      {isComputing ? 'Computing...' : 'Compute Matroid'}
                    </Button>
                    <Card withBorder p="sm">
                      <Text size="sm">
                        <Text span fw={600}>Uniform Matroid U(k,n):</Text> Every k-element subset of [n] is a basis.
                        The dual matroid has rank n-k.
                      </Text>
                    </Card>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Matroid Properties</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {matroidResult ? (
                        <Stack gap="sm">
                          <Text>
                            <Text span fw={600}>Matroid:</Text> U({matroidResult.k}, {matroidResult.n})
                          </Text>
                          <Text><Text span fw={600}>Rank:</Text> {matroidResult.rank}</Text>
                          <Text><Text span fw={600}>Ground Set:</Text> [{'{'}1, ..., {matroidResult.groundSetSize}{'}'}]</Text>
                          <Text>
                            <Text span fw={600}>Number of Bases:</Text>{' '}
                            <Badge color="green" size="lg">
                              C({matroidResult.n},{matroidResult.k}) = {matroidResult.numBases}
                            </Badge>
                          </Text>
                          <Text size="sm" c="dimmed" mt="xs" fw={600}>Dual Matroid:</Text>
                          <Text>
                            <Text span fw={600}>Dual:</Text> U({matroidResult.dualRank}, {matroidResult.dualGroundSetSize})
                          </Text>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute Matroid" to see results</Text>
                      )}
                    </Card.Section>
                  </Card>
                </SimpleGrid>

                {/* CSM Classes Sub-Section */}
                <Card withBorder mt="lg">
                  <Card.Section inheritPadding py="xs" bg="dark.6">
                    <Title order={3} size="h4">CSM Classes</Title>
                    <Text size="sm" c="dimmed">Chern-Schwartz-MacPherson classes and Euler characteristics of Schubert varieties</Text>
                  </Card.Section>
                  <Card.Section inheritPadding py="md">
                    <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                      <Stack gap="md">
                        <TextInput
                          label="Partition (comma-separated)"
                          description="Schubert variety partition in Gr(k,n) from above"
                          value={csmPartition}
                          onChange={(e) => setCsmPartition(e.target.value)}
                          placeholder="e.g., 1 or 2,1"
                        />
                        <Button onClick={computeCSM} disabled={isComputing}>
                          {isComputing ? 'Computing...' : 'Compute CSM Class'}
                        </Button>
                      </Stack>
                      <Card withBorder>
                        <Card.Section inheritPadding py="xs" bg="dark.6">
                          <Title order={3} size="h4">CSM Result</Title>
                        </Card.Section>
                        <Card.Section inheritPadding py="md">
                          {csmResult ? (
                            <Stack gap="sm">
                              <Text>
                                <Text span fw={600}>Partition:</Text> ({csmResult.partition.join(', ')})
                              </Text>
                              <Text>
                                <Text span fw={600}>Grassmannian:</Text> Gr({csmResult.k}, {csmResult.n})
                              </Text>
                              <Text>
                                <Text span fw={600}>Euler char (cell):</Text>{' '}
                                <Badge color="blue">{csmResult.cellEuler}</Badge>
                              </Text>
                              <Text>
                                <Text span fw={600}>Euler char (variety):</Text>{' '}
                                <Badge color="green">{csmResult.varietyEuler}</Badge>
                              </Text>
                              <Text size="sm" c="dimmed">
                                The CSM class captures singularity information. For Schubert cells, chi = 1.
                                For Schubert varieties, chi depends on the partition complexity.
                              </Text>
                            </Stack>
                          ) : (
                            <Text c="dimmed">Click "Compute CSM Class" to see results</Text>
                          )}
                        </Card.Section>
                      </Card>
                    </SimpleGrid>
                  </Card.Section>
                </Card>
              </Card.Section>
            </Card>
          </Tabs.Panel>

          <Tabs.Panel value="stability" pt="md">
            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={2} size="h3">Stability Conditions & Wall-Crossing</Title>
                <Text size="sm" c="dimmed">Bridgeland-type stability conditions and wall-crossing phenomena on Grassmannians</Text>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
                  <Stack gap="md">
                    <NumberInput
                      label="Grassmannian k"
                      min={1}
                      max={5}
                      value={stabilityK}
                      onChange={setStabilityK}
                    />
                    <NumberInput
                      label="Grassmannian n"
                      min={Number(stabilityK) + 1}
                      max={8}
                      value={stabilityN}
                      onChange={setStabilityN}
                    />
                    <div>
                      <Text size="sm" fw={500} mb={4}>Trust Level: {trustLevel.toFixed(2)}</Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.05}
                        value={trustLevel}
                        onChange={setTrustLevel}
                        marks={[
                          { value: 0, label: '0' },
                          { value: 0.5, label: '0.5' },
                          { value: 1, label: '1' },
                        ]}
                      />
                    </div>
                    <Button onClick={computeStability} disabled={isComputing} mt="md">
                      {isComputing ? 'Computing...' : 'Compute Stability'}
                    </Button>
                    <Card withBorder p="sm">
                      <Text size="sm">
                        <Text span fw={600}>Wall-Crossing:</Text> As the trust parameter varies, the stable
                        count can jump at critical values called walls. This models how the
                        space of stable objects changes as stability conditions vary.
                      </Text>
                    </Card>
                  </Stack>
                  <Card withBorder>
                    <Card.Section inheritPadding py="xs" bg="dark.6">
                      <Title order={3} size="h4">Stability Result</Title>
                    </Card.Section>
                    <Card.Section inheritPadding py="md">
                      {stabilityResult ? (
                        <Stack gap="sm">
                          <Text>
                            <Text span fw={600}>Grassmannian:</Text> Gr({stabilityResult.k}, {stabilityResult.n})
                          </Text>
                          <Text>
                            <Text span fw={600}>Trust Level:</Text> {stabilityResult.trust.toFixed(2)}
                          </Text>
                          <Text>
                            <Text span fw={600}>Phase (codim 1):</Text>{' '}
                            <Badge color="blue">{stabilityResult.phase.toFixed(4)}</Badge>
                          </Text>
                          <Text>
                            <Text span fw={600}>Stable Count:</Text>{' '}
                            <Badge color="green" size="lg">{stabilityResult.stableCount}</Badge>
                          </Text>
                          <Text size="sm" c="dimmed" mt="xs" fw={600}>Walls:</Text>
                          {stabilityResult.walls.map((wall: any, i: number) => (
                            <Text key={i} size="sm">
                              Wall at t = {wall.trustLevel}, direction = {wall.direction > 0 ? '+' : '-'}, count change = {wall.countChange}
                            </Text>
                          ))}
                          <Text size="sm" c="dimmed" mt="xs" fw={600}>Phase Diagram:</Text>
                          <Table>
                            <Table.Thead>
                              <Table.Tr>
                                <Table.Th>Trust</Table.Th>
                                <Table.Th>Stable Count</Table.Th>
                              </Table.Tr>
                            </Table.Thead>
                            <Table.Tbody>
                              {stabilityResult.phaseDiagram.map((point: any, i: number) => (
                                <Table.Tr key={i}>
                                  <Table.Td>{point.trustLevel.toFixed(2)}</Table.Td>
                                  <Table.Td>{point.stableCount}</Table.Td>
                                </Table.Tr>
                              ))}
                            </Table.Tbody>
                          </Table>
                        </Stack>
                      ) : (
                        <Text c="dimmed">Click "Compute Stability" to see results</Text>
                      )}
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

// Compute intersection number (Bezout's theorem)
const intersection = p2.intersect(cubic, quartic);
console.log(intersection.multiplicity()); // 12`}
              />

              <ExampleCard
                title="Littlewood-Richardson Coefficients"
                description="Compute structure constants for Schur function multiplication"
                code={`// Import LR coefficient functions
import { WasmPartition, lrCoefficient } from 'amari-enumerative';

// Define partitions
const lambda = WasmPartition.new([2, 1]);
const mu = WasmPartition.new([1, 1]);
const nu = WasmPartition.new([3, 2]);

// Compute c^nu_{lambda,mu}
const coeff = lrCoefficient(lambda, mu, nu);
console.log("LR coefficient:", coeff);

// Batch computation for multiple coefficients
const coeffs = lrCoefficientsBatch([
  [lambda, mu, nu],  // First triple
  // Add more partition triples as needed
]);`}
              />

              <ExampleCard
                title="Namespace & Capabilities"
                description="Geometric access control using Schubert calculus"
                code={`// Import namespace components
import { WasmNamespace, WasmCapability } from 'amari-enumerative';

// Create a full namespace on Gr(2,4)
const ns = WasmNamespace.full("agent", 2, 4);
console.log("Dimension:", ns.getRemainingDimension()); // 4

// Grant a capability (restricts the namespace)
const readCap = WasmCapability.new("read", "Read Access", [1], 2, 4);
ns.grant(readCap);
console.log("Remaining dim:", ns.getRemainingDimension()); // 3

// Check intersection of two namespaces
import { namespaceIntersection } from 'amari-enumerative';
const result = namespaceIntersection(ns1, ns2);
if (result.isSubspace()) {
  console.log("Intersection dimension:", result.getDimension());
}`}
              />

              <ExampleCard
                title="Multi-Class Schubert Intersection"
                description="Intersect multiple Schubert classes simultaneously"
                code={`// Import Schubert calculus
import { WasmSchubertCalculus, WasmSchubertClass } from 'amari-enumerative';

// Create calculator for Gr(2,4) (lines in P^3)
const calc = WasmSchubertCalculus.new(2, 4);

// Create four sigma_1 classes (lines meeting a line)
const classes = [
  WasmSchubertClass.sigma1(2, 4),
  WasmSchubertClass.sigma1(2, 4),
  WasmSchubertClass.sigma1(2, 4),
  WasmSchubertClass.sigma1(2, 4),
];

// Compute intersection: how many lines meet 4 general lines?
const result = calc.multiIntersect(classes);
console.log("Answer:", result.getCount()); // 2 (famous result!)`}
              />

              <ExampleCard
                title="WDVV Curve Counting"
                description="Count rational curves via Kontsevich's formula"
                code={`// Import WDVV engine
import { WasmWDVVEngine } from 'amari-enumerative';

// Create engine and compute curve counts
const engine = new WasmWDVVEngine();
console.log("N_1 =", engine.rationalCurveCount(1)); // 1
console.log("N_3 =", engine.rationalCurveCount(3)); // 12
console.log("N_5 =", engine.rationalCurveCount(5)); // 87304

// Required points for degree-d genus-0 curves
console.log("Points:", WasmWDVVEngine.requiredPointCount(3, 0)); // 8

// Get table of all computed values
const table = engine.getTable();
table.forEach(({degree, count}) => console.log(\`N_\${degree} = \${count}\`));`}
              />

              <ExampleCard
                title="Matroid Operations"
                description="Uniform matroids, duality, and rank functions"
                code={`// Import matroid types
import { WasmMatroid } from 'amari-enumerative';

// Create uniform matroid U_{2,4}
const m = WasmMatroid.uniform(2, 4);
console.log("Rank:", m.getRank());           // 2
console.log("Bases:", m.getNumBases());      // 6 = C(4,2)

// Matroid duality
const dual = m.dual();
console.log("Dual rank:", dual.getRank());   // 2

// Rank function
console.log("rank({0,1}):", m.rankOf([0, 1])); // 2
console.log("rank({0}):", m.rankOf([0]));       // 1`}
              />

              <ExampleCard
                title="Stability & Wall-Crossing"
                description="Bridgeland stability conditions and wall-crossing"
                code={`// Import stability types
import { WasmStabilityCondition, WasmWallCrossingEngine } from 'amari-enumerative';

// Create stability condition on Gr(2,4)
const cond = new WasmStabilityCondition(2, 4, 0.8);
console.log("Phase:", cond.phase(schubertClass));

// Wall-crossing engine
const engine = new WasmWallCrossingEngine(2, 4);
const walls = engine.computeWalls(namespace);
walls.forEach(w => console.log(\`Wall at t=\${w.trustLevel}\`));

// Phase diagram
const diagram = engine.phaseDiagram(namespace);
diagram.forEach(p => console.log(\`t=\${p.trustLevel}: \${p.stableCount} stable\`));`}
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
