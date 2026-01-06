import { useState, useEffect, useCallback } from "react";
import { Card, Title, Text, Button, Group, Box, Stack, Slider, SimpleGrid, Badge, NumberInput, Select, Tabs, Progress, SegmentedControl, ActionIcon, Tooltip } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";

// ============================================================================
// Geometric Algebra Visualization
// ============================================================================

interface Vector3D {
  x: number;
  y: number;
  z: number;
}

export function MultivectorVisualization() {
  const [coefficients, setCoefficients] = useState<number[]>([1, 0, 0, 0, 0, 0, 0, 0]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [animationAngle, setAnimationAngle] = useState(0);

  const labels = ['1', 'e₁', 'e₂', 'e₃', 'e₁₂', 'e₁₃', 'e₂₃', 'e₁₂₃'];
  const colors = ['#888', '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#6ab04c', '#be2edd'];

  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setAnimationAngle(prev => (prev + 0.05) % (2 * Math.PI));
      // Animate a rotating vector
      const cos = Math.cos(animationAngle);
      const sin = Math.sin(animationAngle);
      setCoefficients([0, cos, sin, 0, 0, 0, 0, 0]);
    }, 50);
    return () => clearInterval(interval);
  }, [isAnimating, animationAngle]);

  const magnitude = Math.sqrt(coefficients.reduce((sum, c) => sum + c * c, 0));
  const grade = coefficients.findIndex((c, i) => Math.abs(c) > 0.01 && i > 0) || 0;

  const setPreset = (preset: string) => {
    switch (preset) {
      case 'scalar': setCoefficients([2, 0, 0, 0, 0, 0, 0, 0]); break;
      case 'vector': setCoefficients([0, 1, 0, 0, 0, 0, 0, 0]); break;
      case 'bivector': setCoefficients([0, 0, 0, 0, 1, 0, 0, 0]); break;
      case 'rotor': setCoefficients([Math.cos(Math.PI/4), 0, 0, 0, Math.sin(Math.PI/4), 0, 0, 0]); break;
      case 'pseudoscalar': setCoefficients([0, 0, 0, 0, 0, 0, 0, 1]); break;
    }
  };

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Multivector Explorer</Title>
            <Text size="xs" c="dimmed">Interactive Cl(3,0,0) visualization</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant={isAnimating ? 'filled' : 'outline'} onClick={() => setIsAnimating(!isAnimating)}>
              {isAnimating ? 'Stop' : 'Animate'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Coefficient sliders */}
          <Stack gap="xs">
            <Text size="sm" fw={600}>Basis Coefficients</Text>
            {coefficients.map((value, i) => (
              <Group key={i} gap="xs" wrap="nowrap">
                <Badge size="sm" color={colors[i]} w={40} variant="light">{labels[i]}</Badge>
                <Slider
                  value={value}
                  onChange={(v) => {
                    const newCoeffs = [...coefficients];
                    newCoeffs[i] = v;
                    setCoefficients(newCoeffs);
                  }}
                  min={-2}
                  max={2}
                  step={0.01}
                  style={{ flex: 1 }}
                  disabled={isAnimating}
                />
                <Text size="xs" w={45} ta="right" ff="monospace">{value.toFixed(2)}</Text>
              </Group>
            ))}
          </Stack>

          {/* Visualization */}
          <Stack gap="md">
            <Box>
              <Text size="sm" fw={600} mb="xs">Visual Representation</Text>
              <svg viewBox="-120 -120 240 240" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
                {/* Grid */}
                <line x1="-100" y1="0" x2="100" y2="0" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
                <line x1="0" y1="-100" x2="0" y2="100" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />

                {/* Vector components */}
                {coefficients[1] !== 0 && (
                  <line x1="0" y1="0" x2={coefficients[1] * 40} y2="0" stroke={colors[1]} strokeWidth="3" markerEnd="url(#arrow)" />
                )}
                {coefficients[2] !== 0 && (
                  <line x1="0" y1="0" x2="0" y2={-coefficients[2] * 40} stroke={colors[2]} strokeWidth="3" />
                )}
                {coefficients[3] !== 0 && (
                  <line x1="0" y1="0" x2={coefficients[3] * 28} y2={-coefficients[3] * 28} stroke={colors[3]} strokeWidth="3" opacity="0.7" />
                )}

                {/* Bivector plane indicator */}
                {coefficients[4] !== 0 && (
                  <ellipse cx="0" cy="0" rx={Math.abs(coefficients[4]) * 30} ry={Math.abs(coefficients[4]) * 15} fill={colors[4]} opacity="0.3" stroke={colors[4]} strokeWidth="2" />
                )}

                {/* Center point */}
                <circle cx="0" cy="0" r="3" fill="white" />

                {/* Labels */}
                <text x="95" y="15" fontSize="10" fill="var(--mantine-color-dimmed)">e₁</text>
                <text x="5" y="-90" fontSize="10" fill="var(--mantine-color-dimmed)">e₂</text>
              </svg>
            </Box>

            {/* Presets */}
            <Group gap="xs">
              <Button size="xs" variant="light" onClick={() => setPreset('scalar')}>Scalar</Button>
              <Button size="xs" variant="light" onClick={() => setPreset('vector')}>Vector</Button>
              <Button size="xs" variant="light" onClick={() => setPreset('bivector')}>Bivector</Button>
              <Button size="xs" variant="light" onClick={() => setPreset('rotor')}>Rotor</Button>
            </Group>

            {/* Info */}
            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <SimpleGrid cols={2} spacing="xs">
                <Text size="xs"><strong>Magnitude:</strong> {magnitude.toFixed(4)}</Text>
                <Text size="xs"><strong>Dominant grade:</strong> {grade}</Text>
              </SimpleGrid>
              <Text size="xs" mt="xs" ff="monospace" c="dimmed">
                {coefficients.map((c, i) => c !== 0 ? `${c >= 0 && i > 0 ? '+' : ''}${c.toFixed(2)}${labels[i]}` : '').filter(Boolean).join(' ') || '0'}
              </Text>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Tropical Algebra Visualization
// ============================================================================

export function TropicalVisualization() {
  const [values, setValues] = useState<number[]>([3, 5, 2, 7, 4]);
  const [operation, setOperation] = useState<'add' | 'multiply'>('add');
  const [iterations, setIterations] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setIterations(prev => prev + 1);
      setValues(prev => {
        if (operation === 'add') {
          // Tropical addition: all values converge to max
          const maxVal = Math.max(...prev);
          return prev.map(v => v + (maxVal - v) * 0.1 + (Math.random() - 0.5) * 0.1);
        } else {
          // Tropical multiplication: values shift together
          return prev.map(v => v + 0.1);
        }
      });
    }, 100);
    return () => clearInterval(interval);
  }, [isRunning, operation]);

  const tropicalSum = Math.max(...values);
  const tropicalProduct = values.reduce((a, b) => a + b, 0);
  const maxIdx = values.indexOf(Math.max(...values));

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Tropical Algebra</Title>
            <Text size="xs" c="dimmed">⊕ = max, ⊗ = + (max-plus semiring)</Text>
          </div>
          <Group gap="xs">
            <SegmentedControl
              size="xs"
              value={operation}
              onChange={(v) => setOperation(v as 'add' | 'multiply')}
              data={[
                { label: '⊕ Add', value: 'add' },
                { label: '⊗ Mul', value: 'multiply' }
              ]}
            />
            <Button size="xs" variant={isRunning ? 'filled' : 'outline'} onClick={() => setIsRunning(!isRunning)}>
              {isRunning ? 'Pause' : 'Run'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <Stack gap="md">
          {/* Value bars */}
          <Stack gap="xs">
            {values.map((value, i) => (
              <Group key={i} gap="sm" wrap="nowrap">
                <Text size="xs" w={20} ff="monospace">x{i}</Text>
                <Box style={{ flex: 1 }}>
                  <Progress
                    value={(value / 10) * 100}
                    color={i === maxIdx ? 'cyan' : 'dark.3'}
                    size="lg"
                  />
                </Box>
                <Text size="xs" w={50} ta="right" ff="monospace">{value.toFixed(2)}</Text>
                <Slider
                  value={value}
                  onChange={(v) => {
                    const newVals = [...values];
                    newVals[i] = v;
                    setValues(newVals);
                  }}
                  min={0}
                  max={10}
                  step={0.1}
                  w={80}
                  disabled={isRunning}
                />
              </Group>
            ))}
          </Stack>

          {/* Results */}
          <SimpleGrid cols={2} spacing="md">
            <Box p="md" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)', textAlign: 'center' }}>
              <Text size="xs" c="dimmed">Tropical Sum (⊕ = max)</Text>
              <Text size="xl" fw={700} c="cyan">{tropicalSum.toFixed(3)}</Text>
              <Text size="xs" ff="monospace" c="dimmed">max({values.map(v => v.toFixed(1)).join(', ')})</Text>
            </Box>
            <Box p="md" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)', textAlign: 'center' }}>
              <Text size="xs" c="dimmed">Tropical Product (⊗ = +)</Text>
              <Text size="xl" fw={700} c="yellow">{tropicalProduct.toFixed(3)}</Text>
              <Text size="xs" ff="monospace" c="dimmed">{values.map(v => v.toFixed(1)).join(' + ')}</Text>
            </Box>
          </SimpleGrid>

          {/* Iteration counter */}
          {isRunning && (
            <Text size="xs" c="dimmed" ta="center">
              Iteration: {iterations} | {operation === 'add' ? 'Values converging to max' : 'Values shifting together'}
            </Text>
          )}
        </Stack>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Dual Number Autodiff Visualization
// ============================================================================

export function DualNumberVisualization() {
  const [x, setX] = useState(1.0);
  const [funcType, setFuncType] = useState<'quadratic' | 'sin' | 'exp' | 'sigmoid'>('quadratic');
  const [history, setHistory] = useState<{x: number, fx: number, dfx: number}[]>([]);

  const computeDual = useCallback((xVal: number) => {
    switch (funcType) {
      case 'quadratic':
        return { fx: xVal * xVal, dfx: 2 * xVal, label: 'x²', derivative: '2x' };
      case 'sin':
        return { fx: Math.sin(xVal), dfx: Math.cos(xVal), label: 'sin(x)', derivative: 'cos(x)' };
      case 'exp':
        return { fx: Math.exp(xVal), dfx: Math.exp(xVal), label: 'eˣ', derivative: 'eˣ' };
      case 'sigmoid':
        const s = 1 / (1 + Math.exp(-xVal));
        return { fx: s, dfx: s * (1 - s), label: 'σ(x)', derivative: 'σ(x)(1-σ(x))' };
    }
  }, [funcType]);

  const result = computeDual(x);

  useEffect(() => {
    setHistory(prev => {
      const newHistory = [...prev, { x, fx: result.fx, dfx: result.dfx }];
      return newHistory.slice(-50);
    });
  }, [x, result.fx, result.dfx]);

  // Generate function curve data
  const curvePoints = [];
  const derivPoints = [];
  for (let i = -30; i <= 30; i++) {
    const xi = i / 10;
    const res = computeDual(xi);
    curvePoints.push({ x: xi, y: res.fx });
    derivPoints.push({ x: xi, y: res.dfx });
  }

  const xToSvg = (val: number) => 150 + val * 40;
  const yToSvg = (val: number) => 100 - val * 30;

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Dual Number Autodiff</Title>
            <Text size="xs" c="dimmed">Forward-mode automatic differentiation</Text>
          </div>
          <Select
            size="xs"
            value={funcType}
            onChange={(v) => v && setFuncType(v as typeof funcType)}
            data={[
              { value: 'quadratic', label: 'f(x) = x²' },
              { value: 'sin', label: 'f(x) = sin(x)' },
              { value: 'exp', label: 'f(x) = eˣ' },
              { value: 'sigmoid', label: 'f(x) = σ(x)' },
            ]}
            w={150}
          />
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Graph */}
          <Box>
            <svg viewBox="0 0 300 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Grid */}
              <line x1="0" y1="100" x2="300" y2="100" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
              <line x1="150" y1="0" x2="150" y2="200" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />

              {/* Function curve */}
              <polyline
                points={curvePoints.map(p => `${xToSvg(p.x)},${yToSvg(Math.max(-3, Math.min(3, p.y)))}`).join(' ')}
                fill="none"
                stroke="var(--mantine-color-cyan-5)"
                strokeWidth="2"
              />

              {/* Derivative curve */}
              <polyline
                points={derivPoints.map(p => `${xToSvg(p.x)},${yToSvg(Math.max(-3, Math.min(3, p.y)))}`).join(' ')}
                fill="none"
                stroke="var(--mantine-color-yellow-5)"
                strokeWidth="2"
                strokeDasharray="5,3"
              />

              {/* Current point */}
              <circle cx={xToSvg(x)} cy={yToSvg(Math.max(-3, Math.min(3, result.fx)))} r="6" fill="var(--mantine-color-cyan-5)" />

              {/* Tangent line */}
              <line
                x1={xToSvg(x - 0.5)}
                y1={yToSvg(result.fx - 0.5 * result.dfx)}
                x2={xToSvg(x + 0.5)}
                y2={yToSvg(result.fx + 0.5 * result.dfx)}
                stroke="var(--mantine-color-orange-5)"
                strokeWidth="2"
              />

              {/* Labels */}
              <text x="280" y="95" fontSize="10" fill="var(--mantine-color-cyan-5)">f(x)</text>
              <text x="280" y="115" fontSize="10" fill="var(--mantine-color-yellow-5)">f'(x)</text>
            </svg>
          </Box>

          {/* Controls and values */}
          <Stack gap="md">
            <Box>
              <Text size="sm" mb="xs">x = {x.toFixed(3)}</Text>
              <Slider
                value={x}
                onChange={setX}
                min={-3}
                max={3}
                step={0.01}
                marks={[{ value: -3 }, { value: 0 }, { value: 3 }]}
              />
            </Box>

            <SimpleGrid cols={2} spacing="xs">
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">f(x) = {result.label}</Text>
                <Text size="lg" fw={700} c="cyan">{result.fx.toFixed(6)}</Text>
              </Box>
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">f'(x) = {result.derivative}</Text>
                <Text size="lg" fw={700} c="yellow">{result.dfx.toFixed(6)}</Text>
              </Box>
            </SimpleGrid>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Dual Number Representation</Text>
              <Text ff="monospace" size="sm">
                x = {x.toFixed(3)} + 1ε
              </Text>
              <Text ff="monospace" size="sm">
                f(x) = {result.fx.toFixed(3)} + {result.dfx.toFixed(3)}ε
              </Text>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Information Geometry Visualization
// ============================================================================

export function FisherVisualization() {
  const [probabilities, setProbabilities] = useState<number[]>([0.4, 0.35, 0.25]);
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setProbabilities(prev => {
        let newProbs = prev.map(p => Math.max(0.01, p + (Math.random() - 0.5) * 0.03));
        const sum = newProbs.reduce((a, b) => a + b, 0);
        return newProbs.map(p => p / sum);
      });
    }, 150);
    return () => clearInterval(interval);
  }, [isAnimating]);

  // Compute Fisher information matrix (diagonal for categorical)
  const fisherMatrix = probabilities.map(p => 1 / Math.max(p, 0.001));
  const entropy = -probabilities.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);

  // Simplex coordinates for 3 probabilities
  const simplexX = probabilities[0] + probabilities[2] / 2;
  const simplexY = probabilities[2] * Math.sqrt(3) / 2;

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Fisher Information</Title>
            <Text size="xs" c="dimmed">Probability simplex with Fisher metric</Text>
          </div>
          <Button size="xs" variant={isAnimating ? 'filled' : 'outline'} onClick={() => setIsAnimating(!isAnimating)}>
            {isAnimating ? 'Pause' : 'Random Walk'}
          </Button>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Probability simplex visualization */}
          <Box>
            <Text size="sm" fw={600} mb="xs">Probability Simplex</Text>
            <svg viewBox="0 0 200 180" style={{ width: '100%', height: '180px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Triangle */}
              <polygon
                points="100,10 10,160 190,160"
                fill="none"
                stroke="var(--mantine-color-dark-3)"
                strokeWidth="2"
              />

              {/* Grid lines */}
              {[0.25, 0.5, 0.75].map(t => (
                <g key={t}>
                  <line x1={10 + t * 90} y1={160 - t * 150} x2={100 + (1 - t) * 90} y2={10 + (1 - t) * 150} stroke="var(--mantine-color-dark-5)" strokeWidth="0.5" />
                  <line x1={100 - t * 90} y1={10 + t * 150} x2={190 - t * 90} y2={160} stroke="var(--mantine-color-dark-5)" strokeWidth="0.5" />
                  <line x1={10 + t * 180} y1={160} x2={55 + t * 90} y2={85 - t * 75} stroke="var(--mantine-color-dark-5)" strokeWidth="0.5" />
                </g>
              ))}

              {/* Current point */}
              <circle
                cx={10 + simplexX * 180}
                cy={160 - simplexY * 180}
                r="8"
                fill="var(--mantine-color-cyan-5)"
              />

              {/* Labels */}
              <text x="100" y="8" fontSize="10" textAnchor="middle" fill="var(--mantine-color-dimmed)">p₂</text>
              <text x="5" y="170" fontSize="10" fill="var(--mantine-color-dimmed)">p₀</text>
              <text x="185" y="170" fontSize="10" fill="var(--mantine-color-dimmed)">p₁</text>
            </svg>
          </Box>

          {/* Controls and info */}
          <Stack gap="md">
            <Stack gap="xs">
              {probabilities.map((p, i) => (
                <Group key={i} gap="xs" wrap="nowrap">
                  <Text size="xs" w={20}>p{i}</Text>
                  <Box style={{ flex: 1 }}>
                    <Progress value={p * 100} color="cyan" size="lg" />
                  </Box>
                  <Text size="xs" w={45} ff="monospace">{p.toFixed(3)}</Text>
                </Group>
              ))}
            </Stack>

            <SimpleGrid cols={2} spacing="xs">
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Entropy H(p)</Text>
                <Text size="lg" fw={700}>{entropy.toFixed(4)}</Text>
              </Box>
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Max Entropy</Text>
                <Text size="lg" fw={700}>{Math.log(probabilities.length).toFixed(4)}</Text>
              </Box>
            </SimpleGrid>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Fisher Matrix (diagonal)</Text>
              <Group gap="xs" justify="center">
                {fisherMatrix.map((f, i) => (
                  <Badge key={i} size="lg" variant="light">{f.toFixed(2)}</Badge>
                ))}
              </Group>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Rotor Rotation Visualization
// ============================================================================

export function RotorVisualization() {
  const [angle, setAngle] = useState(0);
  const [plane, setPlane] = useState<'xy' | 'xz' | 'yz'>('xy');
  const [isAnimating, setIsAnimating] = useState(false);

  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setAngle(prev => (prev + 2) % 360);
    }, 50);
    return () => clearInterval(interval);
  }, [isAnimating]);

  const rad = (angle * Math.PI) / 180;
  const cos = Math.cos(rad / 2);
  const sin = Math.sin(rad / 2);

  // Rotor components based on plane
  const rotor = plane === 'xy' ? [cos, 0, 0, 0, sin, 0, 0, 0]
    : plane === 'xz' ? [cos, 0, 0, 0, 0, sin, 0, 0]
    : [cos, 0, 0, 0, 0, 0, sin, 0];

  // Apply rotor to basis vector e1
  const rotateVector = () => {
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    switch (plane) {
      case 'xy': return { x: c, y: s, z: 0 };
      case 'xz': return { x: c, y: 0, z: s };
      case 'yz': return { x: 0, y: c, z: s };
    }
  };

  const rotated = rotateVector();

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Rotor Visualization</Title>
            <Text size="xs" c="dimmed">3D rotation via geometric algebra</Text>
          </div>
          <Group gap="xs">
            <SegmentedControl
              size="xs"
              value={plane}
              onChange={(v) => setPlane(v as typeof plane)}
              data={[
                { label: 'XY', value: 'xy' },
                { label: 'XZ', value: 'xz' },
                { label: 'YZ', value: 'yz' },
              ]}
            />
            <Button size="xs" variant={isAnimating ? 'filled' : 'outline'} onClick={() => setIsAnimating(!isAnimating)}>
              {isAnimating ? 'Stop' : 'Spin'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Visualization */}
          <svg viewBox="-120 -120 240 240" style={{ width: '100%', height: '220px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
            {/* Circle */}
            <circle cx="0" cy="0" r="80" fill="none" stroke="var(--mantine-color-dark-4)" strokeWidth="1" />

            {/* Axes */}
            <line x1="-100" y1="0" x2="100" y2="0" stroke="var(--mantine-color-red-5)" strokeWidth="1" opacity="0.5" />
            <line x1="0" y1="-100" x2="0" y2="100" stroke="var(--mantine-color-green-5)" strokeWidth="1" opacity="0.5" />

            {/* Original vector */}
            <line x1="0" y1="0" x2="80" y2="0" stroke="var(--mantine-color-dark-2)" strokeWidth="2" strokeDasharray="5,5" />

            {/* Rotated vector */}
            <line
              x1="0"
              y1="0"
              x2={rotated.x * 80}
              y2={-rotated.y * 80}
              stroke="var(--mantine-color-cyan-5)"
              strokeWidth="3"
            />
            <circle cx={rotated.x * 80} cy={-rotated.y * 80} r="5" fill="var(--mantine-color-cyan-5)" />

            {/* Angle arc */}
            <path
              d={`M 30 0 A 30 30 0 ${angle > 180 ? 1 : 0} 0 ${30 * Math.cos(rad)} ${-30 * Math.sin(rad)}`}
              fill="none"
              stroke="var(--mantine-color-yellow-5)"
              strokeWidth="2"
            />

            {/* Labels */}
            <text x="85" y="5" fontSize="10" fill="var(--mantine-color-red-5)">x</text>
            <text x="5" y="-90" fontSize="10" fill="var(--mantine-color-green-5)">y</text>
            <text x="-100" y="-90" fontSize="10" fill="var(--mantine-color-dimmed)">{angle}°</text>
          </svg>

          {/* Controls */}
          <Stack gap="md">
            <Box>
              <Text size="sm" mb="xs">Rotation Angle: {angle}°</Text>
              <Slider
                value={angle}
                onChange={setAngle}
                min={0}
                max={360}
                step={1}
                marks={[{ value: 0 }, { value: 90 }, { value: 180 }, { value: 270 }, { value: 360 }]}
                disabled={isAnimating}
              />
            </Box>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Rotor R = cos(θ/2) + sin(θ/2)B</Text>
              <Text ff="monospace" size="sm">
                R = {cos.toFixed(3)} + {sin.toFixed(3)}e₁₂
              </Text>
            </Box>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Rotated Vector (RvR̃)</Text>
              <Text ff="monospace" size="sm">
                v' = ({rotated.x.toFixed(3)}, {rotated.y.toFixed(3)}, {rotated.z.toFixed(3)})
              </Text>
            </Box>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Rotor Normalization</Text>
              <Text ff="monospace" size="sm">
                |R| = {Math.sqrt(cos * cos + sin * sin).toFixed(6)} ≈ 1
              </Text>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// MCMC Sampling Visualization
// ============================================================================

export function MCMCVisualization() {
  const [samples, setSamples] = useState<{x: number, y: number}[]>([]);
  const [currentPoint, setCurrentPoint] = useState({ x: 0, y: 0 });
  const [isRunning, setIsRunning] = useState(false);
  const [acceptanceRate, setAcceptanceRate] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [accepted, setAccepted] = useState(0);
  const [proposalStd, setProposalStd] = useState(0.5);

  // Target distribution: 2D Gaussian
  const targetLogProb = (x: number, y: number) => {
    return -0.5 * (x * x + y * y);
  };

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setTotalSteps(prev => prev + 1);

      // Propose new point
      const propX = currentPoint.x + (Math.random() - 0.5) * 2 * proposalStd;
      const propY = currentPoint.y + (Math.random() - 0.5) * 2 * proposalStd;

      // Acceptance ratio
      const logAlpha = targetLogProb(propX, propY) - targetLogProb(currentPoint.x, currentPoint.y);

      if (Math.log(Math.random()) < logAlpha) {
        setCurrentPoint({ x: propX, y: propY });
        setSamples(prev => [...prev.slice(-200), { x: propX, y: propY }]);
        setAccepted(prev => prev + 1);
      } else {
        setSamples(prev => [...prev.slice(-200), { ...currentPoint }]);
      }

      setAcceptanceRate(prev => (accepted + (Math.log(Math.random()) < logAlpha ? 1 : 0)) / (totalSteps + 1));
    }, 50);

    return () => clearInterval(interval);
  }, [isRunning, currentPoint, proposalStd, accepted, totalSteps]);

  const reset = () => {
    setSamples([]);
    setCurrentPoint({ x: 0, y: 0 });
    setTotalSteps(0);
    setAccepted(0);
    setAcceptanceRate(0);
  };

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>MCMC Sampling</Title>
            <Text size="xs" c="dimmed">Metropolis-Hastings on 2D Gaussian</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant="outline" onClick={reset}>Reset</Button>
            <Button size="xs" variant={isRunning ? 'filled' : 'outline'} onClick={() => setIsRunning(!isRunning)}>
              {isRunning ? 'Pause' : 'Run'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Sample scatter plot */}
          <svg viewBox="-4 -4 8 8" style={{ width: '100%', height: '220px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
            {/* Grid */}
            <line x1="-4" y1="0" x2="4" y2="0" stroke="var(--mantine-color-dark-4)" strokeWidth="0.02" />
            <line x1="0" y1="-4" x2="0" y2="4" stroke="var(--mantine-color-dark-4)" strokeWidth="0.02" />

            {/* Target distribution contours */}
            {[1, 2, 3].map(r => (
              <circle key={r} cx="0" cy="0" r={r} fill="none" stroke="var(--mantine-color-dark-3)" strokeWidth="0.02" />
            ))}

            {/* Samples */}
            {samples.map((s, i) => (
              <circle
                key={i}
                cx={s.x}
                cy={-s.y}
                r="0.05"
                fill="var(--mantine-color-cyan-5)"
                opacity={0.3 + (i / samples.length) * 0.7}
              />
            ))}

            {/* Current point */}
            <circle cx={currentPoint.x} cy={-currentPoint.y} r="0.12" fill="var(--mantine-color-red-5)" />
          </svg>

          {/* Stats */}
          <Stack gap="md">
            <Box>
              <Text size="sm" mb="xs">Proposal Std Dev: {proposalStd.toFixed(2)}</Text>
              <Slider
                value={proposalStd}
                onChange={setProposalStd}
                min={0.1}
                max={2}
                step={0.1}
                disabled={isRunning}
              />
            </Box>

            <SimpleGrid cols={2} spacing="xs">
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Samples</Text>
                <Text size="lg" fw={700}>{samples.length}</Text>
              </Box>
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Acceptance Rate</Text>
                <Text size="lg" fw={700} c={acceptanceRate > 0.2 && acceptanceRate < 0.5 ? 'green' : 'yellow'}>
                  {(acceptanceRate * 100).toFixed(1)}%
                </Text>
              </Box>
            </SimpleGrid>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Current Position</Text>
              <Text ff="monospace" size="sm">
                ({currentPoint.x.toFixed(3)}, {currentPoint.y.toFixed(3)})
              </Text>
            </Box>

            <Text size="xs" c="dimmed">
              Optimal acceptance rate for 2D: 23-44%
            </Text>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Network Visualization
// ============================================================================

export function NetworkVisualization() {
  const [nodes, setNodes] = useState<{x: number, y: number, value: number}[]>([]);
  const [edges, setEdges] = useState<{from: number, to: number, weight: number}[]>([]);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);

  useEffect(() => {
    // Initialize random network
    const newNodes = Array.from({ length: 8 }, (_, i) => ({
      x: 100 + 70 * Math.cos((i / 8) * 2 * Math.PI),
      y: 100 + 70 * Math.sin((i / 8) * 2 * Math.PI),
      value: Math.random() * 2 - 1
    }));

    const newEdges: typeof edges = [];
    for (let i = 0; i < 8; i++) {
      for (let j = i + 1; j < 8; j++) {
        if (Math.random() > 0.5) {
          newEdges.push({ from: i, to: j, weight: Math.random() });
        }
      }
    }

    setNodes(newNodes);
    setEdges(newEdges);
  }, []);

  const calculateCentrality = (nodeIdx: number) => {
    return edges.filter(e => e.from === nodeIdx || e.to === nodeIdx).length / edges.length;
  };

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Geometric Network</Title>
            <Text size="xs" c="dimmed">Network with multivector node values</Text>
          </div>
          <Button size="xs" variant="outline" onClick={() => setSelectedNode(null)}>
            Clear Selection
          </Button>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Network graph */}
          <svg viewBox="0 0 200 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
            {/* Edges */}
            {edges.map((edge, i) => (
              <line
                key={i}
                x1={nodes[edge.from]?.x || 0}
                y1={nodes[edge.from]?.y || 0}
                x2={nodes[edge.to]?.x || 0}
                y2={nodes[edge.to]?.y || 0}
                stroke={selectedNode !== null && (edge.from === selectedNode || edge.to === selectedNode)
                  ? 'var(--mantine-color-cyan-5)'
                  : 'var(--mantine-color-dark-3)'}
                strokeWidth={edge.weight * 3}
                opacity={0.6}
              />
            ))}

            {/* Nodes */}
            {nodes.map((node, i) => (
              <g key={i} onClick={() => setSelectedNode(i)} style={{ cursor: 'pointer' }}>
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={10 + calculateCentrality(i) * 10}
                  fill={node.value > 0 ? 'var(--mantine-color-red-5)' : 'var(--mantine-color-blue-5)'}
                  stroke={selectedNode === i ? 'white' : 'none'}
                  strokeWidth="2"
                  opacity={0.8}
                />
                <text x={node.x} y={node.y + 4} fontSize="10" fill="white" textAnchor="middle">{i}</text>
              </g>
            ))}
          </svg>

          {/* Node info */}
          <Stack gap="md">
            {selectedNode !== null && nodes[selectedNode] ? (
              <>
                <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                  <Text size="sm" fw={600}>Node {selectedNode}</Text>
                  <Text size="xs" c="dimmed" mt="xs">Value: {nodes[selectedNode].value.toFixed(4)}</Text>
                  <Text size="xs" c="dimmed">Centrality: {(calculateCentrality(selectedNode) * 100).toFixed(1)}%</Text>
                  <Text size="xs" c="dimmed">Connections: {edges.filter(e => e.from === selectedNode || e.to === selectedNode).length}</Text>
                </Box>
                <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                  <Text size="xs" c="dimmed" mb="xs">Connected Nodes</Text>
                  <Group gap="xs">
                    {edges
                      .filter(e => e.from === selectedNode || e.to === selectedNode)
                      .map((e, i) => (
                        <Badge key={i} size="sm" variant="light">
                          {e.from === selectedNode ? e.to : e.from}
                        </Badge>
                      ))}
                  </Group>
                </Box>
              </>
            ) : (
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)', textAlign: 'center' }}>
                <Text size="sm" c="dimmed">Click a node to see details</Text>
              </Box>
            )}

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Network Stats</Text>
              <SimpleGrid cols={2} spacing="xs">
                <Text size="xs">Nodes: {nodes.length}</Text>
                <Text size="xs">Edges: {edges.length}</Text>
                <Text size="xs">Density: {(edges.length / (nodes.length * (nodes.length - 1) / 2) * 100).toFixed(1)}%</Text>
                <Text size="xs">Avg degree: {(edges.length * 2 / nodes.length).toFixed(1)}</Text>
              </SimpleGrid>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Topology Visualization
// ============================================================================

interface SimplexData {
  vertices: number[];
  dimension: number;
}

export function TopologyVisualization() {
  const [simplices, setSimplices] = useState<SimplexData[]>([]);
  const [bettiNumbers, setBettiNumbers] = useState<number[]>([0, 0, 0]);
  const [selectedPreset, setSelectedPreset] = useState<string>('triangle');
  const [eulerChar, setEulerChar] = useState(0);

  // Node positions for visualization (up to 6 nodes)
  const nodePositions = [
    { x: 100, y: 30 },   // 0 - top
    { x: 170, y: 80 },   // 1 - right-top
    { x: 170, y: 150 },  // 2 - right-bottom
    { x: 100, y: 180 },  // 3 - bottom
    { x: 30, y: 150 },   // 4 - left-bottom
    { x: 30, y: 80 },    // 5 - left-top
  ];

  const presets: { [key: string]: { simplices: SimplexData[], betti: number[], euler: number, name: string } } = {
    'triangle': {
      name: 'Filled Triangle (2-simplex)',
      simplices: [
        { vertices: [0], dimension: 0 },
        { vertices: [1], dimension: 0 },
        { vertices: [2], dimension: 0 },
        { vertices: [0, 1], dimension: 1 },
        { vertices: [1, 2], dimension: 1 },
        { vertices: [0, 2], dimension: 1 },
        { vertices: [0, 1, 2], dimension: 2 },
      ],
      betti: [1, 0, 0],
      euler: 1
    },
    'circle': {
      name: 'Circle (3 edges, no fill)',
      simplices: [
        { vertices: [0], dimension: 0 },
        { vertices: [1], dimension: 0 },
        { vertices: [2], dimension: 0 },
        { vertices: [0, 1], dimension: 1 },
        { vertices: [1, 2], dimension: 1 },
        { vertices: [0, 2], dimension: 1 },
      ],
      betti: [1, 1, 0],
      euler: 0
    },
    'tetrahedron': {
      name: 'Tetrahedron Surface (hollow)',
      simplices: [
        { vertices: [0], dimension: 0 },
        { vertices: [1], dimension: 0 },
        { vertices: [2], dimension: 0 },
        { vertices: [3], dimension: 0 },
        { vertices: [0, 1], dimension: 1 },
        { vertices: [0, 2], dimension: 1 },
        { vertices: [0, 3], dimension: 1 },
        { vertices: [1, 2], dimension: 1 },
        { vertices: [1, 3], dimension: 1 },
        { vertices: [2, 3], dimension: 1 },
        { vertices: [0, 1, 2], dimension: 2 },
        { vertices: [0, 1, 3], dimension: 2 },
        { vertices: [0, 2, 3], dimension: 2 },
        { vertices: [1, 2, 3], dimension: 2 },
      ],
      betti: [1, 0, 1],
      euler: 2
    },
    'torus-approx': {
      name: 'Two Loops (β₁ = 2)',
      simplices: [
        { vertices: [0], dimension: 0 },
        { vertices: [1], dimension: 0 },
        { vertices: [2], dimension: 0 },
        { vertices: [3], dimension: 0 },
        { vertices: [4], dimension: 0 },
        { vertices: [0, 1], dimension: 1 },
        { vertices: [1, 2], dimension: 1 },
        { vertices: [2, 0], dimension: 1 },
        { vertices: [2, 3], dimension: 1 },
        { vertices: [3, 4], dimension: 1 },
        { vertices: [4, 2], dimension: 1 },
      ],
      betti: [1, 2, 0],
      euler: -1
    },
    'disconnected': {
      name: 'Two Components',
      simplices: [
        { vertices: [0], dimension: 0 },
        { vertices: [1], dimension: 0 },
        { vertices: [2], dimension: 0 },
        { vertices: [3], dimension: 0 },
        { vertices: [0, 1], dimension: 1 },
        { vertices: [2, 3], dimension: 1 },
      ],
      betti: [2, 0, 0],
      euler: 2
    }
  };

  useEffect(() => {
    const preset = presets[selectedPreset];
    setSimplices(preset.simplices);
    setBettiNumbers(preset.betti);
    setEulerChar(preset.euler);
  }, [selectedPreset]);

  const getSimplexColor = (dim: number) => {
    switch (dim) {
      case 0: return 'var(--mantine-color-cyan-5)';
      case 1: return 'var(--mantine-color-yellow-5)';
      case 2: return 'var(--mantine-color-red-5)';
      default: return 'var(--mantine-color-gray-5)';
    }
  };

  const edges = simplices.filter(s => s.dimension === 1);
  const triangles = simplices.filter(s => s.dimension === 2);
  const vertices = simplices.filter(s => s.dimension === 0);

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Simplicial Complex</Title>
            <Text size="xs" c="dimmed">Homology and Betti numbers visualization</Text>
          </div>
          <Select
            size="xs"
            value={selectedPreset}
            onChange={(v) => v && setSelectedPreset(v)}
            data={Object.entries(presets).map(([key, val]) => ({ value: key, label: val.name }))}
            w={200}
          />
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Complex visualization */}
          <Box>
            <Text size="sm" fw={600} mb="xs">Complex Visualization</Text>
            <svg viewBox="0 0 200 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Draw 2-simplices (triangles) first */}
              {triangles.map((tri, i) => {
                const [v0, v1, v2] = tri.vertices;
                const p0 = nodePositions[v0];
                const p1 = nodePositions[v1];
                const p2 = nodePositions[v2];
                return (
                  <polygon
                    key={`tri-${i}`}
                    points={`${p0.x},${p0.y} ${p1.x},${p1.y} ${p2.x},${p2.y}`}
                    fill={getSimplexColor(2)}
                    opacity={0.3}
                    stroke={getSimplexColor(2)}
                    strokeWidth="1"
                  />
                );
              })}

              {/* Draw 1-simplices (edges) */}
              {edges.map((edge, i) => {
                const [v0, v1] = edge.vertices;
                const p0 = nodePositions[v0];
                const p1 = nodePositions[v1];
                return (
                  <line
                    key={`edge-${i}`}
                    x1={p0.x}
                    y1={p0.y}
                    x2={p1.x}
                    y2={p1.y}
                    stroke={getSimplexColor(1)}
                    strokeWidth="3"
                  />
                );
              })}

              {/* Draw 0-simplices (vertices) */}
              {vertices.map((vert, i) => {
                const p = nodePositions[vert.vertices[0]];
                return (
                  <g key={`vert-${i}`}>
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r="8"
                      fill={getSimplexColor(0)}
                    />
                    <text
                      x={p.x}
                      y={p.y + 4}
                      fontSize="10"
                      fill="white"
                      textAnchor="middle"
                    >
                      {vert.vertices[0]}
                    </text>
                  </g>
                );
              })}

              {/* Legend */}
              <g transform="translate(5, 185)">
                <circle cx="5" cy="0" r="4" fill={getSimplexColor(0)} />
                <text x="12" y="3" fontSize="8" fill="var(--mantine-color-dimmed)">vertex</text>
                <line x1="45" y1="0" x2="60" y2="0" stroke={getSimplexColor(1)} strokeWidth="2" />
                <text x="65" y="3" fontSize="8" fill="var(--mantine-color-dimmed)">edge</text>
                <rect x="100" y="-5" width="10" height="10" fill={getSimplexColor(2)} opacity="0.5" />
                <text x="115" y="3" fontSize="8" fill="var(--mantine-color-dimmed)">face</text>
              </g>
            </svg>
          </Box>

          {/* Homology info */}
          <Stack gap="md">
            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Betti Numbers</Text>
              <Group gap="md" justify="center">
                {bettiNumbers.map((b, k) => (
                  <Box key={k} style={{ textAlign: 'center' }}>
                    <Text size="xl" fw={700} c={b > 0 ? 'cyan' : 'dimmed'}>β{k} = {b}</Text>
                    <Text size="xs" c="dimmed">
                      {k === 0 ? 'components' : k === 1 ? 'loops' : 'voids'}
                    </Text>
                  </Box>
                ))}
              </Group>
            </Box>

            <SimpleGrid cols={2} spacing="xs">
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Euler Characteristic</Text>
                <Text size="lg" fw={700}>χ = {eulerChar}</Text>
                <Text size="xs" ff="monospace" c="dimmed">V - E + F</Text>
              </Box>
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Simplex Count</Text>
                <Text size="sm" ff="monospace">
                  {vertices.length} vertices, {edges.length} edges, {triangles.length} faces
                </Text>
              </Box>
            </SimpleGrid>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Homology Interpretation</Text>
              <Text size="xs">
                {bettiNumbers[0] === 1 ? '• Connected (1 component)' : `• ${bettiNumbers[0]} connected components`}
              </Text>
              <Text size="xs">
                {bettiNumbers[1] === 0 ? '• No 1D holes (loops)' : `• ${bettiNumbers[1]} loop${bettiNumbers[1] > 1 ? 's' : ''}/tunnel${bettiNumbers[1] > 1 ? 's' : ''}`}
              </Text>
              <Text size="xs">
                {bettiNumbers[2] === 0 ? '• No 2D voids' : `• ${bettiNumbers[2]} enclosed void${bettiNumbers[2] > 1 ? 's' : ''}`}
              </Text>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Main Export: All Visualizations Section
// ============================================================================

export function LiveVisualizationSection() {
  const [activeTab, setActiveTab] = useState<string | null>('geometric');

  return (
    <Card withBorder>
      <Card.Section inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={2}>Live Visualizations</Title>
            <Text size="sm" c="dimmed">Interactive demonstrations of Amari's mathematical concepts</Text>
          </div>
          <Badge size="lg" variant="light" color="green">Interactive</Badge>
        </Group>
      </Card.Section>
      <Card.Section>
        <Tabs value={activeTab} onChange={setActiveTab}>
          <Tabs.List>
            <Tabs.Tab value="geometric">Geometric Algebra</Tabs.Tab>
            <Tabs.Tab value="tropical">Tropical</Tabs.Tab>
            <Tabs.Tab value="dual">Autodiff</Tabs.Tab>
            <Tabs.Tab value="rotor">Rotations</Tabs.Tab>
            <Tabs.Tab value="fisher">Info Geometry</Tabs.Tab>
            <Tabs.Tab value="topology">Topology</Tabs.Tab>
            <Tabs.Tab value="mcmc">MCMC</Tabs.Tab>
            <Tabs.Tab value="network">Networks</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="geometric" p="md">
            <MultivectorVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="tropical" p="md">
            <TropicalVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="dual" p="md">
            <DualNumberVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="rotor" p="md">
            <RotorVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="fisher" p="md">
            <FisherVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="topology" p="md">
            <TopologyVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="mcmc" p="md">
            <MCMCVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="network" p="md">
            <NetworkVisualization />
          </Tabs.Panel>
        </Tabs>
      </Card.Section>
    </Card>
  );
}
