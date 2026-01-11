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
// Clifford Grade Decomposition Visualization
// ============================================================================

export function GradeDecompositionVisualization() {
  const [coefficients, setCoefficients] = useState<number[]>([1, 2, 1, 0.5, 1.5, 0.3, 0.7, 0.4]);
  const [selectedGrade, setSelectedGrade] = useState<number | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);

  const grades = [
    { name: 'Grade 0 (Scalar)', indices: [0], color: '#888888', label: '1' },
    { name: 'Grade 1 (Vector)', indices: [1, 2, 3], color: '#ff6b6b', label: 'e₁, e₂, e₃' },
    { name: 'Grade 2 (Bivector)', indices: [4, 5, 6], color: '#4ecdc4', label: 'e₁₂, e₁₃, e₂₃' },
    { name: 'Grade 3 (Trivector)', indices: [7], color: '#be2edd', label: 'e₁₂₃' },
  ];

  const basisLabels = ['1', 'e₁', 'e₂', 'e₃', 'e₁₂', 'e₁₃', 'e₂₃', 'e₁₂₃'];

  useEffect(() => {
    if (!isAnimating) return;
    const interval = setInterval(() => {
      setCoefficients(prev => prev.map((c, i) => {
        const phase = Date.now() / 1000 + i * 0.5;
        return Math.sin(phase) * (1 + i * 0.3);
      }));
    }, 50);
    return () => clearInterval(interval);
  }, [isAnimating]);

  const getGradeMagnitude = (gradeIdx: number) => {
    const indices = grades[gradeIdx].indices;
    return Math.sqrt(indices.reduce((sum, i) => sum + coefficients[i] ** 2, 0));
  };

  const totalMagnitude = Math.sqrt(coefficients.reduce((sum, c) => sum + c ** 2, 0));

  const setPreset = (preset: string) => {
    switch (preset) {
      case 'vector': setCoefficients([0, 1, 1, 0, 0, 0, 0, 0]); break;
      case 'rotor': setCoefficients([Math.cos(0.5), 0, 0, 0, Math.sin(0.5), 0, 0, 0]); break;
      case 'spinor': setCoefficients([1, 0, 0, 0, 0.5, 0.3, 0.2, 0]); break;
      case 'pseudoscalar': setCoefficients([0, 0, 0, 0, 0, 0, 0, 1]); break;
      case 'mixed': setCoefficients([0.5, 1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1]); break;
    }
  };

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Grade Decomposition Lens</Title>
            <Text size="xs" c="dimmed">Clifford algebra multivector grade analysis</Text>
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
          {/* Grade breakdown bars */}
          <Stack gap="md">
            <Text size="sm" fw={600}>Grade Components</Text>
            {grades.map((grade, gIdx) => {
              const mag = getGradeMagnitude(gIdx);
              const pct = totalMagnitude > 0 ? (mag / totalMagnitude) * 100 : 0;
              return (
                <Box
                  key={gIdx}
                  p="xs"
                  bg={selectedGrade === gIdx ? 'dark.5' : 'dark.7'}
                  style={{ borderRadius: 'var(--mantine-radius-sm)', cursor: 'pointer', border: selectedGrade === gIdx ? `2px solid ${grade.color}` : '2px solid transparent' }}
                  onClick={() => setSelectedGrade(selectedGrade === gIdx ? null : gIdx)}
                >
                  <Group justify="space-between" mb={4}>
                    <Group gap="xs">
                      <Box w={12} h={12} bg={grade.color} style={{ borderRadius: 3 }} />
                      <Text size="xs" fw={500}>{grade.name}</Text>
                    </Group>
                    <Text size="xs" ff="monospace">{mag.toFixed(3)}</Text>
                  </Group>
                  <Progress value={pct} color={grade.color.replace('#', '')} size="sm" />
                  <Text size="xs" c="dimmed" mt={4}>{grade.label}: {grade.indices.map(i => coefficients[i].toFixed(2)).join(', ')}</Text>
                </Box>
              );
            })}
          </Stack>

          {/* Visualization */}
          <Stack gap="md">
            <Box>
              <Text size="sm" fw={600} mb="xs">Radial Grade View</Text>
              <svg viewBox="-120 -120 240 240" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
                {/* Concentric circles for grades */}
                {grades.map((grade, gIdx) => {
                  const radius = 25 + gIdx * 25;
                  const mag = getGradeMagnitude(gIdx);
                  const scaledR = Math.min(radius, mag * 30);
                  return (
                    <g key={gIdx}>
                      <circle cx="0" cy="0" r={radius} fill="none" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" strokeDasharray="3,3" />
                      <circle
                        cx="0"
                        cy="0"
                        r={scaledR}
                        fill={grade.color}
                        opacity={selectedGrade === null || selectedGrade === gIdx ? 0.3 : 0.05}
                        stroke={grade.color}
                        strokeWidth={selectedGrade === gIdx ? 3 : 1}
                      />
                    </g>
                  );
                })}

                {/* Individual basis vectors as rays */}
                {coefficients.map((c, i) => {
                  if (Math.abs(c) < 0.01) return null;
                  const gradeIdx = grades.findIndex(g => g.indices.includes(i));
                  const angle = (i / 8) * 2 * Math.PI - Math.PI / 2;
                  const length = Math.abs(c) * 35;
                  return (
                    <g key={i}>
                      <line
                        x1="0"
                        y1="0"
                        x2={Math.cos(angle) * length}
                        y2={Math.sin(angle) * length}
                        stroke={grades[gradeIdx]?.color || '#888'}
                        strokeWidth="3"
                        opacity={selectedGrade === null || selectedGrade === gradeIdx ? 0.8 : 0.15}
                      />
                      <text
                        x={Math.cos(angle) * (length + 15)}
                        y={Math.sin(angle) * (length + 15)}
                        fontSize="8"
                        fill={grades[gradeIdx]?.color || '#888'}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        opacity={selectedGrade === null || selectedGrade === gradeIdx ? 1 : 0.3}
                      >
                        {basisLabels[i]}
                      </text>
                    </g>
                  );
                })}

                <circle cx="0" cy="0" r="3" fill="white" />
              </svg>
            </Box>

            <Group gap="xs" wrap="wrap">
              <Button size="xs" variant="light" onClick={() => setPreset('vector')}>Vector</Button>
              <Button size="xs" variant="light" onClick={() => setPreset('rotor')}>Rotor</Button>
              <Button size="xs" variant="light" onClick={() => setPreset('spinor')}>Spinor</Button>
              <Button size="xs" variant="light" onClick={() => setPreset('pseudoscalar')}>Pseudoscalar</Button>
              <Button size="xs" variant="light" onClick={() => setPreset('mixed')}>Mixed</Button>
            </Group>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <SimpleGrid cols={2} spacing="xs">
                <Text size="xs"><strong>Total magnitude:</strong> {totalMagnitude.toFixed(4)}</Text>
                <Text size="xs"><strong>Dominant grade:</strong> {grades.reduce((max, g, i) => getGradeMagnitude(i) > getGradeMagnitude(max) ? i : max, 0)}</Text>
              </SimpleGrid>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Möbius Transformation Visualization
// ============================================================================

export function MobiusVisualization() {
  const [a, setA] = useState({ re: 1, im: 0 });
  const [b, setB] = useState({ re: 0, im: 0 });
  const [c, setC] = useState({ re: 0, im: 0 });
  const [d, setD] = useState({ re: 1, im: 0 });
  const [showGrid, setShowGrid] = useState(true);
  const [preset, setPreset] = useState('identity');

  // Complex multiplication
  const cmul = (z1: {re: number, im: number}, z2: {re: number, im: number}) => ({
    re: z1.re * z2.re - z1.im * z2.im,
    im: z1.re * z2.im + z1.im * z2.re
  });

  // Complex division
  const cdiv = (z1: {re: number, im: number}, z2: {re: number, im: number}) => {
    const denom = z2.re * z2.re + z2.im * z2.im;
    if (denom < 0.0001) return { re: Infinity, im: Infinity };
    return {
      re: (z1.re * z2.re + z1.im * z2.im) / denom,
      im: (z1.im * z2.re - z1.re * z2.im) / denom
    };
  };

  // Complex addition
  const cadd = (z1: {re: number, im: number}, z2: {re: number, im: number}) => ({
    re: z1.re + z2.re,
    im: z1.im + z2.im
  });

  // Möbius transformation: f(z) = (az + b) / (cz + d)
  const mobius = (z: {re: number, im: number}) => {
    const num = cadd(cmul(a, z), b);
    const denom = cadd(cmul(c, z), d);
    return cdiv(num, denom);
  };

  // Generate grid points
  const generateGrid = () => {
    const lines: {original: {re: number, im: number}[], transformed: {re: number, im: number}[]}[] = [];

    // Vertical lines
    for (let x = -2; x <= 2; x += 0.5) {
      const original: {re: number, im: number}[] = [];
      const transformed: {re: number, im: number}[] = [];
      for (let y = -2; y <= 2; y += 0.1) {
        const z = { re: x, im: y };
        original.push(z);
        transformed.push(mobius(z));
      }
      lines.push({ original, transformed });
    }

    // Horizontal lines
    for (let y = -2; y <= 2; y += 0.5) {
      const original: {re: number, im: number}[] = [];
      const transformed: {re: number, im: number}[] = [];
      for (let x = -2; x <= 2; x += 0.1) {
        const z = { re: x, im: y };
        original.push(z);
        transformed.push(mobius(z));
      }
      lines.push({ original, transformed });
    }

    // Unit circle
    const circleOriginal: {re: number, im: number}[] = [];
    const circleTransformed: {re: number, im: number}[] = [];
    for (let theta = 0; theta <= 2 * Math.PI; theta += 0.1) {
      const z = { re: Math.cos(theta), im: Math.sin(theta) };
      circleOriginal.push(z);
      circleTransformed.push(mobius(z));
    }
    lines.push({ original: circleOriginal, transformed: circleTransformed });

    return lines;
  };

  const applyPreset = (p: string) => {
    setPreset(p);
    switch (p) {
      case 'identity':
        setA({ re: 1, im: 0 }); setB({ re: 0, im: 0 }); setC({ re: 0, im: 0 }); setD({ re: 1, im: 0 });
        break;
      case 'inversion':
        setA({ re: 0, im: 0 }); setB({ re: 1, im: 0 }); setC({ re: 1, im: 0 }); setD({ re: 0, im: 0 });
        break;
      case 'rotation':
        setA({ re: Math.cos(Math.PI/4), im: Math.sin(Math.PI/4) }); setB({ re: 0, im: 0 }); setC({ re: 0, im: 0 }); setD({ re: 1, im: 0 });
        break;
      case 'translation':
        setA({ re: 1, im: 0 }); setB({ re: 1, im: 0.5 }); setC({ re: 0, im: 0 }); setD({ re: 1, im: 0 });
        break;
      case 'dilation':
        setA({ re: 2, im: 0 }); setB({ re: 0, im: 0 }); setC({ re: 0, im: 0 }); setD({ re: 1, im: 0 });
        break;
      case 'cayley':
        setA({ re: 1, im: 0 }); setB({ re: 0, im: -1 }); setC({ re: 1, im: 0 }); setD({ re: 0, im: 1 });
        break;
    }
  };

  const lines = generateGrid();
  const scale = 50;
  const toSvg = (z: {re: number, im: number}, offset: number) => ({
    x: offset + z.re * scale,
    y: 110 - z.im * scale
  });

  // Determinant for classification
  const det = cmul(a, d).re - cmul(a, d).im * 0 - (cmul(b, c).re - cmul(b, c).im * 0);

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Möbius Transformations</Title>
            <Text size="xs" c="dimmed">Conformal maps of the complex plane: f(z) = (az+b)/(cz+d)</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant={showGrid ? 'filled' : 'outline'} onClick={() => setShowGrid(!showGrid)}>
              Grid
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="md">
          {/* Visualization */}
          <Box>
            <svg viewBox="0 0 520 220" style={{ width: '100%', height: '220px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Original plane */}
              <g>
                <text x="110" y="15" fontSize="10" fill="var(--mantine-color-dimmed)" textAnchor="middle">Original</text>
                <line x1="10" y1="110" x2="210" y2="110" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
                <line x1="110" y1="10" x2="110" y2="210" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />

                {showGrid && lines.slice(0, -1).map((line, i) => (
                  <polyline
                    key={`orig-${i}`}
                    points={line.original.map(z => {
                      const p = toSvg(z, 110);
                      return `${p.x},${p.y}`;
                    }).join(' ')}
                    fill="none"
                    stroke="var(--mantine-color-cyan-7)"
                    strokeWidth="1"
                    opacity="0.4"
                  />
                ))}

                {/* Unit circle */}
                <polyline
                  points={lines[lines.length - 1].original.map(z => {
                    const p = toSvg(z, 110);
                    return `${p.x},${p.y}`;
                  }).join(' ')}
                  fill="none"
                  stroke="var(--mantine-color-yellow-5)"
                  strokeWidth="2"
                />
              </g>

              {/* Arrow between planes */}
              <g>
                <line x1="220" y1="110" x2="290" y2="110" stroke="var(--mantine-color-dimmed)" strokeWidth="1" markerEnd="url(#arrowhead)" />
                <text x="255" y="100" fontSize="10" fill="var(--mantine-color-dimmed)" textAnchor="middle">f(z)</text>
              </g>

              {/* Transformed plane */}
              <g>
                <text x="410" y="15" fontSize="10" fill="var(--mantine-color-dimmed)" textAnchor="middle">Transformed</text>
                <line x1="310" y1="110" x2="510" y2="110" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
                <line x1="410" y1="10" x2="410" y2="210" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />

                {showGrid && lines.slice(0, -1).map((line, i) => (
                  <polyline
                    key={`trans-${i}`}
                    points={line.transformed
                      .filter(z => Math.abs(z.re) < 5 && Math.abs(z.im) < 5)
                      .map(z => {
                        const p = toSvg(z, 410);
                        return `${p.x},${p.y}`;
                      }).join(' ')}
                    fill="none"
                    stroke="var(--mantine-color-red-7)"
                    strokeWidth="1"
                    opacity="0.4"
                  />
                ))}

                {/* Transformed unit circle */}
                <polyline
                  points={lines[lines.length - 1].transformed
                    .filter(z => Math.abs(z.re) < 5 && Math.abs(z.im) < 5)
                    .map(z => {
                      const p = toSvg(z, 410);
                      return `${p.x},${p.y}`;
                    }).join(' ')}
                  fill="none"
                  stroke="var(--mantine-color-green-5)"
                  strokeWidth="2"
                />
              </g>

              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="var(--mantine-color-dimmed)" />
                </marker>
              </defs>
            </svg>
          </Box>

          {/* Controls */}
          <Stack gap="sm">
            <Group gap="xs" wrap="wrap">
              {['identity', 'inversion', 'rotation', 'translation', 'dilation', 'cayley'].map(p => (
                <Button key={p} size="xs" variant={preset === p ? 'filled' : 'light'} onClick={() => applyPreset(p)}>
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </Button>
              ))}
            </Group>

            <SimpleGrid cols={2} spacing="xs">
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed" mb={4}>a = {a.re.toFixed(2)} + {a.im.toFixed(2)}i</Text>
                <Slider value={a.re} onChange={(v) => setA({ ...a, re: v })} min={-2} max={2} step={0.1} size="xs" label="Re" />
                <Slider value={a.im} onChange={(v) => setA({ ...a, im: v })} min={-2} max={2} step={0.1} size="xs" label="Im" mt={4} />
              </Box>
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed" mb={4}>b = {b.re.toFixed(2)} + {b.im.toFixed(2)}i</Text>
                <Slider value={b.re} onChange={(v) => setB({ ...b, re: v })} min={-2} max={2} step={0.1} size="xs" />
                <Slider value={b.im} onChange={(v) => setB({ ...b, im: v })} min={-2} max={2} step={0.1} size="xs" mt={4} />
              </Box>
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed" mb={4}>c = {c.re.toFixed(2)} + {c.im.toFixed(2)}i</Text>
                <Slider value={c.re} onChange={(v) => setC({ ...c, re: v })} min={-2} max={2} step={0.1} size="xs" />
                <Slider value={c.im} onChange={(v) => setC({ ...c, im: v })} min={-2} max={2} step={0.1} size="xs" mt={4} />
              </Box>
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed" mb={4}>d = {d.re.toFixed(2)} + {d.im.toFixed(2)}i</Text>
                <Slider value={d.re} onChange={(v) => setD({ ...d, re: v })} min={-2} max={2} step={0.1} size="xs" />
                <Slider value={d.im} onChange={(v) => setD({ ...d, im: v })} min={-2} max={2} step={0.1} size="xs" mt={4} />
              </Box>
            </SimpleGrid>

            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed">Properties</Text>
              <Text size="xs" ff="monospace">det(M) = ad - bc ≈ {det.toFixed(3)}</Text>
              <Text size="xs" c="dimmed" mt={4}>
                {det > 0 ? '• Orientation preserving' : det < 0 ? '• Orientation reversing' : '• Degenerate'}
              </Text>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Tropical Shortest Path Visualization
// ============================================================================

export function TropicalShortestPathVisualization() {
  const [nodes, setNodes] = useState<{x: number, y: number, label: string}[]>([]);
  const [edges, setEdges] = useState<{from: number, to: number, weight: number}[]>([]);
  const [distances, setDistances] = useState<number[]>([]);
  const [sourceNode, setSourceNode] = useState(0);
  const [pathHighlight, setPathHighlight] = useState<number[]>([]);
  const [targetNode, setTargetNode] = useState(4);
  const [isComputing, setIsComputing] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    // Initialize graph
    const newNodes = [
      { x: 50, y: 100, label: 'A' },
      { x: 120, y: 50, label: 'B' },
      { x: 120, y: 150, label: 'C' },
      { x: 190, y: 80, label: 'D' },
      { x: 190, y: 130, label: 'E' },
      { x: 260, y: 100, label: 'F' },
    ];
    const newEdges = [
      { from: 0, to: 1, weight: 2 },
      { from: 0, to: 2, weight: 4 },
      { from: 1, to: 2, weight: 1 },
      { from: 1, to: 3, weight: 7 },
      { from: 2, to: 4, weight: 3 },
      { from: 3, to: 4, weight: 2 },
      { from: 3, to: 5, weight: 1 },
      { from: 4, to: 5, weight: 5 },
    ];
    setNodes(newNodes);
    setEdges(newEdges);
    setDistances(Array(6).fill(Infinity));
  }, []);

  // Tropical Bellman-Ford: use min (tropical +) and + (tropical ×)
  const computeShortestPaths = useCallback(() => {
    const dist = Array(nodes.length).fill(Infinity);
    const prev: (number | null)[] = Array(nodes.length).fill(null);
    dist[sourceNode] = 0;

    // Relax edges n-1 times
    for (let i = 0; i < nodes.length - 1; i++) {
      for (const edge of edges) {
        // Tropical: dist[to] = min(dist[to], dist[from] ⊗ weight)
        // where ⊗ is classical + in tropical algebra
        const newDist = dist[edge.from] + edge.weight;
        if (newDist < dist[edge.to]) {
          dist[edge.to] = newDist;
          prev[edge.to] = edge.from;
        }
        // Also check reverse direction for undirected
        const newDistRev = dist[edge.to] + edge.weight;
        if (newDistRev < dist[edge.from]) {
          dist[edge.from] = newDistRev;
          prev[edge.from] = edge.to;
        }
      }
    }

    setDistances(dist);

    // Reconstruct path to target
    const path: number[] = [];
    let current: number | null = targetNode;
    while (current !== null) {
      path.unshift(current);
      current = prev[current];
    }
    if (path[0] === sourceNode) {
      setPathHighlight(path);
    } else {
      setPathHighlight([]);
    }
  }, [nodes.length, edges, sourceNode, targetNode]);

  useEffect(() => {
    if (nodes.length > 0) {
      computeShortestPaths();
    }
  }, [computeShortestPaths, nodes.length]);

  const runAnimation = () => {
    setIsComputing(true);
    setStep(0);
    const dist = Array(nodes.length).fill(Infinity);
    dist[sourceNode] = 0;
    setDistances([...dist]);

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep >= edges.length * 2) {
        setIsComputing(false);
        computeShortestPaths();
        clearInterval(interval);
        return;
      }

      const edgeIdx = currentStep % edges.length;
      const edge = edges[edgeIdx];
      const newDist = dist[edge.from] + edge.weight;
      if (newDist < dist[edge.to]) {
        dist[edge.to] = newDist;
      }
      const newDistRev = dist[edge.to] + edge.weight;
      if (newDistRev < dist[edge.from]) {
        dist[edge.from] = newDistRev;
      }

      setDistances([...dist]);
      setStep(currentStep);
      currentStep++;
    }, 300);
  };

  const isOnPath = (from: number, to: number) => {
    for (let i = 0; i < pathHighlight.length - 1; i++) {
      if ((pathHighlight[i] === from && pathHighlight[i + 1] === to) ||
          (pathHighlight[i] === to && pathHighlight[i + 1] === from)) {
        return true;
      }
    }
    return false;
  };

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Tropical Shortest Path</Title>
            <Text size="xs" c="dimmed">Min-plus semiring: ⊕ = min, ⊗ = +</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant={isComputing ? 'filled' : 'outline'} onClick={runAnimation} disabled={isComputing}>
              {isComputing ? 'Computing...' : 'Animate'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Graph visualization */}
          <Box>
            <svg viewBox="0 0 300 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Edges */}
              {edges.map((edge, i) => {
                const from = nodes[edge.from];
                const to = nodes[edge.to];
                if (!from || !to) return null;
                const onPath = isOnPath(edge.from, edge.to);
                const midX = (from.x + to.x) / 2;
                const midY = (from.y + to.y) / 2;
                return (
                  <g key={i}>
                    <line
                      x1={from.x}
                      y1={from.y}
                      x2={to.x}
                      y2={to.y}
                      stroke={onPath ? 'var(--mantine-color-yellow-5)' : 'var(--mantine-color-dark-3)'}
                      strokeWidth={onPath ? 4 : 2}
                    />
                    <rect x={midX - 10} y={midY - 8} width="20" height="16" fill="var(--mantine-color-dark-6)" rx="3" />
                    <text x={midX} y={midY + 4} fontSize="10" fill="var(--mantine-color-cyan-5)" textAnchor="middle">
                      {edge.weight}
                    </text>
                  </g>
                );
              })}

              {/* Nodes */}
              {nodes.map((node, i) => {
                const isSource = i === sourceNode;
                const isTarget = i === targetNode;
                const onPath = pathHighlight.includes(i);
                return (
                  <g key={i} onClick={() => setTargetNode(i)} style={{ cursor: 'pointer' }}>
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r="18"
                      fill={isSource ? 'var(--mantine-color-green-7)' : isTarget ? 'var(--mantine-color-red-7)' : onPath ? 'var(--mantine-color-yellow-7)' : 'var(--mantine-color-dark-4)'}
                      stroke={onPath ? 'var(--mantine-color-yellow-5)' : 'var(--mantine-color-dark-2)'}
                      strokeWidth="2"
                    />
                    <text x={node.x} y={node.y - 3} fontSize="12" fill="white" textAnchor="middle" fontWeight="bold">
                      {node.label}
                    </text>
                    <text x={node.x} y={node.y + 10} fontSize="9" fill="white" textAnchor="middle">
                      {distances[i] === Infinity ? '∞' : distances[i]}
                    </text>
                  </g>
                );
              })}
            </svg>
          </Box>

          {/* Info */}
          <Stack gap="md">
            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Tropical Algebra Operations</Text>
              <Text size="xs" ff="monospace">a ⊕ b = min(a, b)</Text>
              <Text size="xs" ff="monospace">a ⊗ b = a + b</Text>
              <Text size="xs" c="dimmed" mt="xs">
                Distance update: d[v] = d[v] ⊕ (d[u] ⊗ w(u,v))
              </Text>
            </Box>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Distance Vector (from {nodes[sourceNode]?.label || 'A'})</Text>
              <Group gap="xs">
                {distances.map((d, i) => (
                  <Badge key={i} size="lg" variant={pathHighlight.includes(i) ? 'filled' : 'light'} color={i === sourceNode ? 'green' : i === targetNode ? 'red' : 'gray'}>
                    {nodes[i]?.label}: {d === Infinity ? '∞' : d}
                  </Badge>
                ))}
              </Group>
            </Box>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Shortest Path: {nodes[sourceNode]?.label} → {nodes[targetNode]?.label}</Text>
              <Group gap={4}>
                {pathHighlight.map((nodeIdx, i) => (
                  <Group key={i} gap={4}>
                    <Badge size="sm" color="yellow">{nodes[nodeIdx]?.label}</Badge>
                    {i < pathHighlight.length - 1 && <Text size="xs">→</Text>}
                  </Group>
                ))}
              </Group>
              <Text size="xs" mt="xs" c="cyan">
                Total distance: {distances[targetNode] === Infinity ? '∞' : distances[targetNode]}
              </Text>
            </Box>

            <Text size="xs" c="dimmed">Click any node to set as target</Text>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Dynamical Systems Visualization (Combined)
// ============================================================================

type DynamicsSystem = 'lorenz' | 'vanderpol' | 'duffing' | 'rossler' | 'bifurcation';

export function DynamicsVisualization() {
  const [activeSystem, setActiveSystem] = useState<DynamicsSystem>('lorenz');
  const [trajectory, setTrajectory] = useState<{x: number, y: number, z: number}[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentState, setCurrentState] = useState({ x: 1, y: 1, z: 1 });

  // Lorenz parameters
  const [sigma, setSigma] = useState(10);
  const [rho, setRho] = useState(28);
  const [beta, setBeta] = useState(8/3);

  // Van der Pol parameter
  const [mu, setMu] = useState(1.5);

  // Duffing parameters
  const [delta, setDelta] = useState(0.1);
  const [duffingAlpha, setDuffingAlpha] = useState(-1);
  const [duffingBeta, setDuffingBeta] = useState(1);

  // Rossler parameters
  const [rosslerA, setRosslerA] = useState(0.2);
  const [rosslerB, setRosslerB] = useState(0.2);
  const [rosslerC, setRosslerC] = useState(5.7);

  // Bifurcation diagram data
  const [diagramData, setDiagramData] = useState<{r: number, x: number}[]>([]);

  const dt = 0.01;

  // System step functions
  const lorenzStep = (state: {x: number, y: number, z: number}) => {
    const dx = sigma * (state.y - state.x);
    const dy = state.x * (rho - state.z) - state.y;
    const dz = state.x * state.y - beta * state.z;
    return {
      x: state.x + dx * dt,
      y: state.y + dy * dt,
      z: state.z + dz * dt
    };
  };

  const vanDerPolStep = (state: {x: number, y: number, z: number}) => {
    const dx = state.y;
    const dy = mu * (1 - state.x * state.x) * state.y - state.x;
    return {
      x: state.x + dx * dt,
      y: state.y + dy * dt,
      z: 0
    };
  };

  const duffingStep = (state: {x: number, y: number, z: number}) => {
    const dx = state.y;
    const dy = -delta * state.y + duffingAlpha * state.x + duffingBeta * state.x * state.x * state.x;
    return {
      x: state.x + dx * dt,
      y: state.y + dy * dt,
      z: 0
    };
  };

  const rosslerStep = (state: {x: number, y: number, z: number}) => {
    const dx = -state.y - state.z;
    const dy = state.x + rosslerA * state.y;
    const dz = rosslerB + state.z * (state.x - rosslerC);
    return {
      x: state.x + dx * dt,
      y: state.y + dy * dt,
      z: state.z + dz * dt
    };
  };

  const stepFunction = useCallback(() => {
    switch (activeSystem) {
      case 'lorenz': return lorenzStep;
      case 'vanderpol': return vanDerPolStep;
      case 'duffing': return duffingStep;
      case 'rossler': return rosslerStep;
      default: return lorenzStep;
    }
  }, [activeSystem, sigma, rho, beta, mu, delta, duffingAlpha, duffingBeta, rosslerA, rosslerB, rosslerC]);

  useEffect(() => {
    if (!isRunning || activeSystem === 'bifurcation') return;
    const step = stepFunction();
    const interval = setInterval(() => {
      const newState = step(currentState);
      setCurrentState(newState);
      setTrajectory(prev => [...prev.slice(-800), newState]);
    }, 16);
    return () => clearInterval(interval);
  }, [isRunning, currentState, stepFunction, activeSystem]);

  // Compute bifurcation diagram
  useEffect(() => {
    if (activeSystem !== 'bifurcation') return;
    const data: {r: number, x: number}[] = [];
    const rMin = 2.5, rMax = 4.0;
    for (let i = 0; i < 400; i++) {
      const r = rMin + (rMax - rMin) * i / 400;
      let x = 0.5;
      for (let j = 0; j < 200; j++) x = r * x * (1 - x);
      for (let j = 0; j < 50; j++) {
        x = r * x * (1 - x);
        data.push({ r, x });
      }
    }
    setDiagramData(data);
  }, [activeSystem]);

  const reset = () => {
    setTrajectory([]);
    switch (activeSystem) {
      case 'lorenz':
        setCurrentState({ x: 1, y: 1, z: 1 });
        break;
      case 'vanderpol':
        setCurrentState({ x: 0.1, y: 0, z: 0 });
        break;
      case 'duffing':
        setCurrentState({ x: 0.5, y: 0, z: 0 });
        break;
      case 'rossler':
        setCurrentState({ x: 1, y: 1, z: 1 });
        break;
    }
  };

  useEffect(() => {
    reset();
    setIsRunning(false);
  }, [activeSystem]);

  // Projection functions
  const project2D = (point: {x: number, y: number, z: number}) => {
    switch (activeSystem) {
      case 'lorenz':
        return { px: 150 + point.x * 4, py: 180 - point.z * 3 };
      case 'vanderpol':
        return { px: 150 + point.x * 40, py: 110 - point.y * 30 };
      case 'duffing':
        return { px: 150 + point.x * 50, py: 110 - point.y * 40 };
      case 'rossler':
        return { px: 150 + point.x * 6, py: 110 - point.y * 6 };
      default:
        return { px: 150, py: 110 };
    }
  };

  const systemInfo: Record<DynamicsSystem, { title: string, description: string, equations: string }> = {
    lorenz: {
      title: 'Lorenz Attractor',
      description: 'Chaotic strange attractor - the "butterfly"',
      equations: 'dx/dt = σ(y-x), dy/dt = x(ρ-z)-y, dz/dt = xy-βz'
    },
    vanderpol: {
      title: 'Van der Pol Oscillator',
      description: 'Self-sustained oscillations with limit cycle',
      equations: 'dx/dt = y, dy/dt = μ(1-x²)y - x'
    },
    duffing: {
      title: 'Duffing Oscillator',
      description: 'Double-well potential with bistability',
      equations: 'dx/dt = y, dy/dt = -δy + αx + βx³'
    },
    rossler: {
      title: 'Rössler Attractor',
      description: 'Simpler chaotic system with single scroll',
      equations: 'dx/dt = -y-z, dy/dt = x+ay, dz/dt = b+z(x-c)'
    },
    bifurcation: {
      title: 'Bifurcation Diagram',
      description: 'Logistic map period-doubling cascade',
      equations: 'x_{n+1} = rx_n(1 - x_n)'
    }
  };

  const info = systemInfo[activeSystem];

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>{info.title}</Title>
            <Text size="xs" c="dimmed">{info.description}</Text>
          </div>
          <Group gap="xs">
            {activeSystem !== 'bifurcation' && (
              <>
                <Button size="xs" variant="outline" onClick={reset}>Reset</Button>
                <Button size="xs" variant={isRunning ? 'filled' : 'outline'} onClick={() => setIsRunning(!isRunning)}>
                  {isRunning ? 'Pause' : 'Run'}
                </Button>
              </>
            )}
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        {/* System selector */}
        <SegmentedControl
          fullWidth
          size="xs"
          mb="md"
          value={activeSystem}
          onChange={(v) => setActiveSystem(v as DynamicsSystem)}
          data={[
            { label: 'Lorenz', value: 'lorenz' },
            { label: 'Van der Pol', value: 'vanderpol' },
            { label: 'Duffing', value: 'duffing' },
            { label: 'Rössler', value: 'rossler' },
            { label: 'Bifurcation', value: 'bifurcation' },
          ]}
        />

        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          {/* Phase space / diagram visualization */}
          <Box>
            <svg viewBox="0 0 300 220" style={{ width: '100%', height: '220px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {activeSystem === 'bifurcation' ? (
                <>
                  {/* Bifurcation diagram */}
                  {diagramData.map((point, i) => (
                    <circle
                      key={i}
                      cx={10 + (point.r - 2.5) * 186}
                      cy={200 - point.x * 180}
                      r="0.5"
                      fill="var(--mantine-color-cyan-5)"
                      opacity="0.5"
                    />
                  ))}
                  <line x1="10" y1="200" x2="290" y2="200" stroke="var(--mantine-color-dark-3)" strokeWidth="1" />
                  <text x="150" y="215" fontSize="10" fill="var(--mantine-color-dimmed)" textAnchor="middle">r</text>
                  <text x="15" y="215" fontSize="8" fill="var(--mantine-color-dimmed)">2.5</text>
                  <text x="280" y="215" fontSize="8" fill="var(--mantine-color-dimmed)">4.0</text>
                </>
              ) : (
                <>
                  {/* Axes */}
                  <line x1="10" y1="110" x2="290" y2="110" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
                  <line x1="150" y1="10" x2="150" y2="210" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />

                  {/* Trajectory */}
                  {trajectory.length > 1 && (
                    <polyline
                      points={trajectory.map(p => {
                        const { px, py } = project2D(p);
                        return `${px},${py}`;
                      }).join(' ')}
                      fill="none"
                      stroke="var(--mantine-color-cyan-5)"
                      strokeWidth="1.5"
                      opacity="0.8"
                    />
                  )}

                  {/* Current point */}
                  {trajectory.length > 0 && (
                    <circle
                      cx={project2D(currentState).px}
                      cy={project2D(currentState).py}
                      r="5"
                      fill="var(--mantine-color-red-5)"
                    />
                  )}

                  {/* Fixed points for Duffing */}
                  {activeSystem === 'duffing' && duffingAlpha < 0 && (
                    <>
                      <circle cx={150 + Math.sqrt(-duffingAlpha/duffingBeta) * 50} cy={110} r="4" fill="var(--mantine-color-green-5)" opacity="0.5" />
                      <circle cx={150 - Math.sqrt(-duffingAlpha/duffingBeta) * 50} cy={110} r="4" fill="var(--mantine-color-green-5)" opacity="0.5" />
                      <circle cx={150} cy={110} r="4" fill="var(--mantine-color-yellow-5)" opacity="0.5" />
                    </>
                  )}

                  {/* Axis labels */}
                  <text x="285" y="105" fontSize="10" fill="var(--mantine-color-dimmed)">x</text>
                  <text x="155" y="20" fontSize="10" fill="var(--mantine-color-dimmed)">{activeSystem === 'lorenz' ? 'z' : 'y'}</text>
                </>
              )}
            </svg>
          </Box>

          {/* Controls */}
          <Stack gap="sm">
            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" ff="monospace" c="dimmed">{info.equations}</Text>
            </Box>

            {activeSystem === 'lorenz' && (
              <>
                <Box>
                  <Text size="xs" mb={4}>σ = {sigma.toFixed(1)}</Text>
                  <Slider value={sigma} onChange={setSigma} min={1} max={20} step={0.5} size="sm" disabled={isRunning} />
                </Box>
                <Box>
                  <Text size="xs" mb={4}>ρ = {rho.toFixed(1)}</Text>
                  <Slider value={rho} onChange={setRho} min={1} max={50} step={0.5} size="sm" disabled={isRunning} />
                </Box>
                <Box>
                  <Text size="xs" mb={4}>β = {beta.toFixed(2)}</Text>
                  <Slider value={beta} onChange={setBeta} min={0.5} max={5} step={0.1} size="sm" disabled={isRunning} />
                </Box>
              </>
            )}

            {activeSystem === 'vanderpol' && (
              <Box>
                <Text size="xs" mb={4}>μ = {mu.toFixed(2)} (nonlinearity)</Text>
                <Slider value={mu} onChange={setMu} min={0.1} max={5} step={0.1} size="sm" disabled={isRunning} />
                <Text size="xs" c="dimmed" mt="xs">μ = 0: harmonic oscillator</Text>
                <Text size="xs" c="dimmed">μ {'>'} 0: limit cycle attractor</Text>
              </Box>
            )}

            {activeSystem === 'duffing' && (
              <>
                <Box>
                  <Text size="xs" mb={4}>δ = {delta.toFixed(2)} (damping)</Text>
                  <Slider value={delta} onChange={setDelta} min={0} max={1} step={0.05} size="sm" disabled={isRunning} />
                </Box>
                <Box>
                  <Text size="xs" mb={4}>α = {duffingAlpha} (linear stiffness)</Text>
                  <Slider value={duffingAlpha} onChange={setDuffingAlpha} min={-2} max={2} step={0.1} size="sm" disabled={isRunning} />
                </Box>
                <Box>
                  <Text size="xs" mb={4}>β = {duffingBeta} (cubic stiffness)</Text>
                  <Slider value={duffingBeta} onChange={setDuffingBeta} min={-2} max={2} step={0.1} size="sm" disabled={isRunning} />
                </Box>
                <Text size="xs" c="dimmed">α {'<'} 0, β {'>'} 0: double-well potential</Text>
              </>
            )}

            {activeSystem === 'rossler' && (
              <>
                <Box>
                  <Text size="xs" mb={4}>a = {rosslerA.toFixed(2)}</Text>
                  <Slider value={rosslerA} onChange={setRosslerA} min={0.1} max={0.5} step={0.01} size="sm" disabled={isRunning} />
                </Box>
                <Box>
                  <Text size="xs" mb={4}>b = {rosslerB.toFixed(2)}</Text>
                  <Slider value={rosslerB} onChange={setRosslerB} min={0.1} max={0.5} step={0.01} size="sm" disabled={isRunning} />
                </Box>
                <Box>
                  <Text size="xs" mb={4}>c = {rosslerC.toFixed(1)}</Text>
                  <Slider value={rosslerC} onChange={setRosslerC} min={2} max={10} step={0.1} size="sm" disabled={isRunning} />
                </Box>
              </>
            )}

            {activeSystem === 'bifurcation' && (
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed" mb="xs">Period-Doubling Cascade</Text>
                <Stack gap={4}>
                  <Group gap="xs"><Badge size="xs" variant="light">Period 2</Badge><Text size="xs" ff="monospace">r ≈ 3.0</Text></Group>
                  <Group gap="xs"><Badge size="xs" variant="light">Period 4</Badge><Text size="xs" ff="monospace">r ≈ 3.449</Text></Group>
                  <Group gap="xs"><Badge size="xs" variant="light">Period 8</Badge><Text size="xs" ff="monospace">r ≈ 3.544</Text></Group>
                  <Group gap="xs"><Badge size="xs" variant="light" color="red">Chaos</Badge><Text size="xs" ff="monospace">r ≈ 3.57</Text></Group>
                </Stack>
                <Text size="xs" c="dimmed" mt="md">Feigenbaum constant: δ ≈ 4.669</Text>
              </Box>
            )}

            {activeSystem !== 'bifurcation' && (
              <SimpleGrid cols={2} spacing="xs">
                <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                  <Text size="xs" c="dimmed">State</Text>
                  <Text size="xs" ff="monospace">
                    ({currentState.x.toFixed(2)}, {currentState.y.toFixed(2)}{activeSystem !== 'vanderpol' && activeSystem !== 'duffing' ? `, ${currentState.z.toFixed(2)}` : ''})
                  </Text>
                </Box>
                <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                  <Text size="xs" c="dimmed">Points</Text>
                  <Text size="lg" fw={700}>{trajectory.length}</Text>
                </Box>
              </SimpleGrid>
            )}
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Nullcline Phase Portrait Visualization
// ============================================================================

export function NullclineVisualization() {
  const [systemType, setSystemType] = useState<'vanderpol' | 'fitzhugh' | 'lotka'>('vanderpol');
  const [trajectories, setTrajectories] = useState<{x: number, y: number}[][]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [mu, setMu] = useState(1.0);
  const [showNullclines, setShowNullclines] = useState(true);
  const [showVectorField, setShowVectorField] = useState(true);

  // System definitions
  const systems = {
    vanderpol: {
      name: 'Van der Pol',
      dx: (x: number, y: number) => y,
      dy: (x: number, y: number) => mu * (1 - x * x) * y - x,
      xRange: [-3, 3] as [number, number],
      yRange: [-4, 4] as [number, number],
    },
    fitzhugh: {
      name: 'FitzHugh-Nagumo',
      dx: (x: number, y: number) => x - x * x * x / 3 - y,
      dy: (x: number, y: number) => 0.08 * (x + 0.7 - 0.8 * y),
      xRange: [-2.5, 2.5] as [number, number],
      yRange: [-1.5, 1.5] as [number, number],
    },
    lotka: {
      name: 'Lotka-Volterra',
      dx: (x: number, y: number) => x * (1.5 - y),
      dy: (x: number, y: number) => y * (x - 1),
      xRange: [0, 4] as [number, number],
      yRange: [0, 4] as [number, number],
    },
  };

  const sys = systems[systemType];

  // Compute nullcline points
  const computeNullclines = () => {
    const xNull: {x: number, y: number}[] = [];
    const yNull: {x: number, y: number}[] = [];

    const [xMin, xMax] = sys.xRange;
    const [yMin, yMax] = sys.yRange;

    for (let x = xMin; x <= xMax; x += 0.05) {
      for (let y = yMin; y <= yMax; y += 0.05) {
        if (Math.abs(sys.dx(x, y)) < 0.1) xNull.push({ x, y });
        if (Math.abs(sys.dy(x, y)) < 0.1) yNull.push({ x, y });
      }
    }

    return { xNull, yNull };
  };

  // Compute fixed points (approximate)
  const findFixedPoints = () => {
    const fps: {x: number, y: number, type: string}[] = [];
    const [xMin, xMax] = sys.xRange;
    const [yMin, yMax] = sys.yRange;

    for (let x = xMin; x <= xMax; x += 0.2) {
      for (let y = yMin; y <= yMax; y += 0.2) {
        if (Math.abs(sys.dx(x, y)) < 0.15 && Math.abs(sys.dy(x, y)) < 0.15) {
          // Check if already found nearby
          const exists = fps.some(fp => Math.hypot(fp.x - x, fp.y - y) < 0.3);
          if (!exists) {
            // Simple stability classification based on local behavior
            const eps = 0.01;
            const dxdx = (sys.dx(x + eps, y) - sys.dx(x, y)) / eps;
            const dydy = (sys.dy(x, y + eps) - sys.dy(x, y)) / eps;
            const trace = dxdx + dydy;
            const type = trace < -0.1 ? 'stable' : trace > 0.1 ? 'unstable' : 'saddle';
            fps.push({ x, y, type });
          }
        }
      }
    }
    return fps;
  };

  const { xNull, yNull } = computeNullclines();
  const fixedPoints = findFixedPoints();

  // Generate vector field
  const vectorField: {x: number, y: number, dx: number, dy: number}[] = [];
  const [xMin, xMax] = sys.xRange;
  const [yMin, yMax] = sys.yRange;
  const step = (xMax - xMin) / 12;
  for (let x = xMin + step/2; x <= xMax; x += step) {
    for (let y = yMin + step/2; y <= yMax; y += step) {
      const dx = sys.dx(x, y);
      const dy = sys.dy(x, y);
      const mag = Math.hypot(dx, dy);
      if (mag > 0.01) {
        vectorField.push({ x, y, dx: dx / mag * 0.15, dy: dy / mag * 0.15 });
      }
    }
  }

  // Integration for trajectories
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setTrajectories(prev => {
        return prev.map(traj => {
          if (traj.length === 0) return traj;
          const last = traj[traj.length - 1];
          const dt = 0.02;
          const dx = sys.dx(last.x, last.y);
          const dy = sys.dy(last.x, last.y);
          const newPoint = { x: last.x + dx * dt, y: last.y + dy * dt };

          // Bound check
          if (newPoint.x < xMin - 1 || newPoint.x > xMax + 1 ||
              newPoint.y < yMin - 1 || newPoint.y > yMax + 1) {
            return traj;
          }

          return [...traj.slice(-200), newPoint];
        });
      });
    }, 30);

    return () => clearInterval(interval);
  }, [isRunning, systemType, mu]);

  const addTrajectory = (svgX: number, svgY: number) => {
    const x = xMin + (svgX / 300) * (xMax - xMin);
    const y = yMax - (svgY / 200) * (yMax - yMin);
    setTrajectories(prev => [...prev, [{ x, y }]]);
  };

  const toSvg = (x: number, y: number) => ({
    sx: ((x - xMin) / (xMax - xMin)) * 300,
    sy: ((yMax - y) / (yMax - yMin)) * 200
  });

  const reset = () => {
    setTrajectories([]);
    setIsRunning(false);
  };

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Nullcline Phase Portrait</Title>
            <Text size="xs" c="dimmed">Fixed points and flow structure analysis</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant="outline" onClick={reset}>Clear</Button>
            <Button size="xs" variant={isRunning ? 'filled' : 'outline'} onClick={() => setIsRunning(!isRunning)}>
              {isRunning ? 'Pause' : 'Run'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          <Box>
            <svg
              viewBox="0 0 300 200"
              style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)', cursor: 'crosshair' }}
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width * 300;
                const y = (e.clientY - rect.top) / rect.height * 200;
                addTrajectory(x, y);
              }}
            >
              {/* Vector field */}
              {showVectorField && vectorField.map((v, i) => {
                const { sx, sy } = toSvg(v.x, v.y);
                const { sx: ex, sy: ey } = toSvg(v.x + v.dx, v.y + v.dy);
                return (
                  <line key={i} x1={sx} y1={sy} x2={ex} y2={ey}
                    stroke="var(--mantine-color-dark-3)" strokeWidth="1" opacity="0.5" />
                );
              })}

              {/* X-nullcline (dx/dt = 0) */}
              {showNullclines && xNull.map((p, i) => {
                const { sx, sy } = toSvg(p.x, p.y);
                return <circle key={`x-${i}`} cx={sx} cy={sy} r="1.5" fill="var(--mantine-color-red-5)" opacity="0.6" />;
              })}

              {/* Y-nullcline (dy/dt = 0) */}
              {showNullclines && yNull.map((p, i) => {
                const { sx, sy } = toSvg(p.x, p.y);
                return <circle key={`y-${i}`} cx={sx} cy={sy} r="1.5" fill="var(--mantine-color-blue-5)" opacity="0.6" />;
              })}

              {/* Fixed points */}
              {fixedPoints.map((fp, i) => {
                const { sx, sy } = toSvg(fp.x, fp.y);
                return (
                  <circle key={`fp-${i}`} cx={sx} cy={sy} r="6"
                    fill={fp.type === 'stable' ? 'var(--mantine-color-green-5)' : fp.type === 'unstable' ? 'var(--mantine-color-red-5)' : 'var(--mantine-color-yellow-5)'}
                    stroke="white" strokeWidth="2" />
                );
              })}

              {/* Trajectories */}
              {trajectories.map((traj, ti) => (
                <polyline key={ti}
                  points={traj.map(p => {
                    const { sx, sy } = toSvg(p.x, p.y);
                    return `${sx},${sy}`;
                  }).join(' ')}
                  fill="none" stroke="var(--mantine-color-cyan-5)" strokeWidth="1.5" opacity="0.8" />
              ))}

              {/* Trajectory heads */}
              {trajectories.map((traj, ti) => {
                if (traj.length === 0) return null;
                const last = traj[traj.length - 1];
                const { sx, sy } = toSvg(last.x, last.y);
                return <circle key={`head-${ti}`} cx={sx} cy={sy} r="4" fill="var(--mantine-color-cyan-5)" />;
              })}
            </svg>
          </Box>

          <Stack gap="sm">
            <SegmentedControl
              fullWidth size="xs"
              value={systemType}
              onChange={(v) => { setSystemType(v as typeof systemType); reset(); }}
              data={Object.entries(systems).map(([k, v]) => ({ value: k, label: v.name }))}
            />

            {systemType === 'vanderpol' && (
              <Box>
                <Text size="xs" mb={4}>μ = {mu.toFixed(2)} (nonlinearity)</Text>
                <Slider value={mu} onChange={setMu} min={0.1} max={3} step={0.1} size="sm" />
              </Box>
            )}

            <Group gap="xs">
              <Button size="xs" variant={showNullclines ? 'filled' : 'light'} onClick={() => setShowNullclines(!showNullclines)}>
                Nullclines
              </Button>
              <Button size="xs" variant={showVectorField ? 'filled' : 'light'} onClick={() => setShowVectorField(!showVectorField)}>
                Vector Field
              </Button>
            </Group>

            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Legend</Text>
              <Group gap="xs">
                <Group gap={4}><Box w={10} h={10} bg="red.5" style={{ borderRadius: '50%' }} /><Text size="xs">dx/dt=0</Text></Group>
                <Group gap={4}><Box w={10} h={10} bg="blue.5" style={{ borderRadius: '50%' }} /><Text size="xs">dy/dt=0</Text></Group>
              </Group>
              <Text size="xs" c="dimmed" mt="xs">Fixed points: {fixedPoints.length}</Text>
            </Box>

            <Text size="xs" c="dimmed">Click on the phase plane to add initial conditions</Text>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Poincaré Section Visualization
// ============================================================================

export function PoincareVisualization() {
  const [trajectory, setTrajectory] = useState<{x: number, y: number, z: number}[]>([]);
  const [sectionPoints, setSectionPoints] = useState<{x: number, y: number}[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [sectionZ, setSectionZ] = useState(27);
  const [currentState, setCurrentState] = useState({ x: 1, y: 1, z: 1 });

  // Lorenz parameters
  const sigma = 10, rho = 28, beta = 8/3;
  const dt = 0.005;

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setCurrentState(prev => {
        const dx = sigma * (prev.y - prev.x);
        const dy = prev.x * (rho - prev.z) - prev.y;
        const dz = prev.x * prev.y - beta * prev.z;
        const next = {
          x: prev.x + dx * dt,
          y: prev.y + dy * dt,
          z: prev.z + dz * dt
        };

        // Detect Poincaré section crossing (z = sectionZ, dz > 0)
        if (prev.z < sectionZ && next.z >= sectionZ) {
          // Linear interpolation to find crossing point
          const t = (sectionZ - prev.z) / (next.z - prev.z);
          const crossX = prev.x + t * (next.x - prev.x);
          const crossY = prev.y + t * (next.y - prev.y);
          setSectionPoints(pts => [...pts.slice(-500), { x: crossX, y: crossY }]);
        }

        setTrajectory(traj => [...traj.slice(-1500), next]);
        return next;
      });
    }, 5);

    return () => clearInterval(interval);
  }, [isRunning, sectionZ]);

  const reset = () => {
    setTrajectory([]);
    setSectionPoints([]);
    setCurrentState({ x: 1 + Math.random() * 0.1, y: 1, z: 1 });
  };

  // Project 3D to 2D (x-z plane)
  const project = (p: {x: number, y: number, z: number}) => ({
    px: 150 + p.x * 4,
    py: 180 - p.z * 3
  });

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Poincaré Section</Title>
            <Text size="xs" c="dimmed">Cross-sectional analysis of Lorenz attractor</Text>
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
          {/* 3D trajectory projection */}
          <Box>
            <Text size="xs" c="dimmed" mb="xs">Trajectory (x-z projection)</Text>
            <svg viewBox="0 0 300 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Section plane indicator */}
              <line x1="0" y1={180 - sectionZ * 3} x2="300" y2={180 - sectionZ * 3}
                stroke="var(--mantine-color-yellow-5)" strokeWidth="2" strokeDasharray="5,5" opacity="0.7" />
              <text x="5" y={175 - sectionZ * 3} fontSize="10" fill="var(--mantine-color-yellow-5)">z={sectionZ}</text>

              {/* Trajectory */}
              {trajectory.length > 1 && (
                <polyline
                  points={trajectory.map(p => {
                    const { px, py } = project(p);
                    return `${px},${py}`;
                  }).join(' ')}
                  fill="none" stroke="var(--mantine-color-cyan-5)" strokeWidth="0.5" opacity="0.6" />
              )}

              {/* Current point */}
              <circle cx={project(currentState).px} cy={project(currentState).py} r="4" fill="var(--mantine-color-red-5)" />
            </svg>
          </Box>

          {/* Poincaré section */}
          <Box>
            <Text size="xs" c="dimmed" mb="xs">Poincaré Section (x-y at z={sectionZ})</Text>
            <svg viewBox="0 0 300 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Axes */}
              <line x1="150" y1="0" x2="150" y2="200" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
              <line x1="0" y1="100" x2="300" y2="100" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />

              {/* Section points */}
              {sectionPoints.map((p, i) => (
                <circle key={i}
                  cx={150 + p.x * 6}
                  cy={100 - p.y * 4}
                  r="2"
                  fill="var(--mantine-color-cyan-5)"
                  opacity={0.3 + (i / sectionPoints.length) * 0.7}
                />
              ))}

              <text x="290" y="95" fontSize="10" fill="var(--mantine-color-dimmed)">x</text>
              <text x="155" y="15" fontSize="10" fill="var(--mantine-color-dimmed)">y</text>
            </svg>
          </Box>
        </SimpleGrid>

        <Stack gap="sm" mt="md">
          <Box>
            <Text size="xs" mb={4}>Section plane z = {sectionZ}</Text>
            <Slider value={sectionZ} onChange={(v) => { setSectionZ(v); setSectionPoints([]); }}
              min={15} max={40} step={1} size="sm" disabled={isRunning} />
          </Box>

          <SimpleGrid cols={3} spacing="xs">
            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed">Crossings</Text>
              <Text size="lg" fw={700}>{sectionPoints.length}</Text>
            </Box>
            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed">Trajectory pts</Text>
              <Text size="lg" fw={700}>{trajectory.length}</Text>
            </Box>
            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed">State</Text>
              <Text size="xs" ff="monospace">({currentState.x.toFixed(1)}, {currentState.y.toFixed(1)}, {currentState.z.toFixed(1)})</Text>
            </Box>
          </SimpleGrid>

          <Text size="xs" c="dimmed">
            The Poincaré section reveals the fractal structure of the strange attractor
          </Text>
        </Stack>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Lyapunov Spectrum Heatmap Visualization
// ============================================================================

export function LyapunovVisualization() {
  const [heatmapData, setHeatmapData] = useState<{r: number, sigma: number, lambda: number}[]>([]);
  const [isComputing, setIsComputing] = useState(false);
  const [selectedPoint, setSelectedPoint] = useState<{r: number, sigma: number, lambda: number} | null>(null);
  const [resolution, setResolution] = useState(20);

  // Estimate largest Lyapunov exponent for Lorenz system
  const computeLyapunov = (sigma: number, rho: number, beta: number, steps = 1000) => {
    let x = 1, y = 1, z = 1;
    let dx = 0.0001, dy = 0, dz = 0;
    const dt = 0.01;
    let sum = 0;
    let count = 0;

    for (let i = 0; i < steps; i++) {
      // Evolve main trajectory
      const dxdt = sigma * (y - x);
      const dydt = x * (rho - z) - y;
      const dzdt = x * y - beta * z;
      x += dxdt * dt;
      y += dydt * dt;
      z += dzdt * dt;

      // Evolve tangent vector (linearized)
      const ddx = sigma * (dy - dx);
      const ddy = dx * (rho - z) + x * (-dz) - dy;
      const ddz = dx * y + x * dy - beta * dz;
      dx += ddx * dt;
      dy += ddy * dt;
      dz += ddz * dt;

      // Renormalize periodically
      if (i % 10 === 0) {
        const norm = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (norm > 0 && isFinite(norm)) {
          sum += Math.log(norm);
          count++;
          dx /= norm;
          dy /= norm;
          dz /= norm;
        }
      }
    }

    return count > 0 ? sum / (count * 10 * dt) : 0;
  };

  const computeHeatmap = () => {
    setIsComputing(true);
    const data: {r: number, sigma: number, lambda: number}[] = [];
    const beta = 8/3;

    // Compute asynchronously
    setTimeout(() => {
      for (let ri = 0; ri < resolution; ri++) {
        const rho = 10 + (ri / resolution) * 50; // rho from 10 to 60
        for (let si = 0; si < resolution; si++) {
          const sigma = 5 + (si / resolution) * 20; // sigma from 5 to 25
          const lambda = computeLyapunov(sigma, rho, beta, 500);
          data.push({ r: rho, sigma, lambda });
        }
      }
      setHeatmapData(data);
      setIsComputing(false);
    }, 50);
  };

  useEffect(() => {
    computeHeatmap();
  }, [resolution]);

  const lambdaToColor = (lambda: number) => {
    if (lambda > 0.5) return 'var(--mantine-color-red-6)';
    if (lambda > 0) return 'var(--mantine-color-orange-6)';
    if (lambda > -0.5) return 'var(--mantine-color-yellow-6)';
    if (lambda > -2) return 'var(--mantine-color-green-6)';
    return 'var(--mantine-color-blue-6)';
  };

  const cellSize = 300 / resolution;

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Lyapunov Exponent Heatmap</Title>
            <Text size="xs" c="dimmed">Chaos detection in Lorenz parameter space</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant="outline" onClick={computeHeatmap} disabled={isComputing}>
              {isComputing ? 'Computing...' : 'Recompute'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          <Box>
            <svg viewBox="0 0 300 300" style={{ width: '100%', height: '300px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Heatmap cells */}
              {heatmapData.map((d, i) => {
                const xi = Math.floor(i / resolution);
                const yi = i % resolution;
                return (
                  <rect key={i}
                    x={yi * cellSize}
                    y={(resolution - 1 - xi) * cellSize}
                    width={cellSize}
                    height={cellSize}
                    fill={lambdaToColor(d.lambda)}
                    opacity="0.8"
                    style={{ cursor: 'pointer' }}
                    onClick={() => setSelectedPoint(d)}
                  />
                );
              })}

              {/* Classic Lorenz point marker */}
              <circle cx={(10 - 5) / 20 * 300} cy={300 - (28 - 10) / 50 * 300} r="5"
                fill="none" stroke="white" strokeWidth="2" />

              {/* Axis labels */}
              <text x="150" y="295" fontSize="10" fill="var(--mantine-color-dimmed)" textAnchor="middle">σ</text>
              <text x="10" y="150" fontSize="10" fill="var(--mantine-color-dimmed)" transform="rotate(-90, 10, 150)">ρ</text>

              {/* Scale markers */}
              <text x="5" y="295" fontSize="8" fill="var(--mantine-color-dimmed)">5</text>
              <text x="290" y="295" fontSize="8" fill="var(--mantine-color-dimmed)">25</text>
              <text x="5" y="10" fontSize="8" fill="var(--mantine-color-dimmed)">60</text>
              <text x="5" y="290" fontSize="8" fill="var(--mantine-color-dimmed)">10</text>
            </svg>
          </Box>

          <Stack gap="md">
            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Color Legend (λ₁)</Text>
              <Stack gap={4}>
                <Group gap="xs"><Box w={20} h={12} bg="red.6" /><Text size="xs">λ {'>'} 0.5 (strongly chaotic)</Text></Group>
                <Group gap="xs"><Box w={20} h={12} bg="orange.6" /><Text size="xs">0 {'<'} λ {'<'} 0.5 (chaotic)</Text></Group>
                <Group gap="xs"><Box w={20} h={12} bg="yellow.6" /><Text size="xs">-0.5 {'<'} λ {'<'} 0 (weakly stable)</Text></Group>
                <Group gap="xs"><Box w={20} h={12} bg="green.6" /><Text size="xs">-2 {'<'} λ {'<'} -0.5 (stable)</Text></Group>
                <Group gap="xs"><Box w={20} h={12} bg="blue.6" /><Text size="xs">λ {'<'} -2 (strongly stable)</Text></Group>
              </Stack>
            </Box>

            {selectedPoint && (
              <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed" mb="xs">Selected Point</Text>
                <Text size="xs" ff="monospace">σ = {selectedPoint.sigma.toFixed(2)}</Text>
                <Text size="xs" ff="monospace">ρ = {selectedPoint.r.toFixed(2)}</Text>
                <Text size="xs" ff="monospace" c={selectedPoint.lambda > 0 ? 'red' : 'green'}>
                  λ₁ = {selectedPoint.lambda.toFixed(4)}
                </Text>
                <Text size="xs" c="dimmed" mt="xs">
                  {selectedPoint.lambda > 0 ? 'Chaotic behavior' : 'Regular behavior'}
                </Text>
              </Box>
            )}

            <Box>
              <Text size="xs" mb={4}>Resolution: {resolution}x{resolution}</Text>
              <Slider value={resolution} onChange={setResolution}
                min={10} max={40} step={5} size="sm" disabled={isComputing} />
            </Box>

            <Text size="xs" c="dimmed">
              White circle: classic Lorenz (σ=10, ρ=28). Click cells to inspect.
            </Text>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Ergodic Measure Evolution Visualization
// ============================================================================

export function ErgodicVisualization() {
  const [histogram, setHistogram] = useState<number[]>(Array(30).fill(0));
  const [trajectory, setTrajectory] = useState<number[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [mapType, setMapType] = useState<'logistic' | 'tent' | 'circle'>('logistic');
  const [r, setR] = useState(3.9);
  const [iterations, setIterations] = useState(0);
  const [currentX, setCurrentX] = useState(0.5);

  // Map functions
  const maps = {
    logistic: (x: number) => r * x * (1 - x),
    tent: (x: number) => x < 0.5 ? 2 * x : 2 * (1 - x),
    circle: (x: number) => (x + 0.5 + 0.3 * Math.sin(2 * Math.PI * x)) % 1,
  };

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      const map = maps[mapType];
      setCurrentX(prev => {
        const next = map(prev);
        const binIdx = Math.min(29, Math.floor(next * 30));

        setHistogram(h => {
          const newH = [...h];
          newH[binIdx]++;
          return newH;
        });

        setTrajectory(t => [...t.slice(-100), next]);
        setIterations(i => i + 1);

        return next;
      });
    }, 10);

    return () => clearInterval(interval);
  }, [isRunning, mapType, r]);

  const reset = () => {
    setHistogram(Array(30).fill(0));
    setTrajectory([]);
    setIterations(0);
    setCurrentX(Math.random());
  };

  useEffect(() => {
    reset();
  }, [mapType, r]);

  const maxCount = Math.max(...histogram, 1);

  // Theoretical invariant density for logistic map at r=4
  const theoreticalDensity = mapType === 'logistic' && r >= 3.99
    ? Array(30).fill(0).map((_, i) => {
        const x = (i + 0.5) / 30;
        return 1 / (Math.PI * Math.sqrt(x * (1 - x)));
      })
    : null;

  const maxTheoretical = theoreticalDensity ? Math.max(...theoreticalDensity) : 1;

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Ergodic Measure Evolution</Title>
            <Text size="xs" c="dimmed">Convergence to invariant density via Birkhoff average</Text>
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
          <Box>
            <Text size="xs" c="dimmed" mb="xs">Histogram (empirical measure)</Text>
            <svg viewBox="0 0 300 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Histogram bars */}
              {histogram.map((count, i) => (
                <rect key={i}
                  x={i * 10}
                  y={200 - (count / maxCount) * 180}
                  width="9"
                  height={(count / maxCount) * 180}
                  fill="var(--mantine-color-cyan-5)"
                  opacity="0.7"
                />
              ))}

              {/* Theoretical density curve */}
              {theoreticalDensity && (
                <polyline
                  points={theoreticalDensity.map((d, i) =>
                    `${i * 10 + 5},${200 - (d / maxTheoretical) * 180 * (maxCount / iterations || 1) * 30}`
                  ).join(' ')}
                  fill="none"
                  stroke="var(--mantine-color-yellow-5)"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />
              )}

              {/* Current position marker */}
              <line x1={currentX * 300} y1="0" x2={currentX * 300} y2="200"
                stroke="var(--mantine-color-red-5)" strokeWidth="2" opacity="0.8" />

              {/* Axis labels */}
              <text x="0" y="195" fontSize="8" fill="var(--mantine-color-dimmed)">0</text>
              <text x="290" y="195" fontSize="8" fill="var(--mantine-color-dimmed)">1</text>
            </svg>
          </Box>

          <Stack gap="md">
            <SegmentedControl
              fullWidth size="xs"
              value={mapType}
              onChange={(v) => setMapType(v as typeof mapType)}
              data={[
                { label: 'Logistic', value: 'logistic' },
                { label: 'Tent', value: 'tent' },
                { label: 'Circle', value: 'circle' },
              ]}
            />

            {mapType === 'logistic' && (
              <Box>
                <Text size="xs" mb={4}>r = {r.toFixed(2)}</Text>
                <Slider value={r} onChange={setR} min={3.5} max={4} step={0.01} size="sm" disabled={isRunning} />
              </Box>
            )}

            <SimpleGrid cols={2} spacing="xs">
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Iterations</Text>
                <Text size="lg" fw={700}>{iterations.toLocaleString()}</Text>
              </Box>
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Current x</Text>
                <Text size="lg" fw={700}>{currentX.toFixed(4)}</Text>
              </Box>
            </SimpleGrid>

            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Map Equation</Text>
              <Text size="xs" ff="monospace">
                {mapType === 'logistic' && `xₙ₊₁ = ${r.toFixed(2)}·xₙ(1 - xₙ)`}
                {mapType === 'tent' && 'xₙ₊₁ = 2·min(xₙ, 1-xₙ)'}
                {mapType === 'circle' && 'xₙ₊₁ = xₙ + 0.5 + 0.3·sin(2πxₙ) mod 1'}
              </Text>
            </Box>

            <Text size="xs" c="dimmed">
              {mapType === 'logistic' && r >= 3.99 && 'Yellow dashed: theoretical density 1/(π√(x(1-x)))'}
              {mapType === 'tent' && 'Tent map has uniform invariant measure'}
            </Text>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Eigenvalue Path Tracer Visualization
// ============================================================================

export function EigenvalueVisualization() {
  const [parameter, setParameter] = useState(0);
  const [eigenvalues, setEigenvalues] = useState<{re: number, im: number}[]>([]);
  const [paths, setPaths] = useState<{re: number, im: number}[][]>([[], []]);
  const [isAnimating, setIsAnimating] = useState(false);
  const [matrixType, setMatrixType] = useState<'rotation' | 'shear' | 'hopf'>('hopf');

  // Compute eigenvalues for 2x2 matrix
  const computeEigenvalues = (a: number, b: number, c: number, d: number) => {
    const trace = a + d;
    const det = a * d - b * c;
    const discriminant = trace * trace - 4 * det;

    if (discriminant >= 0) {
      const sqrt = Math.sqrt(discriminant);
      return [
        { re: (trace + sqrt) / 2, im: 0 },
        { re: (trace - sqrt) / 2, im: 0 }
      ];
    } else {
      const sqrtAbs = Math.sqrt(-discriminant);
      return [
        { re: trace / 2, im: sqrtAbs / 2 },
        { re: trace / 2, im: -sqrtAbs / 2 }
      ];
    }
  };

  // Matrix as function of parameter
  const getMatrix = (t: number): [number, number, number, number] => {
    switch (matrixType) {
      case 'rotation':
        return [Math.cos(t), -Math.sin(t), Math.sin(t), Math.cos(t)];
      case 'shear':
        return [1, t, 0, 1];
      case 'hopf':
        // Hopf bifurcation normal form Jacobian
        return [t, -1, 1, t];
    }
  };

  useEffect(() => {
    const [a, b, c, d] = getMatrix(parameter);
    const eigs = computeEigenvalues(a, b, c, d);
    setEigenvalues(eigs);
  }, [parameter, matrixType]);

  useEffect(() => {
    if (!isAnimating) return;

    const interval = setInterval(() => {
      setParameter(prev => {
        const next = prev + 0.02;
        if (next > 2) return -2;

        const [a, b, c, d] = getMatrix(next);
        const eigs = computeEigenvalues(a, b, c, d);

        setPaths(p => [
          [...p[0].slice(-200), eigs[0]],
          [...p[1].slice(-200), eigs[1]]
        ]);

        return next;
      });
    }, 30);

    return () => clearInterval(interval);
  }, [isAnimating, matrixType]);

  const reset = () => {
    setParameter(matrixType === 'hopf' ? -1 : 0);
    setPaths([[], []]);
    setIsAnimating(false);
  };

  useEffect(() => {
    reset();
  }, [matrixType]);

  const [a, b, c, d] = getMatrix(parameter);
  const trace = a + d;
  const det = a * d - b * c;

  const toSvg = (re: number, im: number) => ({
    x: 150 + re * 60,
    y: 100 - im * 60
  });

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Eigenvalue Path Tracer</Title>
            <Text size="xs" c="dimmed">Track eigenvalues as matrix parameters vary</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant="outline" onClick={reset}>Reset</Button>
            <Button size="xs" variant={isAnimating ? 'filled' : 'outline'} onClick={() => setIsAnimating(!isAnimating)}>
              {isAnimating ? 'Pause' : 'Animate'}
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          <Box>
            <svg viewBox="0 0 300 200" style={{ width: '100%', height: '200px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
              {/* Unit circle */}
              <circle cx="150" cy="100" r="60" fill="none" stroke="var(--mantine-color-dark-4)" strokeWidth="1" strokeDasharray="3,3" />

              {/* Axes */}
              <line x1="30" y1="100" x2="270" y2="100" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
              <line x1="150" y1="20" x2="150" y2="180" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />

              {/* Stability region (left half-plane) */}
              <rect x="30" y="20" width="120" height="160" fill="var(--mantine-color-green-9)" opacity="0.1" />

              {/* Eigenvalue paths */}
              {paths[0].length > 1 && (
                <polyline
                  points={paths[0].map(e => `${toSvg(e.re, e.im).x},${toSvg(e.re, e.im).y}`).join(' ')}
                  fill="none" stroke="var(--mantine-color-cyan-5)" strokeWidth="1" opacity="0.6" />
              )}
              {paths[1].length > 1 && (
                <polyline
                  points={paths[1].map(e => `${toSvg(e.re, e.im).x},${toSvg(e.re, e.im).y}`).join(' ')}
                  fill="none" stroke="var(--mantine-color-yellow-5)" strokeWidth="1" opacity="0.6" />
              )}

              {/* Current eigenvalues */}
              {eigenvalues.map((e, i) => {
                const { x, y } = toSvg(e.re, e.im);
                return (
                  <circle key={i} cx={x} cy={y} r="6"
                    fill={i === 0 ? 'var(--mantine-color-cyan-5)' : 'var(--mantine-color-yellow-5)'}
                    stroke="white" strokeWidth="2" />
                );
              })}

              {/* Labels */}
              <text x="265" y="95" fontSize="10" fill="var(--mantine-color-dimmed)">Re</text>
              <text x="155" y="25" fontSize="10" fill="var(--mantine-color-dimmed)">Im</text>
              <text x="60" y="180" fontSize="8" fill="var(--mantine-color-green-5)">stable</text>
            </svg>
          </Box>

          <Stack gap="sm">
            <SegmentedControl
              fullWidth size="xs"
              value={matrixType}
              onChange={(v) => setMatrixType(v as typeof matrixType)}
              data={[
                { label: 'Hopf', value: 'hopf' },
                { label: 'Rotation', value: 'rotation' },
                { label: 'Shear', value: 'shear' },
              ]}
            />

            <Box>
              <Text size="xs" mb={4}>Parameter: {parameter.toFixed(3)}</Text>
              <Slider value={parameter} onChange={setParameter}
                min={matrixType === 'rotation' ? 0 : -2}
                max={matrixType === 'rotation' ? 2 * Math.PI : 2}
                step={0.01} size="sm" disabled={isAnimating} />
            </Box>

            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Matrix A</Text>
              <Text size="xs" ff="monospace">
                [{a.toFixed(3)}, {b.toFixed(3)}]
              </Text>
              <Text size="xs" ff="monospace">
                [{c.toFixed(3)}, {d.toFixed(3)}]
              </Text>
            </Box>

            <SimpleGrid cols={2} spacing="xs">
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">λ₁</Text>
                <Text size="xs" ff="monospace" c="cyan">
                  {eigenvalues[0]?.re.toFixed(3)}{eigenvalues[0]?.im !== 0 ? ` + ${eigenvalues[0]?.im.toFixed(3)}i` : ''}
                </Text>
              </Box>
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">λ₂</Text>
                <Text size="xs" ff="monospace" c="yellow">
                  {eigenvalues[1]?.re.toFixed(3)}{eigenvalues[1]?.im !== 0 ? ` + ${eigenvalues[1]?.im.toFixed(3)}i` : ''}
                </Text>
              </Box>
            </SimpleGrid>

            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs">tr(A) = {trace.toFixed(3)}, det(A) = {det.toFixed(3)}</Text>
              <Text size="xs" c={eigenvalues.every(e => e.re < 0) ? 'green' : eigenvalues.some(e => e.re > 0) ? 'red' : 'yellow'}>
                {eigenvalues.every(e => e.re < 0) ? 'Stable (Re < 0)' :
                 eigenvalues.some(e => e.re > 0) ? 'Unstable (Re > 0)' : 'Marginal'}
              </Text>
            </Box>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Geodesic Distance Heatmap Visualization
// ============================================================================

export function GeodesicVisualization() {
  const [selectedPoint, setSelectedPoint] = useState<{x: number, y: number} | null>({ x: 0.5, y: 0.5 });
  const [metric, setMetric] = useState<'euclidean' | 'fisher' | 'hyperbolic'>('fisher');
  const resolution = 25;

  // Distance functions
  const distances = {
    euclidean: (p1: {x: number, y: number}, p2: {x: number, y: number}) =>
      Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2),

    fisher: (p1: {x: number, y: number}, p2: {x: number, y: number}) => {
      // Fisher-Rao distance on probability simplex (2D approx)
      const theta1 = 2 * Math.acos(Math.sqrt(Math.max(0.01, Math.min(0.99, p1.x))));
      const theta2 = 2 * Math.acos(Math.sqrt(Math.max(0.01, Math.min(0.99, p2.x))));
      const phi1 = 2 * Math.acos(Math.sqrt(Math.max(0.01, Math.min(0.99, p1.y))));
      const phi2 = 2 * Math.acos(Math.sqrt(Math.max(0.01, Math.min(0.99, p2.y))));
      return Math.sqrt((theta1 - theta2) ** 2 + (phi1 - phi2) ** 2);
    },

    hyperbolic: (p1: {x: number, y: number}, p2: {x: number, y: number}) => {
      // Poincaré disk distance (map [0,1]² to disk)
      const toD = (p: {x: number, y: number}) => ({
        x: (p.x - 0.5) * 1.6,
        y: (p.y - 0.5) * 1.6
      });
      const z1 = toD(p1);
      const z2 = toD(p2);
      const r1sq = z1.x * z1.x + z1.y * z1.y;
      const r2sq = z2.x * z2.x + z2.y * z2.y;
      if (r1sq >= 0.99 || r2sq >= 0.99) return 10;
      const dx = z1.x - z2.x;
      const dy = z1.y - z2.y;
      const num = dx * dx + dy * dy;
      const denom = (1 - r1sq) * (1 - r2sq);
      return Math.acosh(1 + 2 * num / Math.max(denom, 0.001));
    }
  };

  // Generate heatmap data
  const heatmapData: {x: number, y: number, dist: number}[] = [];
  let maxDist = 0;

  if (selectedPoint) {
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = (i + 0.5) / resolution;
        const y = (j + 0.5) / resolution;
        const dist = distances[metric]({ x, y }, selectedPoint);
        heatmapData.push({ x, y, dist });
        if (dist < Infinity && dist > maxDist) maxDist = dist;
      }
    }
  }

  const distToColor = (dist: number) => {
    const t = Math.min(1, dist / Math.max(maxDist, 0.01));
    // Blue to red gradient
    const r = Math.floor(255 * t);
    const b = Math.floor(255 * (1 - t));
    return `rgb(${r}, 50, ${b})`;
  };

  const cellSize = 300 / resolution;

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Geodesic Distance Heatmap</Title>
            <Text size="xs" c="dimmed">Distance from reference point under different metrics</Text>
          </div>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          <Box>
            <svg
              viewBox="0 0 300 300"
              style={{ width: '100%', height: '300px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)', cursor: 'crosshair' }}
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const x = (e.clientX - rect.left) / rect.width;
                const y = 1 - (e.clientY - rect.top) / rect.height;
                setSelectedPoint({ x, y });
              }}
            >
              {/* Heatmap cells */}
              {heatmapData.map((d, i) => (
                <rect key={i}
                  x={d.x * 300 - cellSize/2}
                  y={(1 - d.y) * 300 - cellSize/2}
                  width={cellSize}
                  height={cellSize}
                  fill={distToColor(d.dist)}
                  opacity="0.8"
                />
              ))}

              {/* Hyperbolic disk boundary */}
              {metric === 'hyperbolic' && (
                <circle cx="150" cy="150" r="120" fill="none" stroke="white" strokeWidth="2" strokeDasharray="5,5" />
              )}

              {/* Selected point */}
              {selectedPoint && (
                <circle
                  cx={selectedPoint.x * 300}
                  cy={(1 - selectedPoint.y) * 300}
                  r="8"
                  fill="white"
                  stroke="black"
                  strokeWidth="2"
                />
              )}

              {/* Geodesic circles (level sets) */}
              {selectedPoint && [0.2, 0.4, 0.6, 0.8].map((level, i) => {
                if (metric === 'euclidean') {
                  const r = level * Math.max(maxDist, 0.01) * 300;
                  return (
                    <circle key={i}
                      cx={selectedPoint.x * 300}
                      cy={(1 - selectedPoint.y) * 300}
                      r={r}
                      fill="none"
                      stroke="white"
                      strokeWidth="0.5"
                      opacity="0.5"
                    />
                  );
                }
                return null;
              })}
            </svg>
          </Box>

          <Stack gap="md">
            <SegmentedControl
              fullWidth size="xs"
              value={metric}
              onChange={(v) => setMetric(v as typeof metric)}
              data={[
                { label: 'Euclidean', value: 'euclidean' },
                { label: 'Fisher-Rao', value: 'fisher' },
                { label: 'Hyperbolic', value: 'hyperbolic' },
              ]}
            />

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Metric Description</Text>
              <Text size="xs">
                {metric === 'euclidean' && 'd(p,q) = √((p₁-q₁)² + (p₂-q₂)²)'}
                {metric === 'fisher' && 'Fisher-Rao distance using √p coordinates on statistical manifold'}
                {metric === 'hyperbolic' && 'Poincaré disk model: d(z,w) = arcosh(1 + 2|z-w|²/((1-|z|²)(1-|w|²)))'}
              </Text>
            </Box>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Reference Point</Text>
              {selectedPoint ? (
                <Text size="xs" ff="monospace">
                  ({selectedPoint.x.toFixed(3)}, {selectedPoint.y.toFixed(3)})
                </Text>
              ) : (
                <Text size="xs" c="dimmed">Click to select</Text>
              )}
              <Text size="xs" c="dimmed" mt="xs">Max distance: {maxDist.toFixed(3)}</Text>
            </Box>

            <Box p="sm" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Color Scale</Text>
              <Group gap={4}>
                <Box w={20} h={12} bg="blue" />
                <Text size="xs">Near (d=0)</Text>
                <Box w={20} h={12} bg="purple" />
                <Text size="xs">→</Text>
                <Box w={20} h={12} bg="red" />
                <Text size="xs">Far (d=max)</Text>
              </Group>
            </Box>

            <Text size="xs" c="dimmed">
              Click anywhere to set a new reference point
            </Text>
          </Stack>
        </SimpleGrid>
      </Card.Section>
    </Card>
  );
}

// ============================================================================
// Holographic Memory Explorer Visualization
// ============================================================================

export function HolographicVisualization() {
  const [memories, setMemories] = useState<number[][]>([]);
  const [queryPattern, setQueryPattern] = useState<number[]>([]);
  const [retrievedPattern, setRetrievedPattern] = useState<number[]>([]);
  const [dimension, setDimension] = useState(16);
  const [noiseLevel, setNoiseLevel] = useState(0.2);
  const [similarity, setSimilarity] = useState(0);
  const [isEncoding, setIsEncoding] = useState(false);

  // Generate random binary pattern
  const randomPattern = (dim: number) => {
    return Array(dim).fill(0).map(() => Math.random() > 0.5 ? 1 : -1);
  };

  // Normalize vector
  const normalize = (v: number[]) => {
    const mag = Math.sqrt(v.reduce((sum, x) => sum + x * x, 0));
    return mag > 0 ? v.map(x => x / mag) : v;
  };

  // Circular convolution (binding operation)
  const bind = (a: number[], b: number[]) => {
    const n = a.length;
    const result = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        result[i] += a[j] * b[(i - j + n) % n];
      }
    }
    return normalize(result);
  };

  // Circular correlation (unbinding operation)
  const unbind = (composite: number[], key: number[]) => {
    const n = composite.length;
    const result = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        result[i] += composite[j] * key[(j - i + n) % n];
      }
    }
    return normalize(result);
  };

  // Cosine similarity
  const cosineSimilarity = (a: number[], b: number[]) => {
    const dot = a.reduce((sum, x, i) => sum + x * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, x) => sum + x * x, 0));
    const magB = Math.sqrt(b.reduce((sum, x) => sum + x * x, 0));
    return magA > 0 && magB > 0 ? dot / (magA * magB) : 0;
  };

  // Add noise to pattern
  const addNoise = (pattern: number[], level: number) => {
    return pattern.map(x => x + (Math.random() - 0.5) * 2 * level);
  };

  // Initialize
  useEffect(() => {
    const patterns = Array(3).fill(0).map(() => randomPattern(dimension));
    setMemories(patterns);
    if (patterns.length > 0) {
      setQueryPattern(addNoise([...patterns[0]], noiseLevel));
    }
  }, [dimension]);

  // Store a new memory
  const storeMemory = () => {
    const newPattern = randomPattern(dimension);
    setMemories(prev => [...prev, newPattern]);
  };

  // Query memory
  const queryMemory = () => {
    if (memories.length === 0 || queryPattern.length === 0) return;

    setIsEncoding(true);

    // Find best match
    let bestMatch = memories[0];
    let bestSim = -Infinity;

    for (const mem of memories) {
      const sim = cosineSimilarity(queryPattern, mem);
      if (sim > bestSim) {
        bestSim = sim;
        bestMatch = mem;
      }
    }

    setRetrievedPattern(bestMatch);
    setSimilarity(bestSim);
    setIsEncoding(false);
  };

  // Generate noisy query from first memory
  const generateQuery = () => {
    if (memories.length > 0) {
      const idx = Math.floor(Math.random() * memories.length);
      setQueryPattern(addNoise([...memories[idx]], noiseLevel));
    }
  };

  const toBarHeight = (val: number) => Math.abs(val) * 30;
  const toBarColor = (val: number) => val > 0 ? 'var(--mantine-color-cyan-5)' : 'var(--mantine-color-red-5)';

  return (
    <Card withBorder>
      <Card.Section withBorder inheritPadding py="sm" bg="dark.6">
        <Group justify="space-between">
          <div>
            <Title order={4}>Holographic Memory Explorer</Title>
            <Text size="xs" c="dimmed">Content-addressable memory with distributed representations</Text>
          </div>
          <Group gap="xs">
            <Button size="xs" variant="outline" onClick={storeMemory}>Store New</Button>
            <Button size="xs" variant="outline" onClick={generateQuery}>Random Query</Button>
            <Button size="xs" variant="filled" onClick={queryMemory} loading={isEncoding}>Retrieve</Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          <Stack gap="md">
            <Box>
              <Text size="xs" c="dimmed" mb="xs">Query Pattern (with noise)</Text>
              <svg viewBox="0 0 300 60" style={{ width: '100%', height: '60px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
                {queryPattern.map((v, i) => (
                  <rect key={i}
                    x={i * (300 / dimension)}
                    y={v > 0 ? 30 - toBarHeight(v) : 30}
                    width={300 / dimension - 1}
                    height={toBarHeight(v)}
                    fill={toBarColor(v)}
                    opacity="0.8"
                  />
                ))}
                <line x1="0" y1="30" x2="300" y2="30" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
              </svg>
            </Box>

            <Box>
              <Text size="xs" c="dimmed" mb="xs">Retrieved Pattern</Text>
              <svg viewBox="0 0 300 60" style={{ width: '100%', height: '60px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
                {retrievedPattern.map((v, i) => (
                  <rect key={i}
                    x={i * (300 / dimension)}
                    y={v > 0 ? 30 - toBarHeight(v) : 30}
                    width={300 / dimension - 1}
                    height={toBarHeight(v)}
                    fill="var(--mantine-color-green-5)"
                    opacity="0.8"
                  />
                ))}
                <line x1="0" y1="30" x2="300" y2="30" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5" />
              </svg>
            </Box>

            <Box>
              <Text size="xs" c="dimmed" mb="xs">Stored Memories ({memories.length})</Text>
              <svg viewBox="0 0 300 80" style={{ width: '100%', height: '80px', background: 'var(--mantine-color-dark-7)', borderRadius: 'var(--mantine-radius-sm)' }}>
                {memories.slice(0, 4).map((mem, mi) => (
                  <g key={mi} transform={`translate(0, ${mi * 20})`}>
                    {mem.map((v, i) => (
                      <rect key={i}
                        x={i * (300 / dimension)}
                        y={v > 0 ? 8 : 10}
                        width={300 / dimension - 1}
                        height={v > 0 ? 4 : 4}
                        fill={v > 0 ? 'var(--mantine-color-blue-5)' : 'var(--mantine-color-orange-5)'}
                        opacity="0.6"
                      />
                    ))}
                  </g>
                ))}
              </svg>
            </Box>
          </Stack>

          <Stack gap="md">
            <Box>
              <Text size="xs" mb={4}>Dimension: {dimension}</Text>
              <Slider value={dimension} onChange={setDimension} min={8} max={64} step={8} size="sm" />
            </Box>

            <Box>
              <Text size="xs" mb={4}>Noise Level: {noiseLevel.toFixed(2)}</Text>
              <Slider value={noiseLevel} onChange={setNoiseLevel} min={0} max={1} step={0.05} size="sm" />
            </Box>

            <SimpleGrid cols={2} spacing="xs">
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Retrieval Similarity</Text>
                <Text size="lg" fw={700} c={similarity > 0.8 ? 'green' : similarity > 0.5 ? 'yellow' : 'red'}>
                  {(similarity * 100).toFixed(1)}%
                </Text>
              </Box>
              <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
                <Text size="xs" c="dimmed">Capacity</Text>
                <Text size="lg" fw={700}>{memories.length} / {Math.floor(dimension * 0.15)}</Text>
              </Box>
            </SimpleGrid>

            <Box p="xs" bg="dark.7" style={{ borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="xs" c="dimmed" mb="xs">Holographic Properties</Text>
              <Text size="xs">• Distributed: each component encodes all memories</Text>
              <Text size="xs">• Robust: partial degradation = graceful retrieval loss</Text>
              <Text size="xs">• Capacity: ~0.15n items for n-dimensional vectors</Text>
            </Box>

            <Progress
              value={(memories.length / Math.floor(dimension * 0.15)) * 100}
              color={memories.length > dimension * 0.15 ? 'red' : 'cyan'}
              size="sm"
            />
            <Text size="xs" c="dimmed">Memory utilization</Text>
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
          <Tabs.List style={{ flexWrap: 'wrap' }}>
            <Tabs.Tab value="geometric">Clifford Algebra</Tabs.Tab>
            <Tabs.Tab value="grades">Grade Decomp</Tabs.Tab>
            <Tabs.Tab value="mobius">Möbius</Tabs.Tab>
            <Tabs.Tab value="tropical">Tropical</Tabs.Tab>
            <Tabs.Tab value="tropical-path">Shortest Path</Tabs.Tab>
            <Tabs.Tab value="dual">Autodiff</Tabs.Tab>
            <Tabs.Tab value="rotor">Rotations</Tabs.Tab>
            <Tabs.Tab value="fisher">Fisher Info</Tabs.Tab>
            <Tabs.Tab value="geodesic">Geodesics</Tabs.Tab>
            <Tabs.Tab value="topology">Topology</Tabs.Tab>
            <Tabs.Tab value="dynamics">Dynamics</Tabs.Tab>
            <Tabs.Tab value="nullclines">Nullclines</Tabs.Tab>
            <Tabs.Tab value="poincare">Poincaré</Tabs.Tab>
            <Tabs.Tab value="lyapunov">Lyapunov</Tabs.Tab>
            <Tabs.Tab value="ergodic">Ergodic</Tabs.Tab>
            <Tabs.Tab value="eigenvalue">Eigenvalues</Tabs.Tab>
            <Tabs.Tab value="holographic">Holographic</Tabs.Tab>
            <Tabs.Tab value="mcmc">MCMC</Tabs.Tab>
            <Tabs.Tab value="network">Networks</Tabs.Tab>
          </Tabs.List>

          <Tabs.Panel value="geometric" p="md">
            <MultivectorVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="grades" p="md">
            <GradeDecompositionVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="mobius" p="md">
            <MobiusVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="tropical" p="md">
            <TropicalVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="tropical-path" p="md">
            <TropicalShortestPathVisualization />
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

          <Tabs.Panel value="geodesic" p="md">
            <GeodesicVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="topology" p="md">
            <TopologyVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="dynamics" p="md">
            <DynamicsVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="nullclines" p="md">
            <NullclineVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="poincare" p="md">
            <PoincareVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="lyapunov" p="md">
            <LyapunovVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="ergodic" p="md">
            <ErgodicVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="eigenvalue" p="md">
            <EigenvalueVisualization />
          </Tabs.Panel>

          <Tabs.Panel value="holographic" p="md">
            <HolographicVisualization />
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
