import { useState, useEffect } from "react";
import { Card, Title, Text, Button, Group, Box, Stack, Progress } from "@mantine/core";
import { ErrorBoundary } from "./ErrorBoundary";

interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface RealTimeVisualizationProps {
  title: string;
  description: string;
  type: 'rotor' | 'tropical' | 'dual' | 'fisher';
  isRunning: boolean;
  onToggle: () => void;
}

// Real-time rotor visualization
function RotorVisualization({ isRunning }: { isRunning: boolean }) {
  const [angle, setAngle] = useState(0);
  const [vectors, setVectors] = useState<Point3D[]>([]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setAngle(prev => (prev + 0.05) % (2 * Math.PI));

      // Generate rotating vectors
      const newVectors: Point3D[] = [];
      for (let i = 0; i < 8; i++) {
        const baseAngle = (i / 8) * 2 * Math.PI;
        const rotatedAngle = baseAngle + angle;
        newVectors.push({
          x: Math.cos(rotatedAngle) * 50,
          y: Math.sin(rotatedAngle) * 50,
          z: Math.sin(angle * 2) * 20
        });
      }
      setVectors(newVectors);
    }, 50);

    return () => clearInterval(interval);
  }, [isRunning, angle]);

  return (
    <Box
      style={{
        position: 'relative',
        width: '100%',
        height: '16rem',
        backgroundColor: 'var(--mantine-color-dark-7)',
        borderRadius: 'var(--mantine-radius-sm)',
        overflow: 'hidden'
      }}
    >
      <svg width="100%" height="100%" viewBox="-100 -100 200 200">
        {/* Grid */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--mantine-color-dark-4)" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect x="-100" y="-100" width="200" height="200" fill="url(#grid)" />

        {/* Center point */}
        <circle cx="0" cy="0" r="2" fill="var(--mantine-color-cyan-5)" opacity="0.5" />

        {/* Rotating vectors */}
        {vectors.map((vector, i) => (
          <g key={i}>
            <line
              x1="0"
              y1="0"
              x2={vector.x}
              y2={vector.y}
              stroke="var(--mantine-color-cyan-5)"
              strokeWidth="2"
              opacity={0.7}
            />
            <circle
              cx={vector.x}
              cy={vector.y}
              r="3"
              fill="var(--mantine-color-cyan-5)"
              opacity={0.8}
            />
          </g>
        ))}

        {/* Rotor info */}
        <text x="-90" y="-80" fontSize="12" fill="var(--mantine-color-dimmed)">
          Angle: {(angle * 180 / Math.PI).toFixed(1)}deg
        </text>
      </svg>
    </Box>
  );
}

// Real-time tropical convergence
function TropicalVisualization({ isRunning }: { isRunning: boolean }) {
  const [values, setValues] = useState<number[]>([]);
  const [iteration, setIteration] = useState(0);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setIteration(prev => prev + 1);

      // Simulate tropical algorithm convergence
      setValues(prev => {
        if (prev.length === 0) {
          // Initialize with random values
          return Array.from({ length: 6 }, () => Math.random() * 10 - 5);
        }

        // Tropical evolution: each value moves toward the max
        const maxVal = Math.max(...prev);
        return prev.map(val => {
          const diff = maxVal - val;
          return val + diff * 0.1 + (Math.random() - 0.5) * 0.2;
        });
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isRunning]);

  const maxVal = Math.max(...values, 0);
  const minVal = Math.min(...values, 0);
  const range = maxVal - minVal || 1;

  return (
    <Box
      p="md"
      style={{
        width: '100%',
        height: '16rem',
        backgroundColor: 'var(--mantine-color-dark-7)',
        borderRadius: 'var(--mantine-radius-sm)'
      }}
    >
      <Text size="sm" mb="md">
        Tropical Convergence (Iteration: {iteration})
      </Text>
      <Stack gap="sm">
        {values.map((value, i) => {
          const normalized = ((value - minVal) / range) * 100;
          const isMax = Math.abs(value - maxVal) < 0.1;

          return (
            <Group key={i} gap="sm" align="center">
              <Text size="xs" w={30}>v{i}</Text>
              <Box style={{ flex: 1 }}>
                <Progress
                  value={Math.max(normalized, 5)}
                  color={isMax ? 'cyan' : 'dark.3'}
                  size="lg"
                />
              </Box>
              <Text size="xs" w={50} ta="right">{value.toFixed(2)}</Text>
            </Group>
          );
        })}
      </Stack>
    </Box>
  );
}

// Real-time dual number derivatives
function DualVisualization({ isRunning }: { isRunning: boolean }) {
  const [x, setX] = useState(0);
  const [functionValues, setFunctionValues] = useState<{x: number, fx: number, fpx: number}[]>([]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setX(prev => prev + 0.1);

      // Compute f(x) = sin(x) + 0.5*x^2 and its derivative
      const fx = Math.sin(x) + 0.5 * x * x;
      const fpx = Math.cos(x) + x; // f'(x) = cos(x) + x

      setFunctionValues(prev => {
        const newValues = [...prev, { x, fx, fpx }];
        return newValues.slice(-50); // Keep last 50 points
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isRunning, x]);

  if (functionValues.length === 0) {
    return (
      <Box
        style={{
          width: '100%',
          height: '16rem',
          backgroundColor: 'var(--mantine-color-dark-7)',
          borderRadius: 'var(--mantine-radius-sm)'
        }}
      />
    );
  }

  const maxFx = Math.max(...functionValues.map(v => v.fx));
  const minFx = Math.min(...functionValues.map(v => v.fx));
  const maxFpx = Math.max(...functionValues.map(v => v.fpx));
  const minFpx = Math.min(...functionValues.map(v => v.fpx));

  return (
    <Box
      p="md"
      style={{
        width: '100%',
        height: '16rem',
        backgroundColor: 'var(--mantine-color-dark-7)',
        borderRadius: 'var(--mantine-radius-sm)'
      }}
    >
      <Text size="sm" mb="md">
        Dual Number AD: f(x) = sin(x) + 0.5x^2 (x = {x.toFixed(2)})
      </Text>
      <svg width="100%" height="180" viewBox="0 0 400 180">
        {/* Grid */}
        <defs>
          <pattern id="dual-grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--mantine-color-dark-5)" strokeWidth="0.5"/>
          </pattern>
        </defs>
        <rect x="0" y="0" width="400" height="180" fill="url(#dual-grid)" />

        {/* Function curve */}
        {functionValues.length > 1 && (
          <g>
            <polyline
              points={functionValues.map((v, i) => {
                const x = (i / (functionValues.length - 1)) * 380 + 10;
                const y = 170 - ((v.fx - minFx) / (maxFx - minFx || 1)) * 70;
                return `${x},${y}`;
              }).join(' ')}
              fill="none"
              stroke="var(--mantine-color-cyan-5)"
              strokeWidth="2"
              opacity="0.8"
            />

            {/* Derivative curve */}
            <polyline
              points={functionValues.map((v, i) => {
                const x = (i / (functionValues.length - 1)) * 380 + 10;
                const y = 170 - ((v.fpx - minFpx) / (maxFpx - minFpx || 1)) * 70 - 90;
                return `${x},${y}`;
              }).join(' ')}
              fill="none"
              stroke="var(--mantine-color-yellow-5)"
              strokeWidth="2"
              strokeDasharray="5,5"
              opacity="0.6"
            />
          </g>
        )}

        {/* Labels */}
        <text x="10" y="15" fontSize="10" fill="var(--mantine-color-cyan-5)">
          f(x) = {functionValues[functionValues.length - 1]?.fx.toFixed(3)}
        </text>
        <text x="10" y="95" fontSize="10" fill="var(--mantine-color-yellow-5)">
          f'(x) = {functionValues[functionValues.length - 1]?.fpx.toFixed(3)}
        </text>
      </svg>
    </Box>
  );
}

// Real-time Fisher information evolution
function FisherVisualization({ isRunning }: { isRunning: boolean }) {
  const [probabilities, setProbabilities] = useState<number[]>([0.33, 0.33, 0.34]);
  const [fisherMatrix, setFisherMatrix] = useState<number[][]>([]);

  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      setProbabilities(prev => {
        // Evolve probabilities with random walk
        let newProbs = prev.map(p => Math.max(0.01, p + (Math.random() - 0.5) * 0.02));

        // Normalize to sum to 1
        const sum = newProbs.reduce((a, b) => a + b, 0);
        newProbs = newProbs.map(p => p / sum);

        return newProbs;
      });
    }, 200);

    return () => clearInterval(interval);
  }, [isRunning]);

  useEffect(() => {
    // Compute Fisher information matrix
    const fisher = probabilities.map((p, i) =>
      probabilities.map((q, j) => i === j ? 1 / Math.max(p, 0.001) : 0)
    );
    setFisherMatrix(fisher);
  }, [probabilities]);

  return (
    <Box
      p="md"
      style={{
        width: '100%',
        height: '16rem',
        backgroundColor: 'var(--mantine-color-dark-7)',
        borderRadius: 'var(--mantine-radius-sm)'
      }}
    >
      <Text size="sm" mb="md">
        Fisher Information Matrix Evolution
      </Text>
      <Stack gap="md">
        {/* Probability bars */}
        <Stack gap="xs">
          {probabilities.map((prob, i) => (
            <Group key={i} gap="sm" align="center">
              <Text size="xs" w={30}>p{i}</Text>
              <Box style={{ flex: 1 }}>
                <Progress value={prob * 100} color="cyan" size="md" />
              </Box>
              <Text size="xs" w={50} ta="right">{prob.toFixed(3)}</Text>
            </Group>
          ))}
        </Stack>

        {/* Fisher matrix visualization */}
        <Box>
          <Text size="xs" c="dimmed" mb="xs">Fisher Matrix (diagonal elements):</Text>
          <Group gap="sm" justify="center">
            {fisherMatrix.map((row, i) => (
              <Box
                key={i}
                p="xs"
                bg="dark.6"
                style={{ borderRadius: 'var(--mantine-radius-sm)', textAlign: 'center' }}
              >
                <Text size="xs" ff="monospace">
                  {row[i]?.toFixed(1) || '0'}
                </Text>
              </Box>
            ))}
          </Group>
        </Box>
      </Stack>
    </Box>
  );
}

export function RealTimeVisualization({ title, description, type, isRunning, onToggle }: RealTimeVisualizationProps) {
  const renderVisualization = () => {
    switch (type) {
      case 'rotor':
        return <RotorVisualization isRunning={isRunning} />;
      case 'tropical':
        return <TropicalVisualization isRunning={isRunning} />;
      case 'dual':
        return <DualVisualization isRunning={isRunning} />;
      case 'fisher':
        return <FisherVisualization isRunning={isRunning} />;
      default:
        return (
          <Box
            style={{
              width: '100%',
              height: '16rem',
              backgroundColor: 'var(--mantine-color-dark-7)',
              borderRadius: 'var(--mantine-radius-sm)'
            }}
          />
        );
    }
  };

  return (
    <ErrorBoundary
      fallback={
        <Card
          withBorder
          style={{
            borderColor: 'var(--mantine-color-red-6)',
            backgroundColor: 'rgba(239, 68, 68, 0.05)'
          }}
        >
          <Card.Section withBorder inheritPadding py="sm">
            <Title order={4} c="red">Visualization Error</Title>
            <Text size="sm" c="dimmed">
              Failed to render {title}. The visualization may be too complex or encountered an error.
            </Text>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Box
              style={{
                width: '100%',
                height: '16rem',
                backgroundColor: 'var(--mantine-color-dark-7)',
                borderRadius: 'var(--mantine-radius-sm)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <Text c="dimmed">Visualization unavailable</Text>
            </Box>
          </Card.Section>
        </Card>
      }
    >
      <Card withBorder>
        <Card.Section withBorder inheritPadding py="sm">
          <Group justify="space-between" align="flex-start">
            <Box>
              <Title order={4}>{title}</Title>
              <Text size="sm" c="dimmed">{description}</Text>
            </Box>
            <Button
              onClick={onToggle}
              variant={isRunning ? "filled" : "outline"}
              size="sm"
            >
              {isRunning ? "Pause" : "Start"}
            </Button>
          </Group>
        </Card.Section>
        <Card.Section inheritPadding py="md">
          {renderVisualization()}
        </Card.Section>
      </Card>
    </ErrorBoundary>
  );
}
