import { useState, useEffect, useRef, useCallback } from "react";
import { Card, CardHeader, CardBody, Button } from "jadis";
import { ErrorBoundary } from "./ErrorBoundary";
import { safeExecute, validateNumbers } from "~/utils/safeExecution";

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
    <div className="relative w-full h-64 bg-muted rounded-lg overflow-hidden">
      <svg width="100%" height="100%" viewBox="-100 -100 200 200">
        {/* Grid */}
        <defs>
          <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.3"/>
          </pattern>
        </defs>
        <rect x="-100" y="-100" width="200" height="200" fill="url(#grid)" />

        {/* Center point */}
        <circle cx="0" cy="0" r="2" fill="currentColor" opacity="0.5" />

        {/* Rotating vectors */}
        {vectors.map((vector, i) => (
          <g key={i}>
            <line
              x1="0"
              y1="0"
              x2={vector.x}
              y2={vector.y}
              stroke="currentColor"
              strokeWidth="2"
              opacity={0.7}
            />
            <circle
              cx={vector.x}
              cy={vector.y}
              r="3"
              fill="currentColor"
              opacity={0.8}
            />
          </g>
        ))}

        {/* Rotor info */}
        <text x="-90" y="-80" fontSize="12" fill="currentColor" opacity="0.7">
          Angle: {(angle * 180 / Math.PI).toFixed(1)}°
        </text>
      </svg>
    </div>
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
    <div className="w-full h-64 bg-muted rounded-lg p-4">
      <div className="text-sm mb-4">
        Tropical Convergence (Iteration: {iteration})
      </div>
      <div className="space-y-3">
        {values.map((value, i) => {
          const normalized = ((value - minVal) / range) * 100;
          const isMax = Math.abs(value - maxVal) < 0.1;

          return (
            <div key={i} className="flex items-center space-x-3">
              <span className="text-xs w-8">v{i}</span>
              <div className="flex-1 bg-background rounded-full h-6 overflow-hidden">
                <div
                  className={`h-full transition-all duration-200 flex items-center justify-end pr-2 ${
                    isMax ? 'bg-primary' : 'bg-primary/60'
                  }`}
                  style={{ width: `${Math.max(normalized, 5)}%` }}
                >
                  <span className="text-xs text-white">
                    {value.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
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

  if (functionValues.length === 0) return <div className="w-full h-64 bg-muted rounded-lg" />;

  const maxFx = Math.max(...functionValues.map(v => v.fx));
  const minFx = Math.min(...functionValues.map(v => v.fx));
  const maxFpx = Math.max(...functionValues.map(v => v.fpx));
  const minFpx = Math.min(...functionValues.map(v => v.fpx));

  return (
    <div className="w-full h-64 bg-muted rounded-lg p-4">
      <div className="text-sm mb-4">
        Dual Number AD: f(x) = sin(x) + 0.5x² (x = {x.toFixed(2)})
      </div>
      <svg width="100%" height="180" viewBox="0 0 400 180">
        {/* Grid */}
        <defs>
          <pattern id="dual-grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="currentColor" strokeWidth="0.5" opacity="0.2"/>
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
              stroke="currentColor"
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
              stroke="currentColor"
              strokeWidth="2"
              strokeDasharray="5,5"
              opacity="0.6"
            />
          </g>
        )}

        {/* Labels */}
        <text x="10" y="15" fontSize="10" fill="currentColor" opacity="0.7">
          f(x) = {functionValues[functionValues.length - 1]?.fx.toFixed(3)}
        </text>
        <text x="10" y="95" fontSize="10" fill="currentColor" opacity="0.7">
          f'(x) = {functionValues[functionValues.length - 1]?.fpx.toFixed(3)}
        </text>
      </svg>
    </div>
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
    <div className="w-full h-64 bg-muted rounded-lg p-4">
      <div className="text-sm mb-4">
        Fisher Information Matrix Evolution
      </div>
      <div className="space-y-4">
        {/* Probability bars */}
        <div className="space-y-2">
          {probabilities.map((prob, i) => (
            <div key={i} className="flex items-center space-x-3">
              <span className="text-xs w-8">p{i}</span>
              <div className="flex-1 bg-background rounded-full h-4 overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-200"
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
              <span className="text-xs w-12">{prob.toFixed(3)}</span>
            </div>
          ))}
        </div>

        {/* Fisher matrix visualization */}
        <div className="space-y-1">
          <div className="text-xs opacity-70">Fisher Matrix (diagonal elements):</div>
          <div className="grid grid-cols-3 gap-2">
            {fisherMatrix.map((row, i) => (
              <div key={i} className="text-center">
                <div className="text-xs bg-background rounded p-1">
                  {row[i]?.toFixed(1) || '0'}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
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
        return <div className="w-full h-64 bg-muted rounded-lg" />;
    }
  };

  return (
    <ErrorBoundary
      fallback={
        <Card className="border-destructive bg-destructive/5">
          <CardHeader>
            <h3 className="text-lg font-semibold text-destructive">Visualization Error</h3>
            <p className="text-sm text-muted-foreground">
              Failed to render {title}. The visualization may be too complex or encountered an error.
            </p>
          </CardHeader>
          <CardBody>
            <div className="w-full h-64 bg-muted rounded-lg flex items-center justify-center">
              <p className="text-muted-foreground">Visualization unavailable</p>
            </div>
          </CardBody>
        </Card>
      }
    >
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold">{title}</h3>
              <p className="text-sm text-muted-foreground">{description}</p>
            </div>
            <Button
              onClick={onToggle}
              variant={isRunning ? "default" : "outline"}
              size="sm"
            >
              {isRunning ? "Pause" : "Start"}
            </Button>
          </div>
        </CardHeader>
        <CardBody>
          {renderVisualization()}
        </CardBody>
      </Card>
    </ErrorBoundary>
  );
}