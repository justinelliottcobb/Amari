import { Container, Stack, Card, Title, Text, SimpleGrid } from "@mantine/core";
import { ExampleCard } from "../components/ExampleCard";

export function Dynamics() {
  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  // RK4 step for 2D systems
  const rk4Step = (
    f: (x: number, y: number) => [number, number],
    x: number,
    y: number,
    dt: number
  ): [number, number] => {
    const [k1x, k1y] = f(x, y);
    const [k2x, k2y] = f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
    const [k3x, k3y] = f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
    const [k4x, k4y] = f(x + dt * k3x, y + dt * k3y);
    return [
      x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x),
      y + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y)
    ];
  };

  // RK4 step for 3D systems
  const rk4Step3D = (
    f: (x: number, y: number, z: number) => [number, number, number],
    x: number,
    y: number,
    z: number,
    dt: number
  ): [number, number, number] => {
    const [k1x, k1y, k1z] = f(x, y, z);
    const [k2x, k2y, k2z] = f(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z);
    const [k3x, k3y, k3z] = f(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z);
    const [k4x, k4y, k4z] = f(x + dt * k3x, y + dt * k3y, z + dt * k3z);
    return [
      x + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x),
      y + (dt / 6) * (k1y + 2 * k2y + 2 * k3y + k4y),
      z + (dt / 6) * (k1z + 2 * k2z + 2 * k3z + k4z)
    ];
  };

  const examples = [
    {
      title: "Lorenz Attractor",
      description: "Classic chaotic system demonstrating sensitivity to initial conditions",
      category: "Chaotic Systems",
      code: `// Lorenz system: a model of atmospheric convection
import { WasmLorenz, WasmTrajectory } from '@justinelliottcobb/amari-wasm';

// Create Lorenz system with classic parameters
// sigma = 10, rho = 28, beta = 8/3
const lorenz = WasmLorenz.classic();

// Compute trajectory from initial condition
const trajectory = lorenz.trajectory(
  1.0, 1.0, 1.0,  // initial (x, y, z)
  0.01,           // time step dt
  5000            // number of steps
);

// Demonstrate butterfly effect
const epsilon = 1e-10;
const traj1 = lorenz.trajectory(1.0, 1.0, 1.0, 0.01, 5000);
const traj2 = lorenz.trajectory(1.0 + epsilon, 1.0, 1.0, 0.01, 5000);

// After 50 time units, the difference grows exponentially`,
      onRun: simulateExample(() => {
        // Lorenz system parameters
        const sigma = 10, rho = 28, beta = 8/3;

        const lorenzField = (x: number, y: number, z: number): [number, number, number] => [
          sigma * (y - x),
          x * (rho - z) - y,
          x * y - beta * z
        ];

        // Simulate trajectory
        let [x, y, z] = [1.0, 1.0, 1.0];
        const dt = 0.01;
        const steps = 5000;

        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;
        let zMin = Infinity, zMax = -Infinity;

        // Skip transient
        for (let i = 0; i < 1000; i++) {
          [x, y, z] = rk4Step3D(lorenzField, x, y, z, dt);
        }

        for (let i = 0; i < steps - 1000; i++) {
          [x, y, z] = rk4Step3D(lorenzField, x, y, z, dt);
          xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
          yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
          zMin = Math.min(zMin, z); zMax = Math.max(zMax, z);
        }

        // Butterfly effect demonstration
        let [x1, y1, z1] = [1.0, 1.0, 1.0];
        let [x2, y2, z2] = [1.0 + 1e-10, 1.0, 1.0];

        const separations: string[] = [];
        for (let i = 0; i <= 5000; i++) {
          if (i % 1000 === 0) {
            const dist = Math.sqrt(
              (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
            );
            separations.push(`  t = ${(i * dt).toFixed(0)}s: separation = ${dist.toExponential(2)}`);
          }
          [x1, y1, z1] = rk4Step3D(lorenzField, x1, y1, z1, dt);
          [x2, y2, z2] = rk4Step3D(lorenzField, x2, y2, z2, dt);
        }

        return [
          "Lorenz Attractor (sigma=10, rho=28, beta=8/3)",
          "",
          "Strange Attractor Bounds (after transient):",
          `  x: [${xMin.toFixed(1)}, ${xMax.toFixed(1)}]`,
          `  y: [${yMin.toFixed(1)}, ${yMax.toFixed(1)}]`,
          `  z: [${zMin.toFixed(1)}, ${zMax.toFixed(1)}]`,
          "",
          "Butterfly Effect (perturbation = 1e-10):",
          ...separations,
          "",
          "The separation grows exponentially - this is chaos!"
        ].join('\n');
      })
    },
    {
      title: "Van der Pol Oscillator",
      description: "Self-sustained oscillations with a stable limit cycle",
      category: "Oscillators",
      code: `// Van der Pol oscillator: models relaxation oscillations
import { WasmVanDerPol } from '@justinelliottcobb/amari-wasm';

// Create oscillator with damping parameter mu
const vdp = new WasmVanDerPol(1.0);

// Equations: dx/dt = y, dy/dt = mu(1-x^2)y - x
// The origin is an unstable spiral (for mu > 0)
// All trajectories converge to a stable limit cycle

// From small initial condition
const traj = vdp.trajectory(0.1, 0.0, 0.01, 5000);`,
      onRun: simulateExample(() => {
        const mu = 1.0;

        const vdpField = (x: number, y: number): [number, number] => [
          y,
          mu * (1 - x * x) * y - x
        ];

        // Trajectories from inside and outside
        const testCases = [
          { x0: 0.1, y0: 0.0, label: "inside (small)" },
          { x0: 0.5, y0: 0.0, label: "inside (medium)" },
          { x0: 3.0, y0: 0.0, label: "outside" },
        ];

        const results: string[] = [];
        const dt = 0.01;

        for (const { x0, y0, label } of testCases) {
          let [x, y] = [x0, y0];

          // Run to convergence
          for (let i = 0; i < 5000; i++) {
            [x, y] = rk4Step(vdpField, x, y, dt);
          }

          // Measure limit cycle amplitude
          let maxX = 0;
          for (let i = 0; i < 1000; i++) {
            [x, y] = rk4Step(vdpField, x, y, dt);
            maxX = Math.max(maxX, Math.abs(x));
          }

          results.push(`  (${x0}, ${y0}) [${label}] -> amplitude = ${maxX.toFixed(3)}`);
        }

        // Jacobian analysis at origin
        const trace = mu;  // df2/dy at origin = mu
        const det = 1.0;    // df1/dy * df2/dx - df1/dx * df2/dy = 1
        const disc = trace * trace - 4 * det;
        const rePart = trace / 2;
        const imPart = Math.sqrt(-disc) / 2;

        return [
          `Van der Pol Oscillator (mu = ${mu})`,
          "",
          "Equations: dx/dt = y, dy/dt = mu(1-x^2)y - x",
          "",
          "Stability at Origin:",
          `  Trace = ${trace}, Det = ${det}`,
          `  Eigenvalues: ${rePart.toFixed(2)} +/- ${imPart.toFixed(2)}i`,
          `  Classification: UNSTABLE SPIRAL (Re > 0)`,
          "",
          "Limit Cycle Convergence:",
          ...results,
          "",
          "All trajectories converge to amplitude ~ 2.0"
        ].join('\n');
      })
    },
    {
      title: "Duffing Oscillator (Double-Well)",
      description: "Bistable system with two attracting fixed points",
      category: "Oscillators",
      code: `// Duffing oscillator with double-well potential
import { WasmDuffing } from '@justinelliottcobb/amari-wasm';

// Equations: dx/dt = y, dy/dt = -delta*y + x - x^3
// Potential: V(x) = -x^2/2 + x^4/4
// Fixed points: x = -1 (stable), x = 0 (saddle), x = +1 (stable)

const duffing = new WasmDuffing(0.05, -1.0, 1.0);

// Initial conditions determine which well the trajectory falls into
const traj1 = duffing.trajectory(-0.5, 0.0, 0.01, 10000);  // -> left well
const traj2 = duffing.trajectory(0.5, 0.0, 0.01, 10000);   // -> right well`,
      onRun: simulateExample(() => {
        const delta = 0.05;

        const duffingField = (x: number, y: number): [number, number] => [
          y,
          -delta * y + x - x * x * x
        ];

        // Test basin of attraction
        const testCases = [
          { x0: -2.0, y0: 0.0 },
          { x0: -0.3, y0: 0.0 },
          { x0: 0.3, y0: 0.0 },
          { x0: 2.0, y0: 0.0 },
        ];

        const results: string[] = [];
        const dt = 0.01;

        for (const { x0, y0 } of testCases) {
          let [x, y] = [x0, y0];

          for (let i = 0; i < 20000; i++) {
            [x, y] = rk4Step(duffingField, x, y, dt);
          }

          const well = x < 0 ? "LEFT" : "RIGHT";
          results.push(`  (${x0 > 0 ? '+' : ''}${x0.toFixed(1)}, ${y0.toFixed(1)}) -> x_final = ${x > 0 ? '+' : ''}${x.toFixed(3)} [${well} well]`);
        }

        return [
          `Duffing Oscillator (delta = ${delta})`,
          "",
          "Equations: dx/dt = y, dy/dt = -delta*y + x - x^3",
          "Potential: V(x) = -x^2/2 + x^4/4",
          "",
          "Fixed Points:",
          "  x = -1 (STABLE SPIRAL) - left well minimum",
          "  x =  0 (SADDLE) - potential maximum",
          "  x = +1 (STABLE SPIRAL) - right well minimum",
          "",
          "Basin of Attraction Test:",
          ...results,
          "",
          "The separatrix passes through x = 0"
        ].join('\n');
      })
    },
    {
      title: "Simple Pendulum",
      description: "Phase portrait showing oscillations and rotations",
      category: "Oscillators",
      code: `// Simple pendulum: d^2(theta)/dt^2 = -(g/L)sin(theta)
import { WasmPendulum } from '@justinelliottcobb/amari-wasm';

// As a 2D system: dtheta/dt = omega, domega/dt = -(g/L)sin(theta)
const pendulum = new WasmPendulum(1.0, 9.8, 0.0);

// Different energy levels show different behaviors:
// - Low energy: oscillations (librations)
// - Critical energy: separatrix
// - High energy: rotations (whirling)

const traj = pendulum.trajectory(0.5, 0.0, 0.01, 2000);`,
      onRun: simulateExample(() => {
        const g_over_L = 9.8;

        const pendulumField = (theta: number, omega: number): [number, number] => [
          omega,
          -g_over_L * Math.sin(theta)
        ];

        // Different initial conditions
        const testCases = [
          { theta0: 0.5, omega0: 0.0, desc: "small oscillation" },
          { theta0: 1.5, omega0: 0.0, desc: "medium oscillation" },
          { theta0: Math.PI - 0.1, omega0: 0.0, desc: "near separatrix" },
          { theta0: 0.0, omega0: 4.0, desc: "rotation (whirling)" },
        ];

        const results: string[] = [];
        const dt = 0.01;

        for (const { theta0, omega0, desc } of testCases) {
          let [theta, omega] = [theta0, omega0];

          // Energy = 0.5*omega^2 - g_over_L*cos(theta)
          const energy = 0.5 * omega0 * omega0 - g_over_L * Math.cos(theta0);
          const criticalEnergy = g_over_L;  // E at separatrix

          // Simulate and track max theta
          let maxTheta = 0;
          for (let i = 0; i < 2000; i++) {
            [theta, omega] = rk4Step(pendulumField, theta, omega, dt);
            maxTheta = Math.max(maxTheta, Math.abs(theta));
          }

          const motionType = maxTheta > Math.PI ? "ROTATING" : "OSCILLATING";

          results.push(`  (theta=${theta0.toFixed(1)}, omega=${omega0.toFixed(1)}) E=${energy.toFixed(2)} -> ${motionType} [${desc}]`);
        }

        return [
          "Simple Pendulum (g/L = 9.8)",
          "",
          "Equations: dtheta/dt = omega, domega/dt = -(g/L)sin(theta)",
          `Separatrix energy: E_crit = g/L = ${g_over_L.toFixed(2)}`,
          "",
          "Phase Portrait Regions:",
          "  E < E_crit: Oscillations (closed orbits)",
          "  E = E_crit: Separatrix (homoclinic orbit)",
          "  E > E_crit: Rotations (unbounded theta)",
          "",
          "Trajectory Analysis:",
          ...results
        ].join('\n');
      })
    },
    {
      title: "Logistic Map Bifurcation",
      description: "Period-doubling route to chaos as parameter increases",
      category: "Bifurcation",
      code: `// Logistic map: x_{n+1} = r * x_n * (1 - x_n)
// Shows period-doubling cascade to chaos as r increases

// Period-1 for r < 3
// Period-2 for r ~ 3.0 - 3.449
// Period-4 for r ~ 3.449 - 3.54
// Chaos for r > 3.57 (approximately)
// Period-3 window at r ~ 3.83

for (const r of [2.5, 3.0, 3.449, 3.54, 3.8, 4.0]) {
  let x = 0.5;
  // Iterate to attractor
  for (let i = 0; i < 1000; i++) {
    x = r * x * (1 - x);
  }
  // Collect attractor values
  const attractor = new Set();
  for (let i = 0; i < 256; i++) {
    x = r * x * (1 - x);
    attractor.add(x.toFixed(6));
  }
  console.log(\`r=\${r}: period-\${attractor.size}\`);
}`,
      onRun: simulateExample(() => {
        const rValues = [2.5, 3.0, 3.2, 3.449, 3.54, 3.57, 3.83, 4.0];
        const results: string[] = [];

        for (const r of rValues) {
          let x = 0.5;

          // Transient removal
          for (let i = 0; i < 500; i++) {
            x = r * x * (1 - x);
          }

          // Collect attractor points
          const attractor: number[] = [];
          for (let i = 0; i < 256; i++) {
            x = r * x * (1 - x);
            const isNew = !attractor.some(v => Math.abs(v - x) < 1e-6);
            if (isNew && attractor.length < 16) {
              attractor.push(x);
            }
          }
          attractor.sort((a, b) => a - b);

          let behavior: string;
          if (attractor.length === 1) {
            behavior = "Period-1 (fixed point)";
          } else if (attractor.length === 2) {
            behavior = "Period-2";
          } else if (attractor.length === 3) {
            behavior = "Period-3 (window)";
          } else if (attractor.length === 4) {
            behavior = "Period-4";
          } else if (attractor.length <= 8) {
            behavior = `Period-${attractor.length}`;
          } else {
            behavior = "CHAOS";
          }

          const vals = attractor.slice(0, 3).map(v => v.toFixed(3)).join(", ");
          const suffix = attractor.length > 3 ? "..." : "";

          results.push(`  r=${r.toFixed(3)}: ${behavior.padEnd(20)} [${vals}${suffix}]`);
        }

        return [
          "Logistic Map: x_{n+1} = r * x_n * (1 - x_n)",
          "",
          "Period-Doubling Cascade:",
          ...results,
          "",
          "Key Transitions:",
          "  r = 3.0: First period-doubling",
          "  r ~ 3.449: Period-2 to Period-4",
          "  r ~ 3.570: Onset of chaos (Feigenbaum point)",
          "  r ~ 3.83: Period-3 window within chaos"
        ].join('\n');
      })
    },
    {
      title: "Lyapunov Exponent",
      description: "Quantifying chaos through exponential separation rates",
      category: "Chaos Theory",
      code: `// Lyapunov exponent measures the rate of separation of nearby trajectories
// For the logistic map: lambda = lim (1/n) sum log|df/dx|
//                              = lim (1/n) sum log|r(1-2x)|

// For the logistic map at r = 4 (fully chaotic):
// lambda = log(2) ~ 0.693 (positive = chaotic)

// For r = 3.5 (period-4):
// lambda < 0 (negative = periodic)

function lyapunovLogistic(r: number, iterations: number): number {
  let x = 0.5;
  let lyapSum = 0;

  // Skip transient
  for (let i = 0; i < 1000; i++) {
    x = r * x * (1 - x);
  }

  // Accumulate log|derivative|
  for (let i = 0; i < iterations; i++) {
    const derivative = Math.abs(r * (1 - 2 * x));
    if (derivative > 0) {
      lyapSum += Math.log(derivative);
    }
    x = r * x * (1 - x);
  }

  return lyapSum / iterations;
}`,
      onRun: simulateExample(() => {
        const lyapunovLogistic = (r: number): number => {
          let x = 0.5;
          let lyapSum = 0;

          // Skip transient
          for (let i = 0; i < 1000; i++) {
            x = r * x * (1 - x);
          }

          // Accumulate
          const n = 5000;
          for (let i = 0; i < n; i++) {
            const derivative = Math.abs(r * (1 - 2 * x));
            if (derivative > 0) {
              lyapSum += Math.log(derivative);
            }
            x = r * x * (1 - x);
          }

          return lyapSum / n;
        };

        const rValues = [2.5, 3.0, 3.5, 3.57, 3.83, 4.0];
        const results: string[] = [];

        for (const r of rValues) {
          const lambda = lyapunovLogistic(r);
          const classification = lambda > 0.01 ? "CHAOTIC" : lambda < -0.01 ? "PERIODIC" : "EDGE OF CHAOS";

          results.push(`  r=${r.toFixed(2)}: lambda = ${lambda > 0 ? '+' : ''}${lambda.toFixed(4)} [${classification}]`);
        }

        return [
          "Lyapunov Exponent for Logistic Map",
          "",
          "Definition: lambda = lim (1/n) sum log|df/dx|",
          "  lambda > 0: Chaotic (exponential divergence)",
          "  lambda < 0: Periodic (convergence to attractor)",
          "  lambda = 0: Edge of chaos (critical point)",
          "",
          "Results:",
          ...results,
          "",
          `Theoretical value at r=4: lambda = log(2) = ${Math.log(2).toFixed(4)}`
        ].join('\n');
      })
    },
    {
      title: "Stability Classification",
      description: "Eigenvalue analysis for fixed point stability",
      category: "Stability Theory",
      code: `// For a 2D system dx/dt = f(x), stability at fixed point x* is determined
// by the Jacobian eigenvalues

// Jacobian: J = [df1/dx  df1/dy]
//              [df2/dx  df2/dy]

// Classification based on trace and determinant:
// tau = tr(J), delta = det(J)

// Stable node: tau < 0, delta > 0, tau^2 > 4*delta
// Stable spiral: tau < 0, delta > 0, tau^2 < 4*delta
// Unstable node: tau > 0, delta > 0, tau^2 > 4*delta
// Unstable spiral: tau > 0, delta > 0, tau^2 < 4*delta
// Saddle: delta < 0
// Center: tau = 0, delta > 0`,
      onRun: simulateExample(() => {
        const classify = (tau: number, delta: number): string => {
          const disc = tau * tau - 4 * delta;

          if (delta < 0) return "SADDLE";
          if (Math.abs(tau) < 0.001) {
            return disc < 0 ? "CENTER" : "DEGENERATE";
          }
          if (tau < 0) {
            return disc < 0 ? "STABLE SPIRAL" : "STABLE NODE";
          } else {
            return disc < 0 ? "UNSTABLE SPIRAL" : "UNSTABLE NODE";
          }
        };

        const examples: [number, number, string][] = [
          [-2, 1, "Van der Pol (mu < 0)"],
          [1, 1, "Van der Pol (mu > 0)"],
          [0, 1, "Harmonic oscillator"],
          [-0.05, 2, "Duffing at wells"],
          [-0.05, -1, "Duffing at origin"],
          [-22.67, -306, "Lorenz origin approx"],
        ];

        const results: string[] = [];
        for (const [tau, delta, name] of examples) {
          const disc = tau * tau - 4 * delta;
          const classification = classify(tau, delta);

          let eigenStr: string;
          if (disc >= 0) {
            const lambda1 = (tau + Math.sqrt(disc)) / 2;
            const lambda2 = (tau - Math.sqrt(disc)) / 2;
            eigenStr = `lambda = ${lambda1.toFixed(2)}, ${lambda2.toFixed(2)}`;
          } else {
            const re = tau / 2;
            const im = Math.sqrt(-disc) / 2;
            eigenStr = `lambda = ${re.toFixed(2)} +/- ${im.toFixed(2)}i`;
          }

          results.push(`  ${name}:`);
          results.push(`    tau=${tau}, delta=${delta} -> ${classification}`);
          results.push(`    ${eigenStr}`);
        }

        return [
          "Fixed Point Stability Classification",
          "",
          "Given Jacobian with trace=tau, det=delta:",
          "",
          "Classification Rules:",
          "  delta < 0          -> SADDLE",
          "  tau < 0, disc < 0  -> STABLE SPIRAL",
          "  tau < 0, disc > 0  -> STABLE NODE",
          "  tau > 0, disc < 0  -> UNSTABLE SPIRAL",
          "  tau > 0, disc > 0  -> UNSTABLE NODE",
          "  tau = 0, delta > 0 -> CENTER",
          "",
          "Examples:",
          ...results
        ].join('\n');
      })
    },
    {
      title: "Rossler Attractor",
      description: "Simpler chaotic system with a single scroll",
      category: "Chaotic Systems",
      code: `// Rossler system: simpler than Lorenz, single-scroll attractor
import { WasmRossler } from '@justinelliottcobb/amari-wasm';

// dx/dt = -y - z
// dy/dt = x + a*y
// dz/dt = b + z*(x - c)

// Standard parameters: a=0.2, b=0.2, c=5.7
const rossler = new WasmRossler(0.2, 0.2, 5.7);

const trajectory = rossler.trajectory(1.0, 1.0, 1.0, 0.01, 10000);`,
      onRun: simulateExample(() => {
        const a = 0.2, b = 0.2, c = 5.7;

        const rosslerField = (x: number, y: number, z: number): [number, number, number] => [
          -y - z,
          x + a * y,
          b + z * (x - c)
        ];

        let [x, y, z] = [1.0, 1.0, 1.0];
        const dt = 0.01;

        // Skip transient
        for (let i = 0; i < 5000; i++) {
          [x, y, z] = rk4Step3D(rosslerField, x, y, z, dt);
        }

        // Collect statistics
        let xMin = Infinity, xMax = -Infinity;
        let yMin = Infinity, yMax = -Infinity;
        let zMin = Infinity, zMax = -Infinity;

        for (let i = 0; i < 10000; i++) {
          [x, y, z] = rk4Step3D(rosslerField, x, y, z, dt);
          xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
          yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
          zMin = Math.min(zMin, z); zMax = Math.max(zMax, z);
        }

        return [
          `Rossler Attractor (a=${a}, b=${b}, c=${c})`,
          "",
          "Equations:",
          "  dx/dt = -y - z",
          "  dy/dt = x + a*y",
          "  dz/dt = b + z*(x - c)",
          "",
          "Attractor Bounds (after transient):",
          `  x: [${xMin.toFixed(1)}, ${xMax.toFixed(1)}]`,
          `  y: [${yMin.toFixed(1)}, ${yMax.toFixed(1)}]`,
          `  z: [${zMin.toFixed(1)}, ${zMax.toFixed(1)}]`,
          "",
          "The Rossler attractor has a single-scroll geometry,",
          "simpler than the Lorenz butterfly but still chaotic.",
          "",
          "Key features:",
          "  - Period-doubling route to chaos (as c increases)",
          "  - Single unstable fixed point",
          "  - Lyapunov exponent ~ 0.07 for these parameters"
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="xl" py="xl">
      <Stack gap="xl">
        <div>
          <Title order={1} mb="md">Dynamical Systems</Title>
          <Text c="dimmed" maw={800}>
            Explore the rich behavior of nonlinear dynamical systems: chaos, stability,
            bifurcations, and attractors. These examples demonstrate the amari-dynamics
            crate capabilities for analyzing continuous and discrete dynamical systems.
          </Text>
        </div>

        <Card withBorder p="lg">
          <Title order={3} mb="sm">Mathematical Framework</Title>
          <SimpleGrid cols={{ base: 1, md: 3 }} spacing="lg">
            <div>
              <Text fw={500} mb="xs">Continuous Systems</Text>
              <Text size="sm" c="dimmed">
                ODEs of the form dx/dt = f(x) integrated using Runge-Kutta methods.
                Analysis includes fixed points, stability, and bifurcations.
              </Text>
            </div>
            <div>
              <Text fw={500} mb="xs">Discrete Maps</Text>
              <Text size="sm" c="dimmed">
                Iterated maps x_{"{n+1}"} = f(x_n) exhibiting period-doubling cascades
                and transition to chaos.
              </Text>
            </div>
            <div>
              <Text fw={500} mb="xs">Chaos Theory</Text>
              <Text size="sm" c="dimmed">
                Strange attractors, Lyapunov exponents, and sensitive dependence on
                initial conditions.
              </Text>
            </div>
          </SimpleGrid>
        </Card>

        <SimpleGrid cols={{ base: 1, lg: 2 }} spacing="lg">
          {examples.map((example, i) => (
            <ExampleCard
              key={i}
              title={example.title}
              description={example.description}
              code={example.code}
              onRun={example.onRun}
            />
          ))}
        </SimpleGrid>
      </Stack>
    </Container>
  );
}
