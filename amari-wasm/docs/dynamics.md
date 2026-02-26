# Dynamical Systems (amari-calculus)

*Added in v0.18.1*

ODE solvers, stability analysis, bifurcation diagrams, Lyapunov exponents, and phase portraits.

## Quick Start

```typescript
import init, {
  WasmLorenzSystem,
  WasmVanDerPolOscillator,
  WasmDuffingOscillator,
  WasmRungeKutta4,
  WasmBifurcationDiagram,
  WasmPhasePortrait,
  WasmStabilityAnalysis,
  computeLyapunovExponents,
  findFixedPoints
} from '@justinelliottcobb/amari-wasm';

async function dynamicsDemo() {
  await init();

  // Lorenz Attractor (sigma=10, rho=28, beta=8/3)
  const lorenz = WasmLorenzSystem.classic();
  const solver = new WasmRungeKutta4();
  const trajectory = solver.solve(lorenz, [1.0, 1.0, 1.0], 0.0, 50.0, 5000);
  console.log(`Trajectory: ${trajectory.length} points`);

  // Lyapunov Exponents
  const lyapunov = computeLyapunovExponents(lorenz, [1.0, 1.0, 1.0], 10000, 0.01);
  console.log(`Exponents: [${lyapunov.exponents.map(e => e.toFixed(4)).join(', ')}]`);
  if (lyapunov.exponents[0] > 0) console.log('System is chaotic!');
  console.log(`Kaplan-Yorke dimension: ${lyapunov.kaplanYorkeDimension().toFixed(3)}`);

  // Bifurcation Diagram (logistic map)
  const bifurcation = WasmBifurcationDiagram.compute('logistic', 2.5, 4.0, 1000, 500, 100);
  console.log(`${bifurcation.parameterCount()} parameter values`);

  // Stability Analysis
  const vdp = WasmVanDerPolOscillator.new(1.0);
  const fixedPoints = findFixedPoints(vdp, [[0.0, 0.0]], 1e-10);
  for (const fp of fixedPoints) {
    const stability = WasmStabilityAnalysis.analyze(vdp, fp.point);
    console.log(`${stability.stabilityType} at (${fp.point})`);
  }

  // Phase Portrait (Duffing oscillator)
  const duffing = WasmDuffingOscillator.new(1.0, -1.0, 0.2, 0.3, 1.2);
  const portrait = WasmPhasePortrait.generate(duffing, [-2, 2], [-2, 2], 20, 5.0, 0.01);
  console.log(`${portrait.trajectoryCount()} trajectories`);

  // Clean up
  lorenz.free(); vdp.free(); duffing.free(); solver.free();
  bifurcation.free(); portrait.free();
}

dynamicsDemo();
```

## API Reference

### Systems

**WasmLorenzSystem:**
- `classic()`: Create with sigma=10, rho=28, beta=8/3
- `new(sigma, rho, beta)`: Create with custom parameters
- `vectorField(state)`: Evaluate dx/dt at state

**WasmVanDerPolOscillator:**
- `new(mu)`: Create with damping parameter mu
- `vectorField(state)`: Evaluate dx/dt at state

**WasmDuffingOscillator:**
- `new(alpha, beta, delta, gamma, omega)`: Create driven Duffing oscillator
- `vectorField(state, t)`: Evaluate dx/dt at state and time t

**WasmRosslerSystem:**
- `new(a, b, c)`: Create Rossler attractor
- `classic()`: Create with a=0.2, b=0.2, c=5.7

**WasmHenonMap:**
- `new(a, b)`: Create Henon map
- `classic()`: Create with a=1.4, b=0.3
- `iterate(state)`: Apply one map iteration

### Solvers

**WasmRungeKutta4:**
- `new()`: Create RK4 solver
- `solve(system, initial, t0, t1, steps)`: Integrate trajectory
- `step(system, state, t, dt)`: Single integration step

**WasmAdaptiveSolver:**
- `rkf45()`: Create RKF45 adaptive solver
- `dormandPrince()`: Create Dormand-Prince solver
- `solve(system, initial, t0, t1, tolerance)`: Adaptive integration

### Analysis

**Lyapunov Functions:**
- `computeLyapunovExponents(system, initial, steps, dt)`: Compute spectrum
- Returns: `{ exponents, sum(), kaplanYorkeDimension(), isChaotic() }`

**WasmBifurcationDiagram:**
- `compute(systemType, paramMin, paramMax, numParams, transient, samples)`: Generate diagram
- `parameterCount()`: Number of parameter values
- `attractorPoints(param)`: Get attractor at specific parameter
- `branches()`: Get all (parameter, points) pairs

**WasmStabilityAnalysis:**
- `analyze(system, point)`: Analyze stability at point
- `stabilityType`: 'stable_node', 'stable_spiral', 'unstable_node', 'unstable_spiral', 'saddle', 'center'
- `eigenvalues`: Array of {real, imag} pairs
- `isStable()`: True if asymptotically stable

**findFixedPoints:**
- `findFixedPoints(system, initialGuesses, tolerance)`: Find via Newton's method
- Returns array of `{ point, converged, iterations }`

**WasmPhasePortrait:**
- `generate(system, xRange, yRange, resolution, tMax, dt)`: Generate portrait
- `trajectoryCount()`: Number of trajectories
- `trajectories()`: Get all trajectory arrays
- `nullclines()`: Get {x, y} nullcline point arrays

## Use Cases

- **Chaos Theory**: Lorenz attractors, bifurcation diagrams, Lyapunov exponents
- **Control Systems**: Stability analysis and phase portraits
- **Climate Modeling**: Sensitivity analysis via Lyapunov spectrum computation
