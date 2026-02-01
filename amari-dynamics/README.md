# amari-dynamics

Dynamical systems analysis on geometric algebra spaces - ODE solvers, stability theory, bifurcations, chaos, and Lyapunov exponents.

## Overview

`amari-dynamics` provides comprehensive tools for analyzing dynamical systems defined on Clifford algebras Cl(P,Q,R). The crate implements ODE solvers (RK4, RKF45, Dormand-Prince), stability analysis via Jacobian eigenvalues, bifurcation detection and continuation, Lyapunov spectrum computation, and phase space analysis.

## Features

### Core Dynamics
- **ODE Solvers**: Runge-Kutta 4, adaptive RKF45, Dormand-Prince, Backward Euler for stiff systems
- **Flow Traits**: Unified interface for continuous and discrete dynamical systems
- **Trajectory Analysis**: Time series with metadata, Poincare sections, recurrence

### Stability Analysis
- **Fixed Point Detection**: Newton's method with damping and line search
- **Linearization**: Numerical Jacobian computation with configurable differentiation
- **Eigenvalue Classification**: Stable/unstable nodes, spirals, saddles, centers
- **Lyapunov Exponents**: QR-based spectrum computation, chaos detection

### Bifurcation Theory
- **Bifurcation Detection**: Saddle-node, transcritical, pitchfork, Hopf
- **Parameter Continuation**: Track fixed points and limit cycles across parameter space
- **Diagram Generation**: Automated bifurcation diagram construction

### Attractors & Phase Space
- **Attractor Classification**: Fixed points, limit cycles, tori, strange attractors
- **Basin of Attraction**: Grid-based basin boundary computation
- **Phase Portraits**: Nullcline computation, vector field visualization
- **Ergodic Measures**: Birkhoff averages, invariant measures

### Built-in Systems
- **Lorenz System**: Classic chaotic attractor (sigma, rho, beta parameters)
- **Van der Pol Oscillator**: Self-sustained relaxation oscillations
- **Duffing Oscillator**: Double-well potential, bistability
- **Rossler System**: Simpler chaotic attractor
- **Simple/Double Pendulum**: Oscillations and rotations
- **Henon Map**: Discrete chaotic system

### Type Safety & Verification
- **Phantom Types**: Compile-time markers for time dependence, stability, chaos
- **Creusot Contracts**: Formal verification for solver correctness and invariants

### Performance
- **GPU Acceleration**: WebGPU/wgpu for batch trajectories and bifurcation diagrams
- **WASM Bindings**: Browser-based visualization and interactive exploration
- **Parallel Computation**: Rayon-based parallelism for basin and ensemble computations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-dynamics = "0.18"
```

### Feature Flags

```toml
[dependencies]
# Default features (std)
amari-dynamics = "0.18"

# With stochastic dynamics (Langevin, Fokker-Planck)
amari-dynamics = { version = "0.18", features = ["stochastic"] }

# With GPU acceleration
amari-dynamics = { version = "0.18", features = ["gpu"] }

# With WASM bindings
amari-dynamics = { version = "0.18", features = ["wasm"] }

# All features
amari-dynamics = { version = "0.18", features = ["stochastic", "gpu", "parallel"] }
```

## Quick Start

### Lorenz Attractor

```rust
use amari_core::Multivector;
use amari_dynamics::{LorenzSystem, RungeKutta4, ODESolver};

// Create classic Lorenz system (sigma=10, rho=28, beta=8/3)
let lorenz = LorenzSystem::classic();

// Initial condition
let mut initial: Multivector<3, 0, 0> = Multivector::zero();
initial.set(1, 1.0);  // x
initial.set(2, 1.0);  // y
initial.set(4, 1.0);  // z

// Integrate trajectory
let solver = RungeKutta4::new();
let trajectory = solver.solve(&lorenz, initial, 0.0, 50.0, 5000)?;

// Analyze attractor
for (t, state) in trajectory.iter() {
    let (x, y, z) = (state.get(1), state.get(2), state.get(4));
    println!("t={:.2}: ({:.3}, {:.3}, {:.3})", t, x, y, z);
}
```

### Van der Pol Limit Cycle

```rust
use amari_core::Multivector;
use amari_dynamics::{VanDerPolOscillator, RungeKutta4, ODESolver};

// Create oscillator with damping parameter mu = 1.0
let vdp = VanDerPolOscillator::new(1.0);

let mut initial: Multivector<2, 0, 0> = Multivector::zero();
initial.set(1, 0.1);  // Small initial displacement

let solver = RungeKutta4::new();
let trajectory = solver.solve(&vdp, initial, 0.0, 50.0, 5000)?;

// All trajectories converge to limit cycle with amplitude ~2
let final_state = trajectory.final_state().unwrap();
println!("Final amplitude: {:.3}", final_state.get(1).abs());
```

### Stability Analysis

```rust
use amari_dynamics::{
    VanDerPolOscillator,
    stability::{find_fixed_point, compute_jacobian, FixedPointConfig, DifferentiationConfig},
};

let vdp = VanDerPolOscillator::new(1.0);

// Find fixed point near origin
let mut guess: Multivector<2, 0, 0> = Multivector::zero();
guess.set(1, 0.1);
guess.set(2, 0.1);

let fp_config = FixedPointConfig::default();
let result = find_fixed_point(&vdp, &guess, &fp_config)?;

if result.converged {
    let fp = &result.point;
    println!("Fixed point: ({:.6}, {:.6})", fp.get(1), fp.get(2));

    // Compute Jacobian for stability analysis
    let diff_config = DifferentiationConfig::default();
    let jac = compute_jacobian(&vdp, fp, &diff_config)?;

    let trace = jac[(0, 0)] + jac[(1, 1)];
    let det = jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)];

    println!("Trace: {:.4}, Det: {:.4}", trace, det);
    // For mu > 0: trace > 0, det > 0 -> unstable spiral
}
```

### Lyapunov Exponents

```rust
use amari_dynamics::{
    LorenzSystem,
    lyapunov::{LyapunovConfig, compute_lyapunov_spectrum},
};

let lorenz = LorenzSystem::classic();

let mut initial: Multivector<3, 0, 0> = Multivector::zero();
initial.set(1, 1.0);
initial.set(2, 1.0);
initial.set(4, 1.0);

let config = LyapunovConfig::default();
let spectrum = compute_lyapunov_spectrum(&lorenz, &initial, &config)?;

println!("Lyapunov exponents: {:?}", spectrum.exponents);
println!("Sum: {:.4} (should be negative for dissipative system)", spectrum.sum());

// Positive largest exponent indicates chaos
if spectrum.exponents[0] > 0.0 {
    println!("System is chaotic!");
}
```

### Bifurcation Diagram

```rust
use amari_dynamics::bifurcation::{BifurcationDiagram, ContinuationConfig};

// Create bifurcation diagram for logistic map
let config = ContinuationConfig {
    parameter_range: (2.5, 4.0),
    num_points: 1000,
    transient: 500,
    samples: 100,
    ..Default::default()
};

// Factory function creates system for each parameter value
let diagram = BifurcationDiagram::compute(
    |r| LogisticMap::new(r),
    &config,
)?;

// Diagram shows period-doubling route to chaos
for (r, values) in diagram.branches() {
    println!("r={:.3}: {} attractor points", r, values.len());
}
```

## Examples

The crate includes several runnable examples:

```bash
# Lorenz attractor demonstration
cargo run --example lorenz_attractor

# Phase portraits of 2D systems
cargo run --example phase_portrait

# Bifurcation diagrams
cargo run --example bifurcation_diagram

# Stability analysis
cargo run --example stability_analysis
```

## Mathematical Background

### Stability Classification

For a 2D system dx/dt = f(x) near a fixed point x*, stability is determined by the Jacobian eigenvalues:

| Stability Type | Eigenvalue Condition | Trace/Det Region |
|---------------|---------------------|------------------|
| Stable Node | λ₁, λ₂ < 0, real | τ < 0, Δ > 0, τ² > 4Δ |
| Stable Spiral | Re(λ) < 0, complex | τ < 0, Δ > 0, τ² < 4Δ |
| Unstable Node | λ₁, λ₂ > 0, real | τ > 0, Δ > 0, τ² > 4Δ |
| Unstable Spiral | Re(λ) > 0, complex | τ > 0, Δ > 0, τ² < 4Δ |
| Saddle | λ₁ < 0 < λ₂ | Δ < 0 |
| Center | Re(λ) = 0 | τ = 0, Δ > 0 |

### Lyapunov Exponents

The Lyapunov exponents measure the average rate of separation of nearby trajectories:

```
λᵢ = lim(t→∞) (1/t) ln |δxᵢ(t)| / |δxᵢ(0)|
```

For n-dimensional systems:
- λ₁ > 0: Chaos (exponential divergence)
- Σλᵢ < 0: Dissipative (volume contracting)
- Kaplan-Yorke dimension: D_KY = k + Σᵢ₌₁ᵏ λᵢ / |λₖ₊₁|

### Bifurcations

Common bifurcation types:

| Type | Normal Form | Condition |
|------|-------------|-----------|
| Saddle-Node | ẋ = r + x² | Fixed point creation/annihilation |
| Transcritical | ẋ = rx - x² | Exchange of stability |
| Pitchfork | ẋ = rx - x³ | Symmetry breaking |
| Hopf | ż = (μ + iω)z - z|z|² | Birth of limit cycle |

## GPU Acceleration

For large-scale computations, enable GPU acceleration:

```rust
use amari_dynamics::gpu::{GpuDynamics, BatchTrajectoryConfig};

// Create GPU context
let gpu = GpuDynamics::new().await?;

// Compute 1000 trajectories in parallel
let config = BatchTrajectoryConfig {
    dt: 0.01,
    steps: 1000,
    dim: 3,
    system_type: GpuSystemType::Lorenz,
};

let result = gpu.batch_trajectories(&initial_conditions, &params, &config).await?;
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
