# Amari Examples Suite v0.17.0

Welcome to the comprehensive Amari examples collection! This suite demonstrates the power of geometric algebra, computational topology, dynamical systems, and functional analysis across multiple domains.

## What's Included

### Rust Examples

#### Physics Simulations (`rust/physics-simulation/`)
Real-world physics demonstrations using geometric algebra's natural representations:

- **Rigid Body Dynamics**: Singularity-free rotations using rotors
- **Electromagnetic Fields**: Unified E+B field treatment with Maxwell's equations
- **Fluid Dynamics**: Vorticity as bivectors, circulation theorems
- **Quantum Mechanics**: Pauli matrices, spin states, Bell's inequality

#### Computer Graphics (`rust/computer-graphics/`)
Modern graphics applications showcasing GA's geometric intuition:

- **3D Transformations**: Gimbal lock-free rotations and interpolation
- **Camera Systems**: Perspective projection and orbital controls
- **Mesh Operations**: Normal calculations and geometric queries
- **Ray Tracing**: Natural ray representation and lighting

#### Machine Learning (`rust/machine-learning/`)
Verified ML algorithms using dual number automatic differentiation:

- **Automatic Differentiation**: Exact gradients without approximation errors
- **Neural Networks**: Verified backpropagation and training
- **Optimization**: Gradient descent, Adam, Newton's method
- **Verified Learning**: Mathematical guarantees and error analysis

#### Dynamical Systems (`rust/dynamical-systems/`) *NEW in v0.17.0*
Comprehensive dynamical systems analysis with `amari-dynamics`:

- **Lorenz Attractor**: Chaotic dynamics, butterfly effect, strange attractors
- **Van der Pol Oscillator**: Limit cycles, relaxation oscillations
- **Bifurcation Analysis**: Period-doubling, Feigenbaum constants, Hopf bifurcations
- **Lyapunov Exponents**: Chaos quantification, Kaplan-Yorke dimension
- **Stability Analysis**: Fixed points, Jacobians, eigenvalue classification
- **Phase Portraits**: Vector fields, nullclines, separatrices
- **Stochastic Dynamics**: Langevin equations, Fokker-Planck, noise-induced transitions

#### Computational Topology (`rust/topology/`) *NEW in v0.17.0*
Algebraic and computational topology with `amari-topology`:

- **Simplicial Complexes**: Building blocks, chain complexes, Euler characteristic
- **Persistent Homology**: Filtrations, persistence diagrams, Betti numbers
- **Morse Theory**: Critical points, gradient flow, level set evolution
- **Topological Data Analysis**: Shape detection, clustering, feature extraction

#### Functional Analysis (`rust/functional-analysis/`) *NEW in v0.17.0*
Hilbert and Banach space theory with `amari-functional`:

- **Hilbert Spaces**: Inner products, orthonormal bases, projections, Parseval's identity
- **Operators**: Bounded operators, adjoints, compact operators, spectral properties
- **Spectral Theory**: Eigenvalue problems, functional calculus, perturbation theory
- **Banach Spaces**: Lp spaces, duality, fixed point theorems
- **Distributions**: Generalized functions, Dirac delta, weak derivatives

### TypeScript Examples (`typescript/`)
Node.js examples using `@justinelliottcobb/amari-wasm`:

- Basic geometric algebra operations
- Dual number computations
- Integration patterns

### PureScript Examples (`purescript/`) *NEW in v0.17.0*
Comprehensive functional programming examples with amari-wasm FFI:

- Type-safe WASM bindings
- Monadic computations with GA
- Effect system integration
- Property-based testing

### Web Demos (`web/interactive-demos/`)
Browser-based interactive visualizations:

- 3D Rotor Manipulator
- Electromagnetic Field Visualizer
- AutoDiff Grapher
- Optimization Tracer

## Quick Start

### Prerequisites
- Rust 1.75+ with Cargo
- Node.js 18+ (for web/TypeScript demos)
- PureScript 0.15+ (for PureScript examples)

### Running Rust Examples

```bash
# Navigate to the examples directory
cd examples

# Dynamical Systems
cd rust/dynamical-systems
cargo run --bin lorenz_attractor
cargo run --bin van_der_pol
cargo run --bin bifurcation_analysis
cargo run --bin lyapunov_exponents
cargo run --bin stability_analysis
cargo run --bin phase_portraits
cargo run --bin stochastic_dynamics

# Computational Topology
cd ../topology
cargo run --bin simplicial_complexes
cargo run --bin persistent_homology
cargo run --bin morse_theory
cargo run --bin topological_data_analysis

# Functional Analysis
cd ../functional-analysis
cargo run --bin hilbert_spaces
cargo run --bin operators
cargo run --bin spectral_theory
cargo run --bin banach_spaces
cargo run --bin distributions

# Physics Simulations
cd ../physics-simulation
cargo run --bin rigid_body_dynamics
cargo run --bin electromagnetic_fields
cargo run --bin fluid_dynamics
cargo run --bin quantum_mechanics

# Computer Graphics
cd ../computer-graphics
cargo run --bin transformations_3d
cargo run --bin camera_projection
cargo run --bin mesh_operations
cargo run --bin ray_tracing

# Machine Learning
cd ../machine-learning
cargo run --bin automatic_differentiation
cargo run --bin neural_networks
cargo run --bin optimization_algorithms
cargo run --bin verified_learning
```

### Running TypeScript Examples

```bash
cd examples/typescript
npm install
npx ts-node src/basic_operations.ts
npx ts-node src/dual_numbers.ts
```

### Running Web Demos

```bash
cd examples/web/interactive-demos
npm install
npm run dev
# Open http://localhost:5173
```

### Running PureScript Examples

```bash
cd examples/purescript
spago build
spago run
```

## Educational Pathways

We've designed structured learning paths for different backgrounds:

### Beginner Track
Start with core GA concepts and build fundamental understanding:
1. Basic multivector operations
2. Dual number automatic differentiation
3. Simple physics simulations

### Intermediate Track
Apply advanced techniques to real problems:
1. Complex transformations and interpolation
2. Machine learning with exact gradients
3. Dynamical systems basics

### Advanced Track
Research-level applications:
1. Persistent homology for data analysis
2. Spectral theory and operators
3. Stochastic dynamics and ergodic theory

### Research Track
Push theoretical boundaries:
1. Novel topological invariants
2. Information geometry applications
3. Formal verification with Creusot

See [LEARNING_PATHS.md](LEARNING_PATHS.md) for detailed curricula.

## Crate Overview

| Crate | Description | Example Directory |
|-------|-------------|-------------------|
| `amari-core` | Geometric algebra fundamentals | All examples |
| `amari-dual` | Automatic differentiation | `machine-learning/` |
| `amari-dynamics` | Dynamical systems | `dynamical-systems/` |
| `amari-topology` | Computational topology | `topology/` |
| `amari-functional` | Functional analysis | `functional-analysis/` |
| `amari-calculus` | Geometric calculus | `physics-simulation/` |
| `amari-gpu` | GPU acceleration | Performance benchmarks |
| `amari-wasm` | WebAssembly bindings | `typescript/`, `web/`, `purescript/` |

## Example Highlights

### Dynamical Systems: Lorenz Attractor
```rust
use amari_dynamics::{
    systems::LorenzSystem,
    solver::{DormandPrince, ODESolver},
    lyapunov::compute_lyapunov_spectrum,
};

let lorenz = LorenzSystem::new(10.0, 28.0, 8.0/3.0);
let solver = DormandPrince::new(1e-8);

// Integrate trajectory
let trajectory = solver.solve(&lorenz, initial_state, 0.0, 100.0, 10000)?;

// Compute Lyapunov exponents
let spectrum = compute_lyapunov_spectrum(&lorenz, 10000, 0.01)?;
println!("Largest Lyapunov exponent: {:.4}", spectrum.exponents[0]);
```

### Topology: Persistent Homology
```rust
use amari_topology::{
    persistence::{Filtration, PersistentHomology},
    tda::VietorisRips,
};

// Build Rips filtration from point cloud
let rips = VietorisRips::new(&point_cloud, 2.0)?;
let ph = PersistentHomology::compute(&rips.filtration())?;

// Analyze persistence diagram
let h1_diagram = ph.diagram(1);  // Loops
for (birth, death) in h1_diagram.pairs() {
    println!("Loop: born at {:.3}, dies at {:.3}", birth, death);
}
```

### Functional Analysis: Spectral Decomposition
```rust
use amari_functional::{
    spectral::SpectralDecomposition,
    operator::MatrixOperator,
    functional_calculus::FunctionalCalculus,
};

let op = MatrixOperator::new(matrix)?;
let decomp = SpectralDecomposition::compute(&op)?;

// Apply function to operator: exp(A)
let fc = FunctionalCalculus::new(&op)?;
let exp_a = fc.apply(|x| x.exp())?;
```

## Technical Details

### Performance Features
- SIMD optimization for geometric products
- GPU acceleration via `amari-gpu` with wgpu
- Parallel computation with Rayon
- WebAssembly compilation for browser deployment

### Formal Verification
- Creusot contracts for mathematical properties
- Type-level guarantees with phantom types
- Comprehensive property-based testing

### Numerical Stability
- Adaptive step-size ODE solvers
- Error-controlled computations
- Condition number monitoring

## Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Comprehensive API reference
- [LEARNING_PATHS.md](LEARNING_PATHS.md) - Structured learning curricula
- [Main README](../README.md) - Project overview

## Contributing

We welcome contributions:
- Bug reports and fixes
- New example programs
- Documentation improvements
- Novel applications

## License

MIT License - see [LICENSE](../LICENSE) for details.

---

**Ready to explore?** Start with [dynamical systems](rust/dynamical-systems/) or dive into [interactive web demos](web/interactive-demos/)!
