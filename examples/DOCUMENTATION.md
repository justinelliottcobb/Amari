# Comprehensive Example Documentation v0.17.0

This document provides detailed explanations of all examples in the Amari suite, including mathematical foundations, implementation details, and educational insights.

## Table of Contents

1. [Dynamical Systems](#dynamical-systems-examples)
2. [Computational Topology](#computational-topology-examples)
3. [Functional Analysis](#functional-analysis-examples)
4. [Physics Simulations](#physics-simulation-examples)
5. [Computer Graphics](#computer-graphics-examples)
6. [Machine Learning](#machine-learning-examples)
7. [Interactive Demos](#interactive-demo-documentation)
8. [Technical Details](#technical-details)

---

## Dynamical Systems Examples

The `amari-dynamics` crate provides comprehensive tools for analyzing dynamical systems, from basic ODE solvers to chaos quantification.

### Lorenz Attractor

**Mathematical Foundation**: The Lorenz system is a paradigmatic example of deterministic chaos:

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz
```

With classic parameters σ = 10, ρ = 28, β = 8/3, the system exhibits a strange attractor.

#### Key Concepts
- **Strange Attractor**: Fractal structure with sensitive dependence on initial conditions
- **Butterfly Effect**: Exponentially diverging trajectories from nearby initial conditions
- **Lyapunov Exponents**: Positive largest exponent indicates chaos (λ₁ ≈ 0.906)
- **Kaplan-Yorke Dimension**: Fractal dimension D_KY ≈ 2.06

#### Implementation Highlights
```rust
use amari_dynamics::{
    systems::LorenzSystem,
    solver::{DormandPrince, ODESolver},
    lyapunov::compute_lyapunov_spectrum,
};

let lorenz = LorenzSystem::new(10.0, 28.0, 8.0/3.0);
let solver = DormandPrince::with_tolerance(1e-8);

// Compute trajectory
let trajectory = solver.solve(&lorenz, initial, 0.0, 100.0, 10000)?;

// Lyapunov spectrum quantifies chaos
let spectrum = compute_lyapunov_spectrum(&lorenz, 10000, 0.01)?;
```

#### Educational Value
- **Chaos Theory**: Understanding deterministic unpredictability
- **Numerical Methods**: Adaptive step-size importance for chaotic systems
- **Attractor Visualization**: 3D phase space structure

---

### Van der Pol Oscillator

**Mathematical Foundation**: Self-sustaining oscillator with nonlinear damping:

```
d²x/dt² - μ(1 - x²)dx/dt + x = 0
```

#### Key Concepts
- **Limit Cycle**: Stable periodic orbit regardless of initial conditions
- **Relaxation Oscillations**: For large μ, characteristic fast-slow dynamics
- **Hopf Bifurcation**: Transition from fixed point to limit cycle at μ = 0

#### Implementation Highlights
```rust
use amari_dynamics::{
    systems::VanDerPol,
    stability::{find_fixed_points, analyze_stability},
    attractor::detect_limit_cycle,
};

let vdp = VanDerPol::new(mu);
let cycle = detect_limit_cycle(&vdp, 1000)?;
println!("Period: {:.4}, Amplitude: {:.4}", cycle.period, cycle.amplitude);
```

#### Educational Value
- **Nonlinear Dynamics**: Self-sustaining oscillations
- **Stability Analysis**: Linearization near fixed points
- **Historical Significance**: Radio circuit modeling

---

### Bifurcation Analysis

**Mathematical Foundation**: Qualitative changes in dynamics as parameters vary.

#### Key Concepts
- **Saddle-Node Bifurcation**: Fixed points appear/disappear
- **Pitchfork Bifurcation**: Symmetry-breaking
- **Hopf Bifurcation**: Fixed point becomes limit cycle
- **Period-Doubling**: Route to chaos
- **Feigenbaum Constant**: δ ≈ 4.669... universal ratio

#### Implementation Highlights
```rust
use amari_dynamics::bifurcation::{
    BifurcationDiagram, ContinuationMethod, classify_bifurcation,
};

// Logistic map period-doubling cascade
let diagram = BifurcationDiagram::compute_map(
    |x, r| r * x * (1.0 - x),  // Logistic map
    (2.5, 4.0),                 // Parameter range
    1000,                       // Points
)?;
```

---

### Lyapunov Exponents

**Mathematical Foundation**: Lyapunov exponents measure the rate of separation of infinitesimally close trajectories:

```
λ = lim_{t→∞} (1/t) ln(|δx(t)|/|δx(0)|)
```

#### Key Concepts
- **Largest Exponent**: λ₁ > 0 indicates chaos
- **Spectrum**: Complete set {λ₁ ≥ λ₂ ≥ ... ≥ λₙ}
- **Sum Rule**: Σλᵢ = trace of Jacobian (dissipation)
- **Kaplan-Yorke Dimension**: D_KY = k + Σᵢ₌₁ᵏ λᵢ / |λₖ₊₁|

#### Implementation Highlights
```rust
use amari_dynamics::lyapunov::{
    LyapunovSpectrum, compute_spectrum_qr, kaplan_yorke_dimension,
};

let spectrum = compute_spectrum_qr(&system, initial, 10000, 0.01)?;
let d_ky = kaplan_yorke_dimension(&spectrum.exponents);
```

---

### Stability Analysis

**Mathematical Foundation**: Linear stability analysis via eigenvalues of Jacobian.

#### Key Concepts
- **Fixed Points**: Solutions where dx/dt = 0
- **Jacobian Matrix**: J_ij = ∂fᵢ/∂xⱼ
- **Stability Classification**:
  - All Re(λ) < 0: Asymptotically stable
  - Any Re(λ) > 0: Unstable
  - Re(λ) = 0: Center (requires nonlinear analysis)

#### Implementation Highlights
```rust
use amari_dynamics::stability::{
    find_fixed_points, compute_jacobian, classify_stability, StabilityType,
};

let fps = find_fixed_points(&system, initial_guesses)?;
for fp in &fps {
    let jacobian = compute_jacobian(&system, fp)?;
    let stability = classify_stability(&jacobian)?;
    match stability {
        StabilityType::AsymptoticallyStable => println!("Stable node"),
        StabilityType::Saddle => println!("Saddle point"),
        StabilityType::StableSpiral => println!("Stable spiral"),
        // ...
    }
}
```

---

### Phase Portraits

**Mathematical Foundation**: Visualization of vector fields and flow structure.

#### Key Concepts
- **Vector Field**: Arrows showing dx/dt direction
- **Nullclines**: Curves where dxᵢ/dt = 0
- **Separatrices**: Boundaries between basins of attraction
- **Phase Space**: Full state space of the system

---

### Stochastic Dynamics

**Mathematical Foundation**: Dynamics with random perturbations.

#### Key Concepts
- **Langevin Equation**: dx = f(x)dt + σdW
- **Fokker-Planck Equation**: Evolution of probability density
- **Noise-Induced Transitions**: Escape over potential barriers
- **Kramers Rate**: k = (ω_a ω_b)/(2πγ) exp(-ΔU/k_B T)

---

## Computational Topology Examples

The `amari-topology` crate provides tools for algebraic and computational topology.

### Simplicial Complexes

**Mathematical Foundation**: Building blocks of algebraic topology.

#### Key Concepts
- **k-Simplex**: Convex hull of k+1 affinely independent points
- **Simplicial Complex**: Collection closed under taking faces
- **Face**: Subset of a simplex (boundary component)
- **f-vector**: (f₀, f₁, ..., fₙ) counting k-simplices

#### Implementation Highlights
```rust
use amari_topology::{
    simplex::Simplex,
    complex::SimplicialComplex,
    chain::ChainComplex,
};

let triangle = Simplex::triangle(0, 1, 2);
let mut complex = SimplicialComplex::new();
complex.add_simplex(triangle)?;  // Automatically adds faces

println!("f-vector: {:?}", complex.f_vector());
println!("Euler characteristic: {}", complex.euler_characteristic());
```

#### Educational Value
- **Discrete Geometry**: Combinatorial representation of spaces
- **Homology Foundation**: Building blocks for algebraic invariants
- **Computational Efficiency**: Finite representation of continuous spaces

---

### Persistent Homology

**Mathematical Foundation**: Tracking topological features across a filtration.

#### Key Concepts
- **Filtration**: Nested sequence K₀ ⊆ K₁ ⊆ ... ⊆ Kₙ
- **Birth/Death**: When features appear and disappear
- **Persistence**: death - birth (lifetime of feature)
- **Persistence Diagram**: Multiset of (birth, death) pairs
- **Betti Numbers**: βₖ = # of k-dimensional holes

#### Implementation Highlights
```rust
use amari_topology::persistence::{
    Filtration, PersistentHomology, PersistenceDiagram,
};

let mut filtration = Filtration::new();
filtration.add(Simplex::vertex(0), 0.0)?;
filtration.add(Simplex::edge(0, 1), 1.0)?;
// ...

let ph = PersistentHomology::compute(&filtration)?;
let h1_diagram = ph.diagram(1);  // 1-dimensional holes (loops)

for (birth, death) in h1_diagram.pairs() {
    println!("Loop: born {:.2}, dies {:.2}, persistence {:.2}",
             birth, death, death - birth);
}
```

#### Educational Value
- **Topological Data Analysis**: Extract shape from data
- **Noise Robustness**: Long-lived features are significant
- **Multi-Scale Analysis**: Features at different scales

---

### Morse Theory

**Mathematical Foundation**: Relating critical points to topology.

#### Key Concepts
- **Critical Points**: Where ∇f = 0
- **Morse Index**: # of negative eigenvalues of Hessian
- **Morse Inequalities**: mₖ ≥ βₖ
- **Level Sets**: f⁻¹(t) and how they change at critical values
- **Gradient Flow**: Trajectories following -∇f

#### Implementation Highlights
```rust
use amari_topology::morse::{
    find_critical_points_2d, MorseComplex, CriticalPointType,
};

let critical = find_critical_points_2d(
    |x, y| (x*x - 1.0).powi(2) + y*y,  // Double-well
    (-2.0, 2.0),
    (-2.0, 2.0),
    100,
)?;

for cp in &critical {
    match cp.point_type {
        CriticalPointType::Minimum => println!("Min at {:?}", cp.position),
        CriticalPointType::Saddle => println!("Saddle at {:?}", cp.position),
        CriticalPointType::Maximum => println!("Max at {:?}", cp.position),
    }
}
```

---

### Topological Data Analysis

**Mathematical Foundation**: Extracting topological features from data.

#### Key Concepts
- **Point Cloud**: Input data as points in metric space
- **Vietoris-Rips Complex**: Add simplex when all pairwise distances ≤ ε
- **Alpha Complex**: Based on Delaunay triangulation (more efficient)
- **Persistence Landscape**: Vectorization of persistence diagrams
- **Bottleneck Distance**: Stability metric for diagrams

#### Implementation Highlights
```rust
use amari_topology::tda::{
    PointCloud, VietorisRips, AlphaComplex,
    bottleneck_distance, PersistenceLandscape,
};

let cloud = PointCloud::new(points)?;
let rips = VietorisRips::new(&cloud, 2.0)?;
let ph = PersistentHomology::compute(&rips.filtration())?;

// Vectorize for machine learning
let landscape = PersistenceLandscape::from_diagram(&ph.diagram(1), 100)?;
let features = landscape.to_vector(50)?;
```

---

## Functional Analysis Examples

The `amari-functional` crate provides Hilbert and Banach space theory.

### Hilbert Spaces

**Mathematical Foundation**: Complete inner product spaces.

#### Key Concepts
- **Inner Product**: ⟨x, y⟩ satisfying linearity, conjugate symmetry, positive definiteness
- **Norm**: ‖x‖ = √⟨x, x⟩
- **Orthogonality**: ⟨x, y⟩ = 0
- **Orthonormal Basis**: {eₙ} with ⟨eₘ, eₙ⟩ = δₘₙ
- **Parseval's Identity**: ‖f‖² = Σₙ |⟨f, eₙ⟩|²

#### Implementation Highlights
```rust
use amari_functional::{
    hilbert::{RealHilbert, L2Space},
    basis::{FourierBasis, OrthonormalBasis},
};

let l2 = L2Space::new(0.0, 2.0 * PI)?;
let fourier = FourierBasis::new(0.0, 2.0 * PI, 20)?;

// Expand function in Fourier basis
let coeffs: Vec<f64> = (0..20)
    .map(|n| l2.inner_product(&f, &fourier.basis_function(n)?))
    .collect::<Result<_, _>>()?;
```

---

### Operators

**Mathematical Foundation**: Bounded linear maps between spaces.

#### Key Concepts
- **Operator Norm**: ‖T‖ = sup{‖Tx‖ : ‖x‖ = 1}
- **Adjoint**: ⟨Tx, y⟩ = ⟨x, T*y⟩
- **Self-Adjoint**: T = T*
- **Compact Operator**: Maps bounded sets to precompact sets
- **Unitary**: U*U = UU* = I

#### Implementation Highlights
```rust
use amari_functional::operator::{
    MatrixOperator, LinearOperator, AdjointOperator,
};

let op = MatrixOperator::new(matrix)?;
let adjoint = op.adjoint()?;

// Verify ⟨Tx, y⟩ = ⟨x, T*y⟩
let lhs = hilbert.inner_product(&op.apply(&x)?, &y)?;
let rhs = hilbert.inner_product(&x, &adjoint.apply(&y)?)?;
assert!((lhs - rhs).abs() < 1e-10);
```

---

### Spectral Theory

**Mathematical Foundation**: Eigenvalue problems and spectral decomposition.

#### Key Concepts
- **Spectrum**: σ(T) = {λ : (T - λI) not invertible}
- **Point Spectrum**: Eigenvalues
- **Spectral Theorem**: Self-adjoint T = Σ λₙ Pₙ
- **Functional Calculus**: f(T) = Σ f(λₙ) Pₙ
- **Spectral Radius**: ρ(T) = max|λ| = lim ‖Tⁿ‖^(1/n)

#### Implementation Highlights
```rust
use amari_functional::{
    spectral::{SpectralDecomposition, FunctionalCalculus},
    operator::SelfAdjoint,
};

let decomp = SpectralDecomposition::compute(&op)?;
let fc = FunctionalCalculus::new(&op)?;

// Compute exp(A) via spectral theorem
let exp_a = fc.apply(|x| x.exp())?;
```

---

### Banach Spaces

**Mathematical Foundation**: Complete normed vector spaces.

#### Key Concepts
- **Lᵖ Spaces**: ‖f‖_p = (∫|f|ᵖ)^(1/p)
- **Dual Space**: X* = bounded linear functionals on X
- **Reflexivity**: X = X** (isometrically)
- **Fixed Point Theorem**: Contraction maps have unique fixed points

#### Implementation Highlights
```rust
use amari_functional::banach::{LpSpace, SequenceLp, BanachFixedPoint};

let l2 = LpSpace::new(0.0, 1.0, 2.0)?;
let l_inf = LpSpace::l_infinity(0.0, 1.0)?;

// Compare norms
println!("‖f‖_2 = {:.4}", l2.norm(&f)?);
println!("‖f‖_∞ = {:.4}", l_inf.norm(&f)?);
```

---

### Distributions

**Mathematical Foundation**: Generalized functions (Schwartz distributions).

#### Key Concepts
- **Test Functions**: Smooth with compact support
- **Distribution**: Continuous linear functional on test functions
- **Dirac Delta**: ⟨δ, φ⟩ = φ(0)
- **Weak Derivative**: ⟨T', φ⟩ = -⟨T, φ'⟩
- **Heaviside**: H' = δ

#### Implementation Highlights
```rust
use amari_functional::distributions::{
    DiracDelta, HeavisideStep, TestFunction,
};

let delta = DiracDelta::at(0.0)?;
let bump = TestFunction::bump(-1.0, 1.0)?;

// ⟨δ, φ⟩ = φ(0)
let result = delta.apply(&bump)?;
assert!((result - bump.evaluate(0.0)).abs() < 1e-10);
```

---

## Physics Simulation Examples

### Rigid Body Dynamics

**Mathematical Foundation**: Rotations using rotors instead of matrices or quaternions.

#### Key Concepts
- **Rotor**: R = cos(θ/2) - sin(θ/2)(n̂ ∧ B)
- **Angular Velocity**: Bivector ω = ωₓe₂₃ + ωᵧe₃₁ + ωᵧe₁₂
- **Torque**: τ = r ∧ F (outer product)

---

### Electromagnetic Fields

**Mathematical Foundation**: Unified field multivector F = E + I·B.

#### Key Concepts
- **Maxwell's Equations**: ∇F = J unified form
- **Lorentz Force**: F = q(E + v × B)
- **Poynting Vector**: S = E × B / μ₀

---

### Fluid Dynamics

**Mathematical Foundation**: Vorticity as bivector ω = ∇ ∧ v.

#### Key Concepts
- **Circulation**: Γ = ∮ v·dl = ∬ ω·dA
- **Helicity**: H = v·ω (topological invariant)

---

### Quantum Mechanics

**Mathematical Foundation**: Pauli matrices as bivectors.

#### Key Concepts
- **Pauli Algebra**: σₓ = e₁₂, σᵧ = e₁₃, σᵧ = e₂₃
- **Spin States**: Multivector representation
- **Rotations**: R|ψ⟩R† using rotors

---

## Computer Graphics Examples

### 3D Transformations
- Gimbal lock-free rotations
- SLERP interpolation
- Hierarchical transforms

### Camera Systems
- View transformations
- Perspective projection
- Orbital controls

### Mesh Operations
- Normal calculation
- Area computation
- Geometric queries

### Ray Tracing
- Ray-surface intersection
- Reflection/refraction
- Lighting calculations

---

## Machine Learning Examples

### Automatic Differentiation
- Dual numbers for exact gradients
- Chain rule automation
- Higher-order derivatives

### Neural Networks
- Verified backpropagation
- Gradient checking
- Error analysis

### Optimization
- Gradient descent variants
- Newton's method
- Convergence analysis

---

## Interactive Demo Documentation

### Architecture
- Frontend: Modern JavaScript with Three.js
- Visualization: Plotly.js for 2D plots
- Controls: Interactive parameter adjustment
- Mathematics: KaTeX equation rendering

### Design Principles
1. Progressive disclosure
2. Immediate feedback
3. Mathematical connection
4. Comparative analysis

---

## Technical Details

### Performance
- SIMD optimization for geometric products
- GPU acceleration via wgpu
- Parallel computation with Rayon
- WebAssembly for browser deployment

### Numerical Stability
- Adaptive step-size solvers
- Error-controlled computations
- Condition number monitoring

### Testing
- Unit tests for individual functions
- Integration tests for workflows
- Property-based testing for invariants
- Performance benchmarks

### Formal Verification
- Creusot contracts for critical properties
- Phantom types for compile-time guarantees
- Mathematical property verification

---

## Further Reading

### Books
- **Dynamical Systems**: Strogatz "Nonlinear Dynamics and Chaos"
- **Topology**: Edelsbrunner & Harer "Computational Topology"
- **Functional Analysis**: Kreyszig "Introductory Functional Analysis"
- **Geometric Algebra**: Doran & Lasenby "Geometric Algebra for Physicists"

### Papers
- Persistent homology and TDA applications
- Lyapunov exponent computation methods
- Spectral theory in operator algebras
- Geometric algebra in physics and graphics

---

*This documentation serves as both a reference and learning guide. Each example is designed to be self-contained while building toward comprehensive understanding.*
