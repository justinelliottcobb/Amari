# Amari Learning Paths v0.17.0

Welcome to the Amari educational journey! This guide provides structured pathways for learning geometric algebra, dynamical systems, computational topology, and functional analysis through our comprehensive example suite.

## Overview

Amari v0.17.0 provides learning tracks across multiple mathematical domains. Each path builds upon previous knowledge while providing practical, hands-on experience.

## Core Tracks

### Beginner Track: Foundations
*Prerequisites: Basic linear algebra, calculus*

**Goal**: Understand fundamental concepts of geometric algebra and dual numbers

#### Module 1: Geometric Algebra Basics
1. **Start Here**: `examples/rust/physics-simulation/`
   - Scalar, vector, bivector operations
   - Geometric product fundamentals
   - Basic rotations with rotors

2. **Visualization**: `examples/web/interactive-demos/`
   - 3D Rotations with Rotors demo
   - Compare with traditional approaches

3. **Practice**: `examples/rust/computer-graphics/`
   - Implement basic transformations
   - Understand gimbal lock avoidance

#### Module 2: Dual Numbers Introduction
1. **Automatic Differentiation**: `examples/rust/machine-learning/`
   - Single-variable derivatives
   - Compare with finite differences
   - Error analysis

2. **Interactive**: `examples/web/interactive-demos/`
   - Automatic Differentiation demo
   - Real-time gradient visualization

#### Module 3: First Applications
1. **Simple Physics**: `examples/rust/physics-simulation/`
   - Basic rigid body motion
   - Rotor-based rotations

2. **Basic Optimization**: `examples/rust/machine-learning/`
   - Gradient descent with exact gradients
   - Simple quadratic functions

**Assessment**: Complete beginner exercises and create a simple rotor-based animation.

---

### Intermediate Track: Applications
*Prerequisites: Beginner track completed*

**Goal**: Apply GA and dual numbers to real-world problems

#### Module 1: Advanced Geometric Algebra
1. **Physics Applications**: `examples/rust/physics-simulation/`
   - Electromagnetic fields as multivectors
   - Fluid dynamics with bivector vorticity

2. **Computer Graphics**: `examples/rust/computer-graphics/`
   - Camera systems and projections
   - Mesh operations and normal calculations
   - Ray tracing with natural ray representation

#### Module 2: Machine Learning
1. **Neural Networks**: `examples/rust/machine-learning/`
   - Verified backpropagation
   - XOR learning demonstration
   - Function approximation

2. **Advanced Optimization**
   - Adam optimizer with exact gradients
   - Newton's method with Hessian

#### Module 3: Quantum Mechanics
1. **Quantum GA**: `examples/rust/physics-simulation/`
   - Pauli matrices as bivectors
   - Spin state evolution
   - Bell's inequality demonstration

**Assessment**: Complete a multi-module project showcasing integrated GA/dual number concepts.

---

## Specialized Tracks

### Dynamical Systems Track
*Prerequisites: Calculus, ODEs, linear algebra*

**Goal**: Master analysis of nonlinear dynamical systems

#### Module 1: ODEs and Numerical Methods
**Examples**: `examples/rust/dynamical-systems/`

1. **Lorenz Attractor** (`lorenz_attractor.rs`)
   - Chaotic dynamics fundamentals
   - Numerical integration (RK4, Dormand-Prince)
   - Phase space visualization

2. **Van der Pol Oscillator** (`van_der_pol.rs`)
   - Limit cycles
   - Relaxation oscillations
   - Parameter dependence

Key concepts:
- State space representation
- Numerical integration methods
- Trajectory visualization

#### Module 2: Stability Analysis
**Examples**: `examples/rust/dynamical-systems/`

1. **Stability Analysis** (`stability_analysis.rs`)
   - Fixed point computation
   - Jacobian linearization
   - Eigenvalue classification

2. **Phase Portraits** (`phase_portraits.rs`)
   - Vector field visualization
   - Nullclines
   - Separatrices and basins of attraction

Key concepts:
- Linear stability theory
- Center manifold theorem
- Hartman-Grobman theorem

#### Module 3: Chaos and Bifurcations
**Examples**: `examples/rust/dynamical-systems/`

1. **Bifurcation Analysis** (`bifurcation_analysis.rs`)
   - Saddle-node bifurcations
   - Hopf bifurcations
   - Period-doubling cascades
   - Feigenbaum universality

2. **Lyapunov Exponents** (`lyapunov_exponents.rs`)
   - QR method computation
   - Chaos quantification
   - Kaplan-Yorke dimension

Key concepts:
- Sensitivity to initial conditions
- Strange attractors
- Routes to chaos

#### Module 4: Stochastic Dynamics
**Examples**: `examples/rust/dynamical-systems/stochastic_dynamics.rs`

1. **Langevin Dynamics**
   - Noise modeling
   - Brownian motion

2. **Fokker-Planck Equation**
   - Probability density evolution
   - Stationary distributions

3. **Noise-Induced Transitions**
   - Kramers escape rate
   - Stochastic resonance

**Capstone Project**: Analyze a real-world dynamical system (climate model, neural network dynamics, population dynamics).

---

### Computational Topology Track
*Prerequisites: Linear algebra, basic topology helpful*

**Goal**: Master topological data analysis and computational topology

#### Module 1: Simplicial Complexes
**Examples**: `examples/rust/topology/simplicial_complexes.rs`

1. **Building Simplices**
   - Vertices, edges, triangles, tetrahedra
   - Faces and boundaries
   - f-vectors

2. **Simplicial Complexes**
   - Complex construction
   - Euler characteristic
   - Skeleton and star operations

Key concepts:
- Combinatorial topology
- Chain complexes
- Boundary operator

#### Module 2: Homology
**Examples**: `examples/rust/topology/`

1. **Chain Complexes**
   - Boundary operator ∂
   - ∂² = 0 property
   - Cycles and boundaries

2. **Betti Numbers**
   - β₀: connected components
   - β₁: loops
   - β₂: voids

Key concepts:
- Kernel and image
- Quotient groups
- Rank-nullity

#### Module 3: Persistent Homology
**Examples**: `examples/rust/topology/persistent_homology.rs`

1. **Filtrations**
   - Nested complex sequences
   - Birth and death times
   - Persistence pairs

2. **Persistence Diagrams**
   - (birth, death) representation
   - Persistence = lifetime
   - Stability theorem

Key concepts:
- Multi-scale topology
- Feature significance
- Noise vs signal

#### Module 4: Topological Data Analysis
**Examples**: `examples/rust/topology/topological_data_analysis.rs`

1. **Point Cloud Analysis**
   - Vietoris-Rips complex
   - Alpha complex
   - Distance matrix input

2. **Shape Detection**
   - Clustering via H₀
   - Loop detection via H₁
   - Void detection via H₂

3. **Feature Extraction**
   - Persistence landscapes
   - Persistence images
   - Bottleneck distance

**Capstone Project**: Apply TDA to real dataset (molecular structure, sensor network, image analysis).

---

### Functional Analysis Track
*Prerequisites: Real analysis, linear algebra, basic topology*

**Goal**: Master Hilbert and Banach space theory

#### Module 1: Hilbert Spaces
**Examples**: `examples/rust/functional-analysis/hilbert_spaces.rs`

1. **Inner Product Spaces**
   - Axioms and examples
   - Cauchy-Schwarz inequality
   - Orthogonality

2. **Completeness**
   - Cauchy sequences
   - L² spaces
   - Sequence spaces ℓ²

3. **Orthonormal Bases**
   - Gram-Schmidt process
   - Fourier basis
   - Parseval's identity

Key concepts:
- Projection theorem
- Best approximation
- Riesz representation

#### Module 2: Operators
**Examples**: `examples/rust/functional-analysis/operators.rs`

1. **Bounded Linear Operators**
   - Operator norm
   - Continuity ↔ boundedness

2. **Adjoint Operators**
   - Definition and properties
   - Self-adjoint operators

3. **Special Operators**
   - Compact operators
   - Unitary operators
   - Integral operators

Key concepts:
- Operator algebra
- Commutativity
- Spectrum preview

#### Module 3: Spectral Theory
**Examples**: `examples/rust/functional-analysis/spectral_theory.rs`

1. **Eigenvalue Problems**
   - Eigenvalues and eigenvectors
   - Rayleigh quotient

2. **Spectral Theorem**
   - Self-adjoint decomposition
   - Projection operators

3. **Functional Calculus**
   - f(A) for operators
   - Square roots, exponentials
   - Inverses

Key concepts:
- Point spectrum
- Continuous spectrum
- Spectral radius

#### Module 4: Banach Spaces
**Examples**: `examples/rust/functional-analysis/banach_spaces.rs`

1. **Lᵖ Spaces**
   - Norms for 1 ≤ p ≤ ∞
   - Embeddings
   - Hölder's inequality

2. **Dual Spaces**
   - Bounded linear functionals
   - Reflexivity

3. **Fixed Point Theory**
   - Banach contraction principle
   - Applications

Key concepts:
- Open mapping theorem
- Closed graph theorem
- Hahn-Banach theorem

#### Module 5: Distributions
**Examples**: `examples/rust/functional-analysis/distributions.rs`

1. **Test Functions**
   - Smooth compact support
   - Bump functions

2. **Distributions**
   - Dirac delta
   - Heaviside function
   - Weak derivatives

3. **Applications**
   - PDEs
   - Fourier transforms
   - Signal processing

**Capstone Project**: Implement a PDE solver using spectral methods and distribution theory.

---

## Cross-Domain Tracks

### Research Track: Innovation
*Prerequisites: At least one specialized track + domain expertise*

**Goal**: Push the boundaries of mathematical computing

#### Research Areas
1. **Theoretical Foundations**
   - Novel algebraic structures
   - Formal verification
   - Proof assistants

2. **High-Performance Computing**
   - GPU acceleration
   - SIMD optimization
   - Parallel algorithms

3. **Novel Applications**
   - Topological machine learning
   - Geometric deep learning
   - Physics-informed networks

4. **Educational Innovation**
   - Visualization techniques
   - Interactive learning tools
   - Curriculum development

---

## Practical Learning Tips

### Setting Up Your Environment
```bash
# Clone the repository
git clone https://github.com/justinelliottcobb/amari.git
cd amari

# Run dynamical systems examples
cd examples/rust/dynamical-systems
cargo run --bin lorenz_attractor

# Run topology examples
cd ../topology
cargo run --bin persistent_homology

# Run functional analysis examples
cd ../functional-analysis
cargo run --bin spectral_theory

# Interactive demos
cd ../../web/interactive-demos
npm install
npm run dev
```

### Study Approach
1. **Read First**: Understand the mathematical concepts
2. **Run Examples**: Execute provided code
3. **Modify**: Change parameters and observe results
4. **Implement**: Write your own versions
5. **Apply**: Create novel applications

### Progress Tracking

#### Beginner Milestones
- [ ] Understand geometric product
- [ ] Implement basic rotor rotation
- [ ] Compute exact derivatives with dual numbers
- [ ] Compare GA vs traditional methods

#### Dynamical Systems Milestones
- [ ] Integrate Lorenz system
- [ ] Compute Lyapunov exponents
- [ ] Generate bifurcation diagram
- [ ] Analyze stability of fixed points

#### Topology Milestones
- [ ] Build simplicial complex
- [ ] Compute Betti numbers
- [ ] Generate persistence diagram
- [ ] Apply TDA to dataset

#### Functional Analysis Milestones
- [ ] Implement L² inner product
- [ ] Compute spectral decomposition
- [ ] Apply functional calculus
- [ ] Solve problem using distributions

---

## Recommended Reading

### Core Texts
- **Geometric Algebra**: Doran & Lasenby "Geometric Algebra for Physicists"
- **Dynamical Systems**: Strogatz "Nonlinear Dynamics and Chaos"
- **Topology**: Edelsbrunner & Harer "Computational Topology"
- **Functional Analysis**: Kreyszig "Introductory Functional Analysis"

### Specialized Topics
- **TDA**: Carlsson "Topology and Data"
- **Chaos**: Ott "Chaos in Dynamical Systems"
- **Spectral Theory**: Reed & Simon "Methods of Modern Mathematical Physics"

---

*Happy learning! The journey into mathematical computing opens new ways of thinking about computation, physics, and data. Take your time, experiment freely, and don't hesitate to explore.*
