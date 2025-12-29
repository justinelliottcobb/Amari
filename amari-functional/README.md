# amari-functional

Functional analysis on multivector spaces - Hilbert spaces, linear operators, spectral theory, and Sobolev spaces for Clifford algebras.

## Overview

`amari-functional` provides the mathematical foundations of functional analysis applied to Clifford algebra-valued function spaces. The crate implements the complete hierarchy from vector spaces to Hilbert spaces, bounded linear operators with spectral decomposition, and Sobolev spaces for weak derivatives.

## Features

- **Space Hierarchy**: VectorSpace → NormedSpace → BanachSpace → InnerProductSpace → HilbertSpace
- **Finite-Dimensional Hilbert Spaces**: Cl(P,Q,R) as 2^(P+Q+R)-dimensional Hilbert spaces
- **L² Function Spaces**: Square-integrable multivector-valued functions
- **Linear Operators**: Bounded, compact, self-adjoint, and Fredholm operators
- **Matrix Operators**: Explicit matrix representations with composition and norms
- **Spectral Theory**: Eigenvalue computation, spectral decomposition, functional calculus
- **Eigenvalue Algorithms**: Power method, inverse iteration, Jacobi algorithm
- **Sobolev Spaces**: H^k spaces with weak derivatives and Poincaré inequalities
- **Phantom Types**: Compile-time verification of operator properties
- **Formal Verification**: Optional Creusot contracts for theorem verification

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-functional = "0.14"
```

### Feature Flags

```toml
[dependencies]
# Default features
amari-functional = "0.14"

# With parallel computation (Rayon)
amari-functional = { version = "0.14", features = ["parallel"] }

# With formal verification (Creusot)
amari-functional = { version = "0.14", features = ["formal-verification"] }
```

## Quick Start

### Hilbert Space Operations

```rust
use amari_functional::space::{MultivectorHilbertSpace, HilbertSpace, InnerProductSpace};

// Create the Hilbert space Cl(2,0,0) ≅ ℝ⁴
let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();

// Create elements: 1 + 2e₁ and 3e₂ + 4e₁₂
let x = space.from_coefficients(&[1.0, 2.0, 0.0, 0.0]).unwrap();
let y = space.from_coefficients(&[0.0, 0.0, 3.0, 4.0]).unwrap();

// Inner product (these are orthogonal)
let ip = space.inner_product(&x, &y);
assert!(ip.abs() < 1e-10);

// Norm and distance
let norm_x = space.norm(&x);
let dist = space.distance(&x, &y);

// Orthogonal projection
let proj = space.project(&x, &y);

// Gram-Schmidt orthonormalization
let basis = space.basis();
let orthonormal = space.gram_schmidt(&basis);
```

### Matrix Operators

```rust
use amari_functional::operator::{MatrixOperator, LinearOperator, BoundedOperator};
use amari_core::Multivector;

// Identity operator
let id: MatrixOperator<2, 0, 0> = MatrixOperator::identity();

// Diagonal operator
let diag = MatrixOperator::<2, 0, 0>::diagonal(&[4.0, 3.0, 2.0, 1.0]).unwrap();

// Apply to a vector
let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 1.0, 1.0, 1.0]);
let result = diag.apply(&x).unwrap();
// result = [4, 3, 2, 1]

// Operator composition
let composed = diag.multiply(&id).unwrap();

// Operator norm ||T||
let norm = diag.operator_norm();

// Check symmetry (self-adjointness)
assert!(diag.is_symmetric(1e-10));
```

### Spectral Decomposition

```rust
use amari_functional::operator::MatrixOperator;
use amari_functional::spectral::{spectral_decompose, SpectralDecomposition};

// Create a symmetric positive-definite matrix
let matrix = MatrixOperator::<2, 0, 0>::diagonal(&[4.0, 3.0, 2.0, 1.0]).unwrap();

// Compute spectral decomposition: T = Σᵢ λᵢ Pᵢ
let decomp = spectral_decompose(&matrix, 100, 1e-10).unwrap();

// Get eigenvalues
let eigenvalues = decomp.eigenvalues();
// [4.0, 3.0, 2.0, 1.0]

// Spectral properties
let radius = decomp.spectral_radius();      // 4.0
let cond = decomp.condition_number();       // Some(4.0)
let pos_def = decomp.is_positive_definite(); // true

// Functional calculus: apply f(T) = √T
let x = Multivector::<2, 0, 0>::from_slice(&[1.0, 0.0, 0.0, 0.0]);
let sqrt_Tx = decomp.apply_function(|lambda| lambda.sqrt(), &x);
// √T · e₀ = 2 · e₀
```

### Eigenvalue Algorithms

```rust
use amari_functional::operator::MatrixOperator;
use amari_functional::spectral::{power_method, inverse_iteration, compute_eigenvalues};

let matrix = MatrixOperator::<2, 0, 0>::diagonal(&[4.0, 3.0, 2.0, 1.0]).unwrap();

// Power method: find dominant eigenvalue
let dominant = power_method(&matrix, None, 100, 1e-10).unwrap();
// dominant.eigenvalue.value ≈ 4.0

// Inverse iteration: find eigenvalue near a shift
let near_2 = inverse_iteration(&matrix, 2.1, None, 100, 1e-10).unwrap();
// near_2.eigenvalue.value ≈ 2.0

// Jacobi algorithm: all eigenvalues
let all_eigs = compute_eigenvalues(&matrix, 100, 1e-10).unwrap();
// [4.0, 3.0, 2.0, 1.0]
```

### Sobolev Spaces

```rust
use amari_functional::sobolev::{SobolevSpace, SobolevFunction, H1Space, poincare_constant_estimate};
use amari_functional::space::Domain;
use amari_core::Multivector;
use std::sync::Arc;

// Create H¹([0,1], Cl(2,0,0))
let h1: H1Space<2, 0, 0> = H1Space::unit_interval().with_quadrature_points(64);

// Define f(x) = x with f'(x) = 1
let f: SobolevFunction<2, 0, 0, 1> = SobolevFunction::new(
    |x| Multivector::<2, 0, 0>::scalar(x[0]),
    vec![Arc::new(|_x| Multivector::<2, 0, 0>::scalar(1.0))],
    1,
);

// Compute H¹ norm: ||f||²_{H¹} = ||f||²_{L²} + ||f'||²_{L²}
let h1_norm = h1.hk_norm(&f);
// ≈ √(1/3 + 1) = √(4/3) ≈ 1.155

// H¹ seminorm: |f|_{H¹} = ||f'||_{L²}
let seminorm = h1.hk_seminorm(&f);
// ≈ 1.0

// Poincaré constant for [0,1]: C = 1/π
let domain = Domain::interval(0.0, 1.0);
let poincare = poincare_constant_estimate(&domain);
// ≈ 0.318
```

### L² Function Spaces

```rust
use amari_functional::space::{MultivectorL2, L2Function, NormedSpace, InnerProductSpace};
use amari_core::Multivector;

// Create L²([0,1], Cl(2,0,0))
let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();

// Define f(x) = x · e₀
let f = L2Function::new(|x| Multivector::<2, 0, 0>::scalar(x[0]));

// Define g(x) = (1-x) · e₀
let g = L2Function::new(|x| Multivector::<2, 0, 0>::scalar(1.0 - x[0]));

// L² norm: ||f||² = ∫₀¹ x² dx = 1/3
let norm = l2.norm(&f);
// ≈ √(1/3) ≈ 0.577

// L² inner product: ⟨f,g⟩ = ∫₀¹ x(1-x) dx = 1/6
let ip = l2.inner_product(&f, &g);
// ≈ 0.167
```

## Mathematical Background

### Space Hierarchy

The crate implements the standard functional analysis hierarchy:

```
VectorSpace         - Linear structure with addition and scalar multiplication
    ↓
NormedSpace         - Adds norm ||·|| satisfying triangle inequality
    ↓
BanachSpace         - Complete normed space (Cauchy sequences converge)
    ↓
InnerProductSpace   - Adds inner product ⟨·,·⟩ with ||x||² = ⟨x,x⟩
    ↓
HilbertSpace        - Complete inner product space
```

### Clifford Algebra Hilbert Spaces

The Clifford algebra Cl(P,Q,R) is a 2^(P+Q+R)-dimensional real vector space equipped with the standard L² inner product on coefficients:

⟨x, y⟩ = Σᵢ xᵢ yᵢ

This makes Cl(P,Q,R) into a finite-dimensional Hilbert space isomorphic to ℝ^(2^(P+Q+R)).

### Spectral Theorem

For a self-adjoint (symmetric) operator T on a finite-dimensional Hilbert space:

T = Σᵢ λᵢ Pᵢ

where λᵢ are real eigenvalues and Pᵢ are orthogonal projections onto eigenspaces.

### Functional Calculus

For any function f: ℝ → ℝ, the operator f(T) is defined via spectral decomposition:

f(T) = Σᵢ f(λᵢ) Pᵢ

This enables computing matrix exponentials, square roots, logarithms, etc.

### Sobolev Spaces

The Sobolev space W^{k,p}(Ω, V) consists of functions f: Ω → V with weak derivatives up to order k in L^p. For p = 2, these are Hilbert spaces (H^k) with inner product:

⟨f, g⟩_{H^k} = Σ_{|α| ≤ k} ∫_Ω ⟨D^α f, D^α g⟩ dx

### Poincaré Inequality

For functions with zero boundary conditions on a bounded domain Ω:

||f||_{L²} ≤ C ||∇f||_{L²}

The Poincaré constant C depends on the domain geometry (C = diam(Ω)/π for intervals).

## Modules

| Module | Description |
|--------|-------------|
| `space` | Vector spaces, normed spaces, Banach spaces, Hilbert spaces |
| `space::hilbert` | MultivectorHilbertSpace for finite-dimensional spaces |
| `space::multivector_l2` | L² spaces of multivector-valued functions |
| `operator` | Linear operator traits and implementations |
| `operator::matrix` | Matrix representation of bounded operators |
| `operator::compact` | Compact and Fredholm operators |
| `spectral` | Eigenvalue algorithms and spectral decomposition |
| `sobolev` | Sobolev spaces H^k with weak derivatives |
| `phantom` | Compile-time property markers for operators |
| `error` | Error types for functional analysis operations |

## Phantom Types

The crate uses phantom types for compile-time verification of operator properties:

```rust
use amari_functional::phantom::{
    Bounded, Unbounded,           // Boundedness
    Compact, NonCompact,          // Compactness
    SelfAdjoint, Normal, Unitary, // Symmetry
    Fredholm, SemiFredholm,       // Fredholm property
    DiscreteSpectrum, ContinuousSpectrum, // Spectral type
};

// Type aliases for common operator types
use amari_functional::phantom::{
    CompactSelfAdjointOperator,
    HilbertSchmidtOperator,
    UnitaryOperator,
};
```

## Applications

### Quantum Mechanics

```rust
use amari_functional::operator::MatrixOperator;
use amari_functional::spectral::spectral_decompose;
use amari_functional::space::{MultivectorHilbertSpace, InnerProductSpace};

// Hamiltonian with energy levels 0, 1, 2, 3
let hamiltonian = MatrixOperator::<2, 0, 0>::diagonal(&[0.0, 1.0, 2.0, 3.0]).unwrap();

// Spectral decomposition for energy eigenstates
let spectral = spectral_decompose(&hamiltonian, 100, 1e-10).unwrap();

// Initial superposition state
let space = MultivectorHilbertSpace::<2, 0, 0>::new();
let psi = space.normalize(
    &space.from_coefficients(&[1.0, 1.0, 0.0, 0.0]).unwrap()
).unwrap();

// Expected energy ⟨ψ|H|ψ⟩
let h_psi = hamiltonian.apply(&psi).unwrap();
let expected_energy = space.inner_product(&psi, &h_psi);

// Time evolution: |ψ(t)⟩ = e^{-iHt}|ψ(0)⟩
let t = 1.0;
let evolved = spectral.apply_function(|E| (-E * t).exp(), &psi);
```

### Numerical PDEs

Sobolev spaces provide the natural setting for weak solutions to PDEs:

```rust
use amari_functional::sobolev::{H1Space, SobolevFunction, poincare_constant_estimate};
use amari_functional::space::Domain;

// H¹₀(Ω) for variational formulations
let h1 = H1Space::<2, 0, 0>::new(Domain::interval(0.0, 1.0));

// Poincaré inequality: ||u||_{L²} ≤ C ||∇u||_{L²}
let domain = Domain::interval(0.0, 1.0);
let poincare = poincare_constant_estimate(&domain);

// Energy norm for elliptic problems
// ||u||_a² = ∫ a(x)|∇u|² dx
```

### Signal Processing

Hilbert space projections for signal approximation:

```rust
use amari_functional::space::{MultivectorHilbertSpace, HilbertSpace, InnerProductSpace};

let space = MultivectorHilbertSpace::<2, 0, 0>::new();

// Signal in the Hilbert space
let signal = space.from_coefficients(&[1.0, 2.0, 3.0, 4.0]).unwrap();

// Basis for approximation subspace
let e0 = space.basis_vector(0).unwrap();
let e1 = space.basis_vector(1).unwrap();

// Best approximation in subspace
let approx = space.best_approximation(&signal, &[e0, e1]);
// Projection onto span{e₀, e₁}
```

### Machine Learning

Kernel methods and reproducing kernel Hilbert spaces:

```rust
use amari_functional::operator::MatrixOperator;
use amari_functional::spectral::spectral_decompose;

// Kernel matrix (Gram matrix)
let kernel = MatrixOperator::<2, 0, 0>::diagonal(&[4.0, 3.0, 2.0, 1.0]).unwrap();

// Spectral decomposition for kernel PCA
let decomp = spectral_decompose(&kernel, 100, 1e-10).unwrap();

// Leading eigenvalues capture most variance
let eigenvalues = decomp.eigenvalues();
let total_variance: f64 = eigenvalues.iter().map(|e| e.value).sum();
let top_2_variance: f64 = eigenvalues.iter().take(2).map(|e| e.value).sum();
let explained_ratio = top_2_variance / total_variance;
```

## Error Handling

The crate provides comprehensive error types:

```rust
use amari_functional::{FunctionalError, Result};

// Dimension mismatch
let err = FunctionalError::dimension_mismatch(4, 5);

// Convergence failure
let err = FunctionalError::convergence_error(100, "Power method did not converge");

// Invalid parameters
let err = FunctionalError::invalid_parameters("Matrix must be symmetric");

// Singular operator
let err = FunctionalError::singular_operator();
```

## Performance

- **Finite-dimensional operations**: O(n²) for matrix-vector, O(n³) for eigenvalue computation
- **Numerical integration**: Configurable quadrature points for accuracy/speed tradeoff
- **Parallel support**: Enable `parallel` feature for Rayon-based parallelism

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library, providing:

- **amari-core**: Clifford algebra fundamentals
- **amari-measure**: Measure theory and integration
- **amari-calculus**: Differential geometry and manifolds
- **amari-functional**: Functional analysis (this crate)
- **amari-info-geom**: Information geometry
- And many more...

## References

- Reed, M. & Simon, B. *Methods of Modern Mathematical Physics* (1980)
- Conway, J.B. *A Course in Functional Analysis* (1990)
- Hestenes, D. & Sobczyk, G. *Clifford Algebra to Geometric Calculus* (1984)
- Adams, R.A. *Sobolev Spaces* (1975)
