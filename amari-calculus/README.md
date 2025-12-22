# amari-calculus

Geometric calculus - a unified framework for differential and integral calculus using geometric algebra.

## Overview

`amari-calculus` provides geometric calculus operations that unify vector calculus, differential forms, and tensor calculus into a single coherent framework. The key insight is that the vector derivative operator ∇ combines the familiar gradient, divergence, and curl operations into one fundamental operation.

## Features

- **Vector Derivative Operator**: The fundamental ∇ operator
- **Classical Operators**: Gradient, divergence, curl, Laplacian
- **Field Types**: Scalar, vector, and multivector fields
- **Coordinate Systems**: Cartesian, spherical, cylindrical, polar
- **Manifold Calculus**: Covariant derivatives, connections
- **Lie Derivatives**: Derivatives along vector fields
- **Integration**: Manifold integration via amari-measure

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-calculus = "0.12"
```

### Feature Flags

```toml
[dependencies]
# Default features
amari-calculus = "0.12"

# Minimal, no-std compatible
amari-calculus = { version = "0.12", default-features = false }
```

## Quick Start

### Gradient of a Scalar Field

```rust
use amari_calculus::{ScalarField, VectorDerivative, CoordinateSystem};

// Define scalar field f(x, y, z) = x² + y²
let f = ScalarField::<3, 0, 0>::new(|coords| {
    coords[0].powi(2) + coords[1].powi(2)
});

// Create vector derivative operator
let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);

// Compute gradient at point (1, 2, 0)
let grad_f = nabla.gradient(&f, &[1.0, 2.0, 0.0]);
// Gradient is approximately (2, 4, 0)
```

### Divergence of a Vector Field

```rust
use amari_calculus::{VectorField, VectorDerivative, CoordinateSystem, vector_from_slice};

// Define vector field F(x, y, z) = (x, y, z)
let f = VectorField::<3, 0, 0>::new(|coords| {
    vector_from_slice(&[coords[0], coords[1], coords[2]])
});

let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);

// Compute divergence (should be 3)
let div_f = nabla.divergence(&f, &[1.0, 1.0, 1.0]);
```

### Curl of a Vector Field

```rust
use amari_calculus::{VectorField, VectorDerivative, CoordinateSystem, vector_from_slice};

// Define vector field F(x, y, z) = (-y, x, 0) (rotation around z-axis)
let f = VectorField::<3, 0, 0>::new(|coords| {
    vector_from_slice(&[-coords[1], coords[0], 0.0])
});

let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);

// Compute curl (returns bivector representing rotation)
let curl_f = nabla.curl(&f, &[0.0, 0.0, 0.0]);
// Curl is (0, 0, 2) as a bivector
```

## Mathematical Foundation

### The Vector Derivative

The vector derivative operator is defined as:

```
∇ = eⁱ ∂ᵢ  (sum over basis vectors)
```

This operator combines:
- **Dot product** → divergence: ∇·F
- **Wedge product** → curl: ∇∧F
- **Full geometric product** → complete derivative: ∇F = ∇·F + ∇∧F

### Unification of Calculus

| Classical | Differential Forms | Geometric Calculus |
|-----------|-------------------|-------------------|
| grad f | df | ∇f |
| div F | ⋆d⋆F | ∇·F |
| curl F | ⋆dF | ∇∧F |
| ∇²f | ⋆d⋆df | ∇²f = ∇·∇f |

### The Fundamental Theorem

Geometric calculus has a single fundamental theorem that unifies all integral theorems:

```
∫_V (∇F) dV = ∮_∂V F dS
```

This single equation encompasses:
- Gradient theorem (fundamental theorem of calculus)
- Divergence theorem (Gauss's theorem)
- Stokes' theorem
- Green's theorem

## Key Types

### ScalarField<P, Q, R>

Scalar-valued functions on Cl(P,Q,R) space:

```rust
use amari_calculus::ScalarField;

// Temperature field T(x, y, z)
let temperature = ScalarField::<3, 0, 0>::new(|coords| {
    100.0 * (-coords[0].powi(2) - coords[1].powi(2)).exp()
});
```

### VectorField<P, Q, R>

Vector-valued functions:

```rust
use amari_calculus::{VectorField, vector_from_slice};

// Velocity field
let velocity = VectorField::<3, 0, 0>::new(|coords| {
    vector_from_slice(&[coords[1], -coords[0], 0.0])
});
```

### MultivectorField<P, Q, R>

General multivector-valued functions:

```rust
use amari_calculus::MultivectorField;

// Electromagnetic field (bivector field)
let em_field = MultivectorField::<3, 0, 0>::new(|coords| {
    // Returns a multivector with bivector components
    // ...
});
```

### VectorDerivative<P, Q, R>

The ∇ operator:

```rust
use amari_calculus::{VectorDerivative, CoordinateSystem};

// Create in Cartesian coordinates
let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);

// Or in spherical coordinates
let nabla_sph = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Spherical);
```

## Coordinate Systems

| System | Coordinates | Use Case |
|--------|-------------|----------|
| Cartesian | (x, y, z) | General computations |
| Spherical | (r, θ, φ) | Central potentials |
| Cylindrical | (ρ, φ, z) | Axial symmetry |
| Polar | (r, θ) | 2D problems |

## Modules

| Module | Description |
|--------|-------------|
| `fields` | ScalarField, VectorField, MultivectorField |
| `operators` | gradient, divergence, curl, laplacian |
| `derivative` | VectorDerivative operator |
| `manifold` | RiemannianManifold, covariant derivatives |
| `lie` | Lie derivatives along vector fields |
| `integration` | ManifoldIntegrator |

## Applications

### Electromagnetism

```rust
use amari_calculus::prelude::*;

// Electric field E = -∇φ
let potential = ScalarField::<3, 0, 0>::new(|r| {
    let r_mag = (r[0].powi(2) + r[1].powi(2) + r[2].powi(2)).sqrt();
    1.0 / r_mag  // Coulomb potential
});

let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);
let e_field = nabla.gradient(&potential, &[1.0, 0.0, 0.0]);
```

### Fluid Dynamics

```rust
use amari_calculus::prelude::*;

// Velocity field
let velocity = VectorField::<3, 0, 0>::new(|r| {
    vector_from_slice(&[r[1], -r[0], 0.0])
});

let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);

// Incompressibility: ∇·v = 0
let div_v = nabla.divergence(&velocity, &[1.0, 1.0, 0.0]);

// Vorticity: ω = ∇∧v
let vorticity = nabla.curl(&velocity, &[0.0, 0.0, 0.0]);
```

### Heat Equation

```rust
use amari_calculus::prelude::*;

// Temperature field
let temp = ScalarField::<3, 0, 0>::new(|r| {
    (-r[0].powi(2) - r[1].powi(2) - r[2].powi(2)).exp()
});

let nabla = VectorDerivative::<3, 0, 0>::new(CoordinateSystem::Cartesian);

// Laplacian for heat diffusion
let laplacian_t = laplacian(&temp, &[0.0, 0.0, 0.0]);
```

## Prelude

For convenient imports:

```rust
use amari_calculus::prelude::*;

// Imports: ScalarField, VectorField, MultivectorField,
//          VectorDerivative, CoordinateSystem,
//          gradient, divergence, curl, laplacian
```

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
