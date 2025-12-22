# amari-enumerative

Enumerative geometry for counting geometric configurations.

## Overview

`amari-enumerative` provides tools for enumerative geometry, the mathematical discipline concerned with counting geometric objects satisfying given conditions. The crate implements intersection theory, Schubert calculus, Gromov-Witten invariants, and tropical curve counting.

## Features

- **Intersection Theory**: Chow rings, intersection multiplicities, Bézout's theorem
- **Schubert Calculus**: Computations on Grassmannians and flag varieties
- **Gromov-Witten Theory**: Curve counting and quantum cohomology
- **Tropical Geometry**: Tropical curve counting via correspondence theorems
- **Moduli Spaces**: Computations on moduli spaces of curves
- **GPU Acceleration**: Optional GPU support for large computations

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-enumerative = "0.12"
```

### Feature Flags

```toml
[dependencies]
# Default features
amari-enumerative = "0.12"

# With serialization
amari-enumerative = { version = "0.12", features = ["serde"] }

# With GPU acceleration
amari-enumerative = { version = "0.12", features = ["gpu"] }

# With parallel computation
amari-enumerative = { version = "0.12", features = ["parallel"] }

# For WASM targets
amari-enumerative = { version = "0.12", features = ["wasm"] }
```

## Quick Start

```rust
use amari_enumerative::{ProjectiveSpace, ChowClass, IntersectionRing};

// Create projective 2-space (P²)
let p2 = ProjectiveSpace::new(2);

// Define two curves by degree
let cubic = ChowClass::hypersurface(3);   // Degree 3 curve
let quartic = ChowClass::hypersurface(4); // Degree 4 curve

// Compute intersection number using Bézout's theorem
let intersection = p2.intersect(&cubic, &quartic);
assert_eq!(intersection.multiplicity(), 12); // 3 × 4 = 12 points
```

## Key Concepts

### Bézout's Theorem

Two plane curves of degrees d and e intersect in d·e points (counting multiplicity):

```rust
use amari_enumerative::{ProjectiveSpace, ChowClass};

let p2 = ProjectiveSpace::new(2);
let line = ChowClass::hypersurface(1);
let conic = ChowClass::hypersurface(2);

// Line meets conic in 1 × 2 = 2 points
let points = p2.intersect(&line, &conic).multiplicity();
assert_eq!(points, 2);
```

### Schubert Calculus

Count linear subspaces satisfying incidence conditions:

```rust
use amari_enumerative::{Grassmannian, SchubertClass};

// Grassmannian Gr(2,4): lines in P³
let gr = Grassmannian::new(2, 4);

// How many lines meet 4 general lines in P³?
let sigma_1 = SchubertClass::sigma(&[1]); // Lines meeting a line
let count = gr.intersect(&[sigma_1; 4]);
assert_eq!(count, 2); // Answer: 2 lines
```

### Gromov-Witten Invariants

Count curves in algebraic varieties:

```rust
use amari_enumerative::{GromovWittenInvariant, QuantumCohomology};

// Count rational curves of degree d in P²
let gw = GromovWittenInvariant::new(variety, degree);
let count = gw.compute_with_insertions(&insertions)?;
```

### Tropical Curves

Use tropical geometry for curve counting:

```rust
use amari_enumerative::tropical_curves::TropicalCurve;

let curve = TropicalCurve::new(degree, genus);
let count = curve.tropical_count()?;
```

## Modules

| Module | Description |
|--------|-------------|
| `intersection` | Chow rings, intersection products, Bézout |
| `schubert` | Schubert calculus on Grassmannians |
| `gromov_witten` | Curve counting, quantum cohomology |
| `tropical_curves` | Tropical geometry methods |
| `moduli_space` | Moduli spaces of curves |
| `higher_genus` | Higher genus curve counting, DT/PT invariants |
| `geometric_algebra` | Integration with geometric algebra |
| `performance` | Optimized computation utilities |

## Mathematical Background

### Chow Rings

The Chow ring A*(X) captures the intersection theory of a variety X:

```
A*(P^n) = Z[H] / (H^(n+1))
```

where H is the hyperplane class.

### Schubert Cells

The Grassmannian Gr(k,n) has a cell decomposition by Schubert cells:

```
Gr(k,n) = ⊔ Ω_λ
```

indexed by partitions λ ⊂ (n-k)^k.

### Gromov-Witten Theory

GW invariants count curves via:

```
⟨τ_a1(γ1),...,τ_an(γn)⟩_{g,β} = ∫_{[M̄_{g,n}(X,β)]^{vir}} ψ_1^{a1} ev_1*(γ1) ∧ ...
```

## Classic Enumerative Problems

| Problem | Answer |
|---------|--------|
| Lines through 2 points | 1 |
| Conics through 5 points | 1 |
| Lines meeting 4 general lines in P³ | 2 |
| Rational cubics through 8 points in P² | 12 |
| Lines on a smooth cubic surface | 27 |

## Performance

- **Parallel Computation**: Rayon-based parallelization
- **GPU Acceleration**: WebGPU for large intersection computations
- **Sparse Matrices**: Efficient representation of Schubert classes
- **Batch Processing**: Process multiple curves simultaneously

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
