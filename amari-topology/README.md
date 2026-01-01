# amari-topology

Topological tools for geometric structures - simplicial complexes, homology computation, persistent homology, Morse theory, and fiber bundles.

## Overview

`amari-topology` provides computational topology primitives integrated with geometric algebra, enabling rigorous analysis of geometric structures through algebraic topology methods. The crate implements simplicial complexes, chain complexes, homology computation, persistent homology for topological data analysis, Morse theory for critical point analysis, and fiber bundle structures.

## Features

### Core Topology
- **Simplicial Complexes**: Abstract simplicial complexes with automatic closure property
- **Chain Groups**: Formal sums of simplices with integer coefficients
- **Boundary Maps**: Sparse matrix representation of boundary operators
- **Homology Computation**: Betti numbers via Gaussian elimination on boundary matrices
- **Persistent Homology**: Track topological features across filtrations (TDA)
- **Morse Theory**: Critical point classification and Morse complex construction
- **Manifold Boundaries**: Detect and characterize manifold boundaries
- **Fiber Bundles**: Vector bundles, principal bundles, sections, and connections

### Type Safety & Verification
- **Phantom Types**: Compile-time verification of orientation, coefficient rings, and filtration validity
- **Verified Contracts**: Creusot-style formal verification for ∂∂ = 0, Euler-Poincaré, Morse inequalities

### Performance
- **Parallel Computation**: Rayon-based parallelism for Betti numbers, Rips filtrations, and grid evaluation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-topology = "0.16"
```

### Feature Flags

```toml
[dependencies]
# Default features (std)
amari-topology = "0.16"

# With parallel computation (Rayon)
amari-topology = { version = "0.16", features = ["parallel"] }

# No-std support
amari-topology = { version = "0.16", default-features = false }
```

## Quick Start

### Simplicial Complexes and Homology

```rust
use amari_topology::{SimplicialComplex, Simplex};

// Create a triangulated torus or sphere
let mut complex = SimplicialComplex::new();

// Add a filled triangle
complex.add_simplex(Simplex::new(vec![0, 1, 2]));

// Closure property: edges and vertices automatically added
assert_eq!(complex.simplex_count(2), 1); // 1 triangle
assert_eq!(complex.simplex_count(1), 3); // 3 edges
assert_eq!(complex.simplex_count(0), 3); // 3 vertices

// Compute Betti numbers
let betti = complex.betti_numbers();
assert_eq!(betti[0], 1); // 1 connected component
assert_eq!(betti[1], 0); // No holes (filled triangle)

// Euler characteristic: χ = V - E + F = 3 - 3 + 1 = 1
assert_eq!(complex.euler_characteristic(), 1);
```

### Hollow vs. Filled Shapes

```rust
use amari_topology::{SimplicialComplex, Simplex};

// Circle (boundary of triangle, has a hole)
let mut circle = SimplicialComplex::new();
circle.add_simplex(Simplex::new(vec![0, 1]));
circle.add_simplex(Simplex::new(vec![1, 2]));
circle.add_simplex(Simplex::new(vec![2, 0]));

let betti = circle.betti_numbers();
assert_eq!(betti[0], 1); // 1 connected component
assert_eq!(betti[1], 1); // 1 hole!

// Sphere (hollow tetrahedron)
let mut sphere = SimplicialComplex::new();
sphere.add_simplex(Simplex::new(vec![0, 1, 2]));
sphere.add_simplex(Simplex::new(vec![0, 1, 3]));
sphere.add_simplex(Simplex::new(vec![0, 2, 3]));
sphere.add_simplex(Simplex::new(vec![1, 2, 3]));

let betti = sphere.betti_numbers();
assert_eq!(betti[0], 1); // 1 connected component
assert_eq!(betti[1], 0); // No 1D holes
assert_eq!(betti[2], 1); // 1 void (enclosed space)
```

### Persistent Homology

```rust
use amari_topology::{Filtration, Simplex, PersistentHomology};

// Build a filtration: points appear, then edges connect them
let mut filt = Filtration::new();
filt.add(0.0, Simplex::new(vec![0]));
filt.add(0.0, Simplex::new(vec![1]));
filt.add(0.0, Simplex::new(vec![2]));
filt.add(1.0, Simplex::new(vec![0, 1])); // Connect 0-1
filt.add(2.0, Simplex::new(vec![1, 2])); // Connect 1-2
filt.add(3.0, Simplex::new(vec![0, 2])); // Creates a loop

// Compute persistent homology
let ph = PersistentHomology::compute(&mut filt).unwrap();
let diagram = ph.diagram();

// At t=0: 3 components, at t=3: 1 component + 1 loop
let betti_0 = diagram.betti_at(0.5);
assert_eq!(betti_0[0], 3); // 3 components at t=0.5

let betti_3 = diagram.betti_at(3.5);
assert_eq!(betti_3[0], 1); // 1 component
assert_eq!(betti_3[1], 1); // 1 loop born at t=3
```

### Rips Filtration

```rust
use amari_topology::rips_filtration;

// Create Rips filtration from points with distance function
let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)];
let distance = |i: usize, j: usize| -> f64 {
    let (x1, y1) = points[i];
    let (x2, y2) = points[j];
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
};

let filt = rips_filtration(points.len(), 2, distance);
```

### Morse Theory

```rust
use amari_topology::{CriticalPoint, CriticalType, MorseComplex, find_critical_points_grid};

// Find critical points of f(x,y) = x² + y² on a grid
let f = |x: f64, y: f64| x * x + y * y;
let bounds = [(-1.0, 1.0), (-1.0, 1.0)];

let cps = find_critical_points_grid(f, &bounds, 20, 0.01).unwrap();

// Should find minimum near origin
let mins: Vec<_> = cps.iter().filter(|cp| cp.is_minimum()).collect();
assert!(!mins.is_empty());

// Build Morse complex
let morse = MorseComplex::new(cps);
let counts = morse.count_by_index();
```

### Manifold Boundaries

```rust
use amari_topology::{SimplicialComplex, Simplex, ManifoldBoundary, is_manifold};

// Single triangle has boundary (3 edges)
let mut triangle = SimplicialComplex::new();
triangle.add_simplex(Simplex::new(vec![0, 1, 2]));

let boundary = ManifoldBoundary::compute(&triangle);
assert!(!boundary.is_closed()); // Has boundary
assert_eq!(boundary.total_simplices(), 3);

// Hollow tetrahedron (sphere) is closed
let mut sphere = SimplicialComplex::new();
sphere.add_simplex(Simplex::new(vec![0, 1, 2]));
sphere.add_simplex(Simplex::new(vec![0, 1, 3]));
sphere.add_simplex(Simplex::new(vec![0, 2, 3]));
sphere.add_simplex(Simplex::new(vec![1, 2, 3]));

let boundary = ManifoldBoundary::compute(&sphere);
assert!(boundary.is_closed()); // No boundary

// Check manifold property
assert!(is_manifold(&triangle));
assert!(is_manifold(&sphere));
```

### Fiber Bundles

```rust
use amari_topology::{VectorBundle, Section, Connection, PrincipalBundle};

// Tangent bundle of a 3-manifold
let tangent = VectorBundle::tangent_bundle(3);
assert_eq!(tangent.rank, 3);

// Line bundle
let line_bundle = VectorBundle::trivial(2, 1);
assert!(line_bundle.is_line_bundle());

// Sections
let section = Section::new(vec![1.0, 2.0, 3.0]);
assert_eq!(section.at(1), Some(&2.0));

// Flat connection
let conn: Connection<f64> = Connection::flat(3);

// Principal bundles
let frame = PrincipalBundle::frame_bundle(3);
assert_eq!(frame.structure_group, "GL(3)");
```

### Phantom Types for Type Safety

The crate provides phantom types for compile-time verification of topological properties:

```rust
use amari_topology::{
    TypedFiltration, Validated, Unvalidated,
    IntegerCoefficients, Mod2Coefficients, RealCoefficients,
    Oriented, Unoriented,
};

// Filtration validation at compile time
let mut filt = TypedFiltration::<Unvalidated>::new();
filt.add(0.0, vec![0]);
filt.add(0.0, vec![1]);
filt.add(1.0, vec![0, 1]);

// Validation converts Unvalidated -> Validated
let validated: Option<TypedFiltration<Validated>> = filt.validate();
assert!(validated.is_some()); // Faces appear before cofaces

// Coefficient rings for homology
// - IntegerCoefficients (ℤ): Standard homology with torsion
// - Mod2Coefficients (ℤ/2ℤ): Non-orientable spaces, simpler signs
// - RealCoefficients (ℝ): When torsion isn't needed
```

### Verified Contracts

Creusot-style formal verification contracts ensure mathematical correctness:

```rust
use amari_topology::{
    VerifiedSimplex, VerifiedChain, VerifiedBoundaryMap,
    verify_boundary_squared_zero, verify_euler_poincare,
    verify_weak_morse_inequalities, verify_strong_morse_inequality,
    SimplicialComplex, Simplex, ChainGroup,
};

// Verified simplex maintains invariants
let simplex = VerifiedSimplex::new(vec![2, 0, 1]);
assert_eq!(simplex.vertices(), &[0, 1, 2]); // Sorted
assert!(simplex.orientation() == 1 || simplex.orientation() == -1);

// Verify the fundamental property ∂∂ = 0
let triangle = Simplex::new(vec![0, 1, 2]);
let c2 = ChainGroup::new(vec![triangle.clone()]);
let c1 = ChainGroup::new(triangle.faces(1));
let c0 = ChainGroup::new(triangle.faces(0));

let d2 = VerifiedBoundaryMap::from_chain_groups(&c2, &c1);
let d1 = VerifiedBoundaryMap::from_chain_groups(&c1, &c0);

assert!(verify_boundary_squared_zero(&d2, &d1));

// Verify Euler-Poincaré formula
let mut complex = SimplicialComplex::new();
complex.add_simplex(Simplex::new(vec![0, 1, 2]));
let betti = vec![1, 0]; // β₀ = 1, β₁ = 0
assert!(verify_euler_poincare(&complex, &betti));

// Verify Morse inequalities
let critical_counts = vec![1, 2, 1]; // Torus: 1 min, 2 saddles, 1 max
let betti_torus = vec![1, 2, 1];     // Torus: β₀=1, β₁=2, β₂=1
assert!(verify_weak_morse_inequalities(&critical_counts, &betti_torus));
assert!(verify_strong_morse_inequality(&critical_counts, &betti_torus));
```

### Parallel Computation

Enable the `parallel` feature for Rayon-based parallelism:

```rust,ignore
use amari_topology::{
    parallel_betti_numbers, parallel_rips_filtration,
    parallel_grid_evaluation, parallel_faces,
    SimplicialComplex, Simplex,
};

// Parallel Betti number computation
let mut complex = SimplicialComplex::new();
complex.add_simplex(Simplex::new(vec![0, 1, 2, 3])); // Tetrahedron
let betti = parallel_betti_numbers(&complex);

// Parallel Rips filtration construction
let points: Vec<(f64, f64)> = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 0.866)];
let distance = |i: usize, j: usize| {
    let (x1, y1) = points[i];
    let (x2, y2) = points[j];
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
};
let filt = parallel_rips_filtration(points.len(), 2, distance);

// Parallel grid evaluation (useful for Morse theory)
let f = |x: f64, y: f64| x * x + y * y;
let values = parallel_grid_evaluation(f, (-1.0, 1.0), (-1.0, 1.0), 100);

// Parallel face enumeration for large simplices
let big_simplex = Simplex::new(vec![0, 1, 2, 3, 4, 5, 6, 7]);
let faces_2 = parallel_faces(&big_simplex, 2); // All 2-faces in parallel
```

## Mathematical Background

### Homology

Homology groups H_k measure "holes" in different dimensions:
- H_0: Connected components
- H_1: 1-dimensional holes (loops)
- H_2: 2-dimensional voids (cavities)

The Betti numbers β_k = dim(H_k) count these features.

### Persistent Homology

Persistent homology tracks how topological features appear (birth) and disappear (death) as we vary a parameter. The persistence diagram plots (birth, death) pairs, with points far from the diagonal representing significant features.

### Morse Theory

Morse theory connects critical points of smooth functions to topology:
- Minima (index 0) create components
- Saddles (index k) create/destroy k-dimensional features
- Maxima (index n) create n-dimensional features

The Morse inequalities relate critical point counts to Betti numbers: c_k ≥ β_k.

### Fiber Bundles

A fiber bundle E → B with fiber F is locally a product B × F but may be globally twisted. Examples:
- Tangent bundle TM (velocity vectors on a manifold)
- Möbius strip (twisted line bundle over a circle)
- Frame bundle (all bases at each point)

### Phantom Types in Topology

The crate uses Rust's type system to encode topological properties at compile time:

| Category | Types | Purpose |
|----------|-------|---------|
| Orientation | `Oriented`, `Unoriented` | Track simplex/complex orientation |
| Coefficients | `IntegerCoefficients`, `Mod2Coefficients`, `RealCoefficients` | Homology coefficient ring |
| Boundary | `Closed`, `WithBoundary` | Complex boundary status |
| Connectivity | `Connected`, `Disconnected` | Path-connectivity |
| Validation | `Validated`, `Unvalidated` | Filtration face ordering |

This prevents runtime errors from invalid operations (e.g., computing oriented homology on unoriented data).

## Integration with Amari

This crate integrates with the broader Amari ecosystem:
- Uses `amari-core` for Clifford algebra types
- Uses `amari-calculus` for differential operators
- Complements `amari-measure` for integration on manifolds
- Works with `amari-functional` for function spaces on topological structures

## License

MIT OR Apache-2.0
