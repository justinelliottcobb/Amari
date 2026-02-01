# Getting Started with Amari

This guide will help you get started with Amari's geometric algebra capabilities.

## Installation

Add Amari to your `Cargo.toml`:

```toml
[dependencies]
amari = "1.0"
```

Or use specific crates:

```toml
[dependencies]
amari-core = "1.0"
amari-dual = "1.0"
```

## Basic Concepts

### Multivectors

The fundamental type in Amari is the `Multivector<P, Q, R>` where:
- `P`: Number of basis vectors that square to +1
- `Q`: Number of basis vectors that square to -1
- `R`: Number of basis vectors that square to 0

```rust
use amari_core::Multivector;

// 3D Euclidean space: Cl(3,0,0)
type Cl3 = Multivector<3, 0, 0>;

// Create basis vectors
let e1 = Cl3::basis_vector(0);
let e2 = Cl3::basis_vector(1);
let e3 = Cl3::basis_vector(2);

// Geometric product
let e12 = e1.geometric_product(&e2);

// Scalar
let scalar = Cl3::scalar(2.5);
```

### Products

Amari supports all standard Clifford algebra products:

```rust
// Geometric product (default)
let ab = a.geometric_product(&b);

// Outer (wedge) product
let a_wedge_b = a.outer_product(&b);

// Inner (dot) product
let a_dot_b = a.inner_product(&b);

// Left/Right contractions
let left = a.left_contraction(&b);
let right = a.right_contraction(&b);

// Scalar product
let scalar = a.scalar_product(&b);
```

### Operators

```rust
// Grade involution
let involuted = mv.grade_involution();

// Reversion
let reversed = mv.reverse();

// Clifford conjugate
let conjugate = mv.clifford_conjugate();

// Dual
let dual = mv.dual();

// Inverse (for invertible elements)
let inv = mv.inverse().expect("Not invertible");
```

## Automatic Differentiation

Use `amari-dual` for automatic differentiation:

```rust
use amari_dual::DualNumber;

// Create a variable
let x = DualNumber::variable(2.0);

// Compute f(x) = x^2 + sin(x)
let f = x * x + x.sin();

// Get value and derivative
println!("f(2) = {}", f.real);      // Value
println!("f'(2) = {}", f.dual);     // Derivative
```

### Multivariate Gradients

```rust
use amari_dual::MultiDualNumber;

// Create variables for gradient computation
let x = MultiDualNumber::variable(1.0, 0, 2);  // Variable 0 of 2
let y = MultiDualNumber::variable(2.0, 1, 2);  // Variable 1 of 2

// f(x,y) = x^2 + x*y
let f = x.clone() * x.clone() + x * y;

// Get gradient
let grad = f.gradient;  // [2x + y, x] = [4, 1]
```

## Tropical Algebra

Use `amari-tropical` for max-plus semiring operations:

```rust
use amari_tropical::TropicalNumber;

let a = TropicalNumber::new(2.0);
let b = TropicalNumber::new(3.0);

// Tropical addition: max(a, b)
let sum = a.tropical_add(&b);  // 3.0

// Tropical multiplication: a + b
let prod = a.tropical_mul(&b);  // 5.0
```

## Geometric Calculus

Use `amari-calculus` for differential operators:

```rust
use amari_calculus::{gradient, divergence, curl, laplacian, ScalarField, VectorField};
use amari_core::Multivector;

type Cl3 = Multivector<3, 0, 0>;

// Define a scalar field: f(x,y,z) = x^2 + y^2 + z^2
let f = ScalarField::<3, 0, 0>::new(|x| {
    x.iter().map(|xi| xi * xi).sum()
});

// Compute gradient at a point
let point = vec![1.0, 2.0, 3.0];
let grad = gradient(&f, &point);

// Define a vector field
let v = VectorField::<3, 0, 0>::new(|x| {
    Cl3::basis_vector(0) * x[0] + Cl3::basis_vector(1) * x[1]
});

// Compute divergence
let div = divergence(&v, &point);
```

## WebAssembly

Amari provides WASM bindings for browser use:

```javascript
import init, { WasmMultivector } from 'amari-wasm';

await init();

const mv = new WasmMultivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
const e1 = WasmMultivector.basis_vector(0);
const result = mv.geometric_product(e1);
```

## GPU Acceleration

For large computations, use GPU acceleration:

```rust
use amari_gpu::{AdaptiveCompute, GpuCliffordAlgebra};

// Adaptive compute automatically chooses CPU or GPU
let adaptive = AdaptiveCompute::new().await?;

// Batch geometric products
let results = adaptive.batch_geometric_product(&coeffs_a, &coeffs_b).await?;
```

## Next Steps

- Explore [[Architecture Overview]] for design details
- See [[v1.0.0 Release Notes]] for the latest features
- Check [API Documentation](https://docs.rs/amari) for complete reference
