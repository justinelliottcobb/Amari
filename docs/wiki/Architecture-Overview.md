# Architecture Overview

This document describes the high-level architecture of the Amari library.

## Design Principles

### 1. Compile-Time Safety

Amari uses Rust's type system extensively to catch errors at compile time:

- **Const generics** for signature parameters `<P, Q, R>`
- **Phantom types** for tracking algebraic properties
- **Sealed traits** for controlled extension points

```rust
// Signature is enforced at compile time
type Cl3 = Multivector<3, 0, 0>;  // 3D Euclidean
type Cl31 = Multivector<3, 1, 0>; // Spacetime algebra

// These cannot be mixed - compile error
// let bad = cl3_value.geometric_product(&cl31_value);
```

### 2. No-Std Compatibility

Core crates work without the standard library:

```rust
#![no_std]
extern crate alloc;

use amari_core::Multivector;
```

Features requiring `std`:
- File I/O
- Threading (use `rayon` feature)
- Some floating-point operations

### 3. Layered Architecture

```
┌─────────────────────────────────────────────────┐
│                  Applications                    │
│         (amari-wasm, examples-suite)            │
├─────────────────────────────────────────────────┤
│                 Domain Crates                    │
│  (dynamics, relativistic, holographic, etc.)    │
├─────────────────────────────────────────────────┤
│              Analysis Crates                     │
│  (calculus, topology, functional, measure)      │
├─────────────────────────────────────────────────┤
│              Algebraic Crates                    │
│     (core, dual, tropical, fusion)              │
├─────────────────────────────────────────────────┤
│              Infrastructure                      │
│           (gpu, optimization)                    │
└─────────────────────────────────────────────────┘
```

## Crate Dependencies

```
amari (umbrella)
├── amari-core
├── amari-dual ← amari-core
├── amari-tropical ← amari-core
├── amari-calculus ← amari-core
├── amari-topology ← amari-core
├── amari-functional ← amari-core, amari-calculus
├── amari-info-geom ← amari-core, amari-functional
├── amari-optimization ← amari-core, amari-dual
├── amari-network ← amari-core, amari-tropical
├── amari-probabilistic ← amari-core, amari-measure
├── amari-measure ← amari-core
├── amari-relativistic ← amari-core, amari-calculus
├── amari-holographic ← amari-core, amari-relativistic
├── amari-enumerative ← amari-core, amari-topology
├── amari-automata ← amari-core, amari-tropical
├── amari-fusion ← amari-core, amari-dual, amari-tropical
├── amari-dynamics ← amari-core, amari-calculus, amari-functional
├── amari-gpu ← all applicable crates
└── amari-wasm ← all crates (WASM bindings)
```

## Core Types

### Multivector<P, Q, R>

The fundamental type representing elements of Clifford algebra Cl(p,q,r):

```rust
pub struct Multivector<const P: usize, const Q: usize, const R: usize> {
    coefficients: [f64; { 1 << (P + Q + R) }],
}
```

Storage is fixed-size array with 2^n elements where n = P + Q + R.

### Phantom Types

Used throughout for compile-time verification:

```rust
// Dimensional markers
pub struct Positive;
pub struct Negative;
pub struct Zero;

// Algebraic property markers
pub struct Verified;
pub struct Unverified;

// Stability markers (amari-dynamics)
pub struct Stable;
pub struct Unstable;
pub struct UnknownStability;
```

## GPU Architecture

### Adaptive Compute

The GPU layer uses adaptive computation:

```rust
pub struct AdaptiveCompute {
    gpu: Option<GpuCliffordAlgebra>,
}

impl AdaptiveCompute {
    pub async fn batch_geometric_product(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if self.gpu.is_some() && a.len() >= GPU_THRESHOLD {
            self.gpu_compute(a, b).await
        } else {
            self.cpu_compute(a, b)
        }
    }
}
```

### Shader Pipeline

WGSL shaders for parallel computation:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Input      │────▶│   Compute    │────▶│   Output     │
│   Buffers    │     │   Shader     │     │   Buffers    │
└──────────────┘     └──────────────┘     └──────────────┘
```

## WASM Architecture

### Binding Strategy

JavaScript bindings via wasm-bindgen:

```rust
#[wasm_bindgen]
pub struct WasmMultivector {
    inner: Multivector<3, 0, 0>,
}

#[wasm_bindgen]
impl WasmMultivector {
    #[wasm_bindgen(constructor)]
    pub fn new(coefficients: &[f64]) -> Result<WasmMultivector, JsValue> {
        // ...
    }

    pub fn geometric_product(&self, other: &WasmMultivector) -> WasmMultivector {
        // ...
    }
}
```

### Memory Management

- WASM linear memory for large arrays
- Automatic cleanup via Drop
- Explicit `free()` methods for JavaScript control

## Testing Strategy

### Unit Tests

Each module has comprehensive unit tests:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_geometric_product_associativity() {
        // Property: (ab)c = a(bc)
    }
}
```

### Integration Tests

Cross-crate integration in `tests/` directory.

### Property-Based Tests

Using proptest for mathematical properties:

```rust
proptest! {
    #[test]
    fn geometric_product_distributes(a: Cl3, b: Cl3, c: Cl3) {
        // a(b + c) = ab + ac
        prop_assert_relative_eq!(
            a.geometric_product(&(b + c)),
            a.geometric_product(&b) + a.geometric_product(&c)
        );
    }
}
```

### Benchmarks

Criterion benchmarks for performance tracking:

```rust
fn bench_geometric_product(c: &mut Criterion) {
    c.bench_function("geo_product_3d", |b| {
        b.iter(|| e1.geometric_product(&e2))
    });
}
```

## Error Handling

### Error Types

Each crate defines domain-specific errors:

```rust
#[derive(Debug, thiserror::Error)]
pub enum DynamicsError {
    #[error("Integration failed: {0}")]
    IntegrationFailed(String),

    #[error("System not in domain")]
    OutOfDomain,

    #[error("Numerical instability detected")]
    NumericalInstability,
}
```

### Result Types

Consistent use of Result with crate-specific errors:

```rust
pub type DynamicsResult<T> = Result<T, DynamicsError>;
```

## Feature Flags

Common features across crates:

| Feature | Description |
|---------|-------------|
| `std` | Standard library support |
| `parallel` | Rayon parallelization |
| `simd` | SIMD optimizations |
| `serde` | Serialization support |
| `contracts` | Creusot formal verification |

## Performance Considerations

### Memory Layout

- Coefficients stored contiguously for cache efficiency
- Aligned allocation for SIMD operations
- Stack allocation for small multivectors

### Computation

- Cayley tables precomputed for common signatures
- Lazy evaluation where beneficial
- GPU offloading for large batches (>100 elements)
