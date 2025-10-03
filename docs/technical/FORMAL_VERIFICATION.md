# Formal Verification Strategy for Amari Mathematical Library

## Overview

This document outlines the systematic approach to formally verify the mathematical correctness of the Amari library using:
1. **Phantom Types** - Encode mathematical invariants at the type level
2. **Creusot** - Deductive verification framework for Rust
3. **Refinement Types** - Express precise mathematical properties

## Goals

- **Prove mathematical invariants** at compile time
- **Prevent invalid operations** through type-level constraints
- **Verify algebraic laws** (associativity, distributivity, etc.)
- **Ensure dimensional consistency** in geometric operations
- **Guarantee numerical stability** properties

## Verification Roadmap

### Phase 1: Foundation (amari-core)
- [ ] Phantom types for Clifford algebra signatures (p,q,r)
- [ ] Invariants for multivector grade consistency
- [ ] Geometric product laws verification
- [ ] Basis blade orthogonality constraints

### Phase 2: Algebraic Structures (amari-tropical, amari-dual)
- [ ] Semiring laws for tropical arithmetic
- [ ] Dual number algebraic properties
- [ ] Automatic differentiation correctness

### Phase 3: Geometric Invariants (amari-enumerative)
- [ ] Dimension bounds for varieties
- [ ] Intersection theory constraints
- [ ] Degree calculations correctness
- [ ] Moduli space stability conditions

### Phase 4: Information Geometry (amari-info-geom)
- [ ] Manifold tangent space properties
- [ ] Metric tensor positive definiteness
- [ ] Connection compatibility conditions

## Phantom Type Design Patterns

### 1. Dimension Tracking
```rust
use std::marker::PhantomData;

// Phantom types for compile-time dimension checking
struct Dim<const N: usize>;

pub struct Vector<T, const D: usize> {
    data: Vec<T>,
    _phantom: PhantomData<Dim<D>>,
}

impl<T, const D: usize> Vector<T, D> {
    // Only allows addition of same-dimension vectors
    pub fn add(self, other: Vector<T, D>) -> Vector<T, D> {
        // Implementation
    }
}
```

### 2. Algebraic Signature Encoding
```rust
// Encode Clifford algebra signature at type level
pub struct Signature<const P: usize, const Q: usize, const R: usize>;

pub struct Multivector<T, const P: usize, const Q: usize, const R: usize> {
    coefficients: Vec<T>,
    _signature: PhantomData<Signature<P, Q, R>>,
}
```

### 3. Grade Constraints
```rust
// Phantom type for multivector grade
pub struct Grade<const G: usize>;

pub struct KVector<T, const G: usize, Sig> {
    data: Vec<T>,
    _grade: PhantomData<Grade<G>>,
    _sig: PhantomData<Sig>,
}
```

## Creusot Annotations

### Pre/Post Conditions
```rust
use creusot_contracts::*;

#[requires(v1.len() == v2.len())]
#[ensures(result.len() == v1.len())]
pub fn vector_add(v1: &Vec<f64>, v2: &Vec<f64>) -> Vec<f64> {
    // Implementation
}
```

### Loop Invariants
```rust
#[invariant(i <= n)]
#[invariant(result == factorial(i))]
for i in 1..=n {
    result *= i;
}
```

### Mathematical Properties
```rust
// Verify associativity
#[ensures(
    geometric_product(a, geometric_product(b, c)) ==
    geometric_product(geometric_product(a, b), c)
)]
pub fn geometric_product(a: &Multivector, b: &Multivector) -> Multivector {
    // Implementation
}
```

## Verification Targets

### amari-core
1. **Clifford Algebra Laws**
   - Anticommutativity of basis vectors
   - Geometric product associativity
   - Distributivity over addition

2. **Rotor Properties**
   - Unit norm preservation
   - Composition as rotation composition
   - Inverse existence

3. **Grade Projection**
   - Grade preservation under projection
   - Orthogonality of different grades

### amari-tropical
1. **Tropical Semiring**
   - Associativity of ⊕ (min/max)
   - Associativity of ⊗ (addition)
   - Distributivity of ⊗ over ⊕

2. **Tropical Polynomial Properties**
   - Newton polygon correspondence
   - Tropical root preservation

### amari-enumerative
1. **Intersection Theory**
   - Bézout's theorem bounds
   - Dimension formula correctness
   - Degree calculations

2. **Moduli Spaces**
   - Stability conditions (2g-2+n > 0)
   - Dimension formulas
   - Compactification properties

## Implementation Strategy

### Step 1: Core Type Definitions
Start with amari-core, adding phantom types to existing structures without breaking the API.

### Step 2: Creusot Setup
1. Install Creusot: `cargo install --git https://github.com/xldenis/creusot`
2. Add creusot-contracts dependency
3. Configure verification targets

### Step 3: Incremental Verification
- Begin with simple invariants (dimension checks)
- Progress to algebraic laws
- Finally tackle complex geometric properties

### Step 4: Documentation
- Document all phantom type invariants
- Explain verification conditions
- Provide proof sketches for complex properties

## Example: Verified Multivector

```rust
use std::marker::PhantomData;
use creusot_contracts::*;

/// Phantom type encoding Clifford algebra signature
pub struct CliffordSignature<const P: usize, const Q: usize, const R: usize>;

/// Type-level grade marker
pub struct Grade<const G: usize>;

/// Verified multivector with compile-time signature and runtime contracts
pub struct VerifiedMultivector<T, const P: usize, const Q: usize, const R: usize> {
    /// Coefficients in lexicographic basis order
    coefficients: Vec<T>,
    /// Phantom data for signature
    _signature: PhantomData<CliffordSignature<P, Q, R>>,
}

impl<T: Field, const P: usize, const Q: usize, const R: usize>
    VerifiedMultivector<T, P, Q, R>
{
    const BASIS_SIZE: usize = 1 << (P + Q + R);

    #[requires(coefficients.len() == Self::BASIS_SIZE)]
    pub fn new(coefficients: Vec<T>) -> Self {
        Self {
            coefficients,
            _signature: PhantomData,
        }
    }

    #[ensures(result.grade() <= P + Q + R)]
    pub fn grade(&self) -> usize {
        // Compute highest non-zero grade
    }

    /// Geometric product with verified associativity
    #[ensures(
        // Associativity property
        forall<a: Self, b: Self, c: Self>
            a.geometric_product(&b.geometric_product(&c)) ==
            a.geometric_product(&b).geometric_product(&c)
    )]
    pub fn geometric_product(&self, other: &Self) -> Self {
        // Implementation with Cayley table
    }

    /// Grade projection with verified idempotence
    #[requires(grade <= P + Q + R)]
    #[ensures(result.project_grade(grade).project_grade(grade) == result.project_grade(grade))]
    pub fn project_grade(&self, grade: usize) -> Self {
        // Extract k-vector component
    }
}

/// Type-safe rotor with unit norm guarantee
pub struct VerifiedRotor<T, const P: usize, const Q: usize, const R: usize> {
    multivector: VerifiedMultivector<T, P, Q, R>,
    _phantom: PhantomData<()>,
}

impl<T: Field, const P: usize, const Q: usize, const R: usize>
    VerifiedRotor<T, P, Q, R>
{
    #[requires(mv.norm() - 1.0 < EPSILON)]
    #[requires(mv.is_even_grade())]
    pub fn new(mv: VerifiedMultivector<T, P, Q, R>) -> Result<Self, Error> {
        // Verify rotor conditions
    }

    #[ensures(result.norm() == 1.0)]
    pub fn compose(&self, other: &Self) -> Self {
        // Rotor composition preserves unit norm
    }
}
```

## Testing Strategy

1. **Property-based testing** with quickcheck
2. **Symbolic verification** for small dimensions
3. **Numerical validation** against known results
4. **Counterexample generation** for failed proofs

## Success Metrics

- [ ] 100% of dimension checks verified at compile time
- [ ] Core algebraic laws proven with Creusot
- [ ] Zero runtime panics from mathematical violations
- [ ] API maintains ergonomics while adding safety

## References

- [Creusot: A Foundry for the Deductive Verification of Rust Programs](https://github.com/xldenis/creusot)
- [Phantom Types in Rust](https://doc.rust-lang.org/nomicon/phantom-data.html)
- [Geometric Algebra for Computer Science](https://geometricalgebra.org)
- [Formal Methods for Cryptographic Protocol Analysis](https://www.di.ens.fr/~blanchet/publications/)