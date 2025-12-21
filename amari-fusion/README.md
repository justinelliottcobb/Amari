# amari-fusion

Tropical-Dual-Clifford fusion system for combining algebraic structures.

## Overview

`amari-fusion` combines three algebraic systems into a unified framework:

- **Tropical Algebra**: Converts softmax operations to efficient max operations
- **Dual Numbers**: Provides automatic differentiation without computational graphs
- **Clifford Algebra**: Handles geometric relationships and rotations

This fusion creates a powerful framework for neural network evaluation, optimization, and geometric machine learning.

## Features

- **TropicalDualClifford**: Unified type combining all three algebras
- **Attention Mechanisms**: Tropical-optimized attention computation
- **Automatic Gradients**: Dual number derivatives through the full system
- **Geometric Features**: Clifford algebra for spatial relationships
- **Sensitivity Analysis**: Analyze parameter importance
- **Optimizer Integration**: Gradient-based optimization with geometric constraints

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-fusion = "0.12"
```

### Feature Flags

```toml
[dependencies]
# Default features
amari-fusion = "0.12"

# With serialization
amari-fusion = { version = "0.12", features = ["serialize"] }

# With GPU acceleration
amari-fusion = { version = "0.12", features = ["gpu"] }

# High-precision arithmetic
amari-fusion = { version = "0.12", features = ["high-precision"] }
```

## Quick Start

```rust
use amari_fusion::TropicalDualClifford;

// Create from logits (common in ML applications)
let logits = vec![1.5, 2.0, 0.8, 1.2];
let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

// Evaluate against another TDC
let other = TropicalDualClifford::from_logits(&[2.0, 1.5, 1.0, 0.9]);
let evaluation = tdc.evaluate(&other);

println!("Combined score: {}", evaluation.combined_score);

// Extract features from each algebra
let tropical_features = tdc.extract_tropical_features();
let dual_features = tdc.extract_dual_features();

// Sensitivity analysis
let sensitivity = tdc.sensitivity_analysis();
let most_sensitive = sensitivity.most_sensitive(2);
println!("Most sensitive components: {:?}", most_sensitive);
```

## The Three Algebras

### Tropical Phase

Fast approximation using max-plus operations:

```
softmax(x) ≈ max(x)  (in log-space)
```

Benefits:
- O(n) complexity instead of O(n log n)
- No overflow issues
- Identifies dominant features quickly

### Dual Phase

Exact gradient computation:

```rust
// Automatic derivatives without backpropagation graphs
let grads = tdc.extract_dual_features();
```

Benefits:
- No computational graph storage
- Exact derivatives (no numerical error)
- Forward-mode efficiency for few outputs

### Clifford Phase

Geometric relationships:

```rust
// Spatial reasoning in the unified system
let geometric = tdc.clifford_repr();
```

Benefits:
- Rotation and reflection handling
- Coordinate-free computations
- Unified treatment of scalars, vectors, bivectors

## Key Types

### TropicalDualClifford<T, DIM>

The main fusion type:

```rust
use amari_fusion::TropicalDualClifford;

// Create from logits
let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);

// Get representations
let tropical = tdc.tropical_repr();
let dual = tdc.dual_repr();
let clifford = tdc.clifford_repr();

// Interpolation between two TDCs
let interpolated = tdc.interpolate(&other, 0.5);
```

### EvaluationResult

Results from evaluating two TDCs:

```rust
let result = tdc.evaluate(&other);

// Scores from each algebra
let tropical_score = result.tropical_score;
let dual_score = result.dual_score;
let clifford_score = result.clifford_score;

// Combined score
let combined = result.combined_score;
```

## Modules

| Module | Description |
|--------|-------------|
| `types` | TropicalDualClifford and related types |
| `attention` | Tropical-optimized attention mechanisms |
| `evaluation` | Evaluation and comparison functions |
| `optimizer` | Gradient-based optimization with TDC |
| `verified` | Compile-time verification |

## Use Cases

### Neural Network Optimization

```rust
use amari_fusion::{TropicalDualClifford, optimizer::TDCOptimizer};

let initial = TropicalDualClifford::from_logits(&params);

let optimizer = TDCOptimizer::new()
    .with_tropical_warmup(5)     // Fast tropical approximation
    .with_dual_refinement(10)    // Exact dual gradients
    .with_clifford_projection(); // Geometric constraints

let optimized = optimizer.optimize(&initial, &loss_fn)?;
```

### Attention Mechanisms

```rust
use amari_fusion::attention;

// Efficient attention using tropical algebra
let attention_weights = attention::tropical_attention(&queries, &keys);
```

### Sensitivity Analysis

```rust
let sensitivity = tdc.sensitivity_analysis();

// Find which parameters matter most
let important = sensitivity.most_sensitive(3);
let least_important = sensitivity.least_sensitive(3);
```

## Performance

The fusion system provides performance benefits from each component:

| Component | Benefit |
|-----------|---------|
| Tropical | O(n) vs O(n log n) for softmax-like operations |
| Dual | No graph storage, O(1) memory overhead |
| Clifford | Unified rotations, no gimbal lock |

## Mathematical Background

The TDC system operates in three phases:

1. **Tropical Warmup**: Fast approximation using max-plus
2. **Dual Refinement**: Compute exact gradients
3. **Clifford Projection**: Enforce geometric constraints

This multi-phase approach combines speed (tropical), accuracy (dual), and geometric awareness (Clifford).

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
