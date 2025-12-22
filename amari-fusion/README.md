# amari-fusion

Tropical-Dual-Clifford fusion algebra with holographic associative memory.

## Overview

`amari-fusion` combines three algebraic systems into a unified framework:

- **Tropical Algebra**: Max-plus semiring operations for optimization and neural attention
- **Dual Numbers**: Forward-mode automatic differentiation for gradient computation
- **Clifford Algebra**: Geometric products and rotations for spatial reasoning

This fusion creates a powerful framework for neural network evaluation, optimization, and geometric machine learning. The `TropicalDualClifford` (TDC) type also serves as the foundation for **holographic associative memory** - a brain-inspired memory system that stores key-value pairs in superposition using Vector Symbolic Architecture (VSA) principles.

## Features

### Core Fusion Algebra

- **TropicalDualClifford**: Unified type combining tropical, dual, and Clifford components
- **Attention Mechanisms**: Tropical-optimized attention computation
- **Automatic Gradients**: Dual number derivatives through the full system
- **Geometric Features**: Clifford algebra for spatial relationships
- **Sensitivity Analysis**: Analyze parameter importance
- **Optimizer Integration**: Gradient-based optimization with geometric constraints

### Holographic Memory (v0.12.0+)

The `holographic` feature provides a complete implementation of holographic reduced representations:

- **Binding Operation** (`⊛`): Associates keys with values using geometric product
- **Bundling Operation** (`⊕`): Superimposes multiple associations into a single trace
- **Content-Addressable Retrieval**: Query with key to retrieve associated value
- **Graceful Degradation**: Partial or noisy queries still retrieve useful information
- **Resonator Cleanup**: Iterative refinement for improved retrieval accuracy
- **Capacity Tracking**: Automatic SNR estimation and capacity warnings

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

# Enable holographic memory
amari-fusion = { version = "0.12", features = ["holographic"] }

# With parallel processing
amari-fusion = { version = "0.12", features = ["rayon"] }
```

## Quick Start

### Basic TropicalDualClifford Usage

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

### Holographic Memory

```rust
use amari_fusion::holographic::{HolographicMemory, BindingAlgebra, Bindable};
use amari_fusion::TropicalDualClifford;

// Create a holographic memory
let mut memory = HolographicMemory::<f64, 8>::new(BindingAlgebra::default());

// Store key-value associations
let key1 = TropicalDualClifford::random();
let value1 = TropicalDualClifford::random();
memory.store(&key1, &value1);

let key2 = TropicalDualClifford::random();
let value2 = TropicalDualClifford::random();
memory.store(&key2, &value2);

// Retrieve with a key
let result = memory.retrieve(&key1);
println!("Confidence: {:.2}", result.confidence);
println!("Retrieved value similarity: {:.2}", result.value.similarity(&value1));

// Check capacity
let info = memory.capacity_info();
println!("Items stored: {}", info.item_count);
println!("Estimated SNR: {:.2}", info.estimated_snr);
```

### Resonator Cleanup

For noisy inputs, the resonator iteratively cleans up retrieved values:

```rust
use amari_fusion::holographic::{Resonator, ResonatorConfig};

// Create a codebook of clean reference vectors
let codebook: Vec<TropicalDualClifford<f64, 8>> = (0..10)
    .map(|_| TropicalDualClifford::random_vector())
    .collect();

// Configure the resonator
let config = ResonatorConfig {
    max_iterations: 10,
    convergence_threshold: 0.99,
    ..Default::default()
};

let resonator = Resonator::new(&codebook, config);

// Clean up a noisy input
let noisy_input = codebook[3].clone(); // Add some noise in practice
let result = resonator.cleanup(&noisy_input);

println!("Converged: {}", result.converged);
println!("Best match index: {}", result.best_match_index);
println!("Iterations: {}", result.iterations);
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

## Holographic Memory Theory

### Capacity

The holographic memory has a theoretical capacity of approximately:

```
C ≈ D / ln(D)
```

where `D` is the dimensionality (e.g., 8 for `TropicalDualClifford<_, 8>`). Beyond this capacity, retrieval quality degrades gracefully.

### Signal-to-Noise Ratio

The estimated SNR is:

```
SNR ≈ √(D / N)
```

where `N` is the number of stored items. Higher SNR means more reliable retrieval.

### Binding Algebra Laws

The binding operation satisfies important algebraic properties:

```rust
use amari_fusion::holographic::Bindable;

let a = TropicalDualClifford::<f64, 8>::random_vector();
let b = TropicalDualClifford::<f64, 8>::random_vector();

// Binding produces dissimilar results
let bound = a.bind(&b);
assert!(bound.similarity(&a).abs() < 0.3);
assert!(bound.similarity(&b).abs() < 0.3);

// Binding is approximately invertible for unit vectors
let recovered_b = bound.unbind(&a);
let b_similarity = recovered_b.similarity(&b);

// Identity element
let identity = TropicalDualClifford::<f64, 8>::binding_identity();
let with_identity = a.bind(&identity);
assert!(with_identity.similarity(&a) > 0.9);
```

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

### Holographic Associative Memory

- **Symbolic AI**: Store and retrieve symbolic associations
- **Memory Networks**: Implement differentiable key-value memory
- **Cognitive Architectures**: Brain-inspired associative memory
- **Embedding Retrieval**: Semantic similarity search
- **Sequence Modeling**: Store temporal associations

## Module Structure

```
amari-fusion/
├── src/
│   ├── lib.rs              # Main entry, TropicalDualClifford type
│   ├── types.rs            # Core fusion types
│   └── holographic/        # Holographic memory module (feature-gated)
│       ├── mod.rs          # Module exports
│       ├── binding.rs      # Bind/bundle operations, Bindable trait
│       ├── memory.rs       # HolographicMemory implementation
│       ├── resonator.rs    # Resonator cleanup
│       └── verified.rs     # Formal verification contracts
```

## Performance

The fusion system provides performance benefits from each component:

| Component | Benefit |
|-----------|---------|
| Tropical | O(n) vs O(n log n) for softmax-like operations |
| Dual | No graph storage, O(1) memory overhead |
| Clifford | Unified rotations, no gimbal lock |
| Holographic | O(D) storage per item, graceful degradation |

- **Batch Operations**: Use `store_batch()` for efficient bulk storage
- **Parallel Support**: Enable `rayon` feature for parallel bundle operations
- **GPU Acceleration**: See `amari-gpu` for GPU-accelerated holographic operations

## References

- Plate, T. A. (2003). *Holographic Reduced Representation*. CSLI Publications.
- Gayler, R. W. (2003). Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../LICENSE-MIT))

at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
