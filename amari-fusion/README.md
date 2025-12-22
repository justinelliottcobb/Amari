# amari-fusion

Tropical-Dual-Clifford fusion algebra with holographic associative memory.

## Overview

`amari-fusion` provides a unified mathematical framework that combines three powerful algebraic systems:

- **Tropical Algebra**: Max-plus semiring operations for optimization and neural attention
- **Dual Numbers**: Forward-mode automatic differentiation for gradient computation
- **Clifford Algebra**: Geometric products and rotations for spatial reasoning

Together, these form the `TropicalDualClifford` (TDC) type, which serves as the foundation for **holographic associative memory** - a brain-inspired memory system that stores key-value pairs in superposition using Vector Symbolic Architecture (VSA) principles.

## Features

### Core Fusion Algebra

- **TropicalDualClifford**: Unified type combining tropical, dual, and Clifford components
- **Seamless Interoperability**: Convert between constituent algebras
- **Rich Arithmetic**: Full support for addition, multiplication, and specialized operations

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

# Enable holographic memory
amari-fusion = { version = "0.12", features = ["holographic"] }
```

## Quick Start

### Basic TropicalDualClifford Usage

```rust
use amari_fusion::TropicalDualClifford;

// Create from logits (common in ML applications)
let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.8, 1.2];
let tdc = TropicalDualClifford::<f64, 8>::from_logits(&logits);

// Access components
let tropical_max = tdc.tropical().max_element();
let dual_gradient = tdc.dual().get(0);
let clifford_scalar = tdc.clifford().get(0);

// Compute similarity
let other = TropicalDualClifford::<f64, 8>::from_logits(&[0.5; 8]);
let similarity = tdc.similarity(&other);
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
if let Some(a_inv) = a.binding_inverse() {
    let recovered_b = bound.unbind(&a);
    let b_similarity = recovered_b.similarity(&b);
    println!("Recovery similarity: {:.2}", b_similarity);
}

// Identity element
let identity = TropicalDualClifford::<f64, 8>::binding_identity();
let with_identity = a.bind(&identity);
assert!(with_identity.similarity(&a) > 0.9);
```

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

### Use Cases

- **Symbolic AI**: Store and retrieve symbolic associations
- **Memory Networks**: Implement differentiable key-value memory
- **Cognitive Architectures**: Brain-inspired associative memory
- **Embedding Retrieval**: Semantic similarity search
- **Sequence Modeling**: Store temporal associations

## Feature Flags

```toml
[features]
default = []
holographic = []  # Enable holographic memory module
rayon = ["dep:rayon"]  # Parallel processing support
```

## Module Structure

```
amari-fusion/
├── src/
│   ├── lib.rs              # Main entry, TropicalDualClifford type
│   ├── types.rs            # Core fusion types
│   └── holographic/        # Holographic memory module
│       ├── mod.rs          # Module exports
│       ├── binding.rs      # Bind/bundle operations, Bindable trait
│       ├── memory.rs       # HolographicMemory implementation
│       ├── resonator.rs    # Resonator cleanup
│       └── verified.rs     # Formal verification contracts
```

## Performance

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
