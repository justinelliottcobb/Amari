# amari-gpu

GPU acceleration for Amari mathematical computations using WebGPU.

## Overview

`amari-gpu` is an integration crate that provides GPU-accelerated implementations of mathematical operations from Amari domain crates. It follows the **progressive enhancement** pattern: operations automatically fall back to CPU computation when GPU is unavailable or for small workloads, scaling to GPU acceleration for large batch operations in production.

## Architecture

As an **integration crate**, `amari-gpu` consumes APIs from domain crates and exposes them to GPU platforms:

```
Domain Crates (provide APIs):
  amari-core → amari-measure → amari-calculus
  amari-info-geom, amari-relativistic, amari-network

Integration Crates (consume APIs):
  amari-gpu → depends on domain crates
  amari-wasm → depends on domain crates
```

**Dependency Rule**: Integration crates depend on domain crates, never the reverse.

## Current Integrations (v0.12.2)

### Implemented GPU Acceleration

| Domain Crate | Module | Operations | Status |
|-------------|--------|------------|--------|
| **amari-core** | `core` | Geometric algebra operations (G2, G3, G4), multivector products | ✅ Implemented |
| **amari-info-geom** | `info_geom` | Fisher metric, divergence computations, statistical manifolds | ✅ Implemented |
| **amari-relativistic** | `relativistic` | Minkowski space operations, Lorentz transformations | ✅ Implemented |
| **amari-network** | `network` | Graph operations, spectral methods | ✅ Implemented |
| **amari-measure** | `measure` | Measure theory computations, sigma-algebras | ✅ Implemented (feature: `measure`) |
| **amari-calculus** | `calculus` | Field evaluation, gradients, divergence, curl | ✅ Implemented (feature: `calculus`) |
| **amari-dual** | `dual` | Automatic differentiation GPU operations | ✅ Implemented (feature: `dual`) |
| **amari-enumerative** | `enumerative` | Intersection theory GPU operations | ✅ Implemented (feature: `enumerative`) |
| **amari-automata** | `automata` | Cellular automata GPU evolution | ✅ Implemented (feature: `automata`) |
| **amari-fusion** | `fusion` | Holographic memory operations, batch binding, similarity matrices | ✅ Implemented (feature: `fusion`) |

### Temporarily Disabled Modules

| Domain Crate | Module | Status | Reason |
|-------------|--------|--------|--------|
| amari-tropical | `tropical` | ❌ Disabled | Orphan impl rules - requires extension traits |

**Note**: If you were using `amari_gpu::tropical` in previous versions, this module is not available in v0.12.2. Use CPU implementations from `amari_tropical` directly until this module is restored in a future release.

## Features

```toml
[features]
default = []
std = ["amari-core/std", "amari-relativistic/std", "amari-info-geom/std"]
webgpu = ["wgpu/webgpu"]
high-precision = ["amari-core/high-precision", "amari-relativistic/high-precision"]
measure = ["dep:amari-measure"]
calculus = ["dep:amari-calculus"]
dual = ["dep:amari-dual"]
enumerative = ["dep:amari-enumerative"]
automata = ["dep:amari-automata"]
fusion = ["dep:amari-fusion"]  # Holographic memory GPU acceleration
# tropical = ["dep:amari-tropical"]  # Disabled - orphan impl rules
```

## Usage

### Basic Setup

```rust
use amari_gpu::unified::GpuContext;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU context
    let context = GpuContext::new().await?;

    // Use GPU-accelerated operations
    // ...

    Ok(())
}
```

### Calculus GPU Acceleration

```rust
use amari_gpu::calculus::GpuCalculus;
use amari_calculus::ScalarField;
use amari_core::Multivector;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU calculus
    let gpu_calculus = GpuCalculus::new().await?;

    // Define a scalar field (e.g., f(x,y,z) = x² + y² + z²)
    let field = ScalarField::new(|pos: &[f64; 3]| -> f64 {
        pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]
    });

    // Batch evaluate at 10,000 points (uses GPU)
    let points: Vec<[f64; 3]> = generate_point_grid(100, 100); // 10,000 points
    let values = gpu_calculus.batch_eval_scalar_field(&field, &points).await?;

    // Batch gradient computation (uses GPU for large batches)
    let gradients = gpu_calculus.batch_gradient(&field, &points, 1e-6).await?;

    Ok(())
}
```

### Holographic Memory GPU Acceleration

```rust
use amari_gpu::fusion::{HolographicGpuOps, GpuHolographicTDC};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GPU holographic operations
    let gpu_ops = HolographicGpuOps::new().await?;

    // Create GPU-compatible vectors
    let keys: Vec<GpuHolographicTDC> = (0..1000)
        .map(|i| GpuHolographicTDC {
            tropical: i as f32,
            dual_real: 1.0,
            dual_dual: 0.0,
            clifford: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            _padding: [0.0; 5],
        })
        .collect();

    let values = keys.clone();

    // Batch bind 1000 key-value pairs on GPU
    let bound = gpu_ops.batch_bind(&keys, &values).await?;
    println!("Bound {} pairs on GPU", bound.len());

    // Compute similarity matrix (1000x1000 = 1M similarities)
    let similarities = gpu_ops.batch_similarity(&keys, &keys, true).await?;
    println!("Computed {} similarities", similarities.len());

    // GPU resonator cleanup
    let noisy_input = &keys[0];
    let codebook = &keys[..100];
    let result = gpu_ops.resonator_cleanup(noisy_input, codebook).await?;
    println!("Best match: index {}, similarity {:.4}",
             result.best_index, result.best_similarity);

    Ok(())
}
```

#### Holographic GPU Operations

| Operation | Description | GPU Threshold |
|-----------|-------------|---------------|
| `batch_bind()` | Parallel geometric product binding | ≥ 100 pairs |
| `batch_similarity()` | Pairwise or matrix similarity computation | ≥ 100 vectors |
| `resonator_cleanup()` | Parallel codebook search for best match | ≥ 100 codebook entries |

#### WGSL Shaders

The holographic module includes optimized WGSL compute shaders:

- **`holographic_batch_bind`**: Cayley table-based geometric product for binding
- **`holographic_batch_similarity`**: Inner product with reverse `<A B̃>₀` for similarity
- **`holographic_bundle_all`**: Parallel reduction for vector superposition
- **`holographic_resonator_step`**: Parallel max-finding for cleanup

### Adaptive CPU/GPU Dispatch

The library automatically selects the optimal execution path:

```rust
// Small batch: Automatically uses CPU (< 1000 points for scalar fields)
let small_points = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
let values = gpu_calculus.batch_eval_scalar_field(&field, &small_points).await?;
// ↑ Executed on CPU (overhead of GPU transfer exceeds benefit)

// Large batch: Automatically uses GPU (≥ 1000 points)
let large_points = generate_point_grid(100, 100); // 10,000 points
let values = gpu_calculus.batch_eval_scalar_field(&field, &large_points).await?;
// ↑ Executed on GPU (parallel processing advantage)
```

### Batch Size Thresholds

| Operation | CPU Threshold | GPU Threshold |
|-----------|--------------|---------------|
| Scalar field evaluation | < 1000 points | ≥ 1000 points |
| Vector field evaluation | < 500 points | ≥ 500 points |
| Gradient computation | < 500 points | ≥ 500 points |
| Divergence/Curl | < 500 points | ≥ 500 points |
| Holographic binding | < 100 pairs | ≥ 100 pairs |
| Holographic similarity | < 100 vectors | ≥ 100 vectors |
| Resonator cleanup | < 100 codebook | ≥ 100 codebook |

## Implementation Status

### Holographic Module (v0.12.2)

**GPU Implementations** (✅ Complete):
- Batch binding with Cayley table geometric product
- Batch similarity using proper inner product `<A B̃>₀`
- Parallel reduction for vector bundling
- Resonator cleanup with parallel codebook search

**Types**:
- `GpuHolographicTDC`: GPU-compatible TropicalDualClifford representation
- `GpuResonatorOutput`: Cleanup result with best match info
- `HolographicGpuOps`: Main GPU operations struct

**Shaders**:
- `HOLOGRAPHIC_BATCH_BIND`: 64-thread workgroups for binding
- `HOLOGRAPHIC_BATCH_SIMILARITY`: 256-thread workgroups for similarity
- `HOLOGRAPHIC_BUNDLE_ALL`: Workgroup-shared memory reduction
- `HOLOGRAPHIC_RESONATOR_STEP`: 256-thread parallel max-finding

### Calculus Module (v0.12.2)

**CPU Implementations** (✅ Complete):
- Central finite differences for numerical derivatives
- Field evaluation at multiple points
- Gradient, divergence, and curl computation
- Step size: h = 1e-6 for numerical stability

**GPU Implementations** (⏸️ Future Work):
- WGSL compute shaders for parallel field evaluation
- Parallel finite difference computation
- Optimized memory layout for GPU transfer

**Current Behavior**:
- Infrastructure and pipelines are in place
- All operations currently use CPU implementations
- Shaders can be added incrementally without API changes

## Examples

See the `examples/` directory for complete examples:

```bash
# Run geometric algebra example
cargo run --example ga_operations

# Run information geometry example
cargo run --example fisher_metric

# Run calculus example (requires 'calculus' feature)
cargo run --features calculus --example field_ops
```

## Development

### Running Tests

```bash
# Run all tests
cargo test

# Run with specific features
cargo test --features calculus
cargo test --features measure

# Run GPU tests (requires GPU access)
cargo test --test gpu_integration
```

### Building Documentation

```bash
cargo doc --all-features --no-deps --open
```

## Future Work

### Short-term (v0.13.x)
1. Implement WGSL shaders for calculus operations
2. Add GPU benchmarks comparing CPU vs GPU performance
3. Optimize memory transfer patterns
4. Add more comprehensive examples
5. **Restore tropical GPU module** using extension traits (orphan impl fix)

### Medium-term (v0.14.x - v0.15.x)
1. Implement tropical algebra GPU operations
2. Multi-GPU support for large holographic memories
3. Performance optimization across all GPU modules
4. Unified GPU context sharing across all modules

### Long-term (v1.0.0+)
1. WebGPU backend for browser deployment
2. Multi-GPU support for distributed computation
3. Kernel fusion optimization
4. Custom WGSL shader compilation pipeline

## Performance Considerations

- **GPU Initialization**: ~100-200ms startup cost for context creation
- **Data Transfer**: Significant overhead for small batches (< 500 elements)
- **Optimal Use Cases**: Large batch operations (> 1000 elements)
- **Memory**: GPU buffers are sized for batch operations (dynamically allocated)

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| Linux | Vulkan | ✅ Tested |
| macOS | Metal | ✅ Supported (not regularly tested) |
| Windows | DirectX 12 / Vulkan | ✅ Supported (not regularly tested) |
| WebAssembly | WebGPU | ⏸️ Requires `webgpu` feature |

## Dependencies

- `wgpu` (v0.19): WebGPU implementation
- `bytemuck`: Zero-cost GPU buffer conversions
- `nalgebra`: Linear algebra operations
- `tokio`: Async runtime for GPU operations
- `futures`, `pollster`: Async utilities

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Areas of particular interest:

1. WGSL shader implementations for calculus operations
2. Performance benchmarks and optimization
3. Platform-specific testing and bug reports
4. Documentation improvements and examples

## References

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [wgpu Documentation](https://docs.rs/wgpu/)
- [Geometric Algebra GPU Acceleration](https://arxiv.org/abs/2103.00123) (example reference)
