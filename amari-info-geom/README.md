# amari-info-geom

Information geometry and statistical manifolds for the Amari library.

## Overview

`amari-info-geom` implements the foundational concepts of information geometry, the study of probability distributions as points on a Riemannian manifold. This crate provides tools for working with Fisher metrics, α-connections, Bregman divergences, and the Amari-Chentsov tensor structure.

Named after Shun-ichi Amari, the pioneer of information geometry.

## Features

- **Fisher Information Metric**: Riemannian metric on statistical manifolds
- **α-Connections**: Family of connections parameterized by α ∈ [-1, +1]
- **Bregman Divergences**: Information-theoretic divergence measures
- **Statistical Manifolds**: Geometric structures on probability spaces
- **GPU Acceleration**: Optional GPU support for large-scale computations
- **Multivector Integration**: Information geometry on Clifford algebra spaces

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-info-geom = "0.12"
```

### Feature Flags

```toml
[dependencies]
# Default features
amari-info-geom = "0.12"

# With GPU acceleration
amari-info-geom = { version = "0.12", features = ["gpu"] }

# High-precision arithmetic
amari-info-geom = { version = "0.12", features = ["high-precision"] }
```

## Quick Start

```rust
use amari_info_geom::{kl_divergence, js_divergence, bregman_divergence};
use amari_core::Multivector;

type Cl3 = Multivector<3, 0, 0>;

// Create probability-like multivectors
let p = Cl3::from_coefficients(&[0.5, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]);
let q = Cl3::from_coefficients(&[0.4, 0.4, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]);

// Compute divergences
let kl = kl_divergence(&p, &q)?;
let js = js_divergence(&p, &q)?;

// Bregman divergence with custom potential
let phi = |mv: &Cl3| mv.norm_squared();
let breg = bregman_divergence(phi, &p, &q)?;
```

## Mathematical Background

### Statistical Manifolds

A statistical manifold S is a set of probability distributions with a smooth structure:

```
S = { p_θ : θ ∈ Θ ⊂ ℝⁿ }
```

### Fisher Information Metric

The Fisher information metric defines a Riemannian structure:

```
g_ij(θ) = E_θ[∂_i log p_θ · ∂_j log p_θ]
```

This metric measures the "distance" between nearby distributions.

### α-Connections

A family of affine connections parameterized by α:

- α = +1: e-connection (exponential family)
- α = 0: Levi-Civita connection
- α = -1: m-connection (mixture family)

### Dually Flat Manifolds

When α = +1 and α = -1 connections are both flat, the manifold is dually flat. This structure underlies:

- Exponential families
- Mixture families
- Bregman divergences

### Divergences

| Divergence | Formula | Properties |
|------------|---------|------------|
| KL | D_KL(p‖q) = Σ p log(p/q) | Asymmetric, non-negative |
| JS | D_JS(p‖q) = ½D_KL(p‖m) + ½D_KL(q‖m) | Symmetric, bounded |
| Bregman | D_φ(p‖q) = φ(p) - φ(q) - ⟨∇φ(q), p-q⟩ | Generalizes squared distance |

## Key Types

### Parameter Trait

Objects usable as parameters on statistical manifolds:

```rust
use amari_info_geom::Parameter;

// Multivectors implement Parameter
let mv = Multivector::<3, 0, 0>::zero();
let dim = mv.dimension();
let component = mv.get_component(0);
```

### GPU Operations

```rust
#[cfg(feature = "gpu")]
use amari_info_geom::{InfoGeomGpuOps, GpuStatisticalManifold};

let gpu_ops = InfoGeomGpuOps::new().await?;
let manifold = GpuStatisticalManifold::new(/* ... */);
let fisher = gpu_ops.compute_fisher_metric(&manifold).await?;
```

## Modules

| Module | Description |
|--------|-------------|
| `gpu` | GPU-accelerated operations (feature-gated) |
| `verified_contracts` | Formal verification contracts |

## Applications

### Machine Learning

- **Natural Gradient Descent**: Optimization respecting manifold geometry
- **Variational Inference**: KL minimization for approximate inference
- **Wasserstein Distance**: Optimal transport metrics

### Statistics

- **Model Selection**: Comparing probability models
- **Hypothesis Testing**: Geometric interpretation of tests
- **Sufficient Statistics**: Exponential family structure

### Physics

- **Thermodynamics**: Free energy and entropy
- **Quantum Information**: Quantum statistical manifolds
- **Maximum Entropy**: Inference with constraints

## Theoretical Background

The crate is based on Amari's foundational work:

1. **Dual Geometry**: e-flat and m-flat coordinate systems
2. **Pythagorean Theorem**: D(p‖r) = D(p‖q) + D(q‖r) for orthogonal q
3. **Amari-Chentsov Tensor**: Unique invariant structure on statistical manifolds

## References

- Amari, S. (2016). *Information Geometry and Its Applications*
- Amari, S., & Nagaoka, H. (2000). *Methods of Information Geometry*

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library, named in honor of Shun-ichi Amari's contributions to information geometry.
