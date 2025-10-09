# Amari Network Analysis

**Geometric network analysis using Clifford algebra and tropical algebra**

[![Crates.io](https://img.shields.io/crates/v/amari-network.svg)](https://crates.io/crates/amari-network)
[![Documentation](https://docs.rs/amari-network/badge.svg)](https://docs.rs/amari-network)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/justinelliottcobb/Amari)

## Overview

`amari-network` provides advanced graph and network analysis tools where nodes are embedded in Clifford algebra (geometric algebra) space. This unique approach enables:

- **Geometric distance metrics** between nodes using multivector norms
- **Community detection** via geometric clustering in high-dimensional spaces
- **Information diffusion** modeling using geometric products
- **Efficient path-finding** with tropical (max-plus) algebra optimization
- **Multi-scale centrality measures** that capture geometric properties

## Mathematical Foundation

### Clifford Algebra (Geometric Algebra)

Networks are embedded in Clifford algebra spaces **Cl(P,Q,R)** with signature **(P,Q,R)**:
- **P**: basis vectors that square to +1 (Euclidean dimensions)
- **Q**: basis vectors that square to -1 (Minkowski-like dimensions)
- **R**: basis vectors that square to 0 (null/degenerate dimensions)

Each node is represented as a **multivector** combining scalars, vectors, bivectors, and higher-grade elements.

### Tropical Algebra (Max-Plus)

Shortest path optimization uses tropical arithmetic where:
- **Addition** ⊕ becomes **max** operation
- **Multiplication** ⊗ becomes **addition**
- **Zero** element is **-∞** (no connection)
- **One** element is **0** (self-distance)

This transforms shortest path problems into elegant matrix operations in the tropical semiring.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-network = "0.9.0"
```

### Basic Example

```rust
use amari_network::{GeometricNetwork, NodeMetadata};
use amari_core::Vector;

// Create a network in 3D Euclidean space (signature 3,0,0)
let mut network = GeometricNetwork::<3, 0, 0>::new();

// Add nodes at specific geometric positions
let node1 = network.add_node_with_metadata(
    Vector::from_components(1.0, 0.0, 0.0).mv,
    NodeMetadata::with_label("Node 1").with_property("importance", 0.8)
);

let node2 = network.add_node_with_metadata(
    Vector::from_components(0.0, 1.0, 0.0).mv,
    NodeMetadata::with_label("Node 2").with_property("importance", 0.9)
);

// Connect nodes with weighted edges
network.add_edge(node1, node2, 1.0)?;

// Compute geometric distance using Clifford algebra
let distance = network.geometric_distance(node1, node2)?;
println!("Geometric distance: {:.2}", distance); // Should be 1.414 (√2)

// Find communities using geometric clustering
let communities = network.find_communities(2)?;

// Simulate information diffusion
let diffusion = network.simulate_diffusion(&[node1], 10, 0.5)?;

// Convert to tropical network for efficient path operations
let tropical_net = network.to_tropical_network()?;
```

## Core Features

### Network Construction

- **Type-safe construction** with const generics for any Clifford algebra signature
- **Node metadata** support for labels and numerical properties
- **Directed and undirected edges** with flexible weight assignment
- **Capacity pre-allocation** for performance optimization

### Geometric Operations

- **Geometric distances** using natural norms in Clifford algebra space
- **Geometric centrality** based on inverse distance sums
- **Geometric similarity** via geometric products between node positions
- **Multi-signature support** (Euclidean, Minkowski, projective spaces)

### Path Finding

- **Dijkstra's algorithm** for weighted shortest paths
- **Geometric path finding** using geometric distances
- **Tropical optimization** for efficient all-pairs shortest paths
- **Path reconstruction** with detailed route information

### Community Detection

- **Geometric clustering** using k-means++ initialization in multivector space
- **Spectral clustering** via graph Laplacian eigendecomposition
- **Cohesion scoring** based on intra-cluster geometric distances
- **Multi-scale analysis** with configurable cluster numbers

### Information Diffusion

- **Geometric product-based** transmission strength calculation
- **Convergence analysis** with configurable decay rates
- **Influence scoring** to identify key information spreaders
- **Coverage tracking** over time steps

### Tropical Network Analysis

- **TropicalNetwork** conversion for max-plus optimization
- **Efficient shortest paths** using Floyd-Warshall in tropical semiring
- **Tropical betweenness centrality** for network analysis
- **Matrix-based computation** enabling parallel processing

## Examples

The crate includes comprehensive examples demonstrating different aspects:

### Run Examples

```bash
# Basic network operations and analysis
cargo run --example basic_network

# Community detection using geometric clustering
cargo run --example community_detection

# Information diffusion simulation
cargo run --example information_diffusion

# Tropical algebra path finding
cargo run --example tropical_pathfinding

# Advanced geometric analysis across different spaces
cargo run --example geometric_analysis
```

### Example Applications

- **Social Networks**: Analyze relationships with semantic embeddings
- **Transportation**: Optimize routing in geographic networks
- **Citation Networks**: Detect research communities using document embeddings
- **Biological Networks**: Model protein interactions in geometric space
- **Communication**: Simulate information spread with geometric constraints

## API Documentation

### Core Types

- **`GeometricNetwork<P,Q,R>`**: Main network structure with const generic signature
- **`GeometricEdge`**: Weighted directed edge between nodes
- **`NodeMetadata`**: Optional labels and properties for nodes
- **`Community<P,Q,R>`**: Community detection results with geometric centroids
- **`PropagationAnalysis`**: Information diffusion analysis results
- **`TropicalNetwork`**: Tropical algebra representation for optimization

### Key Methods

- **Construction**: `new()`, `add_node()`, `add_edge()`, `add_undirected_edge()`
- **Geometric**: `geometric_distance()`, `compute_geometric_centrality()`
- **Paths**: `shortest_path()`, `shortest_geometric_path()`, `all_pairs_shortest_paths()`
- **Analysis**: `find_communities()`, `spectral_clustering()`, `simulate_diffusion()`
- **Conversion**: `to_tropical_network()`

## Performance Characteristics

- **Memory**: O(V + E) for network storage, O(V²) for all-pairs computations
- **Path Finding**: O((V + E) log V) for single-source, O(V³) for all-pairs
- **Tropical Optimization**: Matrix operations enable GPU acceleration
- **Community Detection**: O(kVd) per iteration where k=clusters, d=dimensions
- **Scalability**: Efficient for networks up to 10⁴-10⁵ nodes

## Mathematical Properties

### Clifford Algebra Benefits

1. **Unified Framework**: Handle different geometric spaces consistently
2. **Natural Distances**: Multivector norms provide meaningful metrics
3. **Rotational Invariance**: Geometric properties preserved under rotations
4. **Scale Independence**: Ratios and angles remain consistent

### Tropical Algebra Advantages

1. **Optimization Focus**: Direct encoding of shortest path problems
2. **Parallel Computation**: Matrix operations suitable for vectorization
3. **Numerical Stability**: Avoids floating-point precision issues
4. **Theoretical Foundation**: Connects to convex geometry and optimization

## Integration with Amari Ecosystem

`amari-network` seamlessly integrates with other Amari crates:

- **`amari-core`**: Provides Clifford algebra operations and multivector types
- **`amari-tropical`**: Supplies tropical number arithmetic and operations
- **`amari-dual`**: Can be used for automatic differentiation of network metrics
- **`amari-gpu`**: Enables GPU acceleration of matrix computations

## Testing and Verification

The crate includes comprehensive tests:

- **Unit tests**: Individual component functionality
- **Integration tests**: End-to-end workflows and complex scenarios
- **Property tests**: Mathematical invariants and edge cases
- **Example tests**: Verify all documentation examples work correctly

Run tests:

```bash
cargo test --package amari-network
cargo test --package amari-network --test integration
```

## Contributing

Contributions are welcome! Areas of particular interest:

- **Algorithm optimizations** for large-scale networks
- **Additional geometric algebras** (quaternions, octonions, etc.)
- **GPU acceleration** for tropical matrix operations
- **Visualization tools** for geometric networks
- **Domain-specific applications** and use cases

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Citation

If you use this crate in academic work, please cite:

```bibtex
@software{amari_network,
  title = {Amari Network Analysis: Geometric Network Analysis using Clifford Algebra},
  author = {Amari Contributors},
  year = {2024},
  url = {https://github.com/justinelliottcobb/Amari},
  version = {0.9.0}
}
```

## Related Work

- **Geometric Algebra**: Dorst, Fontijne, Mann - "Geometric Algebra for Computer Science"
- **Tropical Geometry**: Maclagan, Sturmfels - "Introduction to Tropical Geometry"
- **Network Analysis**: Newman - "Networks: An Introduction"
- **Graph Theory**: Diestel - "Graph Theory"

---

**Part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing ecosystem**