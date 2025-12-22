# amari-automata

Cellular automata, inverse design, and self-assembly using geometric algebra.

## Overview

`amari-automata` implements geometric cellular automata where cells contain multivectors instead of simple states. This enables rich spatial relationships, natural rotation handling, and automatic differentiation through evolution steps. The crate combines three mathematical frameworks:

1. **Geometric Algebra**: Spatial relationships and rotations
2. **Dual Numbers**: Automatic differentiation through time
3. **Tropical Algebra**: Constraint solving and optimization

## Features

- **GeometricCA**: Cellular automata with multivector cells
- **Inverse Design**: Find seeds that produce target configurations
- **Self-Assembly**: Polyomino tiling with geometric constraints
- **Cayley Navigation**: CA evolution as graph navigation
- **Tropical Solver**: Max-plus algebra for discrete constraints
- **GPU Acceleration**: Optional GPU support for large grids

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-automata = "0.12"
```

### Feature Flags

```toml
[dependencies]
# Default features
amari-automata = "0.12"

# With serialization
amari-automata = { version = "0.12", features = ["serde-support"] }

# With GPU acceleration
amari-automata = { version = "0.12", features = ["gpu"] }
```

## Quick Start

```rust
use amari_automata::{GeometricCA, Evolvable};
use amari_core::Multivector;

// Create a 2D geometric cellular automaton (64x64 grid)
let mut ca = GeometricCA::<3, 0, 0>::new_2d(64, 64);

// Set initial configuration with multivector cells
ca.set_cell_2d(32, 32, Multivector::basis_vector(0)).unwrap();
ca.set_cell_2d(33, 32, Multivector::basis_vector(1)).unwrap();

// Evolve the system one step
ca.step().unwrap();

// Get cell state
let cell = ca.get_cell_2d(32, 32).unwrap();
```

## Key Concepts

### Geometric Cellular Automata

Traditional CA cells hold discrete states (0, 1, 2, ...). Geometric CA cells hold multivectors, enabling:

- **Continuous States**: Smooth transitions between configurations
- **Rotations**: Natural handling via rotors
- **Composition**: Geometric product combines cell states
- **Grading**: Scalar, vector, bivector, trivector components

### Inverse Design

Find seeds that evolve to target patterns:

```rust
use amari_automata::inverse_design::InverseDesigner;

let designer = InverseDesigner::new(&target_pattern);
let seed = designer.find_seed(max_iterations)?;
```

Uses dual numbers to compute gradients through time evolution.

### Self-Assembly

Polyomino tiling with geometric constraints:

```rust
use amari_automata::self_assembly::Assembler;

let assembler = Assembler::new(tiles, constraints);
let solution = assembler.assemble(target_region)?;
```

### Cayley Navigation

Interpret CA evolution as paths in a Cayley graph:

```rust
use amari_automata::cayley_navigation::CayleyNavigator;

let navigator = CayleyNavigator::new(&group);
let path = navigator.find_path(start, target)?;
```

### Tropical Solver

Use max-plus algebra to linearize discrete constraints:

```rust
use amari_automata::tropical_solver::TropicalSolver;

let solver = TropicalSolver::new(constraints);
let solution = solver.solve()?;
```

## Modules

| Module | Description |
|--------|-------------|
| `geometric_ca` | Core geometric cellular automata implementation |
| `inverse_design` | Find seeds producing target configurations |
| `self_assembly` | Polyomino tiling with constraints |
| `cayley_navigation` | CA as Cayley graph navigation |
| `tropical_solver` | Max-plus constraint solving |
| `ui_assembly` | Self-assembling UI primitives |
| `traits` | Core traits (Evolvable, etc.) |
| `error` | Error types |

## Mathematical Background

### Why Geometric CA?

Traditional CA:
```
cell ∈ {0, 1, 2, ...}  (discrete states)
```

Geometric CA:
```
cell ∈ Cl(P,Q,R)  (multivector states)
```

Benefits:
- **Rotations**: Rotor sandwich product handles rotations naturally
- **Gradients**: Dual number extension enables optimization
- **Hierarchy**: Grade structure (scalar, vector, bivector) is meaningful
- **Composition**: Geometric product gives natural cell interaction

### Inverse Design with Dual Numbers

To find a seed s that evolves to target t:

1. Extend cells to dual numbers: `s + εds`
2. Evolve forward: `E(s + εds) = E(s) + εE'(s)·ds`
3. Compute loss gradient: `∂L/∂s = E'(s)ᵀ · ∂L/∂E(s)`
4. Update seed: `s ← s - α · ∂L/∂s`

### Tropical Constraints

Discrete constraints like "cell A must be adjacent to cell B" become linear in tropical algebra:

```
max(A, B) = threshold  →  A ⊕ B = threshold
```

This enables efficient constraint propagation.

## Use Cases

- **Generative Art**: Evolving geometric patterns
- **UI Layout**: Self-assembling interface components
- **Game of Life Variants**: Geometric versions of classic CA
- **Texture Synthesis**: Inverse design for procedural textures
- **Tiling Problems**: Polyomino and constraint satisfaction

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
