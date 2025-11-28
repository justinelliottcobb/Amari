//! # Amari Automata
//!
//! Cellular automata, inverse design, and self-assembly using geometric algebra.
//! This crate implements the mathematical foundation for Cliffy-Alive's self-assembling
//! UI system, where geometric algebra provides natural composition rules for CA cells.
//!
//! ## Key Concepts
//!
//! - **Geometric CA**: Cellular automata where cells contain multivectors
//! - **Inverse Design**: Finding seeds that produce target configurations using dual numbers
//! - **Self-Assembly**: Polyomino tiling with geometric algebra constraints
//! - **Cayley Navigation**: CA evolution as navigation in Cayley graphs
//! - **Tropical Solving**: Using max-plus algebra to linearize discrete constraints
//!
//! ## Architecture
//!
//! The crate combines three mathematical frameworks:
//! 1. Geometric algebra for spatial relationships and rotations
//! 2. Dual numbers for automatic differentiation through time
//! 3. Tropical algebra for constraint solving and optimization
//!
//! ## Usage
//!
//! ```rust
//! use amari_automata::{GeometricCA, Evolvable};
//! use amari_core::Multivector;
//!
//! // Create a 2D geometric cellular automaton
//! let mut ca = GeometricCA::<3, 0, 0>::new_2d(64, 64);
//!
//! // Set initial configuration with multivector cells using 2D coordinates
//! ca.set_cell_2d(32, 32, Multivector::basis_vector(0)).unwrap();
//!
//! // Evolve the system
//! ca.step().unwrap();
//! ```

#![no_std]

#[macro_use]
extern crate alloc;

// Re-export dependencies for convenience
pub use amari_core::{Bivector, Multivector, Vector};
pub use amari_dual::DualMultivector;
pub use amari_tropical::TropicalMultivector;

// Domain types
pub mod error;
pub mod traits;

// Core modules
pub mod cayley_navigation;
pub mod geometric_ca;
pub mod inverse_design;
pub mod self_assembly;
pub mod tropical_solver;
pub mod ui_assembly;

// Re-export main types
pub use cayley_navigation::{
    CayleyGraph, CayleyGraphNavigator, CayleyNavigator, DefaultCayleyNavigator, Generator,
    GroupElement,
};
pub use geometric_ca::{CARule, CellState, GeometricCA, RuleType};
pub use inverse_design::{Configuration, Target};
pub use inverse_design::{
    InverseCADesigner, InverseDesigner, Objective, TargetPattern, TropicalConstraint,
};
pub use self_assembly::{
    Assembly, AssemblyConfig, AssemblyConstraint, AssemblyRule, Component, ComponentType,
    Polyomino, SelfAssembler, SelfAssembly, Shape, TileSet, UIComponentType, WangTileSet,
};
pub use tropical_solver::{
    ConstraintType, SolverConfig, TropicalExpression, TropicalSolver, TropicalSystem,
};
pub use ui_assembly::{Layout, LayoutConstraint, UIAssembler, UIAssemblyConfig, UIComponent};

// Re-export domain types
pub use error::{AutomataError, AutomataResult};
pub use traits::{Evolvable, InverseDesignable, SelfAssembling};

// GPU acceleration exports
