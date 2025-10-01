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

/// Common error types for the automata system
#[derive(Debug, Clone, PartialEq)]
pub enum AutomataError {
    /// Invalid cell coordinates
    InvalidCoordinates(usize, usize),
    /// Configuration not found during inverse design
    ConfigurationNotFound,
    /// Assembly constraint violation
    AssemblyConstraintViolation,
    /// Cayley table cache miss
    CayleyTableMiss,
    /// Tropical solver convergence failure
    SolverConvergenceFailure,
}

/// Result type for automata operations
pub type AutomataResult<T> = Result<T, AutomataError>;

/// Trait for objects that can evolve over time
pub trait Evolvable {
    /// Perform one evolution step
    fn step(&mut self) -> AutomataResult<()>;

    /// Get the current generation/time step
    fn generation(&self) -> usize;

    /// Reset to initial state
    fn reset(&mut self);
}

/// Trait for systems that support inverse design
pub trait InverseDesignable {
    type Target;
    type Configuration;

    /// Find configuration that produces the target after evolution
    fn find_seed(&self, target: &Self::Target) -> AutomataResult<Self::Configuration>;

    /// Measure fitness/distance to target
    fn fitness(&self, config: &Self::Configuration, target: &Self::Target) -> f64;
}

/// Trait for self-assembling systems
pub trait SelfAssembling {
    type Component;
    type Assembly;

    /// Attempt to assemble components
    fn assemble(&self, components: &[Self::Component]) -> AutomataResult<Self::Assembly>;

    /// Check if assembly is stable
    fn is_stable(&self, assembly: &Self::Assembly) -> bool;

    /// Get affinity between components
    fn affinity(&self, a: &Self::Component, b: &Self::Component) -> f64;
}

#[cfg(test)]
pub mod comprehensive_tests_simple;

#[cfg(test)]
pub mod verified_contracts_simple;
