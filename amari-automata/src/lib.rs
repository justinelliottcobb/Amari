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
#[cfg(feature = "gpu")]
pub mod gpu;
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

// GPU acceleration exports
#[cfg(feature = "gpu")]
pub use gpu::{
    AutomataGpuConfig, AutomataGpuError, AutomataGpuOps, AutomataGpuResult, GpuCellData,
    GpuEvolutionParams, GpuRuleConfig,
};

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

impl core::fmt::Display for AutomataError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidCoordinates(x, y) => write!(f, "Invalid coordinates: ({}, {})", x, y),
            Self::ConfigurationNotFound => {
                write!(f, "Configuration not found during inverse design")
            }
            Self::AssemblyConstraintViolation => write!(f, "Assembly constraint violation"),
            Self::CayleyTableMiss => write!(f, "Cayley table cache miss"),
            Self::SolverConvergenceFailure => write!(f, "Tropical solver convergence failure"),
        }
    }
}

/// Result type for automata operations
pub type AutomataResult<T> = Result<T, AutomataError>;

/// **Innovative Evolvable Trait for Automata Theory**
///
/// A foundational abstraction that unifies evolution concepts across different
/// mathematical structures in automata theory. This trait represents one of the
/// key innovations in the Amari library's design philosophy.
///
/// ## Design Philosophy
///
/// The `Evolvable` trait abstracts the concept of "evolution" or "time-stepped
/// progression" across disparate mathematical systems:
///
/// - **Cellular Automata**: Discrete time evolution of cell states
/// - **Cayley Graph Navigation**: Movement through group element spaces
/// - **Self-Assembly Systems**: Progressive component aggregation
/// - **Tropical Algebra**: Iterative constraint solving
/// - **Inverse Design**: Gradient-based optimization steps
///
/// ## Innovation Highlights
///
/// 1. **Mathematical Universality**: Provides a common interface for time-based
///    evolution regardless of the underlying mathematical structure
///
/// 2. **Composability**: Enables complex systems to be built by composing
///    different `Evolvable` components
///
/// 3. **Verification-Friendly**: The discrete step nature allows for
///    invariant checking and formal verification at each evolution step
///
/// 4. **Geometric Algebra Integration**: Naturally supports multivector-based
///    state evolution common in geometric algebra computations
///
/// ## Usage Patterns
///
/// ```rust
/// use amari_automata::{GeometricCA, Evolvable};
///
/// let mut ca = GeometricCA::<3, 0, 0>::new_2d(64, 64);
///
/// // Evolve system through multiple generations
/// for generation in 0..100 {
///     ca.step().unwrap();
///     println!("Generation {}: {}", ca.generation(), ca.total_energy());
/// }
///
/// // Reset and try different evolution
/// ca.reset();
/// ```
///
/// ## Verification Integration
///
/// The trait design supports formal verification by providing discrete
/// checkpoints where mathematical invariants can be verified:
///
/// - Energy conservation laws
/// - Geometric algebra closure properties
/// - Group theory axioms in Cayley navigation
/// - Constraint satisfaction in tropical solving
///
/// This makes `Evolvable` particularly valuable for applications requiring
/// mathematical correctness guarantees.
pub trait Evolvable {
    /// Perform one evolution step
    ///
    /// # Contracts
    /// - `ensures(self.generation() == old(self.generation()) + 1)`
    /// - `ensures(mathematical_invariants_preserved())`
    fn step(&mut self) -> AutomataResult<()>;

    /// Get the current generation/time step
    ///
    /// # Contracts
    /// - `ensures(result >= 0)`
    /// - `ensures(result increases monotonically with each step())`
    fn generation(&self) -> usize;

    /// Reset to initial state
    ///
    /// # Contracts
    /// - `ensures(self.generation() == 0)`
    /// - `ensures(state_equals_initial_configuration())`
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
