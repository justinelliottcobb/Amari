//! Dynamical system traits and definitions
//!
//! This module provides the core abstractions for dynamical systems:
//!
//! - **Continuous-time systems**: [`DynamicalSystem`] for autonomous ODEs
//! - **Non-autonomous systems**: [`NonAutonomousSystem`] with explicit time dependence
//! - **Discrete maps**: [`DiscreteMap`] for iterated maps
//! - **Parametric systems**: [`ParametricSystem`] for bifurcation analysis
//!
//! # Geometric Algebra State Spaces
//!
//! All systems operate on Clifford algebra state spaces Cl(P,Q,R). This provides:
//!
//! - Natural encoding of geometric transformations
//! - Unified treatment of rotations, reflections, and projections
//! - Grade-structured state evolution
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::flow::{DynamicalSystem, HarmonicOscillator};
//! use amari_core::Multivector;
//!
//! let system = HarmonicOscillator::new(1.0);
//! let mut state = Multivector::<2, 0, 0>::zero();
//! state.set(1, 1.0);  // Initial position
//! state.set(2, 0.0);  // Initial velocity
//!
//! let derivative = system.vector_field(&state)?;
//! ```

mod traits;

// Re-export main types
pub use traits::{
    numerical_jacobian, numerical_jacobian_map, numerical_jacobian_nonautonomous,
    AutonomizedSystem, DiscreteMap, DynamicalSystem, HarmonicOscillator, NonAutonomousSystem,
    ParametricSystem,
};
