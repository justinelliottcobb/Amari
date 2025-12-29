//! Linear operators on function spaces.
//!
//! This module provides traits and types for linear operators between
//! Hilbert spaces, including bounded operators, compact operators, and
//! self-adjoint operators.
//!
//! # Operator Hierarchy
//!
//! ```text
//! LinearOperator
//!     ↓
//! BoundedOperator (adds ||T|| < ∞)
//!     ↓
//! CompactOperator (maps bounded sets to precompact sets)
//!     ↓
//! FiniteRankOperator (finite-dimensional range)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use amari_functional::operator::{LinearOperator, IdentityOperator};
//! use amari_functional::space::MultivectorHilbertSpace;
//!
//! let space: MultivectorHilbertSpace<2, 0, 0> = MultivectorHilbertSpace::new();
//! let identity = IdentityOperator::new();
//!
//! let x = space.from_coefficients(&[1.0, 2.0, 0.0, 0.0]).unwrap();
//! let y = identity.apply(&x)?;
//! assert_eq!(x, y);
//! ```

mod basic;
mod compact;
mod matrix;
mod traits;

pub use basic::{
    CompositeOperator, IdentityOperator, ProjectionOperator, ScalingOperator, ZeroOperator,
};
pub use compact::{
    CompactMatrixOperator, CompactOperator, FiniteRankOperator, FredholmMatrixOperator,
    FredholmOperator,
};
pub use matrix::MatrixOperator;
pub use traits::{
    AdjointableOperator, BoundedOperator, LinearOperator, OperatorNorm, SelfAdjointOperator,
};
