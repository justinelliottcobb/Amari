//! Function space abstractions for functional analysis.
//!
//! This module provides the core traits and types for working with
//! function spaces in the context of geometric algebra.
//!
//! # Space Hierarchy
//!
//! The module implements the standard functional analysis hierarchy:
//!
//! ```text
//! VectorSpace
//!     ↓
//! NormedSpace (adds ||·||)
//!     ↓
//! BanachSpace (complete normed space)
//!     ↓
//! InnerProductSpace (adds ⟨·,·⟩)
//!     ↓
//! HilbertSpace (complete inner product space)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use amari_functional::space::{HilbertSpace, MultivectorL2};
//!
//! // Create L²(Cl(3,0,0)) - square-integrable multivector functions
//! let l2: MultivectorL2<3, 0, 0> = MultivectorL2::new();
//!
//! // Compute inner product
//! let f = /* some function */;
//! let g = /* another function */;
//! let inner = l2.inner_product(&f, &g)?;
//! ```

mod hilbert;
mod multivector_l2;
mod traits;

pub use hilbert::MultivectorHilbertSpace;
pub use multivector_l2::{Domain, L2Function, MultivectorL2};
pub use traits::{BanachSpace, HilbertSpace, InnerProductSpace, NormedSpace, VectorSpace};
