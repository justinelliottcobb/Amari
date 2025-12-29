//! Spectral theory for linear operators.
//!
//! This module provides tools for spectral analysis of linear operators,
//! including eigenvalue computation and spectral decomposition.
//!
//! # Mathematical Background
//!
//! For a linear operator T: H → H on a Hilbert space:
//! - An **eigenvalue** λ satisfies Tx = λx for some non-zero x (eigenvector)
//! - The **spectrum** σ(T) is the set of λ where (T - λI) is not invertible
//! - The **spectral theorem** states that self-adjoint operators have
//!   orthogonal eigenvectors and real eigenvalues
//!
//! # Example
//!
//! ```ignore
//! use amari_functional::spectral::{compute_eigenvalues, SpectralDecomposition};
//! use amari_functional::operator::MatrixOperator;
//!
//! let matrix: MatrixOperator<2, 0, 0> = MatrixOperator::identity();
//! let eigenvalues = compute_eigenvalues(&matrix)?;
//! // All eigenvalues of identity are 1
//! ```

mod decomposition;
mod eigenvalue;

pub use decomposition::{spectral_decompose, SpectralDecomposition};
pub use eigenvalue::{compute_eigenvalues, inverse_iteration, power_method, Eigenpair, Eigenvalue};
