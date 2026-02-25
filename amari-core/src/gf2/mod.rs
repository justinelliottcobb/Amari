//! GF(2) algebra module — finite field arithmetic, linear algebra, and Clifford algebra over GF(2).
//!
//! Provides the algebraic foundation for binary Grassmannians, binary matroid
//! representability, and binary-encoded geometric data.

mod clifford;
mod grassmannian;
mod matrix;
mod scalar;
mod vector;

pub use clifford::BinaryMultivector;
pub use grassmannian::{
    binary_grassmannian_size, enumerate_subspaces, gaussian_binomial, schubert_cell_of,
    schubert_cell_size,
};
pub use matrix::GF2Matrix;
pub use scalar::GF2;
pub use vector::GF2Vector;
