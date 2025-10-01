//! Error types for tropical algebra operations

use thiserror::Error;

/// Error types for tropical algebra operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TropicalError {
    /// Matrix dimension mismatch
    #[error("Matrix dimension mismatch: {0}")]
    MatrixDimensionMismatch(String),

    /// Invalid matrix dimensions for operation
    #[error("Invalid matrix dimensions for {operation}: ({rows1}x{cols1}) and ({rows2}x{cols2})")]
    InvalidMatrixOperation {
        operation: String,
        rows1: usize,
        cols1: usize,
        rows2: usize,
        cols2: usize,
    },

    /// Negative infinity arithmetic error
    #[error("Invalid tropical arithmetic: negative infinity operation")]
    NegativeInfinityError,

    /// Invalid tropical polynomial
    #[error("Invalid tropical polynomial: {0}")]
    InvalidPolynomial(String),

    /// Numerical overflow in tropical computation
    #[error("Numerical overflow in tropical computation")]
    Overflow,
}

/// Result type for tropical operations
pub type TropicalResult<T> = Result<T, TropicalError>;
