//! Error types for core geometric algebra operations

use thiserror::Error;

/// Error types for core geometric algebra operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CoreError {
    /// Invalid dimension for operation
    #[error("Invalid dimension: expected {expected}, got {actual}")]
    InvalidDimension { expected: usize, actual: usize },

    /// Division by zero
    #[error("Division by zero")]
    DivisionByZero,

    /// Numerical instability detected
    #[error("Numerical instability detected")]
    NumericalInstability,

    /// Invalid basis vector index
    #[error("Invalid basis vector index: {0} (max: {1})")]
    InvalidBasisIndex(usize, usize),

    /// Matrix is singular (non-invertible)
    #[error("Matrix is singular and cannot be inverted")]
    SingularMatrix,

    /// Invalid metric signature
    #[error("Invalid metric signature: positive {positive} + negative {negative} + zero {zero} != dimension {dimension}")]
    InvalidSignature {
        positive: usize,
        negative: usize,
        zero: usize,
        dimension: usize,
    },
}

/// Result type for core operations
pub type CoreResult<T> = Result<T, CoreError>;
