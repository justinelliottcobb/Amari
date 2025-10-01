//! Error types for dual number operations

use thiserror::Error;

/// Error types for dual number operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DualError {
    /// Dimension mismatch in operations
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Division by zero
    #[error("Division by zero in dual number computation")]
    DivisionByZero,

    /// Invalid gradient dimension
    #[error("Invalid gradient dimension: {0}")]
    InvalidGradientDimension(usize),

    /// Numerical instability detected
    #[error("Numerical instability detected in dual number computation")]
    NumericalInstability,

    /// Invalid operation for the given dual number configuration
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
}

/// Result type for dual number operations
pub type DualResult<T> = Result<T, DualError>;
