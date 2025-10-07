//! Error types for core geometric algebra operations

#[cfg(feature = "std")]
use thiserror::Error;

#[cfg(not(feature = "std"))]
use core::fmt;

/// Error types for core geometric algebra operations
#[cfg(feature = "std")]
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

/// Error types for core geometric algebra operations (no_std version)
#[cfg(not(feature = "std"))]
#[derive(Debug, Clone, PartialEq)]
pub enum CoreError {
    /// Invalid dimension for operation
    InvalidDimension { expected: usize, actual: usize },

    /// Division by zero
    DivisionByZero,

    /// Numerical instability detected
    NumericalInstability,

    /// Invalid basis vector index
    InvalidBasisIndex(usize, usize),

    /// Matrix is singular (non-invertible)
    SingularMatrix,

    /// Invalid metric signature
    InvalidSignature {
        positive: usize,
        negative: usize,
        zero: usize,
        dimension: usize,
    },
}

#[cfg(not(feature = "std"))]
impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::InvalidDimension { expected, actual } => {
                write!(
                    f,
                    "Invalid dimension: expected {}, got {}",
                    expected, actual
                )
            }
            CoreError::DivisionByZero => {
                write!(f, "Division by zero")
            }
            CoreError::NumericalInstability => {
                write!(f, "Numerical instability detected")
            }
            CoreError::InvalidBasisIndex(idx, max) => {
                write!(f, "Invalid basis vector index: {} (max: {})", idx, max)
            }
            CoreError::SingularMatrix => {
                write!(f, "Matrix is singular and cannot be inverted")
            }
            CoreError::InvalidSignature {
                positive,
                negative,
                zero,
                dimension,
            } => {
                write!(
                    f,
                    "Invalid metric signature: positive {} + negative {} + zero {} != dimension {}",
                    positive, negative, zero, dimension
                )
            }
        }
    }
}

/// Result type for core operations
pub type CoreResult<T> = Result<T, CoreError>;
