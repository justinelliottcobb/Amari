//! Error types for functional analysis operations.
//!
//! This module provides comprehensive error handling for functional analysis
//! operations including space construction, operator application, and spectral
//! decomposition.

use thiserror::Error;

/// Result type for functional analysis operations.
pub type Result<T> = core::result::Result<T, FunctionalError>;

/// Errors that can occur during functional analysis operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FunctionalError {
    /// Dimension mismatch between spaces or operators.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension encountered.
        actual: usize,
    },

    /// The space does not satisfy required completeness properties.
    #[error("Space is not complete: {reason}")]
    NotComplete {
        /// Description of why completeness fails.
        reason: String,
    },

    /// The operator is not bounded.
    #[error("Operator is not bounded: {description}")]
    NotBounded {
        /// Description of the unboundedness.
        description: String,
    },

    /// The operator is not compact.
    #[error("Operator is not compact: {reason}")]
    NotCompact {
        /// Reason for non-compactness.
        reason: String,
    },

    /// The operator does not have a valid inverse.
    #[error("Operator is not invertible: {reason}")]
    NotInvertible {
        /// Reason for non-invertibility.
        reason: String,
    },

    /// Eigenvalue computation failed.
    #[error("Eigenvalue computation failed: {reason}")]
    EigenvalueError {
        /// Description of the failure.
        reason: String,
    },

    /// Spectral decomposition failed.
    #[error("Spectral decomposition failed: {reason}")]
    SpectralDecompositionError {
        /// Description of the failure.
        reason: String,
    },

    /// The Fredholm index is undefined.
    #[error("Fredholm index undefined: {reason}")]
    FredholmIndexUndefined {
        /// Reason why the index is undefined.
        reason: String,
    },

    /// Sobolev space construction failed.
    #[error("Sobolev space error: {description}")]
    SobolevError {
        /// Description of the Sobolev space error.
        description: String,
    },

    /// Norm computation failed or is undefined.
    #[error("Norm error: {description}")]
    NormError {
        /// Description of the norm error.
        description: String,
    },

    /// Inner product computation failed.
    #[error("Inner product error: {description}")]
    InnerProductError {
        /// Description of the inner product error.
        description: String,
    },

    /// Convergence conditions not satisfied.
    #[error("Convergence failed after {iterations} iterations: {reason}")]
    ConvergenceError {
        /// Number of iterations attempted.
        iterations: usize,
        /// Reason for convergence failure.
        reason: String,
    },

    /// Numerical instability detected.
    #[error("Numerical instability in {operation}: {details}")]
    NumericalInstability {
        /// The operation that became unstable.
        operation: String,
        /// Details about the instability.
        details: String,
    },

    /// Invalid parameters provided.
    #[error("Invalid parameters: {description}")]
    InvalidParameters {
        /// Description of the invalid parameters.
        description: String,
    },

    /// Error from amari-measure.
    #[error(transparent)]
    MeasureError(#[from] amari_measure::MeasureError),

    /// Error from amari-calculus.
    #[error(transparent)]
    CalculusError(#[from] amari_calculus::CalculusError),
}

impl FunctionalError {
    /// Create a dimension mismatch error.
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a not complete error.
    pub fn not_complete(reason: impl Into<String>) -> Self {
        Self::NotComplete {
            reason: reason.into(),
        }
    }

    /// Create a not bounded error.
    pub fn not_bounded(description: impl Into<String>) -> Self {
        Self::NotBounded {
            description: description.into(),
        }
    }

    /// Create a not compact error.
    pub fn not_compact(reason: impl Into<String>) -> Self {
        Self::NotCompact {
            reason: reason.into(),
        }
    }

    /// Create a not invertible error.
    pub fn not_invertible(reason: impl Into<String>) -> Self {
        Self::NotInvertible {
            reason: reason.into(),
        }
    }

    /// Create an eigenvalue error.
    pub fn eigenvalue_error(reason: impl Into<String>) -> Self {
        Self::EigenvalueError {
            reason: reason.into(),
        }
    }

    /// Create a spectral decomposition error.
    pub fn spectral_decomposition_error(reason: impl Into<String>) -> Self {
        Self::SpectralDecompositionError {
            reason: reason.into(),
        }
    }

    /// Create a Fredholm index undefined error.
    pub fn fredholm_index_undefined(reason: impl Into<String>) -> Self {
        Self::FredholmIndexUndefined {
            reason: reason.into(),
        }
    }

    /// Create a Sobolev error.
    pub fn sobolev_error(description: impl Into<String>) -> Self {
        Self::SobolevError {
            description: description.into(),
        }
    }

    /// Create a norm error.
    pub fn norm_error(description: impl Into<String>) -> Self {
        Self::NormError {
            description: description.into(),
        }
    }

    /// Create an inner product error.
    pub fn inner_product_error(description: impl Into<String>) -> Self {
        Self::InnerProductError {
            description: description.into(),
        }
    }

    /// Create a convergence error.
    pub fn convergence_error(iterations: usize, reason: impl Into<String>) -> Self {
        Self::ConvergenceError {
            iterations,
            reason: reason.into(),
        }
    }

    /// Create a numerical instability error.
    pub fn numerical_instability(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::NumericalInstability {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create an invalid parameters error.
    pub fn invalid_parameters(description: impl Into<String>) -> Self {
        Self::InvalidParameters {
            description: description.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FunctionalError::dimension_mismatch(8, 16);
        assert_eq!(err.to_string(), "Dimension mismatch: expected 8, got 16");
    }

    #[test]
    fn test_error_factory_methods() {
        let err = FunctionalError::not_complete("Cauchy sequence does not converge");
        assert!(matches!(err, FunctionalError::NotComplete { .. }));

        let err = FunctionalError::eigenvalue_error("Matrix is singular");
        assert!(matches!(err, FunctionalError::EigenvalueError { .. }));
    }

    #[test]
    fn test_error_equality() {
        let err1 = FunctionalError::dimension_mismatch(4, 8);
        let err2 = FunctionalError::dimension_mismatch(4, 8);
        let err3 = FunctionalError::dimension_mismatch(4, 16);

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
