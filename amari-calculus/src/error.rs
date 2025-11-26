//! Error types for geometric calculus operations

use thiserror::Error;

/// Errors that can occur during calculus operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CalculusError {
    /// Invalid coordinate dimension
    #[error("Invalid coordinate dimension: expected {expected}, got {got}")]
    InvalidDimension { expected: usize, got: usize },

    /// Numerical differentiation failed
    #[error("Numerical differentiation failed: {reason}")]
    DifferentiationFailed { reason: String },

    /// Integration error
    #[error("Integration failed: {reason}")]
    IntegrationFailed { reason: String },

    /// Invalid coordinate system for operation
    #[error("Invalid coordinate system: {0}")]
    InvalidCoordinateSystem(String),

    /// Manifold computation error
    #[error("Manifold computation failed: {reason}")]
    ManifoldError { reason: String },

    /// Field evaluation error
    #[error("Field evaluation failed at point: {reason}")]
    FieldEvaluationError { reason: String },

    /// Covariant derivative error
    #[error("Covariant derivative computation failed: {reason}")]
    CovariantDerivativeError { reason: String },

    /// Metric tensor error
    #[error("Metric tensor error: {reason}")]
    MetricError { reason: String },

    /// Numerical instability detected
    #[error("Numerical instability: {reason}")]
    NumericalInstability { reason: String },
}

/// Result type for calculus operations
pub type CalculusResult<T> = Result<T, CalculusError>;
