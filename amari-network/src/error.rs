//! Error types for network analysis operations

use thiserror::Error;

/// Error types for network analysis operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum NetworkError {
    /// Node index is out of bounds
    #[error("Node index {0} out of bounds")]
    NodeIndexOutOfBounds(usize),

    /// Invalid edge specification
    #[error("Invalid edge: source={0}, target={1}")]
    InvalidEdge(usize, usize),

    /// Matrix dimension mismatch
    #[error("Matrix dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Network is disconnected when connected network required
    #[error("Network is disconnected")]
    DisconnectedNetwork,

    /// Computation failed with detailed message
    #[error("Computation failed: {0}")]
    ComputationError(String),

    /// Invalid parameter value
    #[error("Invalid parameter {parameter}: {value} (must be {constraint})")]
    InvalidParameter {
        parameter: String,
        value: String,
        constraint: String,
    },

    /// Empty network when non-empty required
    #[error("Operation requires non-empty network")]
    EmptyNetwork,

    /// Convergence failure in iterative algorithm
    #[error("Algorithm failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// Insufficient data for analysis
    #[error("Insufficient data: {reason}")]
    InsufficientData { reason: String },
}

/// Result type for network operations
pub type NetworkResult<T> = Result<T, NetworkError>;

impl NetworkError {
    /// Create a computation error with formatted message
    pub fn computation<T: std::fmt::Display>(msg: T) -> Self {
        NetworkError::ComputationError(msg.to_string())
    }

    /// Create an invalid parameter error
    pub fn invalid_param(param: &str, value: impl std::fmt::Display, constraint: &str) -> Self {
        NetworkError::InvalidParameter {
            parameter: param.to_string(),
            value: value.to_string(),
            constraint: constraint.to_string(),
        }
    }

    /// Create an insufficient data error
    pub fn insufficient_data(reason: &str) -> Self {
        NetworkError::InsufficientData {
            reason: reason.to_string(),
        }
    }
}
