//! Error types for dynamical systems operations
//!
//! This module provides a comprehensive error hierarchy for all dynamical
//! systems computations including solver failures, stability analysis errors,
//! and bifurcation detection issues.

use thiserror::Error;

/// Result type for dynamics operations
pub type Result<T> = core::result::Result<T, DynamicsError>;

/// Errors that can occur in dynamical systems operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum DynamicsError {
    /// Dimension mismatch between expected and actual state dimensions
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension received
        actual: usize,
    },

    /// Solver failed to converge within the specified tolerance
    #[error("Solver failed to converge after {iterations} iterations: {reason}")]
    ConvergenceFailure {
        /// Number of iterations attempted
        iterations: usize,
        /// Reason for failure
        reason: String,
    },

    /// Numerical instability detected during computation
    #[error("Numerical instability in {operation}: {details}")]
    NumericalInstability {
        /// Operation that failed
        operation: String,
        /// Details about the instability
        details: String,
    },

    /// Fixed point could not be found
    #[error("Fixed point not found: {reason}")]
    FixedPointNotFound {
        /// Reason the fixed point was not found
        reason: String,
    },

    /// Invalid parameter value provided
    #[error("Invalid parameter: {description}")]
    InvalidParameter {
        /// Description of the invalid parameter
        description: String,
    },

    /// Bifurcation detection or classification failed
    #[error("Bifurcation detection failed: {reason}")]
    BifurcationError {
        /// Reason for the bifurcation error
        reason: String,
    },

    /// Lyapunov exponent computation failed
    #[error("Lyapunov computation failed: {reason}")]
    LyapunovError {
        /// Reason for the Lyapunov computation failure
        reason: String,
    },

    /// State left the valid domain of the dynamical system
    #[error("Out of domain: state left valid region")]
    OutOfDomain,

    /// Stiff system detected, implicit solver recommended
    #[error("Stiff system detected: consider using an implicit solver")]
    StiffSystem,

    /// Invalid step size (must be positive)
    #[error("Invalid step size: {step_size} (must be positive)")]
    InvalidStepSize {
        /// The invalid step size
        step_size: f64,
    },

    /// Time interval is invalid (t1 must be > t0 for forward integration)
    #[error("Invalid time interval: [{t0}, {t1}]")]
    InvalidTimeInterval {
        /// Start time
        t0: f64,
        /// End time
        t1: f64,
    },

    /// Jacobian computation failed
    #[error("Jacobian computation failed: {reason}")]
    JacobianError {
        /// Reason for the failure
        reason: String,
    },

    /// Eigenvalue computation failed
    #[error("Eigenvalue computation failed: {reason}")]
    EigenvalueError {
        /// Reason for the failure
        reason: String,
    },

    /// Error from amari-functional crate
    #[error(transparent)]
    FunctionalError(#[from] amari_functional::FunctionalError),

    /// Error from amari-calculus crate
    #[error(transparent)]
    CalculusError(#[from] amari_calculus::CalculusError),

    /// Error from amari-topology crate
    #[error(transparent)]
    TopologyError(#[from] amari_topology::TopologyError),
}

impl DynamicsError {
    /// Create a convergence failure error
    pub fn convergence_failure(iterations: usize, reason: impl Into<String>) -> Self {
        Self::ConvergenceFailure {
            iterations,
            reason: reason.into(),
        }
    }

    /// Create a numerical instability error
    pub fn numerical_instability(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::NumericalInstability {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create a fixed point not found error
    pub fn fixed_point_not_found(reason: impl Into<String>) -> Self {
        Self::FixedPointNotFound {
            reason: reason.into(),
        }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter(description: impl Into<String>) -> Self {
        Self::InvalidParameter {
            description: description.into(),
        }
    }

    /// Create a bifurcation error
    pub fn bifurcation_error(reason: impl Into<String>) -> Self {
        Self::BifurcationError {
            reason: reason.into(),
        }
    }

    /// Create a Lyapunov error
    pub fn lyapunov_error(reason: impl Into<String>) -> Self {
        Self::LyapunovError {
            reason: reason.into(),
        }
    }

    /// Create a Jacobian error
    pub fn jacobian_error(reason: impl Into<String>) -> Self {
        Self::JacobianError {
            reason: reason.into(),
        }
    }

    /// Create an eigenvalue error
    pub fn eigenvalue_error(reason: impl Into<String>) -> Self {
        Self::EigenvalueError {
            reason: reason.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create an invalid step size error
    pub fn invalid_step_size(step_size: f64) -> Self {
        Self::InvalidStepSize { step_size }
    }

    /// Create an invalid time interval error
    pub fn invalid_time_interval(t0: f64, t1: f64) -> Self {
        Self::InvalidTimeInterval { t0, t1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = DynamicsError::convergence_failure(100, "tolerance not met");
        assert!(matches!(
            err,
            DynamicsError::ConvergenceFailure {
                iterations: 100,
                ..
            }
        ));

        let err = DynamicsError::numerical_instability("RK4 step", "NaN detected");
        assert!(matches!(err, DynamicsError::NumericalInstability { .. }));

        let err = DynamicsError::dimension_mismatch(3, 4);
        assert!(matches!(
            err,
            DynamicsError::DimensionMismatch {
                expected: 3,
                actual: 4
            }
        ));
    }

    #[test]
    fn test_error_display() {
        let err = DynamicsError::convergence_failure(50, "step size too large");
        let msg = format!("{}", err);
        assert!(msg.contains("50"));
        assert!(msg.contains("step size too large"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = DynamicsError::OutOfDomain;
        let err2 = DynamicsError::OutOfDomain;
        assert_eq!(err1, err2);

        let err3 = DynamicsError::StiffSystem;
        assert_ne!(err1, err3);
    }
}
