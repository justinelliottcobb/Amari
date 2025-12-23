//! Error types for probabilistic operations on geometric algebra spaces
//!
//! This module defines error types for probability distributions, sampling,
//! stochastic processes, and Bayesian inference on multivector spaces.

use thiserror::Error;

/// Result type alias for probabilistic operations
pub type Result<T> = core::result::Result<T, ProbabilisticError>;

/// Errors that can occur in probabilistic operations on geometric algebra
#[derive(Error, Debug)]
pub enum ProbabilisticError {
    /// Distribution is not normalized (total probability ≠ 1)
    ///
    /// Probability distributions must integrate to 1 over their support.
    #[error("Distribution not normalized: total probability is {total}, expected 1.0")]
    NotNormalized {
        /// The actual total probability
        total: f64,
    },

    /// Sample is outside the distribution's support
    ///
    /// Occurs when evaluating log-probability at an invalid point.
    #[error("Sample out of support: {sample}")]
    OutOfSupport {
        /// Description of the invalid sample
        sample: String,
    },

    /// MCMC sampler failed to converge
    ///
    /// Occurs when diagnostic criteria (R-hat, ESS) are not satisfied.
    #[error("Sampler not converged after {iterations} iterations: {reason}")]
    SamplerNotConverged {
        /// Number of iterations attempted
        iterations: usize,
        /// Reason for non-convergence
        reason: String,
    },

    /// SDE numerical solver became unstable
    ///
    /// Occurs when step size is too large or drift/diffusion are ill-conditioned.
    #[error("SDE instability at time {time}: {details}")]
    SDEInstability {
        /// Time at which instability occurred
        time: f64,
        /// Details about the instability
        details: String,
    },

    /// Invalid distribution parameters
    ///
    /// Occurs when distribution parameters don't satisfy constraints
    /// (e.g., negative variance, non-positive-definite covariance).
    #[error("Invalid parameters: {description}")]
    InvalidParameters {
        /// Description of the parameter issue
        description: String,
    },

    /// Posterior computation failed
    ///
    /// Occurs when Bayesian posterior cannot be computed.
    #[error("Posterior computation failed: {reason}")]
    PosteriorComputationFailed {
        /// Reason for failure
        reason: String,
    },

    /// Dimension mismatch in probabilistic operation
    ///
    /// Occurs when multivector dimensions don't match.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension provided
        actual: usize,
    },

    /// Numerical computation error
    ///
    /// Occurs when numerical algorithms fail (overflow, underflow, NaN).
    #[error("Numerical error in {operation}: {details}")]
    NumericalError {
        /// Operation that failed
        operation: String,
        /// Details about the numerical issue
        details: String,
    },

    /// Grade mismatch in geometric operation
    ///
    /// Occurs when grade constraints are violated.
    #[error("Grade mismatch: expected grade {expected}, got {actual}")]
    GradeMismatch {
        /// Expected grade
        expected: usize,
        /// Actual grade
        actual: usize,
    },

    /// Insufficient samples for estimation
    ///
    /// Occurs when sample size is too small for reliable estimation.
    #[error("Insufficient samples: need at least {required}, got {actual}")]
    InsufficientSamples {
        /// Required number of samples
        required: usize,
        /// Actual number of samples
        actual: usize,
    },

    /// Error from underlying measure theory operations
    #[error(transparent)]
    MeasureError(#[from] amari_measure::MeasureError),

    /// Error from information geometry operations
    #[error(transparent)]
    InfoGeomError(#[from] amari_info_geom::InfoGeomError),
}

impl ProbabilisticError {
    /// Create a not normalized error
    pub fn not_normalized(total: f64) -> Self {
        Self::NotNormalized { total }
    }

    /// Create an out of support error
    pub fn out_of_support(sample: impl Into<String>) -> Self {
        Self::OutOfSupport {
            sample: sample.into(),
        }
    }

    /// Create a sampler not converged error
    pub fn sampler_not_converged(iterations: usize, reason: impl Into<String>) -> Self {
        Self::SamplerNotConverged {
            iterations,
            reason: reason.into(),
        }
    }

    /// Create an SDE instability error
    pub fn sde_instability(time: f64, details: impl Into<String>) -> Self {
        Self::SDEInstability {
            time,
            details: details.into(),
        }
    }

    /// Create an invalid parameters error
    pub fn invalid_parameters(description: impl Into<String>) -> Self {
        Self::InvalidParameters {
            description: description.into(),
        }
    }

    /// Create a posterior computation failed error
    pub fn posterior_failed(reason: impl Into<String>) -> Self {
        Self::PosteriorComputationFailed {
            reason: reason.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a numerical error
    pub fn numerical(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::NumericalError {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create a grade mismatch error
    pub fn grade_mismatch(expected: usize, actual: usize) -> Self {
        Self::GradeMismatch { expected, actual }
    }

    /// Create an insufficient samples error
    pub fn insufficient_samples(required: usize, actual: usize) -> Self {
        Self::InsufficientSamples { required, actual }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_normalized_error() {
        let err = ProbabilisticError::not_normalized(0.95);
        assert_eq!(
            err.to_string(),
            "Distribution not normalized: total probability is 0.95, expected 1.0"
        );
    }

    #[test]
    fn test_out_of_support_error() {
        let err = ProbabilisticError::out_of_support("negative variance");
        assert_eq!(err.to_string(), "Sample out of support: negative variance");
    }

    #[test]
    fn test_sampler_not_converged_error() {
        let err = ProbabilisticError::sampler_not_converged(10000, "R-hat > 1.1");
        assert_eq!(
            err.to_string(),
            "Sampler not converged after 10000 iterations: R-hat > 1.1"
        );
    }

    #[test]
    fn test_sde_instability_error() {
        let err = ProbabilisticError::sde_instability(1.5, "step size too large");
        assert_eq!(
            err.to_string(),
            "SDE instability at time 1.5: step size too large"
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = ProbabilisticError::dimension_mismatch(8, 4);
        assert_eq!(err.to_string(), "Dimension mismatch: expected 8, got 4");
    }

    #[test]
    fn test_numerical_error() {
        let err = ProbabilisticError::numerical("log_prob", "underflow");
        assert_eq!(err.to_string(), "Numerical error in log_prob: underflow");
    }
}
