//! Error types for measure theory operations
//!
//! This module defines error types for measure-theoretic operations,
//! following the Amari error handling design principles.

use thiserror::Error;

/// Result type alias for measure theory operations
pub type Result<T> = core::result::Result<T, MeasureError>;

/// Errors that can occur in measure theory operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MeasureError {
    /// Set is not measurable with respect to the σ-algebra
    ///
    /// Occurs when attempting to measure a set that is not in the σ-algebra.
    ///
    /// # Examples
    ///
    /// - Attempting to measure a Vitali set with Lebesgue measure
    /// - Measuring a set not in the Borel σ-algebra with Borel measure
    #[error("Set is not measurable: {description}")]
    NotMeasurable {
        /// Description of why the set is not measurable
        description: String,
    },

    /// Function is not integrable
    ///
    /// Occurs when attempting to integrate a function that is not integrable.
    ///
    /// # Examples
    ///
    /// - Function with infinite integral
    /// - Non-measurable function
    /// - Integral does not converge
    #[error("Function is not integrable: {reason}")]
    NotIntegrable {
        /// Reason why the function is not integrable
        reason: String,
    },

    /// Measure is not absolutely continuous with respect to another measure
    ///
    /// Occurs when attempting to compute Radon-Nikodym derivative when ν ⊥ μ
    /// (ν is singular with respect to μ).
    ///
    /// # Mathematical Background
    ///
    /// For measures ν and μ, ν ≪ μ (ν absolutely continuous w.r.t. μ) means:
    /// μ(A) = 0 ⟹ ν(A) = 0 for all measurable A
    ///
    /// Radon-Nikodym theorem requires ν ≪ μ to guarantee existence of dν/dμ.
    #[error("Measure ν is not absolutely continuous with respect to μ: {details}")]
    NotAbsolutelyContinuous {
        /// Details about the singularity
        details: String,
    },

    /// Measure is not σ-finite
    ///
    /// Occurs when an operation requires σ-finiteness but the measure is not σ-finite.
    ///
    /// # Examples
    ///
    /// - Radon-Nikodym theorem requires σ-finite measures
    /// - Fubini's theorem requires σ-finite product measures
    /// - Some convergence theorems require σ-finiteness
    #[error("Measure is not σ-finite: {reason}")]
    NotSigmaFinite {
        /// Reason why σ-finiteness is required
        reason: String,
    },

    /// Dimension mismatch between measure spaces
    ///
    /// Occurs when dimensions don't match in operations requiring compatible spaces.
    ///
    /// # Examples
    ///
    /// - Integrating a function on ℝ³ with measure on ℝ²
    /// - Product measure of incompatible spaces
    /// - Pushforward with dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension provided
        actual: usize,
    },

    /// Convergence theorem conditions not satisfied
    ///
    /// Occurs when attempting to apply a convergence theorem without meeting the hypotheses.
    ///
    /// # Examples
    ///
    /// - Dominated convergence without dominating function
    /// - Monotone convergence for non-monotone sequence
    /// - Fatou's lemma for non-nonnegative functions
    #[error("Convergence theorem conditions not satisfied: {theorem} requires {condition}")]
    ConvergenceConditionsNotSatisfied {
        /// Name of the theorem (Monotone, Dominated, Fatou)
        theorem: String,
        /// Condition that was not satisfied
        condition: String,
    },

    /// Invalid probability measure
    ///
    /// Occurs when a probability measure does not satisfy μ(X) = 1.
    ///
    /// # Mathematical Requirement
    ///
    /// Probability measures must satisfy:
    /// - μ(A) ≥ 0 for all measurable A (non-negativity)
    /// - μ(X) = 1 (normalization)
    /// - μ(⋃ Aₙ) = ∑ μ(Aₙ) for disjoint Aₙ (countable additivity)
    #[error("Invalid probability measure: total measure is {total}, expected 1.0")]
    InvalidProbabilityMeasure {
        /// Total measure of the space
        total: f64,
    },

    /// Operation requires a finite measure
    ///
    /// Occurs when attempting an operation that requires finite measure on infinite measure.
    ///
    /// # Examples
    ///
    /// - Some integration operations require finite base measure
    /// - Certain product measures require finiteness
    #[error("Operation requires finite measure, but measure is infinite")]
    RequiresFiniteMeasure,

    /// Numerical computation error
    ///
    /// Occurs when numerical integration or computation fails.
    ///
    /// # Examples
    ///
    /// - Integration algorithm fails to converge
    /// - Numerical instability in computation
    /// - Overflow/underflow in calculations
    #[error("Numerical error in {operation}: {details}")]
    NumericalError {
        /// Operation that failed
        operation: String,
        /// Details about the numerical issue
        details: String,
    },

    /// Invalid set operation
    ///
    /// Occurs when attempting an invalid set-theoretic operation.
    ///
    /// # Examples
    ///
    /// - Complement operation on non-complementable set
    /// - Union of sets from different σ-algebras
    /// - Invalid partition construction
    #[error("Invalid set operation: {description}")]
    InvalidSetOperation {
        /// Description of the invalid operation
        description: String,
    },

    /// Generic computation error with context
    ///
    /// Used for other errors that don't fit specific categories.
    #[error("Computation error: {message}")]
    ComputationError {
        /// Error message
        message: String,
    },
}

impl MeasureError {
    /// Create a not measurable error
    pub fn not_measurable(description: impl Into<String>) -> Self {
        Self::NotMeasurable {
            description: description.into(),
        }
    }

    /// Create a not integrable error
    pub fn not_integrable(reason: impl Into<String>) -> Self {
        Self::NotIntegrable {
            reason: reason.into(),
        }
    }

    /// Create a not absolutely continuous error
    pub fn not_absolutely_continuous(details: impl Into<String>) -> Self {
        Self::NotAbsolutelyContinuous {
            details: details.into(),
        }
    }

    /// Create a not σ-finite error
    pub fn not_sigma_finite(reason: impl Into<String>) -> Self {
        Self::NotSigmaFinite {
            reason: reason.into(),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a convergence conditions not satisfied error
    pub fn convergence_conditions(
        theorem: impl Into<String>,
        condition: impl Into<String>,
    ) -> Self {
        Self::ConvergenceConditionsNotSatisfied {
            theorem: theorem.into(),
            condition: condition.into(),
        }
    }

    /// Create an invalid probability measure error
    pub fn invalid_probability(total: f64) -> Self {
        Self::InvalidProbabilityMeasure { total }
    }

    /// Create a requires finite measure error
    pub fn requires_finite() -> Self {
        Self::RequiresFiniteMeasure
    }

    /// Create a numerical error
    pub fn numerical(operation: impl Into<String>, details: impl Into<String>) -> Self {
        Self::NumericalError {
            operation: operation.into(),
            details: details.into(),
        }
    }

    /// Create an invalid set operation error
    pub fn invalid_set_operation(description: impl Into<String>) -> Self {
        Self::InvalidSetOperation {
            description: description.into(),
        }
    }

    /// Create a generic computation error
    pub fn computation(message: impl Into<String>) -> Self {
        Self::ComputationError {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_measurable_error() {
        let err = MeasureError::not_measurable("Vitali set");
        assert_eq!(err.to_string(), "Set is not measurable: Vitali set");
    }

    #[test]
    fn test_not_integrable_error() {
        let err = MeasureError::not_integrable("infinite integral");
        assert_eq!(
            err.to_string(),
            "Function is not integrable: infinite integral"
        );
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = MeasureError::dimension_mismatch(3, 2);
        assert_eq!(err.to_string(), "Dimension mismatch: expected 3, got 2");
    }

    #[test]
    fn test_convergence_conditions_error() {
        let err =
            MeasureError::convergence_conditions("Dominated Convergence", "dominating function");
        assert_eq!(
            err.to_string(),
            "Convergence theorem conditions not satisfied: Dominated Convergence requires dominating function"
        );
    }

    #[test]
    fn test_invalid_probability_error() {
        let err = MeasureError::invalid_probability(0.95);
        assert_eq!(
            err.to_string(),
            "Invalid probability measure: total measure is 0.95, expected 1.0"
        );
    }

    #[test]
    fn test_numerical_error() {
        let err = MeasureError::numerical("integration", "overflow");
        assert_eq!(err.to_string(), "Numerical error in integration: overflow");
    }

    #[test]
    fn test_error_clone() {
        let err = MeasureError::not_measurable("test");
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}
