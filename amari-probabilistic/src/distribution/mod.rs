//! Distribution trait and core abstractions
//!
//! This module provides the foundational `Distribution` trait for probability
//! distributions over geometric algebra spaces.
//!
//! # Core Concept
//!
//! A distribution over a Clifford algebra Cl(P,Q,R) assigns probabilities to
//! measurable subsets of the multivector space. The `Distribution` trait abstracts
//! the essential operations: sampling, density evaluation, and support queries.
//!
//! # Examples
//!
//! ```ignore
//! use amari_probabilistic::distribution::{Distribution, MultivectorDistribution};
//! use amari_core::Multivector;
//!
//! // Create a Gaussian distribution on Cl(3,0,0)
//! let dist = MultivectorDistribution::<3, 0, 0>::gaussian(
//!     Multivector::zero(),  // mean
//!     1.0,                   // variance per component
//! );
//!
//! // Sample from the distribution
//! let mut rng = rand::thread_rng();
//! let sample: Multivector<3, 0, 0> = dist.sample(&mut rng);
//!
//! // Evaluate log probability
//! let log_prob = dist.log_prob(&sample)?;
//! ```

mod multivector;

pub use multivector::{
    GaussianMultivector, GradeProjectedDistribution, MultivectorDistribution, UniformMultivector,
};

use crate::error::Result;
use rand::Rng;

/// Core trait for probability distributions over geometric algebra
///
/// This trait defines the fundamental operations for any probability distribution
/// on a multivector space. Implementations may be parametric (e.g., Gaussian with
/// mean and variance) or non-parametric (e.g., kernel density estimates).
///
/// # Type Parameters
///
/// - `T`: The output type of the distribution (typically `Multivector<P,Q,R>`)
///
/// # Required Methods
///
/// - `sample`: Draw a random sample from the distribution
/// - `log_prob`: Compute the log-probability density at a point
///
/// # Provided Methods
///
/// - `prob`: Compute the probability density (exp of log_prob)
/// - `sample_n`: Draw multiple independent samples
pub trait Distribution<T> {
    /// Draw a random sample from the distribution
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// A sample from the distribution
    fn sample<R: Rng>(&self, rng: &mut R) -> T;

    /// Compute the log-probability density at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the density
    ///
    /// # Returns
    ///
    /// Log of the probability density, or error if point is outside support
    ///
    /// # Notes
    ///
    /// For discrete distributions, this returns log of the probability mass.
    /// For continuous distributions, this returns log of the density.
    fn log_prob(&self, x: &T) -> Result<f64>;

    /// Compute the probability density at a point
    ///
    /// # Arguments
    ///
    /// * `x` - Point at which to evaluate the density
    ///
    /// # Returns
    ///
    /// Probability density (or mass for discrete distributions)
    fn prob(&self, x: &T) -> Result<f64> {
        self.log_prob(x).map(|lp| lp.exp())
    }

    /// Draw multiple independent samples from the distribution
    ///
    /// # Arguments
    ///
    /// * `rng` - Random number generator
    /// * `n` - Number of samples to draw
    ///
    /// # Returns
    ///
    /// Vector of n independent samples
    fn sample_n<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<T> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

/// Trait for distributions with known support
///
/// Provides methods to query the support of the distribution.
pub trait HasSupport<T> {
    /// Check if a point is in the support of the distribution
    ///
    /// # Arguments
    ///
    /// * `x` - Point to check
    ///
    /// # Returns
    ///
    /// True if x is in the support (has positive probability density)
    fn in_support(&self, x: &T) -> bool;

    /// Get bounds on the support (if bounded)
    ///
    /// Returns `None` if the support is unbounded.
    fn support_bounds(&self) -> Option<(T, T)>;
}

/// Trait for distributions with finite moments
///
/// Provides methods to compute moments of the distribution.
pub trait HasMoments<T>: Distribution<T> {
    /// Compute the mean (first moment)
    fn mean(&self) -> T;

    /// Compute the variance (second central moment)
    fn variance(&self) -> T;

    /// Compute a raw moment of given order
    ///
    /// # Arguments
    ///
    /// * `order` - Order of the moment to compute
    fn raw_moment(&self, order: usize) -> Result<T>;

    /// Compute a central moment of given order
    ///
    /// # Arguments
    ///
    /// * `order` - Order of the central moment to compute
    fn central_moment(&self, order: usize) -> Result<T>;
}

/// Trait for distributions in exponential family form
///
/// Exponential families have the form: p(x|θ) = h(x) exp(η(θ)·T(x) - A(θ))
///
/// This structure enables efficient inference and natural gradient methods.
pub trait ExponentialFamily<T>: Distribution<T> {
    /// Type of the natural parameters η
    type NaturalParams;

    /// Type of the sufficient statistics T(x)
    type SufficientStats;

    /// Get the natural parameters
    fn natural_params(&self) -> &Self::NaturalParams;

    /// Compute sufficient statistics for a sample
    fn sufficient_stats(&self, x: &T) -> Self::SufficientStats;

    /// Compute the log-partition function A(η)
    fn log_partition(&self) -> f64;

    /// Create from natural parameters
    fn from_natural(params: Self::NaturalParams) -> Self;
}

/// Trait for reparameterizable distributions
///
/// Distributions that support the reparameterization trick for gradient estimation.
/// Essential for variational inference and gradient-based optimization.
pub trait Reparameterizable<T>: Distribution<T> {
    /// Type of the base noise distribution
    type BaseNoise;

    /// Sample base noise
    fn sample_noise<R: Rng>(&self, rng: &mut R) -> Self::BaseNoise;

    /// Transform noise to sample (deterministic given noise)
    fn transform(&self, noise: &Self::BaseNoise) -> T;

    /// Sample with reparameterization
    ///
    /// Returns both the sample and the noise used to generate it.
    fn sample_reparam<R: Rng>(&self, rng: &mut R) -> (T, Self::BaseNoise) {
        let noise = self.sample_noise(rng);
        let sample = self.transform(&noise);
        (sample, noise)
    }
}

/// Trait for conjugate prior-likelihood pairs
///
/// When the prior and posterior are in the same family, we have conjugacy.
pub trait ConjugateTo<Likelihood>: Distribution<Likelihood> {
    /// Type of the posterior distribution (same family as prior)
    type Posterior: Distribution<Likelihood>;

    /// Update prior with observed data to get posterior
    fn posterior(&self, data: &[Likelihood]) -> Self::Posterior;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock distribution for testing trait definitions
    struct MockDistribution {
        mean: f64,
    }

    impl Distribution<f64> for MockDistribution {
        fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
            self.mean + rng.gen::<f64>() - 0.5
        }

        fn log_prob(&self, x: &f64) -> Result<f64> {
            // Simple uniform-ish distribution
            let diff = (x - self.mean).abs();
            if diff > 0.5 {
                Ok(f64::NEG_INFINITY)
            } else {
                Ok(0.0) // log(1)
            }
        }
    }

    #[test]
    fn test_distribution_trait() {
        let dist = MockDistribution { mean: 5.0 };
        let mut rng = rand::thread_rng();

        let sample = dist.sample(&mut rng);
        assert!((sample - 5.0).abs() <= 0.5);

        let log_p = dist.log_prob(&5.0).unwrap();
        assert_eq!(log_p, 0.0);

        let p = dist.prob(&5.0).unwrap();
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_sample_n() {
        let dist = MockDistribution { mean: 0.0 };
        let mut rng = rand::thread_rng();

        let samples = dist.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);

        for s in samples {
            assert!(s.abs() <= 0.5);
        }
    }
}
