//! Posterior computation and priors for Bayesian inference

use crate::distribution::Distribution;
use crate::error::{ProbabilisticError, Result};
use crate::sampling::{MetropolisHastings, Sampler};
use amari_core::Multivector;
use rand::Rng;
use std::marker::PhantomData;

/// Bayesian model on geometric algebra spaces
///
/// Combines a prior distribution with a likelihood to enable posterior inference.
///
/// # Type Parameters
///
/// * `P, Q, R` - Clifford algebra signature
/// * `Prior` - Prior distribution type
/// * `Likelihood` - Likelihood function type
#[derive(Debug, Clone)]
pub struct BayesianGA<const P: usize, const Q: usize, const R: usize, Prior, Likelihood> {
    /// Prior distribution over parameters
    prior: Prior,
    /// Likelihood function
    likelihood: Likelihood,
    /// Observations
    observations: Vec<Multivector<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize, Prior, Likelihood>
    BayesianGA<P, Q, R, Prior, Likelihood>
where
    Prior: Distribution<Multivector<P, Q, R>>,
    Likelihood: Fn(&Multivector<P, Q, R>, &[Multivector<P, Q, R>]) -> Result<f64>,
{
    /// Create a new Bayesian model
    ///
    /// # Arguments
    ///
    /// * `prior` - Prior distribution over parameters
    /// * `likelihood` - Function computing log-likelihood of data given parameters
    pub fn new(prior: Prior, likelihood: Likelihood) -> Self {
        Self {
            prior,
            likelihood,
            observations: Vec::new(),
        }
    }

    /// Add observations to the model
    pub fn observe(&mut self, data: Vec<Multivector<P, Q, R>>) {
        self.observations.extend(data);
    }

    /// Compute log-posterior (up to normalizing constant)
    ///
    /// log p(θ|D) ∝ log p(D|θ) + log p(θ)
    pub fn log_posterior(&self, theta: &Multivector<P, Q, R>) -> Result<f64> {
        let log_prior = self.prior.log_prob(theta)?;
        let log_likelihood = (self.likelihood)(theta, &self.observations)?;
        Ok(log_prior + log_likelihood)
    }

    /// Sample from the posterior using MCMC
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to draw
    /// * `burnin` - Number of burnin iterations
    /// * `proposal_std` - Standard deviation for Metropolis-Hastings proposals
    pub fn sample_posterior<RNG: Rng>(
        &self,
        rng: &mut RNG,
        num_samples: usize,
        burnin: usize,
        proposal_std: f64,
    ) -> Result<Vec<Multivector<P, Q, R>>> {
        // Create a distribution adapter for the posterior
        let posterior_dist = PosteriorDistribution {
            model: self,
            _phantom: PhantomData,
        };

        let mut sampler = MetropolisHastings::new(&posterior_dist, proposal_std);
        sampler.run(rng, num_samples, burnin)
    }

    /// Get the prior
    pub fn prior(&self) -> &Prior {
        &self.prior
    }

    /// Get the observations
    pub fn observations(&self) -> &[Multivector<P, Q, R>] {
        &self.observations
    }
}

/// Wrapper to make BayesianGA usable as a Distribution
struct PosteriorDistribution<'a, const P: usize, const Q: usize, const R: usize, Prior, Likelihood>
{
    model: &'a BayesianGA<P, Q, R, Prior, Likelihood>,
    _phantom: PhantomData<(Prior, Likelihood)>,
}

impl<'a, const P: usize, const Q: usize, const R: usize, Prior, Likelihood>
    Distribution<Multivector<P, Q, R>> for PosteriorDistribution<'a, P, Q, R, Prior, Likelihood>
where
    Prior: Distribution<Multivector<P, Q, R>>,
    Likelihood: Fn(&Multivector<P, Q, R>, &[Multivector<P, Q, R>]) -> Result<f64>,
{
    fn sample<R_: Rng>(&self, rng: &mut R_) -> Multivector<P, Q, R> {
        // Sample from prior as initial point
        self.model.prior.sample(rng)
    }

    fn log_prob(&self, x: &Multivector<P, Q, R>) -> Result<f64> {
        self.model.log_posterior(x)
    }
}

/// Jeffreys prior for location parameters
///
/// The Jeffreys prior is an objective, non-informative prior that is
/// invariant under reparameterization. For location parameters, it is flat.
#[derive(Debug, Clone, Default)]
pub struct JeffreysPrior<const P: usize, const Q: usize, const R: usize> {
    /// Lower bound for parameters (for numerical stability)
    lower_bound: f64,
    /// Upper bound for parameters
    upper_bound: f64,
    _phantom: PhantomData<((), (), ())>,
}

impl<const P: usize, const Q: usize, const R: usize> JeffreysPrior<P, Q, R> {
    /// Create a Jeffreys prior with default bounds
    pub fn new() -> Self {
        Self {
            lower_bound: -1e10,
            upper_bound: 1e10,
            _phantom: PhantomData,
        }
    }

    /// Create with specified bounds
    pub fn with_bounds(lower: f64, upper: f64) -> Self {
        Self {
            lower_bound: lower,
            upper_bound: upper,
            _phantom: PhantomData,
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> Distribution<Multivector<P, Q, R>>
    for JeffreysPrior<P, Q, R>
{
    fn sample<RNG: Rng>(&self, rng: &mut RNG) -> Multivector<P, Q, R> {
        // Sample uniformly from bounds
        let dim = 1 << (P + Q + R);
        let range = self.upper_bound - self.lower_bound;

        let coeffs: Vec<f64> = (0..dim)
            .map(|_| self.lower_bound + rng.gen::<f64>() * range)
            .collect();

        Multivector::from_coefficients(coeffs)
    }

    fn log_prob(&self, x: &Multivector<P, Q, R>) -> Result<f64> {
        // Flat prior - check bounds
        let dim = 1 << (P + Q + R);

        for i in 0..dim {
            let xi = x.get(i);
            if xi < self.lower_bound || xi > self.upper_bound {
                return Ok(f64::NEG_INFINITY);
            }
        }

        // Constant log probability (improper prior normalization)
        Ok(0.0)
    }
}

/// Conjugate Gaussian prior
///
/// For Gaussian likelihood with known variance, the conjugate prior is Gaussian.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GaussianPrior<const P: usize, const Q: usize, const R: usize> {
    /// Prior mean
    mean: Multivector<P, Q, R>,
    /// Prior variance (per component)
    variance: f64,
}

#[allow(dead_code)]
impl<const P: usize, const Q: usize, const R: usize> GaussianPrior<P, Q, R> {
    /// Create a new Gaussian prior
    pub fn new(mean: Multivector<P, Q, R>, variance: f64) -> Result<Self> {
        if variance <= 0.0 {
            return Err(ProbabilisticError::invalid_parameters(
                "Variance must be positive",
            ));
        }
        Ok(Self { mean, variance })
    }

    /// Create a diffuse prior (large variance)
    pub fn diffuse() -> Self {
        Self {
            mean: Multivector::zero(),
            variance: 1e6,
        }
    }

    /// Compute posterior mean and variance given data
    ///
    /// For Gaussian-Gaussian conjugacy with known likelihood variance σ²:
    /// - Posterior mean: (τ⁻²μ₀ + nσ⁻²x̄) / (τ⁻² + nσ⁻²)
    /// - Posterior variance: 1 / (τ⁻² + nσ⁻²)
    pub fn posterior(
        &self,
        observations: &[Multivector<P, Q, R>],
        likelihood_variance: f64,
    ) -> Result<(Multivector<P, Q, R>, f64)> {
        if observations.is_empty() {
            return Ok((self.mean.clone(), self.variance));
        }

        let n = observations.len() as f64;
        let dim = 1 << (P + Q + R);

        // Compute sample mean
        let mut sum: Multivector<P, Q, R> = Multivector::zero();
        for obs in observations {
            sum = sum.add(obs);
        }
        let scale = 1.0 / n;
        let sample_mean_coeffs: Vec<f64> = sum.to_vec().iter().map(|&x| x * scale).collect();
        let sample_mean: Multivector<P, Q, R> = Multivector::from_coefficients(sample_mean_coeffs);

        // Posterior precision = prior precision + data precision
        let prior_precision = 1.0 / self.variance;
        let data_precision = n / likelihood_variance;
        let posterior_precision = prior_precision + data_precision;
        let posterior_variance = 1.0 / posterior_precision;

        // Posterior mean (weighted average)
        let prior_mean_vec = self.mean.to_vec();
        let sample_mean_vec = sample_mean.to_vec();

        let posterior_mean_coeffs: Vec<f64> = (0..dim)
            .map(|i| {
                (prior_precision * prior_mean_vec[i] + data_precision * sample_mean_vec[i])
                    / posterior_precision
            })
            .collect();

        Ok((
            Multivector::from_coefficients(posterior_mean_coeffs),
            posterior_variance,
        ))
    }
}

impl<const P: usize, const Q: usize, const R: usize> Distribution<Multivector<P, Q, R>>
    for GaussianPrior<P, Q, R>
{
    fn sample<RNG: Rng>(&self, rng: &mut RNG) -> Multivector<P, Q, R> {
        use rand_distr::{Distribution as RandDist, Normal};

        let dim = 1 << (P + Q + R);
        let std_dev = self.variance.sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();

        let coeffs: Vec<f64> = (0..dim)
            .map(|i| self.mean.get(i) + normal.sample(rng))
            .collect();

        Multivector::from_coefficients(coeffs)
    }

    fn log_prob(&self, x: &Multivector<P, Q, R>) -> Result<f64> {
        let dim = 1 << (P + Q + R);
        let n = dim as f64;

        let mut log_p = -0.5 * n * (2.0 * std::f64::consts::PI * self.variance).ln();

        for i in 0..dim {
            let diff = x.get(i) - self.mean.get(i);
            log_p -= 0.5 * diff * diff / self.variance;
        }

        Ok(log_p)
    }
}

/// Helper type for posterior sampling
pub struct PosteriorSampler<const P: usize, const Q: usize, const R: usize> {
    /// Samples from the posterior
    samples: Vec<Multivector<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize> PosteriorSampler<P, Q, R> {
    /// Create from samples
    pub fn new(samples: Vec<Multivector<P, Q, R>>) -> Self {
        Self { samples }
    }

    /// Get the samples
    pub fn samples(&self) -> &[Multivector<P, Q, R>] {
        &self.samples
    }

    /// Compute posterior mean
    pub fn mean(&self) -> Result<Multivector<P, Q, R>> {
        if self.samples.is_empty() {
            return Err(ProbabilisticError::insufficient_samples(1, 0));
        }

        let _dim = 1 << (P + Q + R);
        let n = self.samples.len() as f64;

        let mut sum = Multivector::zero();
        for sample in &self.samples {
            sum = sum.add(sample);
        }

        let coeffs: Vec<f64> = sum.to_vec().iter().map(|&x| x / n).collect();
        Ok(Multivector::from_coefficients(coeffs))
    }

    /// Compute credible interval for a specific component
    pub fn credible_interval(&self, component: usize, level: f64) -> Result<(f64, f64)> {
        if self.samples.is_empty() {
            return Err(ProbabilisticError::insufficient_samples(1, 0));
        }

        let dim = 1 << (P + Q + R);
        if component >= dim {
            return Err(ProbabilisticError::dimension_mismatch(dim, component + 1));
        }

        let mut values: Vec<f64> = self.samples.iter().map(|s| s.get(component)).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = (1.0 - level) / 2.0;
        let n = values.len();
        let lower_idx = (alpha * n as f64).floor() as usize;
        let upper_idx = ((1.0 - alpha) * n as f64).ceil() as usize;

        Ok((values[lower_idx], values[upper_idx.min(n - 1)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jeffreys_prior() {
        let prior = JeffreysPrior::<2, 0, 0>::new();
        let mut rng = rand::thread_rng();

        let sample = prior.sample(&mut rng);
        let log_p = prior.log_prob(&sample).unwrap();

        // Flat prior has log_prob = 0
        assert_eq!(log_p, 0.0);
    }

    #[test]
    fn test_gaussian_prior() {
        let mean = Multivector::<2, 0, 0>::scalar(1.0);
        let prior = GaussianPrior::new(mean.clone(), 1.0).unwrap();
        let mut rng = rand::thread_rng();

        let samples = prior.sample_n(&mut rng, 100);
        assert_eq!(samples.len(), 100);

        // Log prob at mean should be highest
        let log_p_mean = prior.log_prob(&mean).unwrap();
        let other = Multivector::scalar(5.0);
        let log_p_other = prior.log_prob(&other).unwrap();
        assert!(log_p_mean > log_p_other);
    }

    #[test]
    fn test_gaussian_conjugacy() {
        let prior = GaussianPrior::<2, 0, 0>::new(Multivector::zero(), 10.0).unwrap();

        // Generate some observations near 2.0
        let obs: Vec<Multivector<2, 0, 0>> = vec![
            Multivector::scalar(2.1),
            Multivector::scalar(1.9),
            Multivector::scalar(2.0),
        ];

        let (post_mean, post_var) = prior.posterior(&obs, 1.0).unwrap();

        // Posterior mean should be between prior mean (0) and data mean (~2)
        assert!(post_mean.get(0) > 0.0);
        assert!(post_mean.get(0) < 2.0);

        // Posterior variance should be less than prior variance
        assert!(post_var < 10.0);
    }

    #[test]
    fn test_bayesian_model() {
        let prior = GaussianPrior::<2, 0, 0>::diffuse();

        // Simple likelihood: Gaussian centered at parameter
        let likelihood =
            |theta: &Multivector<2, 0, 0>, data: &[Multivector<2, 0, 0>]| -> Result<f64> {
                let mut log_lik = 0.0;
                for x in data {
                    for i in 0..4 {
                        let diff = x.get(i) - theta.get(i);
                        log_lik -= 0.5 * diff * diff;
                    }
                }
                Ok(log_lik)
            };

        let mut model = BayesianGA::new(prior, likelihood);
        model.observe(vec![Multivector::scalar(1.0), Multivector::scalar(1.1)]);

        // Test log posterior
        let theta = Multivector::scalar(1.0);
        let log_post = model.log_posterior(&theta).unwrap();
        assert!(log_post.is_finite());
    }

    #[test]
    fn test_posterior_sampler() {
        let samples = vec![
            Multivector::<2, 0, 0>::scalar(1.0),
            Multivector::<2, 0, 0>::scalar(2.0),
            Multivector::<2, 0, 0>::scalar(3.0),
        ];

        let sampler = PosteriorSampler::new(samples);

        let mean = sampler.mean().unwrap();
        assert!((mean.get(0) - 2.0).abs() < 1e-10);

        let (lower, upper) = sampler.credible_interval(0, 0.5).unwrap();
        assert!(lower <= upper);
    }
}
