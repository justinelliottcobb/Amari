//! Markov Chain Monte Carlo sampling algorithms
//!
//! This module implements MCMC methods for sampling from distributions
//! over multivector spaces.

use crate::distribution::Distribution;
use crate::error::{ProbabilisticError, Result};
use amari_core::Multivector;
use rand::Rng;
use rand_distr::{Distribution as RandDist, Normal};

/// Trait for MCMC samplers
pub trait Sampler<T> {
    /// Take a single MCMC step
    fn step<RNG: Rng>(&mut self, rng: &mut RNG) -> T;

    /// Run the sampler for a given number of steps
    fn run<RNG: Rng>(
        &mut self,
        rng: &mut RNG,
        num_samples: usize,
        burnin: usize,
    ) -> Result<Vec<T>> {
        // Burnin phase
        for _ in 0..burnin {
            self.step(rng);
        }

        // Collection phase
        let samples: Vec<T> = (0..num_samples).map(|_| self.step(rng)).collect();

        Ok(samples)
    }

    /// Get diagnostics about the sampling run
    fn diagnostics(&self) -> MCMCDiagnostics;
}

/// State of an MCMC sampler
#[derive(Debug, Clone)]
pub struct SamplerState<const P: usize, const Q: usize, const R: usize> {
    /// Current sample
    pub current: Multivector<P, Q, R>,
    /// Log probability at current sample
    pub log_prob: f64,
    /// Number of steps taken
    pub num_steps: usize,
    /// Number of accepted proposals
    pub num_accepted: usize,
}

impl<const P: usize, const Q: usize, const R: usize> SamplerState<P, Q, R> {
    /// Compute acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.num_steps == 0 {
            0.0
        } else {
            self.num_accepted as f64 / self.num_steps as f64
        }
    }
}

/// Diagnostics for MCMC sampling
#[derive(Debug, Clone)]
pub struct MCMCDiagnostics {
    /// Total number of steps
    pub num_steps: usize,
    /// Acceptance rate
    pub acceptance_rate: f64,
    /// Effective sample size estimate
    pub effective_sample_size: Option<f64>,
    /// R-hat convergence diagnostic
    pub r_hat: Option<f64>,
}

impl MCMCDiagnostics {
    /// Check if the sampler has converged (R-hat < 1.1)
    pub fn is_converged(&self) -> bool {
        self.r_hat.map_or(true, |r| r < 1.1)
    }
}

/// Metropolis-Hastings sampler for multivector distributions
///
/// Uses isotropic Gaussian proposals centered at the current state.
pub struct MetropolisHastings<'a, const P: usize, const Q: usize, const R: usize, D>
where
    D: Distribution<Multivector<P, Q, R>>,
{
    /// Target distribution
    target: &'a D,
    /// Proposal standard deviation
    proposal_std: f64,
    /// Current state
    state: SamplerState<P, Q, R>,
}

impl<'a, const P: usize, const Q: usize, const R: usize, D> MetropolisHastings<'a, P, Q, R, D>
where
    D: Distribution<Multivector<P, Q, R>>,
{
    /// Dimension of the multivector space
    #[allow(dead_code)]
    const DIM: usize = 1 << (P + Q + R);

    /// Create a new Metropolis-Hastings sampler
    ///
    /// # Arguments
    ///
    /// * `target` - Target distribution to sample from
    /// * `proposal_std` - Standard deviation of the Gaussian proposal
    pub fn new(target: &'a D, proposal_std: f64) -> Self {
        let mut rng = rand::thread_rng();
        let initial = target.sample(&mut rng);
        let log_prob = target.log_prob(&initial).unwrap_or(f64::NEG_INFINITY);

        Self {
            target,
            proposal_std,
            state: SamplerState {
                current: initial,
                log_prob,
                num_steps: 0,
                num_accepted: 0,
            },
        }
    }

    /// Create with a specific initial state
    pub fn with_initial(
        target: &'a D,
        proposal_std: f64,
        initial: Multivector<P, Q, R>,
    ) -> Result<Self> {
        let log_prob = target.log_prob(&initial)?;

        Ok(Self {
            target,
            proposal_std,
            state: SamplerState {
                current: initial,
                log_prob,
                num_steps: 0,
                num_accepted: 0,
            },
        })
    }

    /// Get the current state
    pub fn current(&self) -> &Multivector<P, Q, R> {
        &self.state.current
    }

    /// Get the acceptance rate so far
    pub fn acceptance_rate(&self) -> f64 {
        self.state.acceptance_rate()
    }

    /// Propose a new state
    fn propose<RNG: Rng>(&self, rng: &mut RNG) -> Multivector<P, Q, R> {
        let normal = Normal::new(0.0, self.proposal_std).unwrap();
        let mut coeffs = self.state.current.to_vec();

        for c in coeffs.iter_mut() {
            *c += normal.sample(rng);
        }

        Multivector::from_coefficients(coeffs)
    }
}

impl<'a, const P: usize, const Q: usize, const R: usize, D> Sampler<Multivector<P, Q, R>>
    for MetropolisHastings<'a, P, Q, R, D>
where
    D: Distribution<Multivector<P, Q, R>>,
{
    fn step<RNG: Rng>(&mut self, rng: &mut RNG) -> Multivector<P, Q, R> {
        // Propose new state
        let proposal = self.propose(rng);

        // Compute acceptance probability
        let proposal_log_prob = self.target.log_prob(&proposal).unwrap_or(f64::NEG_INFINITY);

        let log_accept_ratio = proposal_log_prob - self.state.log_prob;
        let accept = if log_accept_ratio >= 0.0 {
            true
        } else {
            let u: f64 = rng.gen();
            u.ln() < log_accept_ratio
        };

        self.state.num_steps += 1;

        if accept {
            self.state.current = proposal;
            self.state.log_prob = proposal_log_prob;
            self.state.num_accepted += 1;
        }

        self.state.current.clone()
    }

    fn diagnostics(&self) -> MCMCDiagnostics {
        MCMCDiagnostics {
            num_steps: self.state.num_steps,
            acceptance_rate: self.state.acceptance_rate(),
            effective_sample_size: None,
            r_hat: None,
        }
    }
}

/// Hamiltonian Monte Carlo sampler for multivector distributions
///
/// Uses the geometric product to update momentum in a geometric-aware way.
#[allow(dead_code)]
pub struct HamiltonianMonteCarlo<'a, const P: usize, const Q: usize, const R: usize, D>
where
    D: Distribution<Multivector<P, Q, R>>,
{
    /// Target distribution (defines potential energy)
    target: &'a D,
    /// Step size for leapfrog integration
    step_size: f64,
    /// Number of leapfrog steps
    num_leapfrog: usize,
    /// Mass matrix diagonal (inverse)
    mass_inv: Vec<f64>,
    /// Current state
    state: SamplerState<P, Q, R>,
}

#[allow(dead_code)]
impl<'a, const P: usize, const Q: usize, const R: usize, D> HamiltonianMonteCarlo<'a, P, Q, R, D>
where
    D: Distribution<Multivector<P, Q, R>>,
{
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Create a new HMC sampler
    ///
    /// # Arguments
    ///
    /// * `target` - Target distribution (log prob defines potential energy)
    /// * `step_size` - Step size for leapfrog integration
    /// * `num_leapfrog` - Number of leapfrog steps per proposal
    pub fn new(target: &'a D, step_size: f64, num_leapfrog: usize) -> Self {
        let mut rng = rand::thread_rng();
        let initial = target.sample(&mut rng);
        let log_prob = target.log_prob(&initial).unwrap_or(f64::NEG_INFINITY);

        Self {
            target,
            step_size,
            num_leapfrog,
            mass_inv: vec![1.0; Self::DIM],
            state: SamplerState {
                current: initial,
                log_prob,
                num_steps: 0,
                num_accepted: 0,
            },
        }
    }

    /// Set mass matrix diagonal
    pub fn with_mass(mut self, mass_diag: Vec<f64>) -> Result<Self> {
        if mass_diag.len() != Self::DIM {
            return Err(ProbabilisticError::dimension_mismatch(
                Self::DIM,
                mass_diag.len(),
            ));
        }

        self.mass_inv = mass_diag.iter().map(|&m| 1.0 / m).collect();
        Ok(self)
    }

    /// Compute gradient of log probability via finite differences
    fn gradient(&self, x: &Multivector<P, Q, R>) -> Multivector<P, Q, R> {
        let eps = 1e-6;
        let mut grad = vec![0.0; Self::DIM];
        let x_vec = x.to_vec();

        for i in 0..Self::DIM {
            let mut x_plus = x_vec.clone();
            let mut x_minus = x_vec.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;

            let mv_plus = Multivector::from_coefficients(x_plus);
            let mv_minus = Multivector::from_coefficients(x_minus);

            let log_p_plus = self.target.log_prob(&mv_plus).unwrap_or(f64::NEG_INFINITY);
            let log_p_minus = self.target.log_prob(&mv_minus).unwrap_or(f64::NEG_INFINITY);

            grad[i] = (log_p_plus - log_p_minus) / (2.0 * eps);
        }

        Multivector::from_coefficients(grad)
    }

    /// Sample momentum from prior
    fn sample_momentum<RNG: Rng>(&self, rng: &mut RNG) -> Multivector<P, Q, R> {
        let normal = Normal::new(0.0, 1.0).unwrap();

        let p: Vec<f64> = self
            .mass_inv
            .iter()
            .map(|&m_inv| normal.sample(rng) / m_inv.sqrt())
            .collect();

        Multivector::from_coefficients(p)
    }

    /// Compute kinetic energy
    fn kinetic_energy(&self, p: &Multivector<P, Q, R>) -> f64 {
        let p_vec = p.to_vec();

        p_vec
            .iter()
            .zip(self.mass_inv.iter())
            .map(|(&pi, &m_inv)| 0.5 * m_inv * pi * pi)
            .sum()
    }

    /// Leapfrog integration step
    fn leapfrog(
        &self,
        q: &Multivector<P, Q, R>,
        p: &Multivector<P, Q, R>,
    ) -> (Multivector<P, Q, R>, Multivector<P, Q, R>) {
        let mut q_vec = q.to_vec();
        let mut p_vec = p.to_vec();

        // Half step for momentum
        let grad = self.gradient(&Multivector::from_coefficients(q_vec.clone()));
        let grad_vec = grad.to_vec();
        for i in 0..Self::DIM {
            p_vec[i] += 0.5 * self.step_size * grad_vec[i];
        }

        // Full steps
        for _ in 0..self.num_leapfrog {
            // Full step for position
            for i in 0..Self::DIM {
                q_vec[i] += self.step_size * self.mass_inv[i] * p_vec[i];
            }

            // Full step for momentum (except last)
            let grad = self.gradient(&Multivector::from_coefficients(q_vec.clone()));
            let grad_vec = grad.to_vec();
            for i in 0..Self::DIM {
                p_vec[i] += self.step_size * grad_vec[i];
            }
        }

        // Undo last half momentum step
        let grad = self.gradient(&Multivector::from_coefficients(q_vec.clone()));
        let grad_vec = grad.to_vec();
        for i in 0..Self::DIM {
            p_vec[i] -= 0.5 * self.step_size * grad_vec[i];
        }

        (
            Multivector::from_coefficients(q_vec),
            Multivector::from_coefficients(p_vec),
        )
    }
}

impl<'a, const P: usize, const Q: usize, const R: usize, D> Sampler<Multivector<P, Q, R>>
    for HamiltonianMonteCarlo<'a, P, Q, R, D>
where
    D: Distribution<Multivector<P, Q, R>>,
{
    fn step<RNG: Rng>(&mut self, rng: &mut RNG) -> Multivector<P, Q, R> {
        // Sample momentum
        let p0 = self.sample_momentum(rng);

        // Compute initial Hamiltonian
        let ke0 = self.kinetic_energy(&p0);
        let h0 = -self.state.log_prob + ke0; // H = U + K = -log p + K

        // Leapfrog integration
        let (q_new, p_new) = self.leapfrog(&self.state.current, &p0);

        // Compute final Hamiltonian
        let log_prob_new = self.target.log_prob(&q_new).unwrap_or(f64::NEG_INFINITY);
        let ke_new = self.kinetic_energy(&p_new);
        let h_new = -log_prob_new + ke_new;

        // Accept/reject
        let log_accept = h0 - h_new;
        let accept = if log_accept >= 0.0 {
            true
        } else {
            let u: f64 = rng.gen();
            u.ln() < log_accept
        };

        self.state.num_steps += 1;

        if accept {
            self.state.current = q_new;
            self.state.log_prob = log_prob_new;
            self.state.num_accepted += 1;
        }

        self.state.current.clone()
    }

    fn diagnostics(&self) -> MCMCDiagnostics {
        MCMCDiagnostics {
            num_steps: self.state.num_steps,
            acceptance_rate: self.state.acceptance_rate(),
            effective_sample_size: None,
            r_hat: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::GaussianMultivector;

    #[test]
    fn test_metropolis_hastings() {
        let target = GaussianMultivector::<2, 0, 0>::standard();
        let mut sampler = MetropolisHastings::new(&target, 0.5);
        let mut rng = rand::thread_rng();

        // Run for a few steps
        let samples = sampler.run(&mut rng, 100, 50).unwrap();
        assert_eq!(samples.len(), 100);

        // Acceptance rate should be reasonable (not 0 or 1)
        let rate = sampler.acceptance_rate();
        assert!(rate > 0.1);
        assert!(rate < 0.9);
    }

    #[test]
    fn test_hmc() {
        let target = GaussianMultivector::<2, 0, 0>::standard();
        let mut sampler = HamiltonianMonteCarlo::new(&target, 0.1, 10);
        let mut rng = rand::thread_rng();

        // Run for a few steps
        let samples = sampler.run(&mut rng, 50, 10).unwrap();
        assert_eq!(samples.len(), 50);

        // HMC typically has high acceptance rate
        let rate = sampler.diagnostics().acceptance_rate;
        assert!(rate > 0.3);
    }

    #[test]
    fn test_sampler_diagnostics() {
        let target = GaussianMultivector::<2, 0, 0>::standard();
        let mut sampler = MetropolisHastings::new(&target, 0.5);
        let mut rng = rand::thread_rng();

        // Initial diagnostics
        let diag0 = sampler.diagnostics();
        assert_eq!(diag0.num_steps, 0);

        // After some steps
        let _ = sampler.run(&mut rng, 100, 0).unwrap();
        let diag1 = sampler.diagnostics();
        assert_eq!(diag1.num_steps, 100);
    }
}
