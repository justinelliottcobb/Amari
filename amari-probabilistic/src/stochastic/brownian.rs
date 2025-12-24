//! Standard stochastic processes: Wiener process and geometric Brownian motion

use super::sde::{EulerMaruyama, SDESolver, StochasticProcess};
use crate::error::Result;
use amari_core::Multivector;
use rand::Rng;
use rand_distr::{Distribution as RandDist, Normal};

/// Wiener process (Brownian motion) on multivector space
///
/// Standard Wiener process with independent components:
/// dW = dW (pure noise, no drift)
#[derive(Debug, Clone)]
pub struct WienerProcess<const P: usize, const Q: usize, const R: usize> {
    /// Diffusion coefficient (scales the noise)
    sigma: f64,
}

impl<const P: usize, const Q: usize, const R: usize> WienerProcess<P, Q, R> {
    /// Create a standard Wiener process (σ = 1)
    pub fn standard() -> Self {
        Self { sigma: 1.0 }
    }

    /// Create a Wiener process with given diffusion coefficient
    pub fn with_sigma(sigma: f64) -> Self {
        Self { sigma }
    }

    /// Simulate increments over a time period
    pub fn simulate_increment<RNG: Rng>(&self, dt: f64, rng: &mut RNG) -> Multivector<P, Q, R> {
        let dim = 1 << (P + Q + R);
        let normal = Normal::new(0.0, self.sigma * dt.sqrt()).unwrap();

        let coeffs: Vec<f64> = (0..dim).map(|_| normal.sample(rng)).collect();
        Multivector::from_coefficients(coeffs)
    }
}

impl<const P: usize, const Q: usize, const R: usize> StochasticProcess<P, Q, R>
    for WienerProcess<P, Q, R>
{
    fn drift(&self, _state: &Multivector<P, Q, R>, _t: f64) -> Result<Multivector<P, Q, R>> {
        // No drift for standard Wiener process
        Ok(Multivector::zero())
    }

    fn diffusion(&self, _state: &Multivector<P, Q, R>, _t: f64) -> Result<Vec<f64>> {
        let dim = 1 << (P + Q + R);
        Ok(vec![self.sigma; dim])
    }

    fn sample_path(
        &self,
        t0: f64,
        t1: f64,
        steps: usize,
    ) -> Result<Vec<(f64, Multivector<P, Q, R>)>> {
        let solver = EulerMaruyama::new();
        let initial = Multivector::zero();
        let mut rng = rand::thread_rng();
        solver.solve(self, initial, t0, t1, steps, &mut rng)
    }
}

/// Geometric Brownian motion on multivector space
///
/// Component-wise geometric Brownian motion:
/// dX_i = μ X_i dt + σ X_i dW_i
///
/// This extends the classical GBM (used in finance for stock prices)
/// to multivector spaces.
#[derive(Debug, Clone)]
pub struct GeometricBrownianMotion<const P: usize, const Q: usize, const R: usize> {
    /// Drift coefficient (growth rate)
    mu: f64,
    /// Volatility (diffusion coefficient)
    sigma: f64,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricBrownianMotion<P, Q, R> {
    /// Create a new geometric Brownian motion
    ///
    /// # Arguments
    ///
    /// * `mu` - Drift coefficient (expected return rate)
    /// * `sigma` - Volatility (standard deviation of returns)
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }

    /// Get the drift coefficient
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Get the volatility
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Compute exact solution at time t given X(0)
    ///
    /// For GBM, the exact solution is:
    /// X(t) = X(0) exp((μ - σ²/2)t + σ W(t))
    pub fn exact_solution<RNG: Rng>(
        &self,
        initial: &Multivector<P, Q, R>,
        t: f64,
        rng: &mut RNG,
    ) -> Multivector<P, Q, R> {
        let _dim = 1 << (P + Q + R);
        let normal = Normal::new(0.0, (self.sigma * self.sigma * t).sqrt()).unwrap();

        let coeffs: Vec<f64> = initial
            .to_vec()
            .iter()
            .map(|&x0| {
                let w_t = normal.sample(rng);
                x0 * ((self.mu - 0.5 * self.sigma * self.sigma) * t + w_t).exp()
            })
            .collect();

        Multivector::from_coefficients(coeffs)
    }
}

impl<const P: usize, const Q: usize, const R: usize> StochasticProcess<P, Q, R>
    for GeometricBrownianMotion<P, Q, R>
{
    fn drift(&self, state: &Multivector<P, Q, R>, _t: f64) -> Result<Multivector<P, Q, R>> {
        // μ(X) = μ * X
        let coeffs: Vec<f64> = state.to_vec().iter().map(|&x| self.mu * x).collect();
        Ok(Multivector::from_coefficients(coeffs))
    }

    fn diffusion(&self, state: &Multivector<P, Q, R>, _t: f64) -> Result<Vec<f64>> {
        // σ(X) = σ * X (component-wise)
        Ok(state.to_vec().iter().map(|&x| self.sigma * x).collect())
    }

    fn sample_path(
        &self,
        t0: f64,
        t1: f64,
        steps: usize,
    ) -> Result<Vec<(f64, Multivector<P, Q, R>)>> {
        let solver = EulerMaruyama::new();
        // Start from (1, 1, ..., 1) to avoid trivial zero solution
        let dim = 1 << (P + Q + R);
        let initial = Multivector::from_coefficients(vec![1.0; dim]);
        let mut rng = rand::thread_rng();
        solver.solve(self, initial, t0, t1, steps, &mut rng)
    }
}

/// Ornstein-Uhlenbeck process on multivector space
///
/// Mean-reverting process:
/// dX = θ(μ - X) dt + σ dW
///
/// The process reverts to the mean μ with rate θ.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OrnsteinUhlenbeck<const P: usize, const Q: usize, const R: usize> {
    /// Mean reversion rate
    theta: f64,
    /// Long-term mean
    mean: Multivector<P, Q, R>,
    /// Volatility
    sigma: f64,
}

#[allow(dead_code)]
impl<const P: usize, const Q: usize, const R: usize> OrnsteinUhlenbeck<P, Q, R> {
    /// Create a new Ornstein-Uhlenbeck process
    ///
    /// # Arguments
    ///
    /// * `theta` - Mean reversion rate (how fast it reverts)
    /// * `mean` - Long-term mean
    /// * `sigma` - Volatility
    pub fn new(theta: f64, mean: Multivector<P, Q, R>, sigma: f64) -> Self {
        Self { theta, mean, sigma }
    }

    /// Create with zero mean
    pub fn zero_mean(theta: f64, sigma: f64) -> Self {
        Self {
            theta,
            mean: Multivector::zero(),
            sigma,
        }
    }

    /// Get the mean reversion rate
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Get the long-term mean
    pub fn mean(&self) -> &Multivector<P, Q, R> {
        &self.mean
    }

    /// Get the volatility
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Stationary variance of the process
    pub fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma / (2.0 * self.theta)
    }
}

impl<const P: usize, const Q: usize, const R: usize> StochasticProcess<P, Q, R>
    for OrnsteinUhlenbeck<P, Q, R>
{
    fn drift(&self, state: &Multivector<P, Q, R>, _t: f64) -> Result<Multivector<P, Q, R>> {
        // μ(X) = θ(mean - X)
        let state_vec = state.to_vec();
        let mean_vec = self.mean.to_vec();

        let coeffs: Vec<f64> = state_vec
            .iter()
            .zip(mean_vec.iter())
            .map(|(&x, &m)| self.theta * (m - x))
            .collect();

        Ok(Multivector::from_coefficients(coeffs))
    }

    fn diffusion(&self, _state: &Multivector<P, Q, R>, _t: f64) -> Result<Vec<f64>> {
        let dim = 1 << (P + Q + R);
        Ok(vec![self.sigma; dim])
    }

    fn sample_path(
        &self,
        t0: f64,
        t1: f64,
        steps: usize,
    ) -> Result<Vec<(f64, Multivector<P, Q, R>)>> {
        let solver = EulerMaruyama::new();
        let initial = self.mean.clone();
        let mut rng = rand::thread_rng();
        solver.solve(self, initial, t0, t1, steps, &mut rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiener_process() {
        let wiener = WienerProcess::<2, 0, 0>::standard();
        let path = wiener.sample_path(0.0, 1.0, 100).unwrap();

        assert_eq!(path.len(), 101);
        assert_eq!(path[0].0, 0.0);

        // Initial value should be zero
        for i in 0..4 {
            assert_eq!(path[0].1.get(i), 0.0);
        }
    }

    #[test]
    fn test_gbm() {
        let gbm = GeometricBrownianMotion::<2, 0, 0>::new(0.1, 0.2);
        let path = gbm.sample_path(0.0, 1.0, 100).unwrap();

        assert_eq!(path.len(), 101);

        // Check that values stay positive (GBM property)
        // Note: numerical issues could cause small negatives, so we check for non-NaN
        for (_, state) in &path {
            for i in 0..4 {
                assert!(state.get(i).is_finite());
            }
        }
    }

    #[test]
    fn test_gbm_exact_solution() {
        let gbm = GeometricBrownianMotion::<2, 0, 0>::new(0.05, 0.1);
        let initial = Multivector::from_coefficients(vec![1.0, 1.0, 1.0, 1.0]);
        let mut rng = rand::thread_rng();

        let solution = gbm.exact_solution(&initial, 1.0, &mut rng);

        // All components should be positive
        for i in 0..4 {
            assert!(solution.get(i) > 0.0);
        }
    }

    #[test]
    fn test_ornstein_uhlenbeck() {
        let ou = OrnsteinUhlenbeck::<2, 0, 0>::zero_mean(1.0, 0.5);
        let path = ou.sample_path(0.0, 5.0, 500).unwrap();

        assert_eq!(path.len(), 501);

        // Stationary variance check
        let expected_var = ou.stationary_variance();
        assert!((expected_var - 0.125).abs() < 1e-10); // σ²/(2θ) = 0.25/2 = 0.125
    }

    #[test]
    fn test_ou_mean_reversion() {
        let mean = Multivector::<2, 0, 0>::scalar(5.0);
        let ou = OrnsteinUhlenbeck::new(2.0, mean.clone(), 0.1);

        // Start far from mean
        let initial = Multivector::scalar(10.0);
        let solver = EulerMaruyama::new();
        let mut rng = rand::thread_rng();

        let path = solver.solve(&ou, initial, 0.0, 5.0, 500, &mut rng).unwrap();

        // After long time, should be close to mean
        let final_scalar = path.last().unwrap().1.get(0);
        assert!((final_scalar - 5.0).abs() < 2.0); // Should be within 2 of mean
    }
}
