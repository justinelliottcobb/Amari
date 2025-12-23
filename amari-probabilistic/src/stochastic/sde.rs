//! Stochastic Differential Equation traits and solvers

use crate::error::{ProbabilisticError, Result};
use amari_core::Multivector;
use rand::Rng;
use rand_distr::{Distribution as RandDist, Normal};

/// Trait for stochastic processes on multivector spaces
///
/// Defines the drift and diffusion coefficients of an SDE.
pub trait StochasticProcess<const P: usize, const Q: usize, const R: usize> {
    /// Dimension of the multivector space
    const DIM: usize = 1 << (P + Q + R);

    /// Compute the drift coefficient μ(X, t)
    ///
    /// # Arguments
    ///
    /// * `state` - Current state of the process
    /// * `t` - Current time
    fn drift(&self, state: &Multivector<P, Q, R>, t: f64) -> Result<Multivector<P, Q, R>>;

    /// Compute the diffusion coefficient σ(X, t)
    ///
    /// Returns a vector of diffusion coefficients (one per component).
    /// For matrix-valued diffusion, this would need to be extended.
    ///
    /// # Arguments
    ///
    /// * `state` - Current state of the process
    /// * `t` - Current time
    fn diffusion(&self, state: &Multivector<P, Q, R>, t: f64) -> Result<Vec<f64>>;

    /// Sample a path from the process
    ///
    /// # Arguments
    ///
    /// * `t0` - Start time
    /// * `t1` - End time
    /// * `steps` - Number of time steps
    fn sample_path(
        &self,
        t0: f64,
        t1: f64,
        steps: usize,
    ) -> Result<Vec<(f64, Multivector<P, Q, R>)>>;
}

/// Trait for SDE numerical solvers
pub trait SDESolver<const P: usize, const Q: usize, const R: usize> {
    /// Take a single step of the solver
    ///
    /// # Arguments
    ///
    /// * `process` - The stochastic process being solved
    /// * `state` - Current state
    /// * `t` - Current time
    /// * `dt` - Time step
    /// * `rng` - Random number generator
    fn step<S, RNG: Rng>(
        &self,
        process: &S,
        state: &Multivector<P, Q, R>,
        t: f64,
        dt: f64,
        rng: &mut RNG,
    ) -> Result<Multivector<P, Q, R>>
    where
        S: StochasticProcess<P, Q, R>;

    /// Convergence order of the solver
    fn convergence_order(&self) -> f64;

    /// Solve the SDE over a time interval
    fn solve<S, RNG: Rng>(
        &self,
        process: &S,
        initial: Multivector<P, Q, R>,
        t0: f64,
        t1: f64,
        steps: usize,
        rng: &mut RNG,
    ) -> Result<Vec<(f64, Multivector<P, Q, R>)>>
    where
        S: StochasticProcess<P, Q, R>,
    {
        if steps == 0 {
            return Err(ProbabilisticError::invalid_parameters(
                "Number of steps must be positive",
            ));
        }

        let dt = (t1 - t0) / steps as f64;
        let mut path = Vec::with_capacity(steps + 1);
        let mut state = initial;
        let mut t = t0;

        path.push((t, state.clone()));

        for _ in 0..steps {
            state = self.step(process, &state, t, dt, rng)?;
            t += dt;
            path.push((t, state.clone()));
        }

        Ok(path)
    }
}

/// Euler-Maruyama solver for SDEs
///
/// First-order explicit solver:
/// X_{n+1} = X_n + μ(X_n, t_n)Δt + σ(X_n, t_n)ΔW_n
///
/// where ΔW_n ~ N(0, Δt)
#[derive(Debug, Clone, Default)]
pub struct EulerMaruyama;

impl EulerMaruyama {
    /// Create a new Euler-Maruyama solver
    pub fn new() -> Self {
        Self
    }
}

impl<const P: usize, const Q: usize, const R: usize> SDESolver<P, Q, R> for EulerMaruyama {
    fn step<S, RNG: Rng>(
        &self,
        process: &S,
        state: &Multivector<P, Q, R>,
        t: f64,
        dt: f64,
        rng: &mut RNG,
    ) -> Result<Multivector<P, Q, R>>
    where
        S: StochasticProcess<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);

        // Compute drift and diffusion
        let drift = process.drift(state, t)?;
        let diffusion = process.diffusion(state, t)?;

        // Generate Brownian increments
        let sqrt_dt = dt.sqrt();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let state_vec = state.to_vec();
        let drift_vec = drift.to_vec();

        let mut new_state = Vec::with_capacity(dim);

        for i in 0..dim {
            let dw = normal.sample(rng) * sqrt_dt;
            let sigma_i = if i < diffusion.len() {
                diffusion[i]
            } else {
                0.0
            };
            new_state.push(state_vec[i] + drift_vec[i] * dt + sigma_i * dw);
        }

        // Check for numerical instability
        for (i, &x) in new_state.iter().enumerate() {
            if !x.is_finite() {
                return Err(ProbabilisticError::sde_instability(
                    t,
                    format!("Component {} became non-finite", i),
                ));
            }
        }

        Ok(Multivector::from_coefficients(new_state))
    }

    fn convergence_order(&self) -> f64 {
        0.5 // Strong order 0.5 for Euler-Maruyama
    }
}

/// Milstein solver for SDEs
///
/// Higher-order solver that includes the derivative of the diffusion:
/// X_{n+1} = X_n + μ Δt + σ ΔW + ½ σ σ' (ΔW² - Δt)
///
/// Achieves strong order 1.0 for scalar SDEs.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Milstein;

#[allow(dead_code)]
impl Milstein {
    /// Create a new Milstein solver
    pub fn new() -> Self {
        Self
    }

    /// Estimate diffusion derivative via finite differences
    fn diffusion_derivative<const P: usize, const Q: usize, const R: usize, S>(
        &self,
        process: &S,
        state: &Multivector<P, Q, R>,
        t: f64,
        component: usize,
    ) -> Result<f64>
    where
        S: StochasticProcess<P, Q, R>,
    {
        let eps = 1e-6;
        let dim = 1 << (P + Q + R);

        let mut state_plus = state.to_vec();
        let mut state_minus = state.to_vec();

        if component < dim {
            state_plus[component] += eps;
            state_minus[component] -= eps;
        }

        let diff_plus = process.diffusion(&Multivector::from_coefficients(state_plus), t)?;
        let diff_minus = process.diffusion(&Multivector::from_coefficients(state_minus), t)?;

        let sigma_plus = if component < diff_plus.len() {
            diff_plus[component]
        } else {
            0.0
        };
        let sigma_minus = if component < diff_minus.len() {
            diff_minus[component]
        } else {
            0.0
        };

        Ok((sigma_plus - sigma_minus) / (2.0 * eps))
    }
}

impl<const P: usize, const Q: usize, const R: usize> SDESolver<P, Q, R> for Milstein {
    fn step<S, RNG: Rng>(
        &self,
        process: &S,
        state: &Multivector<P, Q, R>,
        t: f64,
        dt: f64,
        rng: &mut RNG,
    ) -> Result<Multivector<P, Q, R>>
    where
        S: StochasticProcess<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);

        // Compute drift and diffusion
        let drift = process.drift(state, t)?;
        let diffusion = process.diffusion(state, t)?;

        // Generate Brownian increments
        let sqrt_dt = dt.sqrt();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let state_vec = state.to_vec();
        let drift_vec = drift.to_vec();

        let mut new_state = Vec::with_capacity(dim);

        for i in 0..dim {
            let dw = normal.sample(rng) * sqrt_dt;
            let sigma_i = if i < diffusion.len() {
                diffusion[i]
            } else {
                0.0
            };

            // Milstein correction term
            let sigma_prime = self.diffusion_derivative(process, state, t, i)?;
            let milstein_correction = 0.5 * sigma_i * sigma_prime * (dw * dw - dt);

            new_state.push(state_vec[i] + drift_vec[i] * dt + sigma_i * dw + milstein_correction);
        }

        // Check for numerical instability
        for (i, &x) in new_state.iter().enumerate() {
            if !x.is_finite() {
                return Err(ProbabilisticError::sde_instability(
                    t,
                    format!("Component {} became non-finite", i),
                ));
            }
        }

        Ok(Multivector::from_coefficients(new_state))
    }

    fn convergence_order(&self) -> f64 {
        1.0 // Strong order 1.0 for Milstein
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test process: dX = -X dt + σ dW (Ornstein-Uhlenbeck)
    struct OrnsteinUhlenbeck<const P: usize, const Q: usize, const R: usize> {
        theta: f64,
        sigma: f64,
    }

    impl<const P: usize, const Q: usize, const R: usize> StochasticProcess<P, Q, R>
        for OrnsteinUhlenbeck<P, Q, R>
    {
        fn drift(&self, state: &Multivector<P, Q, R>, _t: f64) -> Result<Multivector<P, Q, R>> {
            // μ(X) = -θX (mean reversion)
            let coeffs: Vec<f64> = state.to_vec().iter().map(|&x| -self.theta * x).collect();
            Ok(Multivector::from_coefficients(coeffs))
        }

        fn diffusion(&self, _state: &Multivector<P, Q, R>, _t: f64) -> Result<Vec<f64>> {
            // Constant diffusion
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

    #[test]
    fn test_euler_maruyama() {
        let ou = OrnsteinUhlenbeck::<2, 0, 0> {
            theta: 1.0,
            sigma: 0.5,
        };
        let solver = EulerMaruyama::new();
        let initial = Multivector::scalar(1.0);
        let mut rng = rand::thread_rng();

        let path = solver.solve(&ou, initial, 0.0, 1.0, 100, &mut rng).unwrap();

        assert_eq!(path.len(), 101); // 100 steps + initial
        assert_eq!(path[0].0, 0.0);
        assert!((path[100].0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_milstein() {
        let ou = OrnsteinUhlenbeck::<2, 0, 0> {
            theta: 1.0,
            sigma: 0.5,
        };
        let solver = Milstein::new();
        let initial = Multivector::scalar(1.0);
        let mut rng = rand::thread_rng();

        let path = solver.solve(&ou, initial, 0.0, 1.0, 100, &mut rng).unwrap();

        assert_eq!(path.len(), 101);
    }

    #[test]
    fn test_convergence_order() {
        // Need to specify const generics for SDESolver
        assert_eq!(
            <EulerMaruyama as SDESolver<2, 0, 0>>::convergence_order(&EulerMaruyama::new()),
            0.5
        );
        assert_eq!(
            <Milstein as SDESolver<2, 0, 0>>::convergence_order(&Milstein::new()),
            1.0
        );
    }

    #[test]
    fn test_sample_path() {
        let ou = OrnsteinUhlenbeck::<2, 0, 0> {
            theta: 1.0,
            sigma: 0.5,
        };
        let path = ou.sample_path(0.0, 1.0, 50).unwrap();

        assert_eq!(path.len(), 51);

        // Check times are monotonically increasing
        for i in 1..path.len() {
            assert!(path[i].0 > path[i - 1].0);
        }
    }
}
