//! Backward Euler implicit solver
//!
//! The backward (implicit) Euler method is unconditionally stable for
//! dissipative systems, making it suitable for stiff ODEs.
//!
//! # Method
//!
//! ```text
//! y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
//! ```
//!
//! This requires solving the nonlinear equation at each step using Newton iteration.
//!
//! # Stability
//!
//! The backward Euler method is A-stable (unconditionally stable for any
//! step size when applied to dissipative systems). This makes it ideal for:
//!
//! - Stiff chemical kinetics
//! - Systems with very fast modes that should be damped
//! - Problems where explicit methods fail due to stability constraints
//!
//! # Performance
//!
//! While more expensive per step (due to Newton iteration), backward Euler
//! can take much larger steps than explicit methods on stiff problems,
//! often resulting in faster overall computation.

use amari_core::Multivector;
use nalgebra::{DMatrix, DVector};

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;
use crate::solver::traits::{ImplicitODESolver, ODESolver, StepResult};

/// Backward Euler implicit solver
///
/// Uses Newton iteration to solve the implicit equation at each step.
#[derive(Debug, Clone)]
pub struct BackwardEuler {
    /// Maximum Newton iterations per step
    pub max_iterations: usize,
    /// Convergence tolerance for Newton iteration
    pub tolerance: f64,
    /// Jacobian finite difference step
    pub jacobian_epsilon: f64,
}

impl Default for BackwardEuler {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-10,
            jacobian_epsilon: 1e-8,
        }
    }
}

impl BackwardEuler {
    /// Create a new backward Euler solver with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom Newton iteration parameters
    pub fn with_parameters(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
            jacobian_epsilon: 1e-8,
        }
    }

    /// Create a high-precision variant
    pub fn high_precision() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-14,
            jacobian_epsilon: 1e-10,
        }
    }

    /// Compute Jacobian of the vector field numerically
    fn compute_jacobian<S, const P: usize, const Q: usize, const R: usize>(
        &self,
        system: &S,
        state: &Multivector<P, Q, R>,
    ) -> Result<DMatrix<f64>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let dim = 1 << (P + Q + R);
        let mut jacobian = DMatrix::zeros(dim, dim);
        let eps = self.jacobian_epsilon;

        for j in 0..dim {
            // Forward perturbation
            let mut forward = state.clone();
            forward.set(j, forward.get(j) + eps);
            let f_forward = system.vector_field(&forward)?;

            // Backward perturbation
            let mut backward = state.clone();
            backward.set(j, backward.get(j) - eps);
            let f_backward = system.vector_field(&backward)?;

            // Central difference
            for i in 0..dim {
                jacobian[(i, j)] = (f_forward.get(i) - f_backward.get(i)) / (2.0 * eps);
            }
        }

        Ok(jacobian)
    }

    /// Convert Multivector to DVector
    fn mv_to_vec<const P: usize, const Q: usize, const R: usize>(
        mv: &Multivector<P, Q, R>,
    ) -> DVector<f64> {
        let dim = 1 << (P + Q + R);
        let mut vec = DVector::zeros(dim);
        for i in 0..dim {
            vec[i] = mv.get(i);
        }
        vec
    }

    /// Convert DVector to Multivector
    fn vec_to_mv<const P: usize, const Q: usize, const R: usize>(
        vec: &DVector<f64>,
    ) -> Multivector<P, Q, R> {
        let dim = 1 << (P + Q + R);
        let coeffs: Vec<f64> = (0..dim).map(|i| vec[i]).collect();
        Multivector::from_coefficients(coeffs)
    }
}

impl<const P: usize, const Q: usize, const R: usize> ODESolver<P, Q, R> for BackwardEuler {
    fn step<S: DynamicalSystem<P, Q, R>>(
        &self,
        system: &S,
        state: &Multivector<P, Q, R>,
        _t: f64,
        dt: f64,
    ) -> Result<StepResult<P, Q, R>> {
        if dt <= 0.0 {
            return Err(DynamicsError::invalid_step_size(dt));
        }

        let dim = 1 << (P + Q + R);

        // Initial guess: Forward Euler step
        let f_n = system.vector_field(state)?;
        let mut y_next = state + &(&f_n * dt);

        let y_n_vec = Self::mv_to_vec(state);

        // Newton iteration to solve: G(y) = y - y_n - h*f(y) = 0
        for iteration in 0..self.max_iterations {
            // Compute f(y_next)
            let f_next = system.vector_field(&y_next)?;
            let f_next_vec = Self::mv_to_vec(&f_next);

            // Residual: G = y_next - y_n - h*f(y_next)
            let y_next_vec = Self::mv_to_vec(&y_next);
            let residual = &y_next_vec - &y_n_vec - &f_next_vec * dt;
            let residual_norm = residual.norm();

            // Check convergence
            if residual_norm < self.tolerance {
                return Ok(StepResult::new(y_next));
            }

            // Compute Jacobian of f at y_next
            let jac_f = self.compute_jacobian(system, &y_next)?;

            // Jacobian of G: dG/dy = I - h*J_f
            let identity = DMatrix::identity(dim, dim);
            let jac_g = &identity - &jac_f * dt;

            // Solve: (I - h*J_f) * delta_y = -G
            let lu = jac_g.lu();
            match lu.solve(&(-&residual)) {
                Some(delta_y) => {
                    // Update: y_next = y_next + delta_y
                    let new_y_next_vec = &y_next_vec + &delta_y;
                    y_next = Self::vec_to_mv(&new_y_next_vec);
                }
                None => {
                    return Err(DynamicsError::numerical_instability(
                        "Backward Euler",
                        "Jacobian is singular in Newton iteration",
                    ));
                }
            }

            // Check for divergence
            if iteration > 5 && residual_norm > 1e10 {
                return Err(DynamicsError::numerical_instability(
                    "Backward Euler",
                    "Newton iteration diverging",
                ));
            }
        }

        // Max iterations reached
        Err(DynamicsError::convergence_failure(
            self.max_iterations,
            "Newton iteration did not converge",
        ))
    }

    fn order(&self) -> u32 {
        1
    }

    fn name(&self) -> &'static str {
        "Backward Euler"
    }
}

impl<const P: usize, const Q: usize, const R: usize> ImplicitODESolver<P, Q, R> for BackwardEuler {
    fn max_newton_iterations(&self) -> usize {
        self.max_iterations
    }

    fn newton_tolerance(&self) -> f64 {
        self.tolerance
    }

    fn implicit_step<S: DynamicalSystem<P, Q, R>>(
        &self,
        system: &S,
        state: &Multivector<P, Q, R>,
        t: f64,
        dt: f64,
    ) -> Result<StepResult<P, Q, R>> {
        // Delegate to the main step function
        self.step(system, state, t, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::HarmonicOscillator;

    // Stiff test system: y' = -λy with large λ
    struct StiffDecay {
        lambda: f64,
    }

    impl StiffDecay {
        fn new(lambda: f64) -> Self {
            Self { lambda }
        }
    }

    impl DynamicalSystem<1, 0, 0> for StiffDecay {
        fn vector_field(&self, state: &Multivector<1, 0, 0>) -> Result<Multivector<1, 0, 0>> {
            let y = state.get(0);

            let mut result = Multivector::zero();
            result.set(0, -self.lambda * y);

            Ok(result)
        }
    }

    // Coupled stiff system with fast and slow modes (reserved for future tests)
    #[allow(dead_code)]
    struct StiffCoupled;

    impl DynamicalSystem<2, 0, 0> for StiffCoupled {
        fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
            let x = state.get(1);
            let y = state.get(2);

            // dx/dt = -1000*x + 999*y (fast)
            // dy/dt = -x (slow)
            let mut result = Multivector::zero();
            result.set(1, -1000.0 * x + 999.0 * y);
            result.set(2, -x);

            Ok(result)
        }
    }

    #[test]
    fn test_backward_euler_creation() {
        let solver = BackwardEuler::new();
        assert_eq!(<BackwardEuler as ODESolver<2, 0, 0>>::order(&solver), 1);
        assert_eq!(
            <BackwardEuler as ODESolver<2, 0, 0>>::name(&solver),
            "Backward Euler"
        );
        assert_eq!(solver.max_iterations, 50);
    }

    #[test]
    fn test_backward_euler_stiff_decay() {
        // Test with very stiff decay: y' = -λ*y
        let lambda = 1000.0;
        let system = StiffDecay::new(lambda);
        let solver = BackwardEuler::new();

        let mut initial = Multivector::<1, 0, 0>::zero();
        initial.set(0, 1.0);

        // Backward Euler should handle large steps
        let dt = 0.1;
        let result = solver.step(&system, &initial, 0.0, dt).unwrap();

        // Backward Euler: y_{n+1} = y_n / (1 + h*λ) = 1 / (1 + 100) ≈ 0.0099
        let expected = 1.0 / (1.0 + dt * lambda);
        let y = result.state.get(0);
        assert!(
            (y - expected).abs() < 1e-6,
            "Expected y ≈ {}, got {}",
            expected,
            y
        );
    }

    #[test]
    fn test_backward_euler_vs_forward_euler_stiff() {
        // Forward Euler would need dt < 2/λ = 0.002 for stability
        // Backward Euler can use much larger steps
        let system = StiffDecay::new(1000.0);
        let solver = BackwardEuler::new();

        let mut initial = Multivector::<1, 0, 0>::zero();
        initial.set(0, 1.0);

        // Use step 10x larger than Forward Euler stability limit
        let result = solver.step(&system, &initial, 0.0, 0.02);

        // Should still work (Forward Euler would explode)
        assert!(result.is_ok());
    }

    #[test]
    fn test_backward_euler_trajectory() {
        let system = StiffDecay::new(100.0);
        let solver = BackwardEuler::new();

        let mut initial = Multivector::<1, 0, 0>::zero();
        initial.set(0, 1.0);

        // Integrate from 0 to 0.1 (10 decay times)
        let trajectory = solver.solve(&system, initial, 0.0, 0.1, 10).unwrap();

        // Should decay to near zero
        let final_y = trajectory.final_state().unwrap().get(0);
        let expected = (-10.0_f64).exp(); // exp(-100*0.1) = exp(-10)

        assert!(
            (final_y - expected).abs() < 0.1,
            "Expected {} (exp(-10)), got {}",
            expected,
            final_y
        );
    }

    #[test]
    fn test_backward_euler_harmonic() {
        // Harmonic oscillator (not stiff, but tests correctness)
        let system = HarmonicOscillator::new(1.0);
        let solver = BackwardEuler::new();

        let mut initial = Multivector::<2, 0, 0>::zero();
        initial.set(1, 1.0);

        // Note: Backward Euler has numerical damping for oscillatory systems
        let trajectory = solver.solve(&system, initial, 0.0, 1.0, 100).unwrap();

        let final_state = trajectory.final_state().unwrap();
        let x = final_state.get(1);

        // First-order method with numerical damping, so not very accurate
        // Just check it doesn't explode
        assert!(x.abs() < 2.0, "Expected |x| < 2, got {}", x);
    }

    #[test]
    fn test_backward_euler_implicit_trait() {
        let solver = BackwardEuler::new();

        assert_eq!(
            <BackwardEuler as ImplicitODESolver<2, 0, 0>>::max_newton_iterations(&solver),
            50
        );
        assert!(<BackwardEuler as ImplicitODESolver<2, 0, 0>>::newton_tolerance(&solver) < 1e-8);
    }

    #[test]
    fn test_backward_euler_parameters() {
        let solver = BackwardEuler::with_parameters(100, 1e-12);
        assert_eq!(solver.max_iterations, 100);
        assert_eq!(solver.tolerance, 1e-12);

        let hp = BackwardEuler::high_precision();
        assert!(hp.max_iterations > 50);
        assert!(hp.tolerance < 1e-12);
    }

    #[test]
    fn test_backward_euler_invalid_step() {
        let system = StiffDecay::new(1.0);
        let solver = BackwardEuler::new();
        let state = Multivector::<1, 0, 0>::zero();

        assert!(solver.step(&system, &state, 0.0, -0.1).is_err());
        assert!(solver.step(&system, &state, 0.0, 0.0).is_err());
    }
}
