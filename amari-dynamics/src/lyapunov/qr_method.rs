//! QR-based Lyapunov exponent computation
//!
//! This module implements the standard QR method (Benettin et al., 1980) for
//! computing the full Lyapunov spectrum of a dynamical system.
//!
//! # Algorithm
//!
//! The method works by:
//! 1. Integrating the reference trajectory
//! 2. Simultaneously integrating n perturbation vectors
//! 3. Periodically orthonormalizing using QR decomposition
//! 4. Accumulating the logarithms of the diagonal of R
//!
//! The Lyapunov exponents are the time-averages of these logarithms.
//!
//! # References
//!
//! - Benettin, G., Galgani, L., Giorgilli, A., & Strelcyn, J. M. (1980).
//!   Lyapunov characteristic exponents for smooth dynamical systems and for
//!   Hamiltonian systems. Meccanica, 15(1), 9-30.
//! - Shimada, I., & Nagashima, T. (1979). A numerical approach to ergodic
//!   problem of dissipative dynamical systems. Progress of Theoretical Physics,
//!   61(6), 1605-1616.

use amari_core::Multivector;
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;
use crate::solver::{DormandPrince, ODESolver};

use super::spectrum::{LyapunovConfig, LyapunovSpectrum};

/// Convert a flat Vec<f64> Jacobian to a DMatrix
fn vec_to_matrix(vec: &[f64]) -> Result<DMatrix<f64>> {
    let n_sq = vec.len();
    let n = (n_sq as f64).sqrt() as usize;

    if n * n != n_sq {
        return Err(DynamicsError::numerical_instability(
            "vec_to_matrix",
            "Jacobian vector length is not a perfect square",
        ));
    }

    Ok(DMatrix::from_row_slice(n, n, vec))
}

/// Compute the full Lyapunov spectrum using QR decomposition
///
/// This implements the standard algorithm of Benettin et al. for computing
/// all Lyapunov exponents of a continuous-time dynamical system.
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `initial` - Initial condition on the attractor
/// * `config` - Configuration parameters
///
/// # Returns
///
/// The full Lyapunov spectrum ordered from largest to smallest
pub fn compute_lyapunov_spectrum<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial: &Multivector<P, Q, R>,
    config: &LyapunovConfig,
) -> Result<LyapunovSpectrum>
where
    S: DynamicalSystem<P, Q, R>,
{
    // Determine system dimension from Jacobian
    let jac_vec = system.jacobian(initial)?;
    let jac = vec_to_matrix(&jac_vec)?;
    let dim = jac.nrows();

    if dim == 0 {
        return Err(DynamicsError::invalid_parameter("Zero-dimensional system"));
    }

    let solver = DormandPrince::new();

    // Skip transient
    let steps = (config.transient_time / config.dt) as usize;
    let trajectory = solver.solve(system, initial.clone(), 0.0, config.transient_time, steps)?;
    let mut state = trajectory
        .final_state()
        .ok_or_else(|| {
            DynamicsError::numerical_instability("lyapunov", "Failed to integrate transient")
        })?
        .clone();

    // Initialize orthonormal perturbation vectors (identity matrix columns)
    let mut q_matrix = DMatrix::<f64>::identity(dim, dim);

    // Accumulator for Lyapunov sums
    let mut lyapunov_sums = DVector::<f64>::zeros(dim);

    // Convergence history
    let mut convergence_history = Vec::new();

    let renorm_steps = (config.renorm_time / config.dt) as usize;
    let mut total_time = 0.0;

    for renorm_idx in 0..config.num_renormalizations {
        // Integrate the reference trajectory
        let ref_traj =
            solver.solve(system, state.clone(), 0.0, config.renorm_time, renorm_steps)?;
        state = ref_traj
            .final_state()
            .ok_or_else(|| {
                DynamicsError::numerical_instability("lyapunov", "Failed to integrate reference")
            })?
            .clone();

        // Integrate each perturbation vector using the variational equation
        let new_q = integrate_perturbations(system, &ref_traj.states, &q_matrix, config.dt)?;

        // QR decomposition
        let qr = new_q.clone().qr();
        q_matrix = qr.q();
        let r_matrix = qr.r();

        // Accumulate logarithms of diagonal elements of R
        for i in 0..dim {
            let diag = r_matrix[(i, i)].abs();
            if diag > 1e-300 {
                lyapunov_sums[i] += diag.ln();
            }
        }

        total_time += config.renorm_time;

        // Record convergence
        if config.track_convergence && renorm_idx % config.convergence_interval == 0 {
            let current_exponents: Vec<f64> =
                lyapunov_sums.iter().map(|&s| s / total_time).collect();
            convergence_history.push((total_time, current_exponents));
        }
    }

    // Compute final exponents
    let exponents: Vec<f64> = lyapunov_sums.iter().map(|&s| s / total_time).collect();

    let mut spectrum = LyapunovSpectrum::new(exponents, total_time, config.num_renormalizations);

    if config.track_convergence {
        spectrum.convergence_history = Some(convergence_history);
    }

    Ok(spectrum)
}

/// Integrate perturbation vectors along a trajectory
fn integrate_perturbations<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    trajectory: &[Multivector<P, Q, R>],
    q_matrix: &DMatrix<f64>,
    dt: f64,
) -> Result<DMatrix<f64>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let _dim = q_matrix.nrows();
    let n_vecs = q_matrix.ncols();

    // Start with current Q matrix
    let mut result = q_matrix.clone();

    // Integrate along trajectory using RK4 for the variational equation
    for window in trajectory.windows(2) {
        let state = &window[0];
        let jac_vec = system.jacobian(state)?;
        let jacobian = vec_to_matrix(&jac_vec)?;

        // For each perturbation vector, do one RK4 step
        // dδx/dt = J(x) * δx
        for j in 0..n_vecs {
            let delta = result.column(j).clone_owned();

            // RK4 step for variational equation
            let k1 = &jacobian * &delta;
            let k2 = &jacobian * (&delta + &k1 * (dt / 2.0));
            let k3 = &jacobian * (&delta + &k2 * (dt / 2.0));
            let k4 = &jacobian * (&delta + &k3 * dt);

            let delta_new = delta + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);

            result.set_column(j, &delta_new);
        }
    }

    Ok(result)
}

/// Compute only the largest Lyapunov exponent (faster than full spectrum)
///
/// Uses a single perturbation vector and doesn't require QR decomposition.
pub fn compute_largest_lyapunov<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial: &Multivector<P, Q, R>,
    config: &LyapunovConfig,
) -> Result<f64>
where
    S: DynamicalSystem<P, Q, R>,
{
    let jac_vec = system.jacobian(initial)?;
    let jac = vec_to_matrix(&jac_vec)?;
    let dim = jac.nrows();

    if dim == 0 {
        return Err(DynamicsError::invalid_parameter("Zero-dimensional system"));
    }

    let solver = DormandPrince::new();

    // Skip transient
    let steps = (config.transient_time / config.dt) as usize;
    let trajectory = solver.solve(system, initial.clone(), 0.0, config.transient_time, steps)?;
    let mut state = trajectory
        .final_state()
        .ok_or_else(|| {
            DynamicsError::numerical_instability("lyapunov", "Failed to integrate transient")
        })?
        .clone();

    // Initialize perturbation vector (random direction, normalized)
    let mut delta = DVector::<f64>::from_fn(dim, |i, _| if i == 0 { 1.0 } else { 0.0 });
    delta.normalize_mut();

    let mut lyapunov_sum = 0.0;
    let renorm_steps = (config.renorm_time / config.dt) as usize;
    let mut total_time = 0.0;

    for _ in 0..config.num_renormalizations {
        // Integrate reference trajectory
        let ref_traj =
            solver.solve(system, state.clone(), 0.0, config.renorm_time, renorm_steps)?;
        state = ref_traj
            .final_state()
            .ok_or_else(|| DynamicsError::numerical_instability("lyapunov", "Failed to integrate"))?
            .clone();

        // Integrate perturbation
        for window in ref_traj.states.windows(2) {
            let s = &window[0];
            let jac_vec = system.jacobian(s)?;
            let jacobian = vec_to_matrix(&jac_vec)?;

            // RK4 step
            let k1 = &jacobian * &delta;
            let k2 = &jacobian * (&delta + &k1 * (config.dt / 2.0));
            let k3 = &jacobian * (&delta + &k2 * (config.dt / 2.0));
            let k4 = &jacobian * (&delta + &k3 * config.dt);

            delta = delta.clone() + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (config.dt / 6.0);
        }

        // Renormalize and accumulate
        let norm = delta.norm();
        if norm > 1e-300 {
            lyapunov_sum += norm.ln();
            delta /= norm;
        }

        total_time += config.renorm_time;
    }

    Ok(lyapunov_sum / total_time)
}

/// Compute Lyapunov spectrum for an ensemble of initial conditions
#[cfg(feature = "parallel")]
pub fn compute_lyapunov_ensemble<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial_conditions: &[Multivector<P, Q, R>],
    config: &LyapunovConfig,
) -> Vec<Result<LyapunovSpectrum>>
where
    S: DynamicalSystem<P, Q, R> + Sync,
{
    initial_conditions
        .par_iter()
        .map(|ic| compute_lyapunov_spectrum(system, ic, config))
        .collect()
}

/// Check for chaos by computing the largest Lyapunov exponent
pub fn is_chaotic<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    initial: &Multivector<P, Q, R>,
    config: &LyapunovConfig,
) -> Result<bool>
where
    S: DynamicalSystem<P, Q, R>,
{
    let lambda_max = compute_largest_lyapunov(system, initial, config)?;
    Ok(lambda_max > config.zero_tolerance)
}

/// Finite-time Lyapunov exponent (FTLE) field computation
///
/// Computes the FTLE for a grid of initial conditions, useful for
/// visualizing Lagrangian coherent structures.
pub fn compute_ftle_field<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    grid_points: &[Multivector<P, Q, R>],
    integration_time: f64,
    dt: f64,
) -> Result<Vec<f64>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let solver = DormandPrince::new();
    let steps = (integration_time / dt) as usize;

    let mut ftle_values = Vec::with_capacity(grid_points.len());

    for point in grid_points {
        // Get Jacobian at initial point
        let jac_vec = system.jacobian(point)?;
        let jac = vec_to_matrix(&jac_vec)?;
        let dim = jac.nrows();

        // Initialize identity matrix for tracking deformation
        let mut deformation = DMatrix::<f64>::identity(dim, dim);

        // Integrate and track deformation
        let trajectory = solver.solve(system, point.clone(), 0.0, integration_time, steps)?;

        for window in trajectory.states.windows(2) {
            let state = &window[0];
            let jac_vec = system.jacobian(state)?;
            let jacobian = vec_to_matrix(&jac_vec)?;

            // Update deformation gradient: dF/dt = J * F
            let df = &jacobian * &deformation * dt;
            deformation += df;
        }

        // FTLE = (1/T) * ln(sqrt(max eigenvalue of F^T * F))
        let c = deformation.transpose() * &deformation;

        // Find largest eigenvalue (using power iteration)
        let lambda_max = power_iteration_largest_eigenvalue(&c, 100)?;
        let ftle = lambda_max.sqrt().ln() / integration_time;

        ftle_values.push(ftle);
    }

    Ok(ftle_values)
}

/// Power iteration to find largest eigenvalue
fn power_iteration_largest_eigenvalue(matrix: &DMatrix<f64>, max_iter: usize) -> Result<f64> {
    let n = matrix.nrows();
    if n == 0 {
        return Err(DynamicsError::invalid_parameter("Empty matrix"));
    }

    let mut v = DVector::<f64>::from_fn(n, |i, _| if i == 0 { 1.0 } else { 0.0 });
    let mut lambda = 1.0;

    for _ in 0..max_iter {
        let w = matrix * &v;
        let new_lambda = w.norm();

        if new_lambda < 1e-300 {
            return Ok(0.0);
        }

        v = w / new_lambda;

        if (new_lambda - lambda).abs() / lambda.max(1e-10) < 1e-10 {
            return Ok(new_lambda);
        }

        lambda = new_lambda;
    }

    Ok(lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple linear system for testing: dx/dt = A*x
    struct LinearSystem {
        matrix: DMatrix<f64>,
    }

    impl LinearSystem {
        fn new(matrix: DMatrix<f64>) -> Self {
            Self { matrix }
        }
    }

    impl DynamicalSystem<3, 0, 0> for LinearSystem {
        fn vector_field(&self, state: &Multivector<3, 0, 0>) -> Result<Multivector<3, 0, 0>> {
            let mut result = Multivector::<3, 0, 0>::zero();
            let x = DVector::from_vec(vec![state.get(0), state.get(1), state.get(2)]);
            let dx = &self.matrix * &x;
            result.set(0, dx[0]);
            result.set(1, dx[1]);
            result.set(2, dx[2]);
            Ok(result)
        }

        fn jacobian(&self, _state: &Multivector<3, 0, 0>) -> Result<Vec<f64>> {
            // Return flattened row-major matrix
            let mut vec = Vec::with_capacity(9);
            for i in 0..3 {
                for j in 0..3 {
                    vec.push(self.matrix[(i, j)]);
                }
            }
            Ok(vec)
        }
    }

    #[test]
    fn test_vec_to_matrix() {
        let vec = vec![1.0, 2.0, 3.0, 4.0];
        let matrix = vec_to_matrix(&vec).unwrap();

        assert_eq!(matrix.nrows(), 2);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(0, 1)], 2.0);
        assert_eq!(matrix[(1, 0)], 3.0);
        assert_eq!(matrix[(1, 1)], 4.0);
    }

    #[test]
    fn test_linear_system_lyapunov() {
        // Simple damped system: diagonal with eigenvalues -1, -2, -3
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, -2.0, -3.0]));
        let system = LinearSystem::new(matrix);

        let initial = Multivector::<3, 0, 0>::zero();

        let config = LyapunovConfig {
            renorm_time: 0.5,
            num_renormalizations: 50,
            transient_time: 10.0,
            dt: 0.01,
            ..Default::default()
        };

        let spectrum = compute_lyapunov_spectrum(&system, &initial, &config).unwrap();

        // For linear systems, Lyapunov exponents equal eigenvalues
        assert!(!spectrum.is_chaotic());
        assert_eq!(spectrum.dimension(), 3);

        // Should be approximately -1, -2, -3
        assert!((spectrum.largest() - (-1.0)).abs() < 0.2);
    }

    #[test]
    fn test_largest_lyapunov_only() {
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![-0.5, -1.0, -2.0]));
        let system = LinearSystem::new(matrix);

        let initial = Multivector::<3, 0, 0>::zero();

        let config = LyapunovConfig {
            renorm_time: 0.5,
            num_renormalizations: 50,
            transient_time: 5.0,
            dt: 0.01,
            ..Default::default()
        };

        let lambda_max = compute_largest_lyapunov(&system, &initial, &config).unwrap();

        // Should be approximately -0.5
        assert!((lambda_max - (-0.5)).abs() < 0.2);
    }

    #[test]
    fn test_chaos_detection() {
        // Stable system: not chaotic
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, -2.0, -3.0]));
        let system = LinearSystem::new(matrix);

        let initial = Multivector::<3, 0, 0>::zero();

        let config = LyapunovConfig::fast();
        let chaotic = is_chaotic(&system, &initial, &config).unwrap();

        assert!(!chaotic);
    }

    #[test]
    fn test_power_iteration() {
        // 2x2 diagonal matrix with known eigenvalues 4 and 2
        let simple = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 2.0]));
        let lambda_max = power_iteration_largest_eigenvalue(&simple, 100).unwrap();

        assert!((lambda_max - 4.0).abs() < 1e-8);
    }

    #[test]
    fn test_ftle_field() {
        // Linear system with known stretching rate
        let matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![0.5, -0.5, -0.5]));
        let system = LinearSystem::new(matrix);

        let mut points = Vec::new();
        let mut p = Multivector::<3, 0, 0>::zero();
        p.set(0, 1.0);
        points.push(p);

        let ftle = compute_ftle_field(&system, &points, 5.0, 0.01).unwrap();

        assert_eq!(ftle.len(), 1);
        // FTLE should be approximately 0.5 (the positive Lyapunov exponent)
        assert!((ftle[0] - 0.5).abs() < 0.2);
    }
}
