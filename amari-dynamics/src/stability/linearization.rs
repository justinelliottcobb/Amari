//! Linearization of dynamical systems
//!
//! This module provides tools for computing the linearization of nonlinear
//! dynamical systems around fixed points and trajectories.
//!
//! # Mathematical Background
//!
//! For a nonlinear system dx/dt = f(x), the linearization around a point x* is:
//!
//! ```text
//! dξ/dt = J(x*) ξ
//! ```
//!
//! where J is the Jacobian matrix ∂f_i/∂x_j and ξ = x - x*.
//!
//! The stability of the fixed point is determined by the eigenvalues of J.

use amari_core::Multivector;
use nalgebra::{DMatrix, DVector};

use crate::error::{DynamicsError, Result};
use crate::flow::DynamicalSystem;

/// Configuration for numerical differentiation
#[derive(Debug, Clone)]
pub struct DifferentiationConfig {
    /// Step size for finite differences
    pub step_size: f64,
    /// Whether to use central differences (more accurate but slower)
    pub central_differences: bool,
    /// Relative step size (step = relative * |x| + absolute)
    pub relative_step: f64,
}

impl Default for DifferentiationConfig {
    fn default() -> Self {
        Self {
            step_size: 1e-8,
            central_differences: true,
            relative_step: 1e-8,
        }
    }
}

impl DifferentiationConfig {
    /// High precision configuration
    pub fn high_precision() -> Self {
        Self {
            step_size: 1e-10,
            central_differences: true,
            relative_step: 1e-10,
        }
    }

    /// Fast configuration (forward differences)
    pub fn fast() -> Self {
        Self {
            step_size: 1e-6,
            central_differences: false,
            relative_step: 1e-6,
        }
    }

    /// Compute adaptive step size for a given value
    pub fn adaptive_step(&self, value: f64) -> f64 {
        self.step_size + self.relative_step * value.abs()
    }
}

/// Result of linearization around a point
#[derive(Debug, Clone)]
pub struct LinearizedSystem<const P: usize, const Q: usize, const R: usize> {
    /// The point around which the system was linearized
    pub base_point: Multivector<P, Q, R>,
    /// The Jacobian matrix at the base point
    pub jacobian: DMatrix<f64>,
    /// The constant term f(x*)
    pub constant_term: DVector<f64>,
}

impl<const P: usize, const Q: usize, const R: usize> LinearizedSystem<P, Q, R> {
    /// Dimension of the system
    pub fn dimension(&self) -> usize {
        self.jacobian.nrows()
    }

    /// Evaluate the linearized system at a displacement ξ
    ///
    /// Returns J * ξ + f(x*)
    pub fn evaluate(&self, displacement: &DVector<f64>) -> DVector<f64> {
        &self.jacobian * displacement + &self.constant_term
    }

    /// Evaluate the linear part only (J * ξ)
    pub fn linear_part(&self, displacement: &DVector<f64>) -> DVector<f64> {
        &self.jacobian * displacement
    }

    /// Get the trace of the Jacobian (sum of eigenvalues)
    pub fn trace(&self) -> f64 {
        self.jacobian.trace()
    }

    /// Get the determinant of the Jacobian (product of eigenvalues)
    pub fn determinant(&self) -> f64 {
        self.jacobian.determinant()
    }

    /// Check if the Jacobian is singular
    pub fn is_singular(&self, tolerance: f64) -> bool {
        self.determinant().abs() < tolerance
    }
}

/// Compute the Jacobian matrix of a dynamical system at a point
///
/// Uses numerical differentiation with the specified configuration.
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `point` - The point at which to compute the Jacobian
/// * `config` - Differentiation configuration
///
/// # Returns
///
/// The Jacobian matrix as a DMatrix
pub fn compute_jacobian<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
) -> Result<DMatrix<f64>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let dim = S::DIM;
    let mut jacobian = DMatrix::zeros(dim, dim);

    if config.central_differences {
        // Central difference: (f(x+h) - f(x-h)) / (2h)
        for j in 0..dim {
            let h = config.adaptive_step(point.get(j));

            // Forward perturbation
            let mut x_plus = point.clone();
            x_plus.set(j, x_plus.get(j) + h);
            let f_plus = system.vector_field(&x_plus)?;

            // Backward perturbation
            let mut x_minus = point.clone();
            x_minus.set(j, x_minus.get(j) - h);
            let f_minus = system.vector_field(&x_minus)?;

            // Compute column
            for i in 0..dim {
                jacobian[(i, j)] = (f_plus.get(i) - f_minus.get(i)) / (2.0 * h);
            }
        }
    } else {
        // Forward difference: (f(x+h) - f(x)) / h
        let f_base = system.vector_field(point)?;

        for j in 0..dim {
            let h = config.adaptive_step(point.get(j));

            let mut x_plus = point.clone();
            x_plus.set(j, x_plus.get(j) + h);
            let f_plus = system.vector_field(&x_plus)?;

            for i in 0..dim {
                jacobian[(i, j)] = (f_plus.get(i) - f_base.get(i)) / h;
            }
        }
    }

    Ok(jacobian)
}

/// Compute the linearization of a system around a point
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `point` - The point around which to linearize
/// * `config` - Differentiation configuration
///
/// # Returns
///
/// A LinearizedSystem containing the Jacobian and constant term
pub fn linearize<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
) -> Result<LinearizedSystem<P, Q, R>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let jacobian = compute_jacobian(system, point, config)?;

    // Compute f(x*)
    let f_star = system.vector_field(point)?;
    let dim = S::DIM;
    let mut constant_term = DVector::zeros(dim);
    for i in 0..dim {
        constant_term[i] = f_star.get(i);
    }

    Ok(LinearizedSystem {
        base_point: point.clone(),
        jacobian,
        constant_term,
    })
}

/// Compute the Hessian tensor (second derivatives) at a point
///
/// Returns H[i,j,k] = ∂²f_i / (∂x_j ∂x_k)
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `point` - The point at which to compute the Hessian
/// * `config` - Differentiation configuration
///
/// # Returns
///
/// The Hessian as a flattened vector [H[0,0,0], H[0,0,1], ..., H[n-1,n-1,n-1]]
pub fn compute_hessian<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
) -> Result<Vec<f64>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let dim = S::DIM;
    let mut hessian = vec![0.0; dim * dim * dim];

    // Use second-order central differences
    for j in 0..dim {
        for k in j..dim {
            let h_j = config.adaptive_step(point.get(j));
            let h_k = config.adaptive_step(point.get(k));

            if j == k {
                // Diagonal: (f(x+2h) - 2f(x+h) + f(x)) / h²
                let mut x_2h = point.clone();
                x_2h.set(j, x_2h.get(j) + 2.0 * h_j);
                let f_2h = system.vector_field(&x_2h)?;

                let mut x_h = point.clone();
                x_h.set(j, x_h.get(j) + h_j);
                let f_h = system.vector_field(&x_h)?;

                let f_0 = system.vector_field(point)?;

                for i in 0..dim {
                    let value = (f_2h.get(i) - 2.0 * f_h.get(i) + f_0.get(i)) / (h_j * h_j);
                    hessian[i * dim * dim + j * dim + k] = value;
                }
            } else {
                // Off-diagonal: (f(x+hj+hk) - f(x+hj) - f(x+hk) + f(x)) / (hj * hk)
                let mut x_jk = point.clone();
                x_jk.set(j, x_jk.get(j) + h_j);
                x_jk.set(k, x_jk.get(k) + h_k);
                let f_jk = system.vector_field(&x_jk)?;

                let mut x_j = point.clone();
                x_j.set(j, x_j.get(j) + h_j);
                let f_j = system.vector_field(&x_j)?;

                let mut x_k = point.clone();
                x_k.set(k, x_k.get(k) + h_k);
                let f_k = system.vector_field(&x_k)?;

                let f_0 = system.vector_field(point)?;

                for i in 0..dim {
                    let value = (f_jk.get(i) - f_j.get(i) - f_k.get(i) + f_0.get(i)) / (h_j * h_k);
                    hessian[i * dim * dim + j * dim + k] = value;
                    hessian[i * dim * dim + k * dim + j] = value; // Symmetry
                }
            }
        }
    }

    Ok(hessian)
}

/// Compute the divergence of the vector field at a point
///
/// div(f) = ∂f_1/∂x_1 + ∂f_2/∂x_2 + ... = tr(J)
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `point` - The point at which to compute divergence
/// * `config` - Differentiation configuration
///
/// # Returns
///
/// The divergence (trace of Jacobian)
pub fn divergence<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
) -> Result<f64>
where
    S: DynamicalSystem<P, Q, R>,
{
    let dim = S::DIM;
    let mut div = 0.0;

    // Only compute diagonal elements
    for j in 0..dim {
        let h = config.adaptive_step(point.get(j));

        let mut x_plus = point.clone();
        x_plus.set(j, x_plus.get(j) + h);
        let f_plus = system.vector_field(&x_plus)?;

        let mut x_minus = point.clone();
        x_minus.set(j, x_minus.get(j) - h);
        let f_minus = system.vector_field(&x_minus)?;

        div += (f_plus.get(j) - f_minus.get(j)) / (2.0 * h);
    }

    Ok(div)
}

/// Compute the curl of a 3D vector field at a point
///
/// Only valid for 3D systems (P + Q + R where 2^(P+Q+R) >= 3)
///
/// curl(f) = (∂f_z/∂y - ∂f_y/∂z, ∂f_x/∂z - ∂f_z/∂x, ∂f_y/∂x - ∂f_x/∂y)
pub fn curl_3d<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
) -> Result<[f64; 3]>
where
    S: DynamicalSystem<P, Q, R>,
{
    let dim = S::DIM;
    if dim < 3 {
        return Err(DynamicsError::dimension_error(
            "Curl requires at least 3 dimensions",
        ));
    }

    // Indices for x, y, z components (using basis vector indices 1, 2, 4 for e1, e2, e3)
    let (ix, iy, iz) = (1, 2, 4.min(dim - 1));

    let jacobian = compute_jacobian(system, point, config)?;

    // curl = (∂f_z/∂y - ∂f_y/∂z, ∂f_x/∂z - ∂f_z/∂x, ∂f_y/∂x - ∂f_x/∂y)
    Ok([
        jacobian[(iz, iy)] - jacobian[(iy, iz)], // curl_x
        jacobian[(ix, iz)] - jacobian[(iz, ix)], // curl_y
        jacobian[(iy, ix)] - jacobian[(ix, iy)], // curl_z
    ])
}

/// Check if the system is volume-preserving (Hamiltonian) at a point
///
/// A system is volume-preserving if div(f) = tr(J) = 0
pub fn is_volume_preserving<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
    tolerance: f64,
) -> Result<bool>
where
    S: DynamicalSystem<P, Q, R>,
{
    let div = divergence(system, point, config)?;
    Ok(div.abs() < tolerance)
}

/// Check if the system is gradient at a point
///
/// A system f is gradient if curl(f) = 0 (for 3D) or more generally
/// if ∂f_i/∂x_j = ∂f_j/∂x_i for all i, j
pub fn is_gradient<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    point: &Multivector<P, Q, R>,
    config: &DifferentiationConfig,
    tolerance: f64,
) -> Result<bool>
where
    S: DynamicalSystem<P, Q, R>,
{
    let jacobian = compute_jacobian(system, point, config)?;
    let dim = jacobian.nrows();

    // Check symmetry
    for i in 0..dim {
        for j in i + 1..dim {
            if (jacobian[(i, j)] - jacobian[(j, i)]).abs() > tolerance {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::HarmonicOscillator;

    // Linear system for testing: dx/dt = Ax
    // dx/dt = -x + y
    // dy/dt = -x - y
    struct LinearSystem;

    impl DynamicalSystem<2, 0, 0> for LinearSystem {
        fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
            let x = state.get(1);
            let y = state.get(2);

            let mut result = Multivector::zero();
            result.set(1, -x + y);
            result.set(2, -x - y);

            Ok(result)
        }
    }

    // Gradient system: dx/dt = -∇V where V = x² + y²
    struct GradientSystem;

    impl DynamicalSystem<2, 0, 0> for GradientSystem {
        fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
            let x = state.get(1);
            let y = state.get(2);

            let mut result = Multivector::zero();
            result.set(1, -2.0 * x);
            result.set(2, -2.0 * y);

            Ok(result)
        }
    }

    #[test]
    fn test_compute_jacobian_linear() {
        let system = LinearSystem;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let jacobian = compute_jacobian(&system, &point, &config).unwrap();

        // Expected: J = [[-1, 1], [-1, -1]]
        assert!((jacobian[(1, 1)] - (-1.0)).abs() < 1e-6);
        assert!((jacobian[(1, 2)] - 1.0).abs() < 1e-6);
        assert!((jacobian[(2, 1)] - (-1.0)).abs() < 1e-6);
        assert!((jacobian[(2, 2)] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_linearize() {
        let system = GradientSystem;
        let mut point = Multivector::<2, 0, 0>::zero();
        point.set(1, 1.0);
        point.set(2, 2.0);

        let config = DifferentiationConfig::default();
        let lin = linearize(&system, &point, &config).unwrap();

        // At (1, 2): f = (-2, -4)
        assert!((lin.constant_term[1] - (-2.0)).abs() < 1e-6);
        assert!((lin.constant_term[2] - (-4.0)).abs() < 1e-6);

        // Jacobian should be -2I
        assert!((lin.jacobian[(1, 1)] - (-2.0)).abs() < 1e-6);
        assert!((lin.jacobian[(2, 2)] - (-2.0)).abs() < 1e-6);
        assert!(lin.jacobian[(1, 2)].abs() < 1e-6);
        assert!(lin.jacobian[(2, 1)].abs() < 1e-6);
    }

    #[test]
    fn test_divergence() {
        let system = LinearSystem;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let div = divergence(&system, &point, &config).unwrap();

        // tr(J) = -1 + (-1) = -2
        assert!((div - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_is_gradient() {
        let gradient_system = GradientSystem;
        let linear_system = LinearSystem;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        // Gradient system should have symmetric Jacobian
        assert!(is_gradient(&gradient_system, &point, &config, 1e-6).unwrap());

        // Linear system has antisymmetric off-diagonal elements
        assert!(!is_gradient(&linear_system, &point, &config, 1e-6).unwrap());
    }

    #[test]
    fn test_harmonic_oscillator_volume_preserving() {
        // Hamiltonian systems are volume preserving
        let system = HarmonicOscillator::new(1.0);
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        // div = ∂(y)/∂x + ∂(-x)/∂y = 0 + 0 = 0
        assert!(is_volume_preserving(&system, &point, &config, 1e-6).unwrap());
    }

    #[test]
    fn test_forward_vs_central_differences() {
        let system = GradientSystem;
        let mut point = Multivector::<2, 0, 0>::zero();
        point.set(1, 1.0);
        point.set(2, 1.0);

        let central = DifferentiationConfig::default();
        let forward = DifferentiationConfig::fast();

        let jac_central = compute_jacobian(&system, &point, &central).unwrap();
        let jac_forward = compute_jacobian(&system, &point, &forward).unwrap();

        // Both should give similar results for a linear gradient
        for i in 0..4 {
            for j in 0..4 {
                assert!((jac_central[(i, j)] - jac_forward[(i, j)]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_linearized_system_evaluation() {
        let system = GradientSystem;
        let point = Multivector::<2, 0, 0>::zero();
        let config = DifferentiationConfig::default();

        let lin = linearize(&system, &point, &config).unwrap();

        // At origin, f(0) = 0, so linear part dominates
        let mut displacement = DVector::zeros(4);
        displacement[1] = 1.0; // δx = 1

        let result = lin.evaluate(&displacement);

        // J * [0, 1, 0, 0]^T + 0 = [0, -2, 0, 0]^T
        assert!((result[1] - (-2.0)).abs() < 1e-6);
    }
}
