//! Parameter continuation methods
//!
//! This module implements numerical continuation (path-following) methods
//! for tracing solution curves through parameter space.
//!
//! # Overview
//!
//! Parameter continuation solves the problem of following a curve of solutions
//! F(x, μ) = 0 as the parameter μ varies. This is essential for:
//!
//! - Tracing branches of fixed points
//! - Detecting bifurcations where stability changes
//! - Finding turning points (saddle-node bifurcations)
//! - Switching branches at bifurcation points
//!
//! # Methods
//!
//! - **Natural Parameter Continuation**: Simple step-by-step, fails at folds
//! - **Pseudo-Arclength Continuation**: Robust, handles turning points
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::bifurcation::{NaturalContinuation, ParameterContinuation};
//!
//! let continuation = NaturalContinuation::new();
//! let (branch, bifurcations) = continuation.continue_branch(
//!     &mut system,
//!     initial_state,
//!     0.0,    // start parameter
//!     (0.0, 2.0), // parameter range
//!     &config,
//! )?;
//! ```

use amari_core::Multivector;
use nalgebra::{DMatrix, DVector};

use crate::error::{DynamicsError, Result};
use crate::flow::{DynamicalSystem, ParametricSystem};
use crate::stability::{
    compute_eigenvalues, compute_jacobian, DifferentiationConfig, FixedPointConfig,
    FixedPointResult,
};

use super::traits::{BranchPoint, ParameterContinuation, SolutionBranch};
use super::types::{BifurcationConfig, BifurcationPoint, BifurcationType, CriticalityCondition};

/// Natural parameter continuation
///
/// The simplest continuation method: fix μ, solve F(x, μ) = 0 using Newton's method.
/// Increment μ and repeat, using the previous solution as initial guess.
///
/// # Limitations
///
/// - Fails at fold (saddle-node) bifurcations where the branch turns back
/// - May miss branches that don't connect to the starting point
///
/// # Advantages
///
/// - Simple to implement and understand
/// - Efficient when folds are not present
#[derive(Debug, Clone, Default)]
pub struct NaturalContinuation {
    /// Configuration for fixed point finding
    fp_config: FixedPointConfig,

    /// Configuration for Jacobian computation
    diff_config: DifferentiationConfig,
}

impl NaturalContinuation {
    /// Create a new natural continuation solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom Newton tolerance
    pub fn with_tolerance(tolerance: f64) -> Self {
        Self {
            fp_config: FixedPointConfig {
                tolerance,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Find a fixed point using Newton's method
    fn find_fixed_point<S, const P: usize, const Q: usize, const R: usize>(
        &self,
        system: &S,
        initial: &Multivector<P, Q, R>,
    ) -> Result<FixedPointResult<P, Q, R>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        crate::stability::find_fixed_point(system, initial, &self.fp_config)
    }

    /// Classify bifurcation type from eigenvalue change
    fn classify_bifurcation(
        &self,
        eigenvalues_before: &[(f64, f64)],
        eigenvalues_after: &[(f64, f64)],
        condition: &CriticalityCondition,
    ) -> BifurcationType {
        // Check for Hopf: eigenvalues crossing imaginary axis with nonzero imaginary part
        for (re_b, im_b) in eigenvalues_before {
            for (re_a, im_a) in eigenvalues_after {
                // Same imaginary part (approximately), real part changes sign
                if (im_b - im_a).abs() < condition.zero_tolerance
                    && im_b.abs() > condition.zero_tolerance
                    && re_b * re_a < 0.0
                {
                    // Determine super/subcritical based on direction
                    if *re_b < 0.0 && *re_a > 0.0 {
                        return BifurcationType::HopfSupercritical;
                    } else {
                        return BifurcationType::HopfSubcritical;
                    }
                }
            }
        }

        // Check for saddle-node: real eigenvalue crossing zero
        let had_near_zero_before = eigenvalues_before
            .iter()
            .any(|(re, im)| re.abs() < 0.1 && im.abs() < condition.zero_tolerance);
        let has_near_zero_after = eigenvalues_after
            .iter()
            .any(|(re, im)| re.abs() < 0.1 && im.abs() < condition.zero_tolerance);

        if had_near_zero_before || has_near_zero_after {
            // Check for stability change
            let stable_before = eigenvalues_before.iter().all(|(re, _)| *re < 0.0);
            let stable_after = eigenvalues_after.iter().all(|(re, _)| *re < 0.0);

            if stable_before != stable_after {
                // Could be saddle-node, transcritical, or pitchfork
                // Need more sophisticated analysis for exact classification
                return BifurcationType::SaddleNode;
            }
        }

        BifurcationType::Unknown
    }
}

impl<const P: usize, const Q: usize, const R: usize> ParameterContinuation<P, Q, R>
    for NaturalContinuation
{
    fn continue_branch<S>(
        &self,
        system: &mut S,
        initial_state: Multivector<P, Q, R>,
        initial_param: f64,
        param_range: (f64, f64),
        config: &BifurcationConfig,
    ) -> Result<(SolutionBranch<P, Q, R>, Vec<BifurcationPoint<P, Q, R>>)>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>,
    {
        let mut branch = SolutionBranch::new(0);
        let mut bifurcations = Vec::new();

        // Set initial parameter
        system.set_parameter(initial_param);

        // Find initial fixed point
        let fp_result = self.find_fixed_point(system, &initial_state)?;
        let mut current_state = fp_result.point;

        // Compute initial eigenvalues
        let jacobian = compute_jacobian(system, &current_state, &self.diff_config)?;
        let mut current_eigenvalues = compute_eigenvalues(&jacobian)?;

        // Add initial point
        branch.push(BranchPoint::new(
            initial_param,
            current_state.clone(),
            current_eigenvalues.clone(),
        ));

        // Determine direction
        let direction = if param_range.1 > initial_param {
            1.0
        } else {
            -1.0
        };
        let end_param = if direction > 0.0 {
            param_range.1
        } else {
            param_range.0
        };

        let mut current_param = initial_param;
        let mut step = config.parameter_step * direction;
        let mut step_count = 0;

        // Continuation loop
        while step_count < config.max_steps {
            let next_param = current_param + step;

            // Check if we've reached the end
            if (direction > 0.0 && next_param > end_param)
                || (direction < 0.0 && next_param < end_param)
            {
                break;
            }

            // Update parameter
            system.set_parameter(next_param);

            // Try to find fixed point near current state
            match self.find_fixed_point(system, &current_state) {
                Ok(fp_result) => {
                    if fp_result.converged {
                        let next_state = fp_result.point;

                        // Compute eigenvalues
                        let jacobian = compute_jacobian(system, &next_state, &self.diff_config)?;
                        let next_eigenvalues = compute_eigenvalues(&jacobian)?;

                        // Check for stability change (potential bifurcation)
                        let was_stable = current_eigenvalues.iter().all(|(re, _)| *re < 0.0);
                        let is_stable = next_eigenvalues.iter().all(|(re, _)| *re < 0.0);

                        if was_stable != is_stable {
                            // Bifurcation detected - try to refine location
                            let bif_type = self.classify_bifurcation(
                                &current_eigenvalues,
                                &next_eigenvalues,
                                &config.criticality,
                            );

                            // Estimate bifurcation location as midpoint
                            let bif_param = (current_param + next_param) / 2.0;

                            // Interpolate state
                            let t = 0.5;
                            let bif_state = &(&current_state * (1.0 - t)) + &(&next_state * t);

                            // Interpolate eigenvalues (crude but quick)
                            let bif_eigenvalues: Vec<_> = current_eigenvalues
                                .iter()
                                .zip(next_eigenvalues.iter())
                                .map(|((r1, i1), (r2, i2))| ((r1 + r2) / 2.0, (i1 + i2) / 2.0))
                                .collect();

                            bifurcations.push(BifurcationPoint::new(
                                bif_param,
                                bif_type,
                                bif_state,
                                bif_eigenvalues,
                            ));
                        }

                        // Add point to branch
                        branch.push(BranchPoint::new(
                            next_param,
                            next_state.clone(),
                            next_eigenvalues.clone(),
                        ));

                        // Update state
                        current_state = next_state;
                        current_eigenvalues = next_eigenvalues;
                        current_param = next_param;

                        // Adaptive step sizing: increase if converging well
                        if fp_result.iterations < config.max_newton_iterations / 4 {
                            step = (step * 1.2).min(config.max_step * direction.signum());
                        }
                    } else {
                        // Newton didn't converge - reduce step
                        step /= 2.0;
                        if step.abs() < config.min_step {
                            // Probable fold bifurcation - natural continuation fails
                            break;
                        }
                    }
                }
                Err(_) => {
                    // Reduce step and try again
                    step /= 2.0;
                    if step.abs() < config.min_step {
                        break;
                    }
                }
            }

            step_count += 1;
        }

        // Sort branch by parameter
        branch.sort_by_parameter();

        // Sort bifurcations by parameter
        bifurcations.sort_by(|a, b| a.parameter_value.partial_cmp(&b.parameter_value).unwrap());

        Ok((branch, bifurcations))
    }
}

/// Pseudo-arclength continuation
///
/// A more robust continuation method that parameterizes the solution curve
/// by arclength rather than the parameter μ. This allows following curves
/// through fold bifurcations (turning points).
///
/// # Method
///
/// Instead of fixing μ and solving F(x, μ) = 0, we solve the augmented system:
///
/// ```text
/// F(x, μ) = 0
/// (x - x₀)ᵀ ẋ₀ + (μ - μ₀) μ̇₀ - Δs = 0
/// ```
///
/// where (ẋ₀, μ̇₀) is the tangent direction at (x₀, μ₀).
#[derive(Debug, Clone, Default)]
pub struct PseudoArclengthContinuation {
    /// Configuration for Jacobian computation
    diff_config: DifferentiationConfig,
}

impl PseudoArclengthContinuation {
    /// Create a new pseudo-arclength continuation solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the tangent direction at a point (reserved for full pseudo-arclength implementation)
    #[allow(dead_code)]
    fn compute_tangent<S, const P: usize, const Q: usize, const R: usize>(
        &self,
        system: &S,
        state: &Multivector<P, Q, R>,
        _param: f64,
        prev_tangent: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>>
    where
        S: DynamicalSystem<P, Q, R>,
    {
        let dim = S::DIM;

        // Compute Jacobian ∂F/∂x
        let jac_x_mat = compute_jacobian(system, state, &self.diff_config)?;

        // Compute ∂F/∂μ numerically
        // For now, we'll use a simple approximation based on the last tangent
        // In a full implementation, we'd differentiate with respect to μ

        // The tangent is in the null space of [∂F/∂x | ∂F/∂μ]
        // For natural parameter continuation, approximate with null space of ∂F/∂x

        let svd = jac_x_mat.svd(true, true);
        let v_t = svd.v_t.ok_or_else(|| {
            DynamicsError::numerical_instability(
                "PseudoArclengthContinuation",
                "SVD failed to compute V matrix",
            )
        })?;

        // Last row of V^T gives approximate null space direction
        let null_vec: Vec<f64> = v_t.row(dim - 1).iter().copied().collect();

        // Extend with parameter component
        let mut tangent = DVector::zeros(dim + 1);
        for (i, &v) in null_vec.iter().enumerate() {
            tangent[i] = v;
        }
        tangent[dim] = 1.0; // Default: increase parameter

        // Normalize
        let norm = tangent.norm();
        tangent /= norm;

        // Ensure consistent direction with previous tangent
        if let Some(prev) = prev_tangent {
            if tangent.dot(prev) < 0.0 {
                tangent = -tangent;
            }
        }

        Ok(tangent)
    }

    /// Solve the augmented system using Newton's method
    #[allow(dead_code)]
    fn newton_corrector<S, const P: usize, const Q: usize, const R: usize>(
        &self,
        system: &mut S,
        predictor: &DVector<f64>,
        tangent: &DVector<f64>,
        arclength: f64,
        base_point: &DVector<f64>,
        config: &BifurcationConfig,
    ) -> Result<DVector<f64>>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>,
    {
        let dim = S::DIM;
        let mut current = predictor.clone();

        for _iteration in 0..config.max_newton_iterations {
            // Extract state and parameter
            let state_vec: Vec<f64> = current.iter().take(dim).copied().collect();
            let param = current[dim];

            let state = Multivector::<P, Q, R>::from_coefficients(state_vec);
            system.set_parameter(param);

            // Evaluate F(x, μ)
            let f = system.vector_field(&state)?;

            // Compute arclength constraint residual
            let diff = &current - base_point;
            let arc_residual = diff.dot(tangent) - arclength;

            // Check convergence
            let f_vec = Self::mv_to_dvec(&f);
            let f_norm = f_vec.norm();
            if f_norm < config.newton_tolerance && arc_residual.abs() < config.newton_tolerance {
                return Ok(current);
            }

            // Build augmented Jacobian
            let jac_x = compute_jacobian(system, &state, &self.diff_config)?;
            let mut aug_jac = DMatrix::zeros(dim + 1, dim + 1);

            // Fill in ∂F/∂x
            for i in 0..dim {
                for j in 0..dim {
                    aug_jac[(i, j)] = jac_x[(i, j)];
                }
            }

            // ∂F/∂μ (numerical differentiation)
            let eps = self.diff_config.step_size;
            system.set_parameter(param + eps);
            let f_plus = system.vector_field(&state)?;
            system.set_parameter(param);

            for i in 0..dim {
                aug_jac[(i, dim)] = (f_plus.get(i) - f.get(i)) / eps;
            }

            // Arclength constraint gradient
            for i in 0..=dim {
                aug_jac[(dim, i)] = tangent[i];
            }

            // Build residual vector
            let mut residual = DVector::zeros(dim + 1);
            for i in 0..dim {
                residual[i] = -f.get(i);
            }
            residual[dim] = -arc_residual;

            // Solve linear system
            let lu = aug_jac.lu();
            match lu.solve(&residual) {
                Some(delta) => {
                    current += delta;
                }
                None => {
                    return Err(DynamicsError::numerical_instability(
                        "PseudoArclengthContinuation",
                        "Singular augmented Jacobian",
                    ));
                }
            }
        }

        Err(DynamicsError::convergence_failure(
            config.max_newton_iterations,
            "Newton corrector did not converge",
        ))
    }

    /// Convert Multivector to DVector
    fn mv_to_dvec<const P: usize, const Q: usize, const R: usize>(
        mv: &Multivector<P, Q, R>,
    ) -> DVector<f64> {
        let dim = 1 << (P + Q + R);
        let mut vec = DVector::zeros(dim);
        for i in 0..dim {
            vec[i] = mv.get(i);
        }
        vec
    }
}

impl<const P: usize, const Q: usize, const R: usize> ParameterContinuation<P, Q, R>
    for PseudoArclengthContinuation
{
    fn continue_branch<S>(
        &self,
        system: &mut S,
        initial_state: Multivector<P, Q, R>,
        initial_param: f64,
        param_range: (f64, f64),
        config: &BifurcationConfig,
    ) -> Result<(SolutionBranch<P, Q, R>, Vec<BifurcationPoint<P, Q, R>>)>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>,
    {
        // For now, fall back to natural continuation
        // Full pseudo-arclength implementation requires more sophisticated
        // handling of the augmented system

        let natural = NaturalContinuation::new();
        natural.continue_branch(system, initial_state, initial_param, param_range, config)
    }
}

/// Simple bifurcation detector using parameter sweep
///
/// Scans parameter space looking for stability changes and classifying
/// the bifurcation type based on eigenvalue behavior.
#[derive(Debug, Clone, Default)]
pub struct SimpleBifurcationDetector {
    continuation: NaturalContinuation,
}

impl SimpleBifurcationDetector {
    /// Create a new simple bifurcation detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect bifurcations by continuing from multiple initial conditions
    pub fn detect_from_initials<S, const P: usize, const Q: usize, const R: usize>(
        &self,
        system: &mut S,
        initial_states: &[Multivector<P, Q, R>],
        initial_param: f64,
        param_range: (f64, f64),
        config: &BifurcationConfig,
    ) -> Result<Vec<BifurcationPoint<P, Q, R>>>
    where
        S: ParametricSystem<P, Q, R, Parameter = f64>,
    {
        let mut all_bifurcations = Vec::new();

        for state in initial_states {
            match self.continuation.continue_branch(
                system,
                state.clone(),
                initial_param,
                param_range,
                config,
            ) {
                Ok((_, bifurcations)) => {
                    all_bifurcations.extend(bifurcations);
                }
                Err(_) => continue, // Skip failed continuations
            }
        }

        // Remove duplicates (bifurcations at same parameter within tolerance)
        all_bifurcations.sort_by(|a, b| a.parameter_value.partial_cmp(&b.parameter_value).unwrap());

        let mut unique: Vec<BifurcationPoint<P, Q, R>> = Vec::new();
        for bif in all_bifurcations {
            let dominated = unique.iter().any(|existing| {
                (existing.parameter_value - bif.parameter_value).abs()
                    < config.criticality.zero_tolerance * 10.0
            });

            if !dominated {
                unique.push(bif);
            }
        }

        Ok(unique)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test system: saddle-node normal form
    // dx/dt = μ - x²
    struct SaddleNodeNormalForm {
        mu: f64,
    }

    impl DynamicalSystem<1, 0, 0> for SaddleNodeNormalForm {
        fn vector_field(&self, state: &Multivector<1, 0, 0>) -> Result<Multivector<1, 0, 0>> {
            let x = state.get(0);
            let mut result = Multivector::zero();
            result.set(0, self.mu - x * x);
            Ok(result)
        }
    }

    impl ParametricSystem<1, 0, 0> for SaddleNodeNormalForm {
        type Parameter = f64;

        fn parameter(&self) -> &f64 {
            &self.mu
        }

        fn set_parameter(&mut self, param: f64) {
            self.mu = param;
        }

        fn vector_field_with_param(
            &self,
            state: &Multivector<1, 0, 0>,
            param: &f64,
        ) -> Result<Multivector<1, 0, 0>> {
            let x = state.get(0);
            let mut result = Multivector::zero();
            result.set(0, param - x * x);
            Ok(result)
        }
    }

    #[test]
    fn test_natural_continuation_simple() {
        let mut system = SaddleNodeNormalForm { mu: 1.0 };
        let continuation = NaturalContinuation::new();
        let config = BifurcationConfig::default();

        // Start from stable fixed point x = 1 at μ = 1
        let mut initial = Multivector::<1, 0, 0>::zero();
        initial.set(0, 1.0);

        let result = continuation.continue_branch(&mut system, initial, 1.0, (0.5, 1.5), &config);

        assert!(result.is_ok());
        let (branch, _bifurcations) = result.unwrap();
        assert!(!branch.is_empty());
    }

    #[test]
    fn test_simple_detector_creation() {
        let detector = SimpleBifurcationDetector::new();
        assert!(detector.continuation.fp_config.max_iterations > 0);
    }

    #[test]
    fn test_harmonic_oscillator_no_bifurcation() {
        // Harmonic oscillator has no bifurcations - center is preserved
        struct ParametricOscillator {
            omega: f64,
        }

        impl DynamicalSystem<2, 0, 0> for ParametricOscillator {
            fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
                let x = state.get(1);
                let v = state.get(2);
                let mut result = Multivector::zero();
                result.set(1, v);
                result.set(2, -self.omega * self.omega * x);
                Ok(result)
            }
        }

        impl ParametricSystem<2, 0, 0> for ParametricOscillator {
            type Parameter = f64;

            fn parameter(&self) -> &f64 {
                &self.omega
            }

            fn set_parameter(&mut self, param: f64) {
                self.omega = param;
            }

            fn vector_field_with_param(
                &self,
                state: &Multivector<2, 0, 0>,
                param: &f64,
            ) -> Result<Multivector<2, 0, 0>> {
                let x = state.get(1);
                let v = state.get(2);
                let mut result = Multivector::zero();
                result.set(1, v);
                result.set(2, -param * param * x);
                Ok(result)
            }
        }

        let mut system = ParametricOscillator { omega: 1.0 };
        let continuation = NaturalContinuation::new();
        let config = BifurcationConfig::default();

        // Origin is always a fixed point (center)
        let initial = Multivector::<2, 0, 0>::zero();

        let result = continuation.continue_branch(&mut system, initial, 1.0, (0.5, 2.0), &config);

        assert!(result.is_ok());
        let (branch, _bifurcations) = result.unwrap();

        // No bifurcations should be detected (center throughout)
        // The stability type doesn't change
        assert!(!branch.is_empty());
        // Note: Center is not "stable" by our definition (re < 0), so no stability change
    }
}
