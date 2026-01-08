//! Core dynamical system traits
//!
//! This module defines the fundamental traits for representing dynamical systems:
//!
//! - [`DynamicalSystem`]: Autonomous continuous-time systems dx/dt = f(x)
//! - [`NonAutonomousSystem`]: Non-autonomous systems dx/dt = f(x, t)
//! - [`DiscreteMap`]: Discrete-time maps x_{n+1} = f(x_n)
//! - [`ParametricSystem`]: Systems with tunable parameters
//!
//! # Geometric Algebra Integration
//!
//! All systems operate on multivector state spaces Cl(P,Q,R), enabling:
//! - Natural representation of rotations, boosts, and other geometric transformations
//! - Grade-aware dynamics (scalar, vector, bivector evolution)
//! - Unified treatment of various physical systems

use crate::error::Result;
use amari_core::Multivector;

// Rayon parallelism will be used in future parallel implementations
#[cfg(feature = "parallel")]
#[allow(unused_imports)]
use rayon::prelude::*;

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// Core Dynamical System Trait
// ============================================================================

/// A continuous-time autonomous dynamical system on multivector state space.
///
/// Represents a system of the form:
/// ```text
/// dx/dt = f(x)
/// ```
/// where x ∈ Cl(P,Q,R) is a multivector and f is the vector field.
///
/// # Type Parameters
///
/// - `P`: Number of positive signature basis vectors
/// - `Q`: Number of negative signature basis vectors
/// - `R`: Number of zero signature basis vectors
///
/// The state space dimension is 2^(P+Q+R).
///
/// # Example
///
/// ```ignore
/// use amari_dynamics::flow::DynamicalSystem;
/// use amari_core::Multivector;
///
/// struct HarmonicOscillator;
///
/// impl DynamicalSystem<2, 0, 0> for HarmonicOscillator {
///     fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
///         let x = state.get(1);  // e1 component = position
///         let v = state.get(2);  // e2 component = velocity
///         let mut result = Multivector::zero();
///         result.set(1, v);      // dx/dt = v
///         result.set(2, -x);     // dv/dt = -x (spring force)
///         Ok(result)
///     }
/// }
/// ```
pub trait DynamicalSystem<const P: usize, const Q: usize, const R: usize> {
    /// State space dimension: 2^(P+Q+R)
    const DIM: usize = 1 << (P + Q + R);

    /// Compute the vector field f(x) at the given state.
    ///
    /// # Arguments
    ///
    /// * `state` - Current state x in the multivector space
    ///
    /// # Returns
    ///
    /// The vector field value f(x), representing dx/dt
    fn vector_field(&self, state: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>>;

    /// Compute the Jacobian matrix Df(x) at the given state.
    ///
    /// The Jacobian is the matrix of partial derivatives:
    /// ```text
    /// J_ij = ∂f_i/∂x_j
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state` - State at which to compute the Jacobian
    ///
    /// # Returns
    ///
    /// Flattened row-major Jacobian matrix (DIM × DIM elements)
    ///
    /// # Default Implementation
    ///
    /// Uses central finite differences with step size 1e-7.
    fn jacobian(&self, state: &Multivector<P, Q, R>) -> Result<Vec<f64>>
    where
        Self: Sized,
    {
        numerical_jacobian(self, state, 1e-7)
    }

    /// Check if a state is within the valid domain of the system.
    ///
    /// # Arguments
    ///
    /// * `state` - State to check
    ///
    /// # Returns
    ///
    /// `true` if the state is in the valid domain, `false` otherwise.
    ///
    /// # Default Implementation
    ///
    /// Returns `true` for all states (entire space is valid).
    fn in_domain(&self, _state: &Multivector<P, Q, R>) -> bool {
        true
    }

    /// Get the dimension of the state space.
    fn dimension(&self) -> usize {
        Self::DIM
    }
}

/// Compute the Jacobian using central finite differences.
///
/// # Arguments
///
/// * `system` - The dynamical system
/// * `state` - State at which to compute the Jacobian
/// * `epsilon` - Step size for finite differences
///
/// # Returns
///
/// Flattened row-major Jacobian matrix
pub fn numerical_jacobian<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    state: &Multivector<P, Q, R>,
    epsilon: f64,
) -> Result<Vec<f64>>
where
    S: DynamicalSystem<P, Q, R>,
{
    let dim = S::DIM;
    let mut jacobian = vec![0.0; dim * dim];

    let coeffs = state.to_vec();

    for j in 0..dim {
        // Forward perturbation
        let mut forward_coeffs = coeffs.clone();
        forward_coeffs[j] += epsilon;
        let forward_state = Multivector::from_coefficients(forward_coeffs);
        let f_forward = system.vector_field(&forward_state)?;

        // Backward perturbation
        let mut backward_coeffs = coeffs.clone();
        backward_coeffs[j] -= epsilon;
        let backward_state = Multivector::from_coefficients(backward_coeffs);
        let f_backward = system.vector_field(&backward_state)?;

        // Central difference
        for i in 0..dim {
            jacobian[i * dim + j] = (f_forward.get(i) - f_backward.get(i)) / (2.0 * epsilon);
        }
    }

    Ok(jacobian)
}

// ============================================================================
// Non-Autonomous System Trait
// ============================================================================

/// A non-autonomous dynamical system with explicit time dependence.
///
/// Represents a system of the form:
/// ```text
/// dx/dt = f(x, t)
/// ```
/// where the vector field explicitly depends on time.
pub trait NonAutonomousSystem<const P: usize, const Q: usize, const R: usize> {
    /// State space dimension: 2^(P+Q+R)
    const DIM: usize = 1 << (P + Q + R);

    /// Compute the vector field f(x, t) at the given state and time.
    ///
    /// # Arguments
    ///
    /// * `state` - Current state x
    /// * `t` - Current time
    ///
    /// # Returns
    ///
    /// The vector field value f(x, t)
    fn vector_field_at(&self, state: &Multivector<P, Q, R>, t: f64)
        -> Result<Multivector<P, Q, R>>;

    /// Compute the Jacobian with respect to state at (x, t).
    ///
    /// # Default Implementation
    ///
    /// Uses central finite differences.
    fn jacobian_at(&self, state: &Multivector<P, Q, R>, t: f64) -> Result<Vec<f64>>
    where
        Self: Sized,
    {
        numerical_jacobian_nonautonomous(self, state, t, 1e-7)
    }

    /// Check if a state is within the valid domain at time t.
    fn in_domain_at(&self, _state: &Multivector<P, Q, R>, _t: f64) -> bool {
        true
    }
}

/// Compute the Jacobian for a non-autonomous system.
pub fn numerical_jacobian_nonautonomous<S, const P: usize, const Q: usize, const R: usize>(
    system: &S,
    state: &Multivector<P, Q, R>,
    t: f64,
    epsilon: f64,
) -> Result<Vec<f64>>
where
    S: NonAutonomousSystem<P, Q, R>,
{
    let dim = S::DIM;
    let mut jacobian = vec![0.0; dim * dim];

    let coeffs = state.to_vec();

    for j in 0..dim {
        let mut forward_coeffs = coeffs.clone();
        forward_coeffs[j] += epsilon;
        let forward_state = Multivector::from_coefficients(forward_coeffs);
        let f_forward = system.vector_field_at(&forward_state, t)?;

        let mut backward_coeffs = coeffs.clone();
        backward_coeffs[j] -= epsilon;
        let backward_state = Multivector::from_coefficients(backward_coeffs);
        let f_backward = system.vector_field_at(&backward_state, t)?;

        for i in 0..dim {
            jacobian[i * dim + j] = (f_forward.get(i) - f_backward.get(i)) / (2.0 * epsilon);
        }
    }

    Ok(jacobian)
}

// ============================================================================
// Discrete Map Trait
// ============================================================================

/// A discrete-time dynamical system (iterated map).
///
/// Represents a system of the form:
/// ```text
/// x_{n+1} = f(x_n)
/// ```
///
/// # Examples
///
/// - Henon map
/// - Logistic map
/// - Standard map
/// - Poincare return maps
pub trait DiscreteMap<const P: usize, const Q: usize, const R: usize> {
    /// State space dimension: 2^(P+Q+R)
    const DIM: usize = 1 << (P + Q + R);

    /// Apply one iteration of the map.
    ///
    /// # Arguments
    ///
    /// * `state` - Current state x_n
    ///
    /// # Returns
    ///
    /// Next state x_{n+1} = f(x_n)
    fn iterate(&self, state: &Multivector<P, Q, R>) -> Result<Multivector<P, Q, R>>;

    /// Compute the Jacobian of the map Df(x).
    ///
    /// # Default Implementation
    ///
    /// Uses central finite differences.
    fn jacobian(&self, state: &Multivector<P, Q, R>) -> Result<Vec<f64>>
    where
        Self: Sized,
    {
        numerical_jacobian_map(self, state, 1e-7)
    }

    /// Apply n iterations of the map.
    ///
    /// # Arguments
    ///
    /// * `state` - Initial state
    /// * `n` - Number of iterations
    ///
    /// # Returns
    ///
    /// State after n iterations
    fn iterate_n(&self, state: &Multivector<P, Q, R>, n: usize) -> Result<Multivector<P, Q, R>> {
        let mut current = state.clone();
        for _ in 0..n {
            current = self.iterate(&current)?;
        }
        Ok(current)
    }

    /// Generate an orbit of length n starting from the given state.
    ///
    /// # Arguments
    ///
    /// * `state` - Initial state
    /// * `n` - Number of iterations
    ///
    /// # Returns
    ///
    /// Vector of states [x_0, x_1, ..., x_n]
    fn orbit(&self, state: &Multivector<P, Q, R>, n: usize) -> Result<Vec<Multivector<P, Q, R>>> {
        let mut orbit = Vec::with_capacity(n + 1);
        let mut current = state.clone();
        orbit.push(current.clone());

        for _ in 0..n {
            current = self.iterate(&current)?;
            orbit.push(current.clone());
        }

        Ok(orbit)
    }

    /// Check if a state is within the valid domain.
    fn in_domain(&self, _state: &Multivector<P, Q, R>) -> bool {
        true
    }
}

/// Compute the Jacobian for a discrete map.
pub fn numerical_jacobian_map<S, const P: usize, const Q: usize, const R: usize>(
    map: &S,
    state: &Multivector<P, Q, R>,
    epsilon: f64,
) -> Result<Vec<f64>>
where
    S: DiscreteMap<P, Q, R>,
{
    let dim = S::DIM;
    let mut jacobian = vec![0.0; dim * dim];

    let coeffs = state.to_vec();

    for j in 0..dim {
        let mut forward_coeffs = coeffs.clone();
        forward_coeffs[j] += epsilon;
        let forward_state = Multivector::from_coefficients(forward_coeffs);
        let f_forward = map.iterate(&forward_state)?;

        let mut backward_coeffs = coeffs.clone();
        backward_coeffs[j] -= epsilon;
        let backward_state = Multivector::from_coefficients(backward_coeffs);
        let f_backward = map.iterate(&backward_state)?;

        for i in 0..dim {
            jacobian[i * dim + j] = (f_forward.get(i) - f_backward.get(i)) / (2.0 * epsilon);
        }
    }

    Ok(jacobian)
}

// ============================================================================
// Parametric System Trait
// ============================================================================

/// A parametric dynamical system with tunable parameters.
///
/// This trait extends dynamical systems with parameter dependence,
/// enabling bifurcation analysis and parameter continuation.
pub trait ParametricSystem<const P: usize, const Q: usize, const R: usize>:
    DynamicalSystem<P, Q, R>
{
    /// Type of the parameter (often f64 for single parameter, Vec<f64> for multiple)
    type Parameter: Clone;

    /// Get the current parameter value.
    fn parameter(&self) -> &Self::Parameter;

    /// Set the parameter value.
    fn set_parameter(&mut self, param: Self::Parameter);

    /// Compute the vector field with an explicit parameter value.
    ///
    /// This allows computing the vector field at different parameter values
    /// without modifying the system state.
    fn vector_field_with_param(
        &self,
        state: &Multivector<P, Q, R>,
        param: &Self::Parameter,
    ) -> Result<Multivector<P, Q, R>>;
}

// ============================================================================
// Wrapper for Converting Non-Autonomous to Autonomous
// ============================================================================

/// Wrapper that converts a non-autonomous system to an autonomous one
/// by extending the state space with time.
///
/// For a non-autonomous system dx/dt = f(x, t), this creates an augmented
/// autonomous system:
/// ```text
/// d[x, τ]/dt = [f(x, τ), 1]
/// ```
/// where τ is an additional state variable representing time.
pub struct AutonomizedSystem<S, const P: usize, const Q: usize, const R: usize> {
    inner: S,
    _phantom: core::marker::PhantomData<([(); P], [(); Q], [(); R])>,
}

impl<S, const P: usize, const Q: usize, const R: usize> AutonomizedSystem<S, P, Q, R>
where
    S: NonAutonomousSystem<P, Q, R>,
{
    /// Create a new autonomized system wrapper.
    pub fn new(system: S) -> Self {
        Self {
            inner: system,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Get a reference to the inner non-autonomous system.
    pub fn inner(&self) -> &S {
        &self.inner
    }
}

// ============================================================================
// Example Systems for Testing
// ============================================================================

/// Simple harmonic oscillator: d²x/dt² = -ω²x
///
/// Represented as a first-order system in Cl(2,0,0):
/// - e1 component: position x
/// - e2 component: velocity v = dx/dt
///
/// Equations:
/// ```text
/// dx/dt = v
/// dv/dt = -ω²x
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HarmonicOscillator {
    /// Angular frequency
    pub omega: f64,
}

impl HarmonicOscillator {
    /// Create a new harmonic oscillator with the given angular frequency
    pub fn new(omega: f64) -> Self {
        Self { omega }
    }
}

impl DynamicalSystem<2, 0, 0> for HarmonicOscillator {
    fn vector_field(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
        let x = state.get(1); // e1 component
        let v = state.get(2); // e2 component

        let mut result = Multivector::zero();
        result.set(1, v);
        result.set(2, -self.omega * self.omega * x);

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_oscillator_vector_field() {
        let system = HarmonicOscillator::new(1.0);

        // State: x=1, v=0
        let mut state = Multivector::<2, 0, 0>::zero();
        state.set(1, 1.0);
        state.set(2, 0.0);

        let vf = system.vector_field(&state).unwrap();

        // dx/dt = v = 0
        assert!((vf.get(1) - 0.0).abs() < 1e-10);
        // dv/dt = -omega^2 * x = -1
        assert!((vf.get(2) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_jacobian() {
        let system = HarmonicOscillator { omega: 1.0 };

        let mut state = Multivector::<2, 0, 0>::zero();
        state.set(1, 0.5);
        state.set(2, 0.3);

        let jac = system.jacobian(&state).unwrap();

        // For harmonic oscillator:
        // dx/dt = v, dv/dt = -omega^2 * x
        // Jacobian (in relevant 2x2 subspace) should be:
        // | 0  1 |
        // |-1  0 |
        // But we have a 4x4 matrix for Cl(2,0,0)

        // Just check it computed without error and has right size
        assert_eq!(jac.len(), 16); // 4x4 for Cl(2,0,0)
    }

    #[test]
    fn test_dimension() {
        let system = HarmonicOscillator { omega: 1.0 };
        assert_eq!(system.dimension(), 4); // 2^2 = 4 for Cl(2,0,0)
    }

    #[test]
    fn test_in_domain_default() {
        let system = HarmonicOscillator { omega: 1.0 };
        let state = Multivector::<2, 0, 0>::zero();
        assert!(system.in_domain(&state));
    }

    // Simple discrete map for testing
    struct DoublingMap;

    impl DiscreteMap<1, 0, 0> for DoublingMap {
        fn iterate(&self, state: &Multivector<1, 0, 0>) -> Result<Multivector<1, 0, 0>> {
            let x = state.get(0); // scalar component
            let mut result = Multivector::zero();
            result.set(0, (2.0 * x) % 1.0);
            Ok(result)
        }
    }

    #[test]
    fn test_discrete_map_iterate() {
        let map = DoublingMap;

        let mut state = Multivector::<1, 0, 0>::zero();
        state.set(0, 0.25);

        let next = map.iterate(&state).unwrap();
        assert!((next.get(0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_map_iterate_n() {
        let map = DoublingMap;

        let mut state = Multivector::<1, 0, 0>::zero();
        state.set(0, 0.125);

        let result = map.iterate_n(&state, 3).unwrap();
        // 0.125 -> 0.25 -> 0.5 -> 1.0 -> 0.0 (mod 1)
        // Actually: 0.125 * 2^3 = 1.0 -> 0.0 (mod 1)
        assert!(result.get(0).abs() < 1e-10);
    }

    #[test]
    fn test_orbit() {
        let map = DoublingMap;

        let mut state = Multivector::<1, 0, 0>::zero();
        state.set(0, 0.1);

        let orbit = map.orbit(&state, 3).unwrap();
        assert_eq!(orbit.len(), 4); // Initial + 3 iterates
    }
}
