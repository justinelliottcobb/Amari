//! Rössler attractor
//!
//! The Rössler system is a system of three nonlinear ordinary differential
//! equations that exhibits chaotic dynamics. It was designed to be simpler
//! than the Lorenz system while still exhibiting chaotic behavior.
//!
//! # Equations
//!
//! ```text
//! dx/dt = -y - z
//! dy/dt = x + ay
//! dz/dt = b + z(x - c)
//! ```
//!
//! # Parameters
//!
//! - a: Controls the "tightness" of the spiral
//! - b: Determines the width of the attractor
//! - c: Controls when the trajectory jumps between the two lobes
//!
//! # Classic Parameter Values
//!
//! - a = 0.2, b = 0.2, c = 5.7: Classic chaotic attractor
//! - a = 0.1, b = 0.1, c = 14: More chaotic behavior
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::systems::RosslerSystem;
//! use amari_dynamics::solver::{RungeKutta4, ODESolver};
//!
//! let rossler = RosslerSystem::classic();
//! let solver = RungeKutta4::new();
//!
//! let initial = rossler.default_initial_condition();
//! let trajectory = solver.solve(&rossler, initial, 0.0, 500.0, 50000)?;
//! ```

use amari_core::Multivector;

use crate::error::Result;
use crate::flow::DynamicalSystem;

/// The Rössler system
///
/// A 3D continuous dynamical system that exhibits chaotic behavior.
/// State is represented in Cl(3,0,0) using e1, e2, e4 for x, y, z.
#[derive(Debug, Clone)]
pub struct RosslerSystem {
    /// Parameter a
    pub a: f64,
    /// Parameter b
    pub b: f64,
    /// Parameter c
    pub c: f64,
}

impl RosslerSystem {
    /// Create a new Rössler system with given parameters
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }

    /// Create the classic chaotic Rössler system (a=0.2, b=0.2, c=5.7)
    pub fn classic() -> Self {
        Self::new(0.2, 0.2, 5.7)
    }

    /// Create a Rössler system with stronger chaos (a=0.1, b=0.1, c=14)
    pub fn strong_chaos() -> Self {
        Self::new(0.1, 0.1, 14.0)
    }

    /// Create a Rössler system with parameter c for bifurcation studies
    pub fn with_c(c: f64) -> Self {
        Self::new(0.2, 0.2, c)
    }

    /// Default initial condition near the attractor
    pub fn default_initial_condition(&self) -> Multivector<3, 0, 0> {
        let mut state = Multivector::zero();
        state.set(1, 1.0); // x = 1
        state.set(2, 1.0); // y = 1
        state.set(4, 0.0); // z = 0
        state
    }

    /// Get the fixed points of the system
    ///
    /// The Rössler system has two fixed points:
    /// P± = ((c ± √(c² - 4ab))/(2a), -(c ± √(c² - 4ab))/(2a), (c ± √(c² - 4ab))/(2a))
    pub fn fixed_points(&self) -> Vec<Multivector<3, 0, 0>> {
        let mut fps = Vec::new();

        let discriminant = self.c * self.c - 4.0 * self.a * self.b;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();

            // P+
            let x_plus = (self.c + sqrt_disc) / (2.0 * self.a);
            let mut fp_plus = Multivector::zero();
            fp_plus.set(1, x_plus);
            fp_plus.set(2, -x_plus / self.a);
            fp_plus.set(4, x_plus / self.a);
            fps.push(fp_plus);

            // P-
            let x_minus = (self.c - sqrt_disc) / (2.0 * self.a);
            let mut fp_minus = Multivector::zero();
            fp_minus.set(1, x_minus);
            fp_minus.set(2, -x_minus / self.a);
            fp_minus.set(4, x_minus / self.a);
            fps.push(fp_minus);
        }

        fps
    }

    /// Get x coordinate from state
    pub fn x(state: &Multivector<3, 0, 0>) -> f64 {
        state.get(1)
    }

    /// Get y coordinate from state
    pub fn y(state: &Multivector<3, 0, 0>) -> f64 {
        state.get(2)
    }

    /// Get z coordinate from state
    pub fn z(state: &Multivector<3, 0, 0>) -> f64 {
        state.get(4)
    }

    /// Approximate Lyapunov dimension
    ///
    /// For the classic parameters, the Kaplan-Yorke dimension is approximately 2.01
    pub fn approximate_dimension(&self) -> f64 {
        if (self.a - 0.2).abs() < 0.01 && (self.b - 0.2).abs() < 0.01 && (self.c - 5.7).abs() < 0.1
        {
            2.01
        } else {
            // Rough estimate
            2.0
        }
    }
}

impl Default for RosslerSystem {
    fn default() -> Self {
        Self::classic()
    }
}

impl DynamicalSystem<3, 0, 0> for RosslerSystem {
    const DIM: usize = 3;

    fn vector_field(&self, state: &Multivector<3, 0, 0>) -> Result<Multivector<3, 0, 0>> {
        let x = state.get(1);
        let y = state.get(2);
        let z = state.get(4);

        let mut result = Multivector::zero();
        result.set(1, -y - z); // dx/dt = -y - z
        result.set(2, x + self.a * y); // dy/dt = x + ay
        result.set(4, self.b + z * (x - self.c)); // dz/dt = b + z(x - c)

        Ok(result)
    }

    fn jacobian(&self, state: &Multivector<3, 0, 0>) -> Result<Vec<f64>> {
        let x = state.get(1);
        let z = state.get(4);

        // Jacobian matrix (row-major order):
        // [  0,    -1,    -1   ]
        // [  1,     a,     0   ]
        // [  z,     0,   x - c ]
        Ok(vec![0.0, -1.0, -1.0, 1.0, self.a, 0.0, z, 0.0, x - self.c])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rossler_creation() {
        let rossler = RosslerSystem::new(0.1, 0.2, 0.3);
        assert_eq!(rossler.a, 0.1);
        assert_eq!(rossler.b, 0.2);
        assert_eq!(rossler.c, 0.3);
    }

    #[test]
    fn test_rossler_classic() {
        let rossler = RosslerSystem::classic();
        assert_eq!(rossler.a, 0.2);
        assert_eq!(rossler.b, 0.2);
        assert_eq!(rossler.c, 5.7);
    }

    #[test]
    fn test_rossler_default() {
        let rossler = RosslerSystem::default();
        assert_eq!(rossler.a, 0.2);
        assert_eq!(rossler.b, 0.2);
        assert_eq!(rossler.c, 5.7);
    }

    #[test]
    fn test_rossler_strong_chaos() {
        let rossler = RosslerSystem::strong_chaos();
        assert_eq!(rossler.a, 0.1);
        assert_eq!(rossler.b, 0.1);
        assert_eq!(rossler.c, 14.0);
    }

    #[test]
    fn test_vector_field() {
        let rossler = RosslerSystem::classic();
        let mut state = Multivector::zero();
        state.set(1, 1.0); // x = 1
        state.set(2, 2.0); // y = 2
        state.set(4, 3.0); // z = 3

        let vf = rossler.vector_field(&state).unwrap();

        // dx/dt = -y - z = -2 - 3 = -5
        assert!((vf.get(1) - (-5.0)).abs() < 1e-10);

        // dy/dt = x + ay = 1 + 0.2*2 = 1.4
        assert!((vf.get(2) - 1.4).abs() < 1e-10);

        // dz/dt = b + z(x - c) = 0.2 + 3*(1 - 5.7) = 0.2 - 14.1 = -13.9
        assert!((vf.get(4) - (-13.9)).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian() {
        let rossler = RosslerSystem::new(0.2, 0.2, 5.7);
        let mut state = Multivector::zero();
        state.set(1, 2.0);
        state.set(4, 3.0);

        let jac = rossler.jacobian(&state).unwrap();

        assert_eq!(jac.len(), 9);
        assert_eq!(jac[0], 0.0);
        assert_eq!(jac[1], -1.0);
        assert_eq!(jac[2], -1.0);
        assert_eq!(jac[3], 1.0);
        assert_eq!(jac[4], 0.2);
        assert_eq!(jac[5], 0.0);
        assert_eq!(jac[6], 3.0); // z
        assert_eq!(jac[7], 0.0);
        assert!((jac[8] - (2.0 - 5.7)).abs() < 1e-10); // x - c
    }

    #[test]
    fn test_fixed_points() {
        let rossler = RosslerSystem::classic();
        let fps = rossler.fixed_points();

        // Should have two fixed points for c² > 4ab
        assert_eq!(fps.len(), 2);
    }

    #[test]
    fn test_default_initial_condition() {
        let rossler = RosslerSystem::classic();
        let ic = rossler.default_initial_condition();
        assert_eq!(RosslerSystem::x(&ic), 1.0);
        assert_eq!(RosslerSystem::y(&ic), 1.0);
        assert_eq!(RosslerSystem::z(&ic), 0.0);
    }

    #[test]
    fn test_approximate_dimension() {
        let rossler = RosslerSystem::classic();
        let dim = rossler.approximate_dimension();
        assert!((dim - 2.01).abs() < 0.1);
    }
}
