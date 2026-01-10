//! Hénon map
//!
//! The Hénon map is a discrete-time dynamical system that exhibits chaotic
//! behavior for certain parameter values. It was introduced by Michel Hénon
//! as a simplified model of the Poincaré section of the Lorenz system.
//!
//! # Equations
//!
//! ```text
//! x_{n+1} = 1 - ax_n² + y_n
//! y_{n+1} = bx_n
//! ```
//!
//! # Parameters
//!
//! - a: Controls the nonlinearity (bending of the attractor)
//! - b: Controls the contraction (area-preserving when |b|=1)
//!
//! # Classic Parameter Values
//!
//! - a = 1.4, b = 0.3: Classic chaotic Hénon attractor
//! - a = 1.2, b = 0.3: Different chaotic regime
//!
//! # Properties
//!
//! - Dissipative: |b| < 1 contracts areas
//! - Jacobian determinant: det(J) = -b (constant)
//! - Lyapunov exponents (classic): λ₁ ≈ 0.42, λ₂ ≈ -1.62
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::systems::HenonMap;
//!
//! let henon = HenonMap::classic();
//! let initial = henon.default_initial_condition();
//!
//! // Generate orbit
//! let orbit = henon.orbit(&initial, 10000)?;
//! ```

use amari_core::Multivector;

use crate::error::Result;
use crate::flow::DiscreteMap;

/// The Hénon map
///
/// A 2D discrete-time chaotic system.
/// State is represented in Cl(2,0,0) using e1, e2 for x, y.
#[derive(Debug, Clone)]
pub struct HenonMap {
    /// Parameter a (nonlinearity)
    pub a: f64,
    /// Parameter b (contraction)
    pub b: f64,
}

impl HenonMap {
    /// Create a new Hénon map with given parameters
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Create the classic Hénon map (a=1.4, b=0.3)
    pub fn classic() -> Self {
        Self::new(1.4, 0.3)
    }

    /// Create a conservative Hénon map (|b|=1)
    ///
    /// Area-preserving version with different dynamics.
    pub fn conservative() -> Self {
        Self::new(1.0, 1.0)
    }

    /// Create a weakly chaotic Hénon map (a=1.2, b=0.3)
    pub fn weak_chaos() -> Self {
        Self::new(1.2, 0.3)
    }

    /// Default initial condition on the attractor
    pub fn default_initial_condition(&self) -> Multivector<2, 0, 0> {
        let mut state = Multivector::zero();
        state.set(1, 0.0); // x = 0
        state.set(2, 0.0); // y = 0
        state
    }

    /// Initial condition for exploring the attractor
    pub fn attractor_initial_condition(&self) -> Multivector<2, 0, 0> {
        let mut state = Multivector::zero();
        state.set(1, 0.1); // x = 0.1
        state.set(2, 0.1); // y = 0.1
        state
    }

    /// Get the fixed points of the map
    ///
    /// Fixed points satisfy x = 1 - ax² + y and y = bx.
    /// This gives: x = 1 - ax² + bx, or ax² + (1-b)x - 1 = 0
    /// But the correct form is: x = 1 - ax² + bx, so ax² - (b-1)x - 1 = 0
    /// Actually: ax² + (1-b)x - 1 = 0
    ///
    /// Solutions: x = ((b-1) ± √((1-b)² + 4a)) / (2a)
    pub fn fixed_points(&self) -> Vec<Multivector<2, 0, 0>> {
        let mut fps = Vec::new();

        let discriminant = (1.0 - self.b).powi(2) + 4.0 * self.a;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();

            // First fixed point
            let x1 = ((self.b - 1.0) + sqrt_disc) / (2.0 * self.a);
            let y1 = self.b * x1;
            let mut fp1 = Multivector::zero();
            fp1.set(1, x1);
            fp1.set(2, y1);
            fps.push(fp1);

            // Second fixed point
            let x2 = ((self.b - 1.0) - sqrt_disc) / (2.0 * self.a);
            let y2 = self.b * x2;
            let mut fp2 = Multivector::zero();
            fp2.set(1, x2);
            fp2.set(2, y2);
            fps.push(fp2);
        }

        fps
    }

    /// Jacobian determinant (constant for Hénon map)
    ///
    /// det(J) = -b for all points
    pub fn jacobian_determinant(&self) -> f64 {
        -self.b
    }

    /// Check if the map is dissipative (contracting)
    pub fn is_dissipative(&self) -> bool {
        self.b.abs() < 1.0
    }

    /// Check if the map is area-preserving
    pub fn is_area_preserving(&self) -> bool {
        (self.b.abs() - 1.0).abs() < 1e-10
    }

    /// Approximate largest Lyapunov exponent
    ///
    /// For the classic parameters, this is approximately 0.42
    pub fn approximate_lyapunov(&self) -> f64 {
        if (self.a - 1.4).abs() < 0.01 && (self.b - 0.3).abs() < 0.01 {
            0.42
        } else {
            // Return log|b| as a rough estimate
            -self.b.abs().ln()
        }
    }

    /// Compute the inverse map (if b ≠ 0)
    ///
    /// x_{n-1} = y_n / b
    /// y_{n-1} = x_n - 1 + a(y_n/b)²
    pub fn inverse(&self, state: &Multivector<2, 0, 0>) -> Option<Multivector<2, 0, 0>> {
        if self.b.abs() < 1e-15 {
            return None;
        }

        let x = state.get(1);
        let y = state.get(2);

        let x_prev = y / self.b;
        let y_prev = x - 1.0 + self.a * x_prev * x_prev;

        let mut result = Multivector::zero();
        result.set(1, x_prev);
        result.set(2, y_prev);

        Some(result)
    }

    /// Get x coordinate from state
    pub fn x(state: &Multivector<2, 0, 0>) -> f64 {
        state.get(1)
    }

    /// Get y coordinate from state
    pub fn y(state: &Multivector<2, 0, 0>) -> f64 {
        state.get(2)
    }

    /// Check if a point has escaped (unbounded trajectory)
    ///
    /// For the classic parameters, points with |x| > 10 or |y| > 10
    /// typically escape to infinity.
    pub fn has_escaped(&self, state: &Multivector<2, 0, 0>) -> bool {
        let x = state.get(1);
        let y = state.get(2);
        x.abs() > 10.0 || y.abs() > 10.0
    }
}

impl Default for HenonMap {
    fn default() -> Self {
        Self::classic()
    }
}

impl DiscreteMap<2, 0, 0> for HenonMap {
    fn iterate(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
        let x = state.get(1);
        let y = state.get(2);

        let mut result = Multivector::zero();
        result.set(1, 1.0 - self.a * x * x + y); // x_{n+1} = 1 - ax_n² + y_n
        result.set(2, self.b * x); // y_{n+1} = bx_n

        Ok(result)
    }

    fn jacobian(&self, state: &Multivector<2, 0, 0>) -> Result<Vec<f64>> {
        let x = state.get(1);

        // Jacobian matrix (row-major order):
        // [ -2ax,  1 ]
        // [   b,   0 ]
        Ok(vec![-2.0 * self.a * x, 1.0, self.b, 0.0])
    }

    fn in_domain(&self, state: &Multivector<2, 0, 0>) -> bool {
        !self.has_escaped(state)
    }
}

/// Generalized Hénon map
///
/// A generalization with an additional parameter controlling the power:
/// ```text
/// x_{n+1} = 1 - a|x_n|^p + y_n
/// y_{n+1} = bx_n
/// ```
///
/// Setting p=2 recovers the standard Hénon map.
#[derive(Debug, Clone)]
pub struct GeneralizedHenon {
    /// Parameter a (nonlinearity)
    pub a: f64,
    /// Parameter b (contraction)
    pub b: f64,
    /// Power of the nonlinear term
    pub power: f64,
}

impl GeneralizedHenon {
    /// Create a new generalized Hénon map
    pub fn new(a: f64, b: f64, power: f64) -> Self {
        Self { a, b, power }
    }

    /// Create a cubic Hénon map (p=3)
    pub fn cubic() -> Self {
        Self::new(1.0, 0.3, 3.0)
    }

    /// Default initial condition
    pub fn default_initial_condition(&self) -> Multivector<2, 0, 0> {
        let mut state = Multivector::zero();
        state.set(1, 0.1);
        state.set(2, 0.1);
        state
    }
}

impl DiscreteMap<2, 0, 0> for GeneralizedHenon {
    fn iterate(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
        let x = state.get(1);
        let y = state.get(2);

        let mut result = Multivector::zero();
        result.set(1, 1.0 - self.a * x.abs().powf(self.power) * x.signum() + y);
        result.set(2, self.b * x);

        Ok(result)
    }
}

/// Lozi map
///
/// A piecewise-linear version of the Hénon map:
/// ```text
/// x_{n+1} = 1 - a|x_n| + y_n
/// y_{n+1} = bx_n
/// ```
///
/// The Lozi map is easier to analyze than the Hénon map while
/// exhibiting similar chaotic behavior.
#[derive(Debug, Clone)]
pub struct LoziMap {
    /// Parameter a
    pub a: f64,
    /// Parameter b
    pub b: f64,
}

impl LoziMap {
    /// Create a new Lozi map
    pub fn new(a: f64, b: f64) -> Self {
        Self { a, b }
    }

    /// Create the classic chaotic Lozi map (a=1.7, b=0.5)
    pub fn classic() -> Self {
        Self::new(1.7, 0.5)
    }

    /// Default initial condition
    pub fn default_initial_condition(&self) -> Multivector<2, 0, 0> {
        let mut state = Multivector::zero();
        state.set(1, 0.0);
        state.set(2, 0.0);
        state
    }

    /// Get x coordinate from state
    pub fn x(state: &Multivector<2, 0, 0>) -> f64 {
        state.get(1)
    }

    /// Get y coordinate from state
    pub fn y(state: &Multivector<2, 0, 0>) -> f64 {
        state.get(2)
    }
}

impl Default for LoziMap {
    fn default() -> Self {
        Self::classic()
    }
}

impl DiscreteMap<2, 0, 0> for LoziMap {
    fn iterate(&self, state: &Multivector<2, 0, 0>) -> Result<Multivector<2, 0, 0>> {
        let x = state.get(1);
        let y = state.get(2);

        let mut result = Multivector::zero();
        result.set(1, 1.0 - self.a * x.abs() + y); // x_{n+1} = 1 - a|x_n| + y_n
        result.set(2, self.b * x); // y_{n+1} = bx_n

        Ok(result)
    }

    fn jacobian(&self, state: &Multivector<2, 0, 0>) -> Result<Vec<f64>> {
        let x = state.get(1);

        // The derivative of |x| is sign(x) (undefined at 0, use 0)
        let sign_x = if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        };

        // Jacobian matrix (row-major order):
        // [ -a*sign(x),  1 ]
        // [     b,       0 ]
        Ok(vec![-self.a * sign_x, 1.0, self.b, 0.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hénon map tests

    #[test]
    fn test_henon_creation() {
        let henon = HenonMap::new(1.5, 0.4);
        assert_eq!(henon.a, 1.5);
        assert_eq!(henon.b, 0.4);
    }

    #[test]
    fn test_henon_classic() {
        let henon = HenonMap::classic();
        assert_eq!(henon.a, 1.4);
        assert_eq!(henon.b, 0.3);
    }

    #[test]
    fn test_henon_default() {
        let henon = HenonMap::default();
        assert_eq!(henon.a, 1.4);
        assert_eq!(henon.b, 0.3);
    }

    #[test]
    fn test_henon_iterate() {
        let henon = HenonMap::new(1.0, 0.5);
        let mut state = Multivector::zero();
        state.set(1, 0.5); // x = 0.5
        state.set(2, 0.2); // y = 0.2

        let next = henon.iterate(&state).unwrap();

        // x_{n+1} = 1 - 1*0.5² + 0.2 = 1 - 0.25 + 0.2 = 0.95
        assert!((HenonMap::x(&next) - 0.95).abs() < 1e-10);

        // y_{n+1} = 0.5*0.5 = 0.25
        assert!((HenonMap::y(&next) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_henon_iterate_origin() {
        let henon = HenonMap::classic();
        let origin = Multivector::zero();

        let next = henon.iterate(&origin).unwrap();

        // x_{n+1} = 1 - 0 + 0 = 1
        assert!((HenonMap::x(&next) - 1.0).abs() < 1e-10);

        // y_{n+1} = 0
        assert!(HenonMap::y(&next).abs() < 1e-10);
    }

    #[test]
    fn test_henon_jacobian() {
        let henon = HenonMap::new(1.4, 0.3);
        let mut state = Multivector::zero();
        state.set(1, 1.0);

        let jac = henon.jacobian(&state).unwrap();

        assert_eq!(jac.len(), 4);
        assert!((jac[0] - (-2.8)).abs() < 1e-10); // -2ax = -2*1.4*1
        assert!((jac[1] - 1.0).abs() < 1e-10);
        assert!((jac[2] - 0.3).abs() < 1e-10);
        assert!(jac[3].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_determinant() {
        let henon = HenonMap::classic();
        assert!((henon.jacobian_determinant() - (-0.3)).abs() < 1e-10);
    }

    #[test]
    fn test_is_dissipative() {
        let classic = HenonMap::classic();
        assert!(classic.is_dissipative());

        let conservative = HenonMap::conservative();
        assert!(!conservative.is_dissipative());
    }

    #[test]
    fn test_is_area_preserving() {
        let classic = HenonMap::classic();
        assert!(!classic.is_area_preserving());

        let conservative = HenonMap::conservative();
        assert!(conservative.is_area_preserving());
    }

    #[test]
    fn test_fixed_points() {
        let henon = HenonMap::classic();
        let fps = henon.fixed_points();
        assert_eq!(fps.len(), 2);

        // Verify each fixed point satisfies f(x) = x
        for fp in &fps {
            let next = henon.iterate(fp).unwrap();
            assert!((HenonMap::x(fp) - HenonMap::x(&next)).abs() < 1e-10);
            assert!((HenonMap::y(fp) - HenonMap::y(&next)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inverse() {
        let henon = HenonMap::classic();
        let mut state = Multivector::zero();
        state.set(1, 0.5);
        state.set(2, 0.2);

        // Forward
        let next = henon.iterate(&state).unwrap();

        // Inverse
        let recovered = henon.inverse(&next).unwrap();

        assert!((HenonMap::x(&state) - HenonMap::x(&recovered)).abs() < 1e-10);
        assert!((HenonMap::y(&state) - HenonMap::y(&recovered)).abs() < 1e-10);
    }

    #[test]
    fn test_orbit() {
        let henon = HenonMap::classic();
        let initial = henon.attractor_initial_condition();

        let orbit = henon.orbit(&initial, 100).unwrap();
        assert_eq!(orbit.len(), 101);

        // Most points should stay bounded
        let bounded_count = orbit.iter().filter(|&s| !henon.has_escaped(s)).count();
        assert!(bounded_count > 90);
    }

    // Lozi map tests

    #[test]
    fn test_lozi_creation() {
        let lozi = LoziMap::new(1.8, 0.6);
        assert_eq!(lozi.a, 1.8);
        assert_eq!(lozi.b, 0.6);
    }

    #[test]
    fn test_lozi_classic() {
        let lozi = LoziMap::classic();
        assert_eq!(lozi.a, 1.7);
        assert_eq!(lozi.b, 0.5);
    }

    #[test]
    fn test_lozi_iterate() {
        let lozi = LoziMap::new(1.0, 0.5);
        let mut state = Multivector::zero();
        state.set(1, 0.5);
        state.set(2, 0.2);

        let next = lozi.iterate(&state).unwrap();

        // x_{n+1} = 1 - 1*|0.5| + 0.2 = 1 - 0.5 + 0.2 = 0.7
        assert!((LoziMap::x(&next) - 0.7).abs() < 1e-10);

        // y_{n+1} = 0.5*0.5 = 0.25
        assert!((LoziMap::y(&next) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_lozi_jacobian_positive_x() {
        let lozi = LoziMap::new(1.7, 0.5);
        let mut state = Multivector::zero();
        state.set(1, 1.0);

        let jac = lozi.jacobian(&state).unwrap();

        assert!((jac[0] - (-1.7)).abs() < 1e-10); // -a*sign(x) = -1.7
        assert!((jac[1] - 1.0).abs() < 1e-10);
        assert!((jac[2] - 0.5).abs() < 1e-10);
        assert!(jac[3].abs() < 1e-10);
    }

    #[test]
    fn test_lozi_jacobian_negative_x() {
        let lozi = LoziMap::new(1.7, 0.5);
        let mut state = Multivector::zero();
        state.set(1, -1.0);

        let jac = lozi.jacobian(&state).unwrap();

        assert!((jac[0] - 1.7).abs() < 1e-10); // -a*sign(x) = 1.7
    }

    // Generalized Hénon tests

    #[test]
    fn test_generalized_henon() {
        let gen = GeneralizedHenon::new(1.4, 0.3, 2.0);

        // Should behave like regular Hénon for p=2
        let henon = HenonMap::classic();
        let mut state = Multivector::zero();
        state.set(1, 0.5);
        state.set(2, 0.2);

        let gen_next = gen.iterate(&state).unwrap();
        let henon_next = henon.iterate(&state).unwrap();

        assert!((gen_next.get(1) - henon_next.get(1)).abs() < 1e-10);
        assert!((gen_next.get(2) - henon_next.get(2)).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_henon() {
        let cubic = GeneralizedHenon::cubic();
        assert_eq!(cubic.power, 3.0);
    }
}
