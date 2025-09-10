//! Rotor operations for rotations and reflections

use crate::Multivector;

/// Rotor in a Clifford algebra (even-grade multivector with unit norm)
pub struct Rotor<const P: usize, const Q: usize, const R: usize> {
    multivector: Multivector<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> Rotor<P, Q, R> {
    /// Create a rotor from a bivector using exponential map
    ///
    /// For a bivector B representing a plane and angle θ,
    /// the rotor R = exp(B*θ/2) performs rotation by angle θ in that plane.
    pub fn from_bivector(bivector: &Multivector<P, Q, R>, angle: f64) -> Self {
        let half_angle_bivector = bivector * (angle / 2.0);
        let rotor = half_angle_bivector.exp();
        
        Self {
            multivector: rotor.normalize().unwrap_or(Multivector::scalar(1.0)),
        }
    }
    
    /// Create a rotor that rotates from vector a to vector b
    pub fn from_vectors(a: &Multivector<P, Q, R>, b: &Multivector<P, Q, R>) -> Option<Self> {
        let a_norm = a.normalize()?;
        let b_norm = b.normalize()?;
        
        // R = (1 + b*a) / |1 + b*a|
        let ba = b_norm.geometric_product(&a_norm);
        let rotor_unnorm = Multivector::scalar(1.0) + ba;
        let rotor = rotor_unnorm.normalize()?;
        
        Some(Self { multivector: rotor })
    }
    
    /// Apply rotor to transform a multivector: R * v * R†
    pub fn apply(&self, v: &Multivector<P, Q, R>) -> Multivector<P, Q, R> {
        let r_dagger = self.multivector.reverse();
        self.multivector.geometric_product(v).geometric_product(&r_dagger)
    }
    
    /// Get the underlying multivector
    pub fn as_multivector(&self) -> &Multivector<P, Q, R> {
        &self.multivector
    }
    
    /// Compose two rotors (multiply them)
    pub fn compose(&self, other: &Self) -> Self {
        let composed = self.multivector.geometric_product(&other.multivector);
        Self {
            multivector: composed.normalize().unwrap_or(Multivector::scalar(1.0)),
        }
    }
    
    /// Get the inverse rotor (reverses the rotation)
    pub fn inverse(&self) -> Self {
        Self {
            multivector: self.multivector.reverse(),
        }
    }
}

/// Reflection through a hyperplane defined by a unit vector
pub fn reflect<const P: usize, const Q: usize, const R: usize>(
    v: &Multivector<P, Q, R>,
    n: &Multivector<P, Q, R>,
) -> Multivector<P, Q, R> {
    // Reflection formula: -n * v * n
    let nvn = n.geometric_product(v).geometric_product(n);
    nvn * -1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{Basis, MultivectorBuilder};
    
    type Cl3 = Multivector<3, 0, 0>;
    
    #[test]
    fn test_rotor_90_degrees() {
        let e1: Cl3 = Basis::e1();
        let e2: Cl3 = Basis::e2();
        let e12: Cl3 = Basis::e12();
        
        // Create 90-degree rotation in e1-e2 plane
        let rotor = Rotor::from_bivector(&e12, core::f64::consts::PI / 2.0);
        
        // Apply to e1, should get e2
        let rotated = rotor.apply(&e1);
        assert!((rotated.get(2) - 1.0).abs() < 1e-10); // Should be e2
        assert!(rotated.get(1).abs() < 1e-10); // e1 component should be ~0
    }
    
    #[test]
    fn test_rotor_composition() {
        let e12: Cl3 = Basis::e12();
        
        // Two 45-degree rotations
        let rotor1 = Rotor::from_bivector(&e12, core::f64::consts::PI / 4.0);
        let rotor2 = Rotor::from_bivector(&e12, core::f64::consts::PI / 4.0);
        
        // Compose them
        let composed = rotor1.compose(&rotor2);
        
        // Should equal a single 90-degree rotation
        let rotor90 = Rotor::from_bivector(&e12, core::f64::consts::PI / 2.0);
        
        let diff = composed.as_multivector() - rotor90.as_multivector();
        assert!(diff.norm() < 1e-10);
    }
    
    #[test]
    fn test_reflection() {
        let e1: Cl3 = Basis::e1();
        let v = MultivectorBuilder::<3, 0, 0>::new()
            .e(1, 1.0)
            .e(2, 1.0)
            .build();
        
        // Reflect across e1 (should negate e2 component)
        let reflected = reflect(&v, &e1);
        assert!((reflected.get(1) - 1.0).abs() < 1e-10); // e1 unchanged
        assert!((reflected.get(2) + 1.0).abs() < 1e-10); // e2 negated
    }
}