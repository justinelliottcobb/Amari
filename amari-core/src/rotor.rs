//! Rotor operations for rotations and reflections

use crate::{Multivector, Bivector, Vector};

/// Rotor in a Clifford algebra (even-grade multivector with unit norm)
pub struct Rotor<const P: usize, const Q: usize, const R: usize> {
    multivector: Multivector<P, Q, R>,
}

impl<const P: usize, const Q: usize, const R: usize> Rotor<P, Q, R> {
    /// Create a rotor from a bivector using exponential map
    ///
    /// For a bivector B representing a plane and angle θ,
    /// the rotor R = exp(-B*θ/2) performs a right-handed rotation by angle θ in that plane.
    ///
    /// The negative sign ensures the rotation follows the right-hand rule convention:
    /// when the thumb points in the direction of the bivector orientation,
    /// the fingers curl in the direction of positive rotation.
    pub fn from_bivector(bivector: &Bivector<P, Q, R>, angle: f64) -> Self {
        let half_angle_bivector = &bivector.mv * (-angle / 2.0);  // Negative for right-handed rotation
        let rotor = half_angle_bivector.exp();

        Self {
            multivector: rotor.normalize().unwrap_or(Multivector::scalar(1.0)),
        }
    }
    
    /// Create a rotor from a raw multivector bivector
    ///
    /// Similar to `from_bivector` but accepts a raw `Multivector` representing a bivector.
    /// Uses the same right-handed rotation convention.
    pub fn from_multivector_bivector(bivector: &Multivector<P, Q, R>, angle: f64) -> Self {
        let half_angle_bivector = bivector * (-angle / 2.0);  // Negative for right-handed rotation
        let rotor = half_angle_bivector.exp();

        Self {
            multivector: rotor.normalize().unwrap_or(Multivector::scalar(1.0)),
        }
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
    
    /// Create identity rotor
    pub fn identity() -> Self {
        Self {
            multivector: Multivector::scalar(1.0),
        }
    }
    
    /// Get the magnitude of the rotor
    pub fn magnitude(&self) -> f64 {
        self.multivector.norm()
    }
    
    /// Get scalar part of the rotor
    pub fn scalar_part(&self) -> f64 {
        self.multivector.scalar_part()
    }
    
    /// Get bivector part of the rotor
    pub fn bivector_part(&self) -> Bivector<P, Q, R> {
        self.multivector.bivector_type()
    }
    
    /// Get coefficients as slice
    pub fn as_slice(&self) -> &[f64] {
        self.multivector.as_slice()
    }
    
    /// Apply rotor to a Vector
    pub fn apply_to_vector(&self, v: &Vector<P, Q, R>) -> Vector<P, Q, R> {
        let result = self.apply(&v.mv);
        Vector::from_multivector(&result)
    }
    
    /// Geometric product with another rotor
    pub fn geometric_product(&self, other: &Self) -> Self {
        self.compose(other)
    }
    
    /// Create rotor from vectors (typed version)
    pub fn from_vectors(a: &Vector<P, Q, R>, b: &Vector<P, Q, R>) -> Option<Self> {
        Self::from_vectors_mv(&a.mv, &b.mv)
    }
    
    /// Create rotor from vectors (multivector version)
    pub fn from_vectors_mv(a: &Multivector<P, Q, R>, b: &Multivector<P, Q, R>) -> Option<Self> {
        let a_norm = a.normalize()?;
        let b_norm = b.normalize()?;
        
        // R = (1 + b*a) / |1 + b*a|
        let ba = b_norm.geometric_product(&a_norm);
        let rotor_unnorm = Multivector::scalar(1.0) + ba;
        let rotor = rotor_unnorm.normalize()?;
        
        Some(Self { multivector: rotor })
    }
    
    /// Create rotor from axis-angle representation
    pub fn from_axis_angle(axis: &Vector<P, Q, R>, angle: f64) -> Self {
        // Convert axis to a bivector perpendicular to it
        // For 3D, if axis is (x,y,z), the bivector plane is orthogonal to it
        let normalized_axis = axis.mv.normalize().unwrap_or(axis.mv.clone());
        
        // Use a simple approach: create bivector from axis cross products
        // This is simplified - a full implementation would be more complex
        let bivector = normalized_axis.grade_projection(2);
        
        Self::from_multivector_bivector(&bivector, angle)
    }
    
    /// Spherical linear interpolation between rotors
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        // Simplified slerp - full implementation would handle edge cases
        let log_other = other.logarithm();
        let scaled_log = &log_other * t;
        let interpolated = scaled_log.exp();
        
        Self {
            multivector: self.multivector.geometric_product(&interpolated),
        }
    }
    
    /// Convert rotor to rotation matrix (3x3 for 3D)
    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        // Extract rotor components
        let s = self.multivector.scalar_part();
        let xy = self.multivector.get(3); // e12
        let xz = self.multivector.get(5); // e13  
        let yz = self.multivector.get(6); // e23
        
        // Convert to rotation matrix using standard formulas
        [
            [1.0 - 2.0*(yz*yz + xz*xz), 2.0*(xy*yz - s*xz), 2.0*(xy*xz + s*yz)],
            [2.0*(xy*yz + s*xz), 1.0 - 2.0*(xy*xy + xz*xz), 2.0*(yz*xz - s*xy)],
            [2.0*(xy*xz - s*yz), 2.0*(yz*xz + s*xy), 1.0 - 2.0*(xy*xy + yz*yz)]
        ]
    }
    
    /// Compute logarithm of rotor
    pub fn logarithm(&self) -> Multivector<P, Q, R> {
        // For a unit rotor R = exp(B), log(R) = B
        // This is a simplified implementation
        let bivector_part = self.multivector.grade_projection(2);
        
        // Scale by appropriate factor based on angle
        let angle = 2.0 * self.scalar_part().acos();
        if angle.abs() > 1e-12 {
            bivector_part * (angle / 2.0)
        } else {
            bivector_part
        }
    }
    
    /// Raise rotor to a power
    pub fn power(&self, exponent: f64) -> Self {
        let log_rotor = self.logarithm();
        let scaled_log = log_rotor * exponent;
        
        Self {
            multivector: scaled_log.exp(),
        }
    }
}

/// Reflection through a hyperplane defined by a unit vector
pub fn reflect<const P: usize, const Q: usize, const R: usize>(
    v: &Multivector<P, Q, R>,
    n: &Multivector<P, Q, R>,
) -> Multivector<P, Q, R> {
    // Reflection formula: n * v * n (for unit normal vector)
    n.geometric_product(v).geometric_product(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rotor_90_degrees() {
        let e1 = Vector::<3, 0, 0>::e1();
        let _e2 = Vector::<3, 0, 0>::e2();
        let e12 = Bivector::<3, 0, 0>::e12();
        
        // Create 90-degree rotation in e1-e2 plane
        let rotor = Rotor::from_bivector(&e12, core::f64::consts::PI / 2.0);
        
        // Apply to e1, should get e2
        let rotated = rotor.apply_to_vector(&e1);
        assert!((rotated.mv.vector_component(1) - 1.0).abs() < 1e-10); // Should be e2
        assert!(rotated.mv.vector_component(0).abs() < 1e-10); // e1 component should be ~0
    }
    
    #[test]
    fn test_rotor_composition() {
        let e12 = Bivector::<3, 0, 0>::e12();
        
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
        let e1 = Vector::<3, 0, 0>::e1();
        let mut v = Multivector::<3, 0, 0>::zero();
        v.set_vector_component(0, 1.0); // e1
        v.set_vector_component(1, 1.0); // e2
        
        // Reflect across e1 (should negate e2 component)
        let reflected = reflect(&v, &e1.mv);
        assert!((reflected.vector_component(0) - 1.0).abs() < 1e-10); // e1 unchanged
        assert!((reflected.vector_component(1) + 1.0).abs() < 1e-10); // e2 negated
    }
}