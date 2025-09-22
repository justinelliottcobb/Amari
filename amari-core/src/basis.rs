//! Basis blade utilities and naming conventions

use crate::Multivector;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// Get the grade (number of basis vectors) in a blade
#[inline(always)]
pub fn blade_grade(blade_index: usize) -> usize {
    blade_index.count_ones() as usize
}

/// Get human-readable name for a basis blade
pub fn blade_name(blade_index: usize, dim: usize) -> String {
    if blade_index == 0 {
        return String::from("1"); // Scalar
    }
    
    let mut name = String::new();
    name.push('e');
    
    for i in 0..dim {
        if (blade_index >> i) & 1 == 1 {
            if name.len() > 1 {
                name.push('_');
            }
            name.push_str(&(i + 1).to_string());
        }
    }
    
    name
}

/// Builder for constructing multivectors with named components
pub struct MultivectorBuilder<const P: usize, const Q: usize, const R: usize> {
    coefficients: Vec<f64>,
}

impl<const P: usize, const Q: usize, const R: usize> Default for MultivectorBuilder<P, Q, R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const P: usize, const Q: usize, const R: usize> MultivectorBuilder<P, Q, R> {
    pub fn new() -> Self {
        Self {
            coefficients: vec![0.0; Multivector::<P, Q, R>::BASIS_COUNT],
        }
    }
    
    /// Set scalar component
    pub fn scalar(mut self, value: f64) -> Self {
        self.coefficients[0] = value;
        self
    }
    
    /// Set coefficient for basis vector e_i (1-indexed)
    pub fn e(mut self, i: usize, value: f64) -> Self {
        assert!(i >= 1 && i <= P + Q + R, "Basis vector index out of range");
        self.coefficients[1 << (i - 1)] = value;
        self
    }
    
    /// Set coefficient for bivector e_i ∧ e_j
    pub fn e_wedge(mut self, i: usize, j: usize, value: f64) -> Self {
        assert!(i != j, "Cannot wedge a vector with itself");
        assert!(i >= 1 && i <= P + Q + R, "Basis vector i out of range");
        assert!(j >= 1 && j <= P + Q + R, "Basis vector j out of range");
        
        let index = (1 << (i - 1)) | (1 << (j - 1));
        let sign = if i < j { 1.0 } else { -1.0 };
        self.coefficients[index] = sign * value;
        self
    }
    
    /// Build the multivector
    pub fn build(self) -> Multivector<P, Q, R> {
        Multivector::from_coefficients(self.coefficients)
    }
}

/// Helper to create standard basis vectors
pub struct Basis;

impl Basis {
    /// Create basis vector e1
    pub fn e1<const P: usize, const Q: usize, const R: usize>() -> Multivector<P, Q, R> {
        Multivector::basis_vector(0)
    }
    
    /// Create basis vector e2
    pub fn e2<const P: usize, const Q: usize, const R: usize>() -> Multivector<P, Q, R> {
        Multivector::basis_vector(1)
    }
    
    /// Create basis vector e3
    pub fn e3<const P: usize, const Q: usize, const R: usize>() -> Multivector<P, Q, R> {
        Multivector::basis_vector(2)
    }
    
    /// Create basis bivector e12
    pub fn e12<const P: usize, const Q: usize, const R: usize>() -> Multivector<P, Q, R> {
        let e1 = Self::e1();
        let e2 = Self::e2();
        e1.outer_product(&e2)
    }
    
    /// Create basis bivector e23
    pub fn e23<const P: usize, const Q: usize, const R: usize>() -> Multivector<P, Q, R> {
        let e2 = Self::e2();
        let e3 = Self::e3();
        e2.outer_product(&e3)
    }
    
    /// Create basis bivector e31
    pub fn e31<const P: usize, const Q: usize, const R: usize>() -> Multivector<P, Q, R> {
        let e3 = Self::e3();
        let e1 = Self::e1();
        e3.outer_product(&e1)
    }
    
    /// Create pseudoscalar for 3D (e123)
    pub fn e123<const P: usize, const Q: usize, const R: usize>() -> Multivector<P, Q, R> {
        let e1 = Self::e1();
        let e2 = Self::e2();
        let e3 = Self::e3();
        e1.outer_product(&e2).outer_product(&e3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    type Cl3 = Multivector<3, 0, 0>;
    
    #[test]
    fn test_blade_names() {
        assert_eq!(blade_name(0, 3), "1");
        assert_eq!(blade_name(1, 3), "e1");
        assert_eq!(blade_name(2, 3), "e2");
        assert_eq!(blade_name(3, 3), "e1_2");
        assert_eq!(blade_name(7, 3), "e1_2_3");
    }
    
    #[test]
    fn test_builder() {
        let mv = MultivectorBuilder::<3, 0, 0>::new()
            .scalar(2.0)
            .e(1, 3.0)
            .e(2, 4.0)
            .e_wedge(1, 2, 5.0)
            .build();
        
        assert_eq!(mv.get(0), 2.0); // Scalar
        assert_eq!(mv.get(1), 3.0); // e1
        assert_eq!(mv.get(2), 4.0); // e2
        assert_eq!(mv.get(3), 5.0); // e12
    }
    
    #[test]
    fn test_basis_helpers() {
        let e1: Cl3 = Basis::e1();
        let e2: Cl3 = Basis::e2();
        let e12: Cl3 = Basis::e12();
        
        assert_eq!(e1.get(1), 1.0);
        assert_eq!(e2.get(2), 1.0);
        assert_eq!(e12.get(3), 1.0);
        
        // Verify e1 ∧ e2 = e12
        let computed_e12 = e1.outer_product(&e2);
        assert_eq!(computed_e12.get(3), e12.get(3));
    }
}