//! High-performance Geometric Algebra/Clifford Algebra library
//!
//! This crate provides the core implementation of Clifford algebras with arbitrary signatures,
//! supporting the geometric product and related operations fundamental to geometric algebra.
//!
//! # Basis Blade Indexing
//!
//! Basis blades are indexed using binary representation where bit i indicates whether
//! basis vector e_i is present in the blade. For example:
//! - 0b001 (1) = e1
//! - 0b010 (2) = e2
//! - 0b011 (3) = e1 ∧ e2 (bivector)
//! - 0b111 (7) = e1 ∧ e2 ∧ e3 (trivector)

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::boxed::Box;
use alloc::vec::{self, Vec};
use core::ops::{Add, Sub, Mul, Neg};
use num_traits::Zero;

pub mod cayley;
pub mod basis;
pub mod rotor;

use cayley::CayleyTable;

/// A multivector in a Clifford algebra Cl(P,Q,R)
/// 
/// # Type Parameters
/// - `P`: Number of basis vectors that square to +1
/// - `Q`: Number of basis vectors that square to -1  
/// - `R`: Number of basis vectors that square to 0
#[derive(Debug, Clone, PartialEq)]
#[repr(C, align(64))] // Cache alignment for performance
pub struct Multivector<const P: usize, const Q: usize, const R: usize> {
    /// Coefficients for each basis blade, indexed by binary representation
    coefficients: Box<[f64]>,
}

impl<const P: usize, const Q: usize, const R: usize> Multivector<P, Q, R> {
    /// Total dimension of the algebra's vector space
    pub const DIM: usize = P + Q + R;
    
    /// Total number of basis blades (2^DIM)
    pub const BASIS_COUNT: usize = 1 << Self::DIM;
    
    /// Create a new zero multivector
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            coefficients: vec::from_elem(0.0, Self::BASIS_COUNT).into_boxed_slice(),
        }
    }
    
    /// Create a scalar multivector
    #[inline(always)]
    pub fn scalar(value: f64) -> Self {
        let mut mv = Self::zero();
        mv.coefficients[0] = value;
        mv
    }
    
    /// Create a basis vector e_i (i starts from 0)
    #[inline(always)]
    pub fn basis_vector(i: usize) -> Self {
        assert!(i < Self::DIM, "Basis vector index out of range");
        let mut mv = Self::zero();
        mv.coefficients[1 << i] = 1.0;
        mv
    }
    
    /// Create a multivector from coefficients
    #[inline(always)]
    pub fn from_coefficients(coefficients: Vec<f64>) -> Self {
        assert_eq!(coefficients.len(), Self::BASIS_COUNT, 
                   "Coefficient array must have {} elements", Self::BASIS_COUNT);
        Self {
            coefficients: coefficients.into_boxed_slice(),
        }
    }
    
    /// Get coefficient for a specific basis blade (by index)
    #[inline(always)]
    pub fn get(&self, index: usize) -> f64 {
        self.coefficients.get(index).copied().unwrap_or(0.0)
    }
    
    /// Set coefficient for a specific basis blade
    #[inline(always)]
    pub fn set(&mut self, index: usize, value: f64) {
        if index < self.coefficients.len() {
            self.coefficients[index] = value;
        }
    }
    
    /// Get the scalar part (grade 0)
    #[inline(always)]
    pub fn scalar_part(&self) -> f64 {
        self.coefficients[0]
    }
    
    /// Geometric product with another multivector
    ///
    /// The geometric product is the fundamental operation in geometric algebra,
    /// combining both the inner and outer products.
    pub fn geometric_product(&self, rhs: &Self) -> Self {
        let table = CayleyTable::<P, Q, R>::get();
        let mut result = Self::zero();
        
        for i in 0..Self::BASIS_COUNT {
            if self.coefficients[i].abs() < 1e-14 { continue; }
            
            for j in 0..Self::BASIS_COUNT {
                if rhs.coefficients[j].abs() < 1e-14 { continue; }
                
                let (sign, index) = table.get_product(i, j);
                result.coefficients[index] += sign * self.coefficients[i] * rhs.coefficients[j];
            }
        }
        
        result
    }
    
    /// Inner product (grade-lowering, dot product for vectors)
    pub fn inner_product(&self, rhs: &Self) -> Self {
        let self_grades = self.grade_decomposition();
        let rhs_grades = rhs.grade_decomposition();
        let mut result = Self::zero();
        
        // Inner product selects terms where grade(result) = |grade(a) - grade(b)|
        for (grade_a, mv_a) in self_grades.iter().enumerate() {
            for (grade_b, mv_b) in rhs_grades.iter().enumerate() {
                if !mv_a.is_zero() && !mv_b.is_zero() {
                    let target_grade = grade_a.abs_diff(grade_b);
                    let product = mv_a.geometric_product(mv_b);
                    let projected = product.grade_projection(target_grade);
                    result = result + projected;
                }
            }
        }
        
        result
    }
    
    /// Outer product (wedge product, grade-raising)
    pub fn outer_product(&self, rhs: &Self) -> Self {
        let self_grades = self.grade_decomposition();
        let rhs_grades = rhs.grade_decomposition();
        let mut result = Self::zero();
        
        // Outer product selects terms where grade(result) = grade(a) + grade(b)
        for (grade_a, mv_a) in self_grades.iter().enumerate() {
            for (grade_b, mv_b) in rhs_grades.iter().enumerate() {
                if !mv_a.is_zero() && !mv_b.is_zero() {
                    let target_grade = grade_a + grade_b;
                    if target_grade <= Self::DIM {
                        let product = mv_a.geometric_product(mv_b);
                        let projected = product.grade_projection(target_grade);
                        result = result + projected;
                    }
                }
            }
        }
        
        result
    }
    
    /// Scalar product (grade-0 part of geometric product)
    pub fn scalar_product(&self, rhs: &Self) -> f64 {
        self.geometric_product(rhs).scalar_part()
    }
    
    /// Reverse operation (reverses order of basis vectors in each blade)
    pub fn reverse(&self) -> Self {
        let mut result = Self::zero();
        
        for i in 0..Self::BASIS_COUNT {
            let grade = i.count_ones() as usize;
            // Reverse introduces sign: (-1)^(grade*(grade-1)/2)
            let sign = if (grade * (grade - 1) / 2) % 2 == 0 { 1.0 } else { -1.0 };
            result.coefficients[i] = sign * self.coefficients[i];
        }
        
        result
    }
    
    /// Grade projection - extract components of a specific grade
    pub fn grade_projection(&self, grade: usize) -> Self {
        let mut result = Self::zero();
        
        for i in 0..Self::BASIS_COUNT {
            if i.count_ones() as usize == grade {
                result.coefficients[i] = self.coefficients[i];
            }
        }
        
        result
    }
    
    /// Decompose into grade components
    fn grade_decomposition(&self) -> Vec<Self> {
        let mut grades = Vec::with_capacity(Self::DIM + 1);
        for _ in 0..=Self::DIM {
            grades.push(Self::zero());
        }
        
        for i in 0..Self::BASIS_COUNT {
            let grade = i.count_ones() as usize;
            grades[grade].coefficients[i] = self.coefficients[i];
        }
        
        grades
    }
    
    /// Check if this is (approximately) zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|&c| c.abs() < 1e-14)
    }
    
    /// Compute the norm squared (scalar product with reverse)
    pub fn norm_squared(&self) -> f64 {
        self.scalar_product(&self.reverse())
    }
    
    /// Compute the norm
    pub fn norm(&self) -> f64 {
        self.norm_squared().abs().sqrt()
    }
    
    /// Normalize this multivector
    pub fn normalize(&self) -> Option<Self> {
        let norm = self.norm();
        if norm > 1e-14 {
            Some(self * (1.0 / norm))
        } else {
            None
        }
    }
    
    /// Compute multiplicative inverse if it exists
    pub fn inverse(&self) -> Option<Self> {
        let rev = self.reverse();
        let norm_sq = self.scalar_product(&rev);
        
        if norm_sq.abs() > 1e-14 {
            Some(rev * (1.0 / norm_sq))
        } else {
            None
        }
    }
    
    /// Exponential map for bivectors (creates rotors)
    ///
    /// For a bivector B, exp(B) creates a rotor that performs rotation
    /// in the plane defined by B.
    pub fn exp(&self) -> Self {
        // Check if this is a bivector (grade 2)
        let grade2 = self.grade_projection(2);
        if (self - &grade2).norm() > 1e-10 {
            // For general multivectors, use series expansion
            return self.exp_series();
        }
        
        // For pure bivectors, use closed form
        let b_squared = grade2.geometric_product(&grade2).scalar_part();
        
        if b_squared > -1e-14 {
            // Hyperbolic case: exp(B) = cosh(|B|) + sinh(|B|) * B/|B|
            let norm = b_squared.abs().sqrt();
            if norm < 1e-14 {
                return Self::scalar(1.0);
            }
            let cosh_norm = norm.cosh();
            let sinh_norm = norm.sinh();
            Self::scalar(cosh_norm) + grade2 * (sinh_norm / norm)
        } else {
            // Circular case: exp(B) = cos(|B|) + sin(|B|) * B/|B|
            let norm = (-b_squared).sqrt();
            let cos_norm = norm.cos();
            let sin_norm = norm.sin();
            Self::scalar(cos_norm) + grade2 * (sin_norm / norm)
        }
    }
    
    /// Series expansion for exponential (fallback for general multivectors)
    fn exp_series(&self) -> Self {
        let mut result = Self::scalar(1.0);
        let mut term = Self::scalar(1.0);
        
        for n in 1..20 {
            term = term.geometric_product(self) * (1.0 / n as f64);
            let old_result = result.clone();
            result = result + term.clone();
            
            // Check convergence
            if (result.clone() - old_result).norm() < 1e-14 {
                break;
            }
        }
        
        result
    }
}

// Operator implementations
impl<const P: usize, const Q: usize, const R: usize> Add for Multivector<P, Q, R> {
    type Output = Self;
    
    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..Self::BASIS_COUNT {
            self.coefficients[i] += rhs.coefficients[i];
        }
        self
    }
}

impl<const P: usize, const Q: usize, const R: usize> Add for &Multivector<P, Q, R> {
    type Output = Multivector<P, Q, R>;
    
    #[inline(always)]
    fn add(self, rhs: Self) -> Multivector<P, Q, R> {
        let mut result = self.clone();
        for i in 0..Multivector::<P, Q, R>::BASIS_COUNT {
            result.coefficients[i] += rhs.coefficients[i];
        }
        result
    }
}

impl<const P: usize, const Q: usize, const R: usize> Sub for Multivector<P, Q, R> {
    type Output = Self;
    
    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..Self::BASIS_COUNT {
            self.coefficients[i] -= rhs.coefficients[i];
        }
        self
    }
}

impl<const P: usize, const Q: usize, const R: usize> Sub for &Multivector<P, Q, R> {
    type Output = Multivector<P, Q, R>;
    
    #[inline(always)]
    fn sub(self, rhs: Self) -> Multivector<P, Q, R> {
        let mut result = self.clone();
        for i in 0..Multivector::<P, Q, R>::BASIS_COUNT {
            result.coefficients[i] -= rhs.coefficients[i];
        }
        result
    }
}

impl<const P: usize, const Q: usize, const R: usize> Mul<f64> for Multivector<P, Q, R> {
    type Output = Self;
    
    #[inline(always)]
    fn mul(mut self, scalar: f64) -> Self {
        for i in 0..Self::BASIS_COUNT {
            self.coefficients[i] *= scalar;
        }
        self
    }
}

impl<const P: usize, const Q: usize, const R: usize> Mul<f64> for &Multivector<P, Q, R> {
    type Output = Multivector<P, Q, R>;
    
    #[inline(always)]
    fn mul(self, scalar: f64) -> Multivector<P, Q, R> {
        let mut result = self.clone();
        for i in 0..Multivector::<P, Q, R>::BASIS_COUNT {
            result.coefficients[i] *= scalar;
        }
        result
    }
}

impl<const P: usize, const Q: usize, const R: usize> Neg for Multivector<P, Q, R> {
    type Output = Self;
    
    #[inline(always)]
    fn neg(mut self) -> Self {
        for i in 0..Self::BASIS_COUNT {
            self.coefficients[i] = -self.coefficients[i];
        }
        self
    }
}

impl<const P: usize, const Q: usize, const R: usize> Zero for Multivector<P, Q, R> {
    fn zero() -> Self {
        Self::zero()
    }
    
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    type Cl3 = Multivector<3, 0, 0>; // 3D Euclidean space
    
    #[test]
    fn test_basis_vectors() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        
        // e1 * e1 = 1
        let e1_squared = e1.geometric_product(&e1);
        assert_eq!(e1_squared.scalar_part(), 1.0);
        
        // e1 * e2 = -e2 * e1 (anticommute)
        let e12 = e1.geometric_product(&e2);
        let e21 = e2.geometric_product(&e1);
        assert_eq!(e12.coefficients[3], -e21.coefficients[3]); // e1∧e2 component
    }
    
    #[test]
    fn test_wedge_product() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        
        let e12 = e1.outer_product(&e2);
        assert!(e12.get(3).abs() - 1.0 < 1e-10); // e1∧e2 has index 0b11 = 3
        
        // e1 ∧ e1 = 0
        let e11 = e1.outer_product(&e1);
        assert!(e11.is_zero());
    }
    
    #[test]
    fn test_rotor_from_bivector() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let e12 = e1.outer_product(&e2);
        
        // Create 90 degree rotation in e1-e2 plane
        let angle = core::f64::consts::PI / 4.0; // Half angle for rotor
        let bivector = e12 * angle;
        let rotor = bivector.exp();
        
        // Check that rotor has unit norm
        assert!((rotor.norm() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_algebraic_identities() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let e3 = Cl3::basis_vector(2);
        
        // Associativity: (ab)c = a(bc)
        let a = e1.clone() + e2.clone() * 2.0;
        let b = e2.clone() + e3.clone() * 3.0;
        let c = e3.clone() + e1.clone() * 0.5;
        
        let left = a.geometric_product(&b).geometric_product(&c);
        let right = a.geometric_product(&b.geometric_product(&c));
        assert_relative_eq!(left.norm(), right.norm(), epsilon = 1e-12);
        
        // Distributivity: a(b + c) = ab + ac
        let ab_plus_ac = a.geometric_product(&b) + a.geometric_product(&c);
        let a_times_b_plus_c = a.geometric_product(&(b.clone() + c.clone()));
        assert!((ab_plus_ac - a_times_b_plus_c).norm() < 1e-12);
        
        // Reverse property: (ab)† = b†a†
        let ab_reverse = a.geometric_product(&b).reverse();
        let b_reverse_a_reverse = b.reverse().geometric_product(&a.reverse());
        assert!((ab_reverse - b_reverse_a_reverse).norm() < 1e-12);
    }
    
    #[test]
    fn test_metric_signature() {
        // Test different signatures
        type Spacetime = Multivector<1, 3, 0>; // Minkowski signature
        
        let e0 = Spacetime::basis_vector(0); // timelike
        let e1 = Spacetime::basis_vector(1); // spacelike
        
        // e0^2 = +1, e1^2 = -1
        assert_eq!(e0.geometric_product(&e0).scalar_part(), 1.0);
        assert_eq!(e1.geometric_product(&e1).scalar_part(), -1.0);
    }
    
    #[test]
    fn test_grade_operations() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let scalar = Cl3::scalar(2.0);
        
        let mv = scalar + e1.clone() * 3.0 + e2.clone() * 4.0 + e1.outer_product(&e2) * 5.0;
        
        // Test grade projections
        let grade0 = mv.grade_projection(0);
        let grade1 = mv.grade_projection(1);
        let grade2 = mv.grade_projection(2);
        
        assert_eq!(grade0.scalar_part(), 2.0);
        assert_eq!(grade1.get(1), 3.0); // e1 component
        assert_eq!(grade1.get(2), 4.0); // e2 component
        assert_eq!(grade2.get(3), 5.0); // e12 component
        
        // Sum of grade projections should equal original
        let reconstructed = grade0 + grade1 + grade2;
        assert!((mv - reconstructed).norm() < 1e-12);
    }
    
    #[test]
    fn test_inner_and_outer_products() {
        let e1 = Cl3::basis_vector(0);
        let e2 = Cl3::basis_vector(1);
        let e3 = Cl3::basis_vector(2);
        
        // Inner product of orthogonal vectors is zero
        assert!(e1.inner_product(&e2).norm() < 1e-12);
        
        // Inner product of parallel vectors
        let v1 = e1.clone() + e2.clone();
        let v2 = e1.clone() * 2.0 + e2.clone() * 2.0;
        let inner = v1.inner_product(&v2);
        assert_relative_eq!(inner.scalar_part(), 4.0, epsilon = 1e-12);
        
        // Outer product creates higher grade
        let bivector = e1.outer_product(&e2);
        let trivector = bivector.outer_product(&e3);
        assert_eq!(trivector.get(7), 1.0); // e123 component
    }
}