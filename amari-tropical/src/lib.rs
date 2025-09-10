//! Tropical (max-plus) algebra for efficient LLM operations
//!
//! Tropical algebra replaces traditional (+, ×) with (max, +), which converts
//! expensive softmax operations into simple max operations. This is particularly
//! useful for finding most likely sequences and optimization in neural networks.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg};
use num_traits::{Float, Zero, One};

pub mod polytope;
pub mod viterbi;

/// A number in the tropical (max-plus) semiring
/// 
/// Tropical addition is max, tropical multiplication is addition
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct TropicalNumber<T: Float>(pub T);

impl<T: Float> TropicalNumber<T> {
    /// Additive identity (negative infinity)
    pub const ZERO: Self = Self(T::neg_infinity());
    
    /// Multiplicative identity (zero in regular arithmetic)
    pub const ONE: Self = Self(T::zero());
    
    /// Create from regular number
    pub fn new(value: T) -> Self {
        Self(value)
    }
    
    /// Get the underlying value
    pub fn value(&self) -> T {
        self.0
    }
    
    /// Check if this is the zero element (negative infinity)
    pub fn is_zero(&self) -> bool {
        self.0.is_infinite() && self.0.is_sign_negative()
    }
    
    /// Check if this is the one element (zero)
    pub fn is_one(&self) -> bool {
        self.0.is_zero()
    }
    
    /// Tropical addition (max operation)
    pub fn tropical_add(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }
    
    /// Tropical multiplication (addition)
    pub fn tropical_mul(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
    
    /// Tropical power (scalar multiplication)
    pub fn tropical_pow(self, n: T) -> Self {
        Self(self.0 * n)
    }
    
    /// Convert from log-probability to tropical number
    pub fn from_log_prob(log_p: T) -> Self {
        Self(log_p)
    }
    
    /// Convert tropical number back to probability (via exp)
    pub fn to_prob(self) -> T {
        if self.is_zero() {
            T::zero()
        } else {
            self.0.exp()
        }
    }
}

impl<T: Float> Add for TropicalNumber<T> {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        self.tropical_add(other)
    }
}

impl<T: Float> Mul for TropicalNumber<T> {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        self.tropical_mul(other)
    }
}

impl<T: Float> Neg for TropicalNumber<T> {
    type Output = Self;
    
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<T: Float> Zero for TropicalNumber<T> {
    fn zero() -> Self {
        Self::ZERO
    }
    
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
}

impl<T: Float> One for TropicalNumber<T> {
    fn one() -> Self {
        Self::ONE
    }
}

/// Tropical multivector for geometric operations in tropical algebra
#[derive(Clone, Debug)]
pub struct TropicalMultivector<T: Float, const DIM: usize> {
    coefficients: Vec<TropicalNumber<T>>,
}

impl<T: Float, const DIM: usize> TropicalMultivector<T, DIM> {
    const BASIS_COUNT: usize = 1 << DIM;
    
    /// Create zero tropical multivector
    pub fn zero() -> Self {
        Self {
            coefficients: {
                let mut coeffs = Vec::with_capacity(Self::BASIS_COUNT);
                for _ in 0..Self::BASIS_COUNT {
                    coeffs.push(TropicalNumber::zero());
                }
                coeffs
            },
        }
    }
    
    /// Create from regular coefficients
    pub fn from_coefficients(coeffs: Vec<T>) -> Self {
        assert_eq!(coeffs.len(), Self::BASIS_COUNT);
        Self {
            coefficients: coeffs.into_iter().map(TropicalNumber::new).collect(),
        }
    }
    
    /// Create from log probabilities
    pub fn from_log_probs(log_probs: &[T]) -> Self {
        let mut coeffs = Vec::with_capacity(Self::BASIS_COUNT);
        for i in 0..Self::BASIS_COUNT {
            if i < log_probs.len() {
                coeffs.push(TropicalNumber::from_log_prob(log_probs[i]));
            } else {
                coeffs.push(TropicalNumber::zero());
            }
        }
        Self { coefficients: coeffs }
    }
    
    /// Get coefficient at index
    pub fn get(&self, index: usize) -> TropicalNumber<T> {
        self.coefficients.get(index).copied().unwrap_or(TropicalNumber::zero())
    }
    
    /// Set coefficient at index
    pub fn set(&mut self, index: usize, value: TropicalNumber<T>) {
        if index < self.coefficients.len() {
            self.coefficients[index] = value;
        }
    }
    
    /// Tropical geometric product
    pub fn geometric_product(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        
        // Simplified tropical geometric product
        // In tropical algebra, we use max for addition and + for multiplication
        for i in 0..Self::BASIS_COUNT {
            for j in 0..Self::BASIS_COUNT {
                let index = i ^ j; // XOR for basis blade combination
                let product = self.coefficients[i] * other.coefficients[j];
                result.coefficients[index] = result.coefficients[index] + product;
            }
        }
        
        result
    }
    
    /// Find the maximum element (tropical sum)
    pub fn max_element(&self) -> TropicalNumber<T> {
        self.coefficients.iter().copied().fold(TropicalNumber::zero(), |acc, x| acc + x)
    }
    
    /// Get indices of non-zero (non-negative-infinity) elements
    pub fn support(&self) -> Vec<usize> {
        self.coefficients
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| !coeff.is_zero())
            .map(|(i, _)| i)
            .collect()
    }
    
    /// Tropical norm (maximum absolute value)
    pub fn tropical_norm(&self) -> TropicalNumber<T> {
        self.coefficients
            .iter()
            .copied()
            .map(|x| TropicalNumber::new(x.value().abs()))
            .fold(TropicalNumber::zero(), |acc, x| acc + x)
    }
}

/// Tropical matrix operations for attention mechanisms
#[derive(Clone, Debug)]
pub struct TropicalMatrix<T: Float> {
    data: Vec<Vec<TropicalNumber<T>>>,
    rows: usize,
    cols: usize,
}

impl<T: Float> TropicalMatrix<T> {
    /// Create new tropical matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        let mut data = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for _ in 0..cols {
                row.push(TropicalNumber::zero());
            }
            data.push(row);
        }
        
        Self { data, rows, cols }
    }
    
    /// Create from log-probability matrix (common in attention)
    pub fn from_log_probs(log_probs: &[Vec<T>]) -> Self {
        let rows = log_probs.len();
        let cols = log_probs[0].len();
        let mut matrix = Self::new(rows, cols);
        
        for (i, row) in log_probs.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                matrix.data[i][j] = TropicalNumber::from_log_prob(value);
            }
        }
        
        matrix
    }
    
    /// Tropical matrix multiplication
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        
        let mut result = Self::new(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = TropicalNumber::zero();
                for k in 0..self.cols {
                    // Tropical: (A*B)[i,j] = max_k(A[i,k] + B[k,j])
                    sum = sum + (self.data[i][k] * other.data[k][j]);
                }
                result.data[i][j] = sum;
            }
        }
        
        result
    }
    
    /// Tropical determinant (maximum over all permutations)
    pub fn determinant(&self) -> TropicalNumber<T> {
        assert_eq!(self.rows, self.cols);
        
        if self.rows == 1 {
            return self.data[0][0];
        }
        
        if self.rows == 2 {
            return self.data[0][0] * self.data[1][1] + self.data[0][1] * self.data[1][0];
        }
        
        // For larger matrices, use recursive expansion
        let mut det = TropicalNumber::zero();
        for j in 0..self.cols {
            let minor = self.minor(0, j);
            let cofactor = self.data[0][j] * minor.determinant();
            det = det + cofactor;
        }
        
        det
    }
    
    /// Extract minor matrix
    fn minor(&self, row: usize, col: usize) -> Self {
        let mut minor_data = Vec::new();
        
        for i in 0..self.rows {
            if i == row { continue; }
            let mut minor_row = Vec::new();
            for j in 0..self.cols {
                if j == col { continue; }
                minor_row.push(self.data[i][j]);
            }
            minor_data.push(minor_row);
        }
        
        Self {
            data: minor_data,
            rows: self.rows - 1,
            cols: self.cols - 1,
        }
    }
    
    /// Convert to attention scores (softmax → max operation)
    pub fn to_attention_scores(&self) -> Vec<Vec<T>> {
        let mut scores = Vec::with_capacity(self.rows);
        
        for row in &self.data {
            let max_val = row.iter().copied().fold(TropicalNumber::zero(), |acc, x| acc + x);
            let row_scores: Vec<T> = row.iter().map(|&val| {
                if max_val.is_zero() {
                    T::zero()
                } else if val == max_val {
                    T::one()
                } else {
                    T::zero()
                }
            }).collect();
            scores.push(row_scores);
        }
        
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_tropical_number_operations() {
        let a = TropicalNumber::new(2.0);
        let b = TropicalNumber::new(3.0);
        
        // Tropical addition is max
        assert_eq!(a + b, TropicalNumber::new(3.0));
        assert_eq!(b + a, TropicalNumber::new(3.0));
        
        // Tropical multiplication is addition
        assert_eq!(a * b, TropicalNumber::new(5.0));
        assert_eq!(b * a, TropicalNumber::new(5.0));
        
        // Identity elements
        assert_eq!(a + TropicalNumber::ZERO, a);
        assert_eq!(a * TropicalNumber::ONE, a);
    }
    
    #[test]
    fn test_tropical_multivector() {
        let mv1 = TropicalMultivector::<f64, 2>::from_coefficients(vec![1.0, 2.0, 3.0, 4.0]);
        let mv2 = TropicalMultivector::<f64, 2>::from_coefficients(vec![0.5, 1.5, 2.5, 3.5]);
        
        let product = mv1.geometric_product(&mv2);
        
        // Verify the result has correct structure
        assert!(!product.max_element().is_zero());
        
        // Check support (non-zero elements)
        let support = mv1.support();
        assert!(support.len() > 0);
    }
    
    #[test]
    fn test_tropical_matrix() {
        let log_probs = vec![
            vec![0.0, -1.0, -2.0],
            vec![-1.0, 0.0, -1.0],
            vec![-2.0, -1.0, 0.0],
        ];
        
        let matrix = TropicalMatrix::from_log_probs(&log_probs);
        let det = matrix.determinant();
        
        // Determinant should not be zero (negative infinity)
        assert!(!det.is_zero());
        
        // Test attention scores conversion
        let scores = matrix.to_attention_scores();
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[0].len(), 3);
    }
    
    #[test]
    fn test_viterbi_equivalence() {
        // Tropical multiplication chain should equal Viterbi path probability
        let transitions = vec![
            TropicalNumber::from_log_prob(-0.5),
            TropicalNumber::from_log_prob(-1.0),
            TropicalNumber::from_log_prob(-0.3),
        ];
        
        let path_prob = transitions.into_iter().fold(TropicalNumber::ONE, |acc, x| acc * x);
        
        // Should equal sum of log probabilities
        assert_relative_eq!(path_prob.value(), -1.8, epsilon = 1e-10);
    }
}