//! Tropical (max-plus) algebra for efficient LLM operations
//!
//! Tropical algebra replaces traditional (+, ×) with (max, +), which converts
//! expensive softmax operations into simple max operations. This is particularly
//! useful for finding most likely sequences and optimization in neural networks.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg};
use num_traits::Float;

pub mod polytope;
pub mod viterbi;

// Phantom types and formal verification modules
#[cfg(feature = "formal-verification")]
pub mod verified;

#[cfg(feature = "formal-verification")]
pub mod verified_contracts;

// Re-export phantom types for tropical algebra
pub use amari_core::verified::VerifiedMultivector as CoreVerifiedMultivector;

/// A number in the tropical (max-plus) semiring
///
/// Tropical addition is max, tropical multiplication is addition
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct TropicalNumber<T: Float>(pub T);

impl<T: Float> TropicalNumber<T> {
    /// Additive identity (negative infinity)
    pub fn neg_infinity() -> Self {
        Self(T::neg_infinity())
    }

    /// Multiplicative identity (zero in regular arithmetic)
    pub fn zero() -> Self {
        Self(T::zero())
    }

    /// Additive identity (same as neg_infinity for tropical)
    pub fn tropical_zero() -> Self {
        Self::neg_infinity()
    }

    /// Multiplicative identity (same as zero for tropical)
    pub fn tropical_one() -> Self {
        Self::zero()
    }

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

    /// Check if this is infinite (either positive or negative)
    pub fn is_infinity(&self) -> bool {
        self.0.is_infinite()
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

// Note: We don't implement Zero and One traits for TropicalNumber
// because tropical zero is negative infinity and tropical one is zero,
// which conflicts with the standard definitions

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

// Convenient constants for f64
impl TropicalNumber<f64> {
    pub const ZERO: Self = Self(f64::NEG_INFINITY);
    pub const ONE: Self = Self(0.0);
}

// Removed duplicate Zero/One implementations

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
        Self {
            coefficients: coeffs,
        }
    }

    /// Get coefficient at index
    pub fn get(&self, index: usize) -> TropicalNumber<T> {
        self.coefficients
            .get(index)
            .copied()
            .unwrap_or(TropicalNumber::tropical_zero())
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
        self.coefficients
            .iter()
            .copied()
            .fold(TropicalNumber::zero(), |acc, x| acc + x)
    }

    /// Check if the multivector is zero
    pub fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|c| c.is_zero())
    }

    /// Add two tropical multivectors
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..Self::BASIS_COUNT {
            result.coefficients[i] = self.coefficients[i] + other.coefficients[i];
        }
        result
    }

    /// Tropical addition (alias for add)
    pub fn tropical_add(&self, other: &Self) -> Self {
        self.add(other)
    }

    /// Tropical multiplication (element-wise multiplication of coefficients)
    pub fn tropical_mul(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..Self::BASIS_COUNT {
            result.coefficients[i] = self.coefficients[i].tropical_mul(other.coefficients[i]);
        }
        result
    }

    /// Tropical scaling (multiply all coefficients by scalar)
    pub fn tropical_scale(&self, scalar: T) -> Self {
        let mut result = Self::zero();
        let tropical_scalar = TropicalNumber::new(scalar);
        for i in 0..Self::BASIS_COUNT {
            result.coefficients[i] = self.coefficients[i].tropical_mul(tropical_scalar);
        }
        result
    }

    /// Scale by a regular scalar
    pub fn scale(&self, factor: T) -> Self {
        let mut result = Self::zero();
        for i in 0..Self::BASIS_COUNT {
            result.coefficients[i] = TropicalNumber(self.coefficients[i].0 + factor);
        }
        result
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

    /// Create from logits (log probabilities)
    pub fn from_logits(logits: &[T]) -> Self {
        Self::from_log_probs(logits)
    }

    /// Viterbi algorithm using tropical algebra
    /// Returns the most likely path through states
    pub fn viterbi(
        transitions: &TropicalMatrix<T>,
        emissions: &TropicalMatrix<T>,
        initial_probs: &[T],
        sequence_length: usize,
    ) -> Vec<usize> {
        let num_states = initial_probs.len();
        let mut path = Vec::with_capacity(sequence_length);

        // Dynamic programming tables
        let mut current_probs = Vec::with_capacity(num_states);
        let mut prev_states = Vec::with_capacity(sequence_length);
        for _ in 0..sequence_length {
            prev_states.push(vec![0; num_states]);
        }

        // Initialize with first observation
        #[allow(clippy::needless_range_loop)]
        for i in 0..num_states {
            let init_prob = TropicalNumber::from_log_prob(initial_probs[i]);
            let emit_prob = emissions.data[i][0]; // First observation
            current_probs.push(init_prob * emit_prob);
        }

        // Forward pass through sequence
        #[allow(clippy::needless_range_loop)]
        for t in 1..sequence_length {
            let mut new_probs = Vec::with_capacity(num_states);

            #[allow(clippy::needless_range_loop)]
            for curr_state in 0..num_states {
                let mut best_prob = TropicalNumber::zero(); // -infinity
                let mut best_prev = 0;

                for prev_state in 0..num_states {
                    let transition_prob = transitions.data[prev_state][curr_state];
                    let emission_prob = emissions.data[curr_state][t.min(emissions.cols - 1)];
                    let total_prob = current_probs[prev_state] * transition_prob * emission_prob;

                    if total_prob.value() > best_prob.value() {
                        best_prob = total_prob;
                        best_prev = prev_state;
                    }
                }

                new_probs.push(best_prob);
                prev_states[t][curr_state] = best_prev;
            }

            current_probs = new_probs;
        }

        // Find best final state
        let mut best_final_state = 0;
        let mut best_final_prob = current_probs[0];
        for (i, &prob) in current_probs.iter().enumerate().skip(1) {
            if prob.value() > best_final_prob.value() {
                best_final_prob = prob;
                best_final_state = i;
            }
        }

        // Backtrack to reconstruct path
        path.push(best_final_state);
        let mut current_state = best_final_state;

        for t in (1..sequence_length).rev() {
            current_state = prev_states[t][current_state];
            path.push(current_state);
        }

        path.reverse();
        path
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
                let mut sum = TropicalNumber::tropical_zero();
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
            if i == row {
                continue;
            }
            let mut minor_row = Vec::new();
            for j in 0..self.cols {
                if j == col {
                    continue;
                }
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
            let max_val = row
                .iter()
                .copied()
                .fold(TropicalNumber::zero(), |acc, x| acc + x);
            let row_scores: Vec<T> = row
                .iter()
                .map(|&val| {
                    if max_val.is_zero() {
                        T::zero()
                    } else if val == max_val {
                        T::one()
                    } else {
                        T::zero()
                    }
                })
                .collect();
            scores.push(row_scores);
        }

        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
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
        let mv1 = TropicalMultivector::<f64, 2>::from_coefficients(Vec::from([1.0, 2.0, 3.0, 4.0]));
        let mv2 = TropicalMultivector::<f64, 2>::from_coefficients(Vec::from([0.5, 1.5, 2.5, 3.5]));

        let product = mv1.geometric_product(&mv2);

        // Verify the result has correct structure
        assert!(!product.max_element().is_zero());

        // Check support (non-zero elements)
        let support = mv1.support();
        assert!(!support.is_empty());
    }

    #[test]
    fn test_tropical_matrix() {
        let log_probs = vec![
            Vec::from([0.0, -1.0, -2.0]),
            Vec::from([-1.0, 0.0, -1.0]),
            Vec::from([-2.0, -1.0, 0.0]),
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
        let transitions = Vec::from([
            TropicalNumber::from_log_prob(-0.5),
            TropicalNumber::from_log_prob(-1.0),
            TropicalNumber::from_log_prob(-0.3),
        ]);

        let path_prob = transitions
            .into_iter()
            .fold(TropicalNumber::ONE, |acc, x| acc * x);

        // Should equal sum of log probabilities
        assert_relative_eq!(path_prob.value(), -1.8, epsilon = 1e-10);
    }

    // Comprehensive TropicalNumber tests
    mod tropical_number_tests {
        use super::*;

        #[test]
        fn test_tropical_number_constructors() {
            let n1 = TropicalNumber::new(5.0);
            assert_eq!(n1.value(), 5.0);

            let zero = TropicalNumber::<f64>::neg_infinity();
            assert!(zero.is_zero());
            assert!(zero.is_infinity());

            let one = TropicalNumber::<f64>::zero();
            assert!(one.is_one());
            assert!(!one.is_infinity());

            let tropical_zero = TropicalNumber::<f64>::tropical_zero();
            assert!(tropical_zero.is_zero());

            let tropical_one = TropicalNumber::<f64>::tropical_one();
            assert!(tropical_one.is_one());
        }

        #[test]
        fn test_tropical_number_constants() {
            assert!(TropicalNumber::<f64>::ZERO.is_zero());
            assert!(TropicalNumber::<f64>::ONE.is_one());
            assert_eq!(TropicalNumber::<f64>::ZERO.value(), f64::NEG_INFINITY);
            assert_eq!(TropicalNumber::<f64>::ONE.value(), 0.0);
        }

        #[test]
        fn test_tropical_predicates() {
            let finite = TropicalNumber::new(3.0);
            assert!(!finite.is_zero());
            assert!(!finite.is_one());
            assert!(!finite.is_infinity());

            let zero = TropicalNumber::new(0.0);
            assert!(zero.is_one());
            assert!(!zero.is_zero());

            let pos_inf = TropicalNumber::new(f64::INFINITY);
            assert!(pos_inf.is_infinity());
            assert!(!pos_inf.is_zero());

            let neg_inf = TropicalNumber::new(f64::NEG_INFINITY);
            assert!(neg_inf.is_zero());
            assert!(neg_inf.is_infinity());
        }

        #[test]
        fn test_tropical_arithmetic_properties() {
            let a = TropicalNumber::new(2.0);
            let b = TropicalNumber::new(3.0);
            let c = TropicalNumber::new(1.0);

            // Commutativity
            assert_eq!(a + b, b + a);
            assert_eq!(a * b, b * a);

            // Associativity
            assert_eq!((a + b) + c, a + (b + c));
            assert_eq!((a * b) * c, a * (b * c));

            // Identity elements
            assert_eq!(a + TropicalNumber::ZERO, a);
            assert_eq!(TropicalNumber::ZERO + a, a);
            assert_eq!(a * TropicalNumber::ONE, a);
            assert_eq!(TropicalNumber::ONE * a, a);

            // Distributivity: a * (b + c) = (a * b) + (a * c)
            let left = a * (b + c);
            let right = (a * b) + (a * c);
            assert_eq!(left, right);
        }

        #[test]
        fn test_tropical_add_operation() {
            let a = TropicalNumber::new(5.0);
            let b = TropicalNumber::new(3.0);

            // Tropical add is max
            let result = a.tropical_add(b);
            assert_eq!(result.value(), 5.0);

            let result2 = b.tropical_add(a);
            assert_eq!(result2.value(), 5.0);

            // Test with infinity
            let inf = TropicalNumber::new(f64::INFINITY);
            let result3 = a.tropical_add(inf);
            assert!(result3.value().is_infinite() && result3.value().is_sign_positive());
        }

        #[test]
        fn test_tropical_mul_operation() {
            let a = TropicalNumber::new(2.0);
            let b = TropicalNumber::new(3.0);

            // Tropical mul is addition
            let result = a.tropical_mul(b);
            assert_eq!(result.value(), 5.0);

            // Test with zero (neg infinity)
            let zero = TropicalNumber::ZERO;
            let result2 = a.tropical_mul(zero);
            assert!(result2.is_zero());
        }

        #[test]
        fn test_tropical_pow() {
            let a = TropicalNumber::new(2.0);
            let result = a.tropical_pow(3.0);
            assert_eq!(result.value(), 6.0); // 2 * 3 = 6

            let zero = TropicalNumber::ZERO;
            let result2 = zero.tropical_pow(5.0);
            assert!(result2.is_zero());
        }

        #[test]
        fn test_probability_conversion() {
            let log_prob = -1.0;
            let trop = TropicalNumber::from_log_prob(log_prob);
            assert_eq!(trop.value(), -1.0);

            let prob = trop.to_prob();
            assert_relative_eq!(prob, (-1.0f64).exp(), epsilon = 1e-10);

            // Test zero conversion
            let zero = TropicalNumber::ZERO;
            assert_eq!(zero.to_prob(), 0.0);
        }

        #[test]
        fn test_negation() {
            let a = TropicalNumber::new(3.0);
            let neg_a = -a;
            assert_eq!(neg_a.value(), -3.0);

            let zero = TropicalNumber::new(0.0);
            let neg_zero = -zero;
            assert_eq!(neg_zero.value(), 0.0);
        }

        #[test]
        fn test_operator_overloads() {
            let a = TropicalNumber::new(4.0);
            let b = TropicalNumber::new(2.0);

            // Addition operator (tropical add = max)
            let sum = a + b;
            assert_eq!(sum.value(), 4.0);

            // Multiplication operator (tropical mul = add)
            let product = a * b;
            assert_eq!(product.value(), 6.0);

            // Negation
            let neg = -a;
            assert_eq!(neg.value(), -4.0);
        }

        #[test]
        fn test_edge_cases() {
            let inf = TropicalNumber::new(f64::INFINITY);
            let neg_inf = TropicalNumber::new(f64::NEG_INFINITY);
            let finite = TropicalNumber::new(1.0);

            // Infinity cases
            assert_eq!((inf + finite).value(), f64::INFINITY);
            assert_eq!((inf * finite).value(), f64::INFINITY);

            // Negative infinity cases
            assert_eq!((neg_inf + finite).value(), 1.0); // max(-∞, 1) = 1
            assert_eq!((neg_inf * finite).value(), f64::NEG_INFINITY); // -∞ + 1 = -∞

            // NaN handling
            let nan = TropicalNumber::new(f64::NAN);
            assert!(nan.value().is_nan());
        }
    }

    // Comprehensive TropicalMultivector tests
    mod tropical_multivector_tests {
        use super::*;

        #[test]
        fn test_multivector_constructors() {
            let zero = TropicalMultivector::<f64, 2>::zero();
            // The zero() constructor creates multiplicative identities (0.0), not additive identities (-∞)
            assert!(!zero.is_zero()); // is_zero() checks for all -∞, but zero() creates all 0.0

            let coeffs = vec![1.0, 2.0, 3.0, 4.0];
            let mv = TropicalMultivector::<f64, 2>::from_coefficients(coeffs.clone());

            for (i, &coeff) in coeffs.iter().enumerate() {
                assert_eq!(mv.get(i).value(), coeff);
            }
        }

        #[test]
        fn test_from_log_probs() {
            let log_probs = vec![-1.0, -2.0, -0.5, -3.0];
            let mv = TropicalMultivector::<f64, 2>::from_log_probs(&log_probs);

            for (i, &log_prob) in log_probs.iter().enumerate() {
                assert_eq!(mv.get(i).value(), log_prob);
            }
        }

        #[test]
        fn test_get_set_operations() {
            let mut mv = TropicalMultivector::<f64, 2>::zero();

            let val = TropicalNumber::new(5.0);
            mv.set(1, val);
            assert_eq!(mv.get(1), val);

            // Test bounds
            assert_eq!(mv.get(999).value(), f64::NEG_INFINITY); // Out of bounds returns zero
        }

        #[test]
        fn test_max_element() {
            let coeffs = vec![1.0, 5.0, 2.0, 3.0];
            let mv = TropicalMultivector::<f64, 2>::from_coefficients(coeffs);

            let max_elem = mv.max_element();
            assert_eq!(max_elem.value(), 5.0);
        }

        #[test]
        fn test_tropical_operations() {
            let coeffs1 = vec![1.0, 2.0, 3.0, 4.0];
            let coeffs2 = vec![0.5, 3.0, 1.5, 2.0];

            let mv1 = TropicalMultivector::<f64, 2>::from_coefficients(coeffs1);
            let mv2 = TropicalMultivector::<f64, 2>::from_coefficients(coeffs2);

            // Tropical addition (element-wise max)
            let sum = mv1.tropical_add(&mv2);
            assert_eq!(sum.get(0).value(), 1.0); // max(1.0, 0.5)
            assert_eq!(sum.get(1).value(), 3.0); // max(2.0, 3.0)
            assert_eq!(sum.get(2).value(), 3.0); // max(3.0, 1.5)
            assert_eq!(sum.get(3).value(), 4.0); // max(4.0, 2.0)

            // Regular addition (for comparison)
            let add = mv1.add(&mv2);
            assert_eq!(add.get(0).value(), 1.0); // Still max operation
        }

        #[test]
        fn test_tropical_scaling() {
            let coeffs = vec![1.0, 2.0, 3.0, 4.0];
            let mv = TropicalMultivector::<f64, 2>::from_coefficients(coeffs);

            let scaled = mv.tropical_scale(2.0);
            assert_eq!(scaled.get(0).value(), 3.0); // 1.0 + 2.0
            assert_eq!(scaled.get(1).value(), 4.0); // 2.0 + 2.0
            assert_eq!(scaled.get(2).value(), 5.0); // 3.0 + 2.0
            assert_eq!(scaled.get(3).value(), 6.0); // 4.0 + 2.0

            let regular_scaled = mv.scale(2.0);
            assert_eq!(regular_scaled.get(0).value(), 3.0); // Same operation
        }

        #[test]
        fn test_support() {
            let coeffs = vec![f64::NEG_INFINITY, 1.0, f64::NEG_INFINITY, 2.0];
            let mv = TropicalMultivector::<f64, 2>::from_coefficients(coeffs);

            let support = mv.support();
            assert_eq!(support, vec![1, 3]); // Only non-negative-infinity elements
        }

        #[test]
        fn test_tropical_norm() {
            let coeffs = vec![1.0, 3.0, 2.0, 4.0];
            let mv = TropicalMultivector::<f64, 2>::from_coefficients(coeffs);

            let norm = mv.tropical_norm();
            assert_eq!(norm.value(), 4.0); // Maximum element
        }

        #[test]
        fn test_from_logits() {
            let logits = vec![1.0, 2.0, 0.5, 3.0];
            let mv = TropicalMultivector::<f64, 2>::from_logits(&logits);

            // Should just copy the logits directly (from_logits calls from_log_probs)
            assert_eq!(mv.get(0).value(), 1.0);
            assert_eq!(mv.get(1).value(), 2.0);
            assert_eq!(mv.get(2).value(), 0.5);
            assert_eq!(mv.get(3).value(), 3.0);
        }

        #[test]
        fn test_geometric_product() {
            let mv1 = TropicalMultivector::<f64, 2>::from_coefficients(vec![1.0, 2.0, 3.0, 4.0]);
            let mv2 = TropicalMultivector::<f64, 2>::from_coefficients(vec![0.5, 1.0, 1.5, 2.0]);

            let product = mv1.geometric_product(&mv2);

            // Verify it's not zero and has reasonable structure
            assert!(!product.is_zero());
            assert!(!product.max_element().is_zero());
        }

        #[test]
        fn test_viterbi() {
            // Test the static viterbi function exists and works
            // Note: This is a simplified test since viterbi is complex
            let transitions = TropicalMatrix::<f64>::new(2, 2);
            let emissions = TropicalMatrix::<f64>::new(3, 2);
            let initial_probs = vec![-0.5, -1.0];

            let path =
                TropicalMultivector::<f64, 2>::viterbi(&transitions, &emissions, &initial_probs, 3);
            assert_eq!(path.len(), 3); // Should match sequence length
        }

        #[test]
        fn test_zero_detection() {
            // Create a true tropical zero (all negative infinity)
            let true_zero_mv = TropicalMultivector::<f64, 2>::from_coefficients(vec![
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ]);
            assert!(true_zero_mv.is_zero());

            let non_zero_mv = TropicalMultivector::<f64, 2>::from_coefficients(vec![
                1.0,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ]);
            assert!(!non_zero_mv.is_zero());
        }
    }

    // Comprehensive TropicalMatrix tests
    mod tropical_matrix_tests {
        use super::*;

        #[test]
        fn test_matrix_constructor() {
            let matrix = TropicalMatrix::<f64>::new(3, 3);
            assert_eq!(matrix.rows, 3);
            assert_eq!(matrix.cols, 3);
            assert_eq!(matrix.data.len(), 3);

            // Should be initialized with multiplicative identity (0.0)
            for row in &matrix.data {
                assert_eq!(row.len(), 3);
                for &val in row {
                    assert!(val.is_one()); // TropicalNumber::zero() creates multiplicative identity
                }
            }
        }

        #[test]
        fn test_from_log_probs() {
            let log_probs = vec![
                vec![0.0, -1.0, -2.0],
                vec![-0.5, 0.0, -1.5],
                vec![-1.0, -0.5, 0.0],
            ];

            let matrix = TropicalMatrix::from_log_probs(&log_probs);
            assert_eq!(matrix.rows, 3);
            assert_eq!(matrix.cols, 3);

            // Check values are correctly set
            assert_eq!(matrix.data[0][0].value(), 0.0);
            assert_eq!(matrix.data[0][1].value(), -1.0);
            assert_eq!(matrix.data[1][0].value(), -0.5);
        }

        #[test]
        fn test_matrix_multiplication() {
            let log_probs1 = vec![vec![0.0, -1.0], vec![-0.5, 0.0]];

            let log_probs2 = vec![vec![-0.2, -0.8], vec![-0.3, 0.0]];

            let m1 = TropicalMatrix::from_log_probs(&log_probs1);
            let m2 = TropicalMatrix::from_log_probs(&log_probs2);

            let result = m1.mul(&m2);
            assert_eq!(result.rows, 2);
            assert_eq!(result.cols, 2);

            // Check first element: max(0 + (-0.2), (-1) + (-0.3)) = max(-0.2, -1.3) = -0.2
            assert_relative_eq!(result.data[0][0].value(), -0.2, epsilon = 1e-10);
        }

        #[test]
        fn test_determinant() {
            let log_probs = vec![
                vec![0.0, -1.0, -2.0],
                vec![-1.0, 0.0, -1.0],
                vec![-2.0, -1.0, 0.0],
            ];

            let matrix = TropicalMatrix::from_log_probs(&log_probs);
            let det = matrix.determinant();

            // Should not be zero (negative infinity)
            assert!(!det.is_zero());

            // Test 2x2 determinant
            let small_probs = vec![vec![0.0, -1.0], vec![-0.5, 0.0]];
            let small_matrix = TropicalMatrix::from_log_probs(&small_probs);
            let small_det = small_matrix.determinant();

            // det = max(0 + 0, (-1) + (-0.5)) = max(0, -1.5) = 0
            assert_eq!(small_det.value(), 0.0);
        }

        #[test]
        fn test_attention_scores() {
            let log_probs = vec![
                vec![0.0, -1.0, -2.0],
                vec![-0.5, 0.0, -1.0],
                vec![-1.5, -0.5, 0.0],
            ];

            let matrix = TropicalMatrix::from_log_probs(&log_probs);
            let scores = matrix.to_attention_scores();

            assert_eq!(scores.len(), 3);
            for row in &scores {
                assert_eq!(row.len(), 3);

                // Each row should sum to approximately 1.0 (attention weights)
                let sum: f64 = row.iter().sum();
                assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
            }
        }

        #[test]
        fn test_matrix_edge_cases() {
            // Empty matrix
            let empty_matrix = TropicalMatrix::<f64>::new(0, 0);
            assert_eq!(empty_matrix.rows, 0);
            assert_eq!(empty_matrix.cols, 0);

            // Single element matrix
            let single = TropicalMatrix::from_log_probs(&[vec![-0.5]]);
            let det = single.determinant();
            assert_eq!(det.value(), -0.5);

            // Matrix with infinities
            let inf_probs = vec![vec![0.0, f64::NEG_INFINITY], vec![f64::NEG_INFINITY, 0.0]];
            let inf_matrix = TropicalMatrix::from_log_probs(&inf_probs);
            let inf_det = inf_matrix.determinant();
            assert_eq!(inf_det.value(), 0.0); // max(0, -∞) = 0
        }

        #[test]
        fn test_matrix_properties() {
            let log_probs = vec![vec![0.0, -1.0], vec![-0.5, 0.0]];

            let matrix = TropicalMatrix::from_log_probs(&log_probs);

            // Test that matrix operations are consistent
            let identity_probs = vec![vec![0.0, f64::NEG_INFINITY], vec![f64::NEG_INFINITY, 0.0]];
            let identity = TropicalMatrix::from_log_probs(&identity_probs);

            let result = matrix.mul(&identity);

            // Result should be close to original matrix
            assert_relative_eq!(
                result.data[0][0].value(),
                matrix.data[0][0].value(),
                epsilon = 1e-10
            );
            assert_relative_eq!(
                result.data[1][1].value(),
                matrix.data[1][1].value(),
                epsilon = 1e-10
            );
        }
    }
}
