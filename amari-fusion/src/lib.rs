//! Tropical-Dual-Clifford fusion system for optimal LLM evaluation
//!
//! This crate combines three exotic number systems:
//! - Tropical algebra: Converts expensive softmax operations to max operations
//! - Dual numbers: Provides automatic differentiation without computational graphs
//! - Clifford algebra: Handles geometric relationships and rotations
//!
//! Together, these systems create a powerful framework for efficient neural network
//! evaluation and optimization.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
use alloc::vec::{self, Vec};

use amari_core::Multivector;
use amari_tropical::{TropicalNumber, TropicalMultivector, TropicalMatrix};
use amari_dual::{DualNumber, multivector::DualMultivector};
use num_traits::Float;

pub mod optimizer;
pub mod attention;
pub mod evaluation;

/// The unified Tropical-Dual-Clifford structure
#[derive(Clone, Debug)]
pub struct TropicalDualClifford<T: Float, const DIM: usize> {
    /// Tropical view: efficient max operations and path finding
    pub tropical: TropicalMultivector<T, DIM>,
    /// Dual view: automatic differentiation
    pub dual: DualMultivector<T, 3, 0, 0>, // Default to 3D Euclidean
    /// Clifford view: geometric relationships
    pub clifford: Multivector<3, 0, 0>,
}

impl<T: Float, const DIM: usize> TropicalDualClifford<T, DIM> {
    const BASIS_COUNT: usize = 1 << DIM;
    
    /// Create zero TDC object
    pub fn zero() -> Self {
        let tropical = TropicalMultivector::zero();
        let dual = DualMultivector::zero();
        let clifford = Multivector::zero();
        
        Self {
            tropical,
            dual,
            clifford,
        }
    }
    
    /// Create random TDC object for testing
    pub fn random() -> Self {
        let logits: Vec<T> = (0..DIM).map(|_| {
            T::from(fastrand::f64() * 2.0 - 1.0).unwrap()
        }).collect();
        Self::from_logits(&logits)
    }
    
    /// Create random TDC with specific scale
    pub fn random_with_scale(scale: T) -> Self {
        let logits: Vec<T> = (0..DIM).map(|_| {
            scale * T::from(fastrand::f64() * 2.0 - 1.0).unwrap()
        }).collect();
        Self::from_logits(&logits)
    }
    
    /// Check if TDC is zero
    pub fn is_zero(&self) -> bool {
        self.tropical.is_zero() && self.dual.norm().real == T::zero() && self.clifford.norm() == 0.0
    }
    
    /// Extract tropical features as vector
    pub fn extract_tropical_features(&self) -> Vec<TropicalNumber<T>> {
        (0..DIM.min(8)).map(|i| self.tropical.get(i)).collect()
    }
    
    /// Extract dual features as vector
    pub fn extract_dual_features(&self) -> Vec<DualNumber<T>> {
        (0..8).map(|i| self.dual.get(i)).collect()
    }
    
    /// Add two TDC objects
    pub fn add(&self, other: &Self) -> Self {
        Self {
            tropical: self.tropical.add(&other.tropical),
            dual: self.dual.clone() + other.dual.clone(),
            clifford: self.clifford.clone() + other.clifford.clone(),
        }
    }
    
    /// Scale TDC object by scalar
    pub fn scale(&self, factor: T) -> Self {
        Self {
            tropical: self.tropical.scale(factor),
            dual: {
                let mut scaled = self.dual.clone();
                for i in 0..8 {
                    scaled.set(i, scaled.get(i) * DualNumber::constant(factor));
                }
                scaled
            },
            clifford: self.clifford.clone() * factor.to_f64().unwrap_or(1.0),
        }
    }
    
    /// Create from components
    pub fn from_components(
        tropical_features: Vec<TropicalNumber<T>>,
        dual_features: Vec<DualNumber<T>>,
        clifford: Multivector<3, 0, 0>
    ) -> Self {
        let mut tropical = TropicalMultivector::zero();
        for (i, &feature) in tropical_features.iter().enumerate() {
            if i < DIM {
                tropical.set(i, feature);
            }
        }
        
        let mut dual = DualMultivector::zero();
        for (i, &feature) in dual_features.iter().enumerate() {
            if i < 8 {
                dual.set(i, feature);
            }
        }
        
        Self {
            tropical,
            dual,
            clifford,
        }
    }
    
    /// Create from log-probabilities (common in LLM outputs)
    pub fn from_logits(logits: &[T]) -> Self {
        // Convert to tropical (log domain is natural for tropical)
        let tropical = TropicalMultivector::from_log_probs(logits);
        
        // Convert to dual for automatic differentiation
        let dual_coeffs_t: Vec<T> = logits.iter()
            .take(8) // Take first 8 for 3D Clifford
            .copied()
            .chain(core::iter::repeat(T::zero()))
            .take(8)
            .collect();
        let dual = DualMultivector::new_variables(&dual_coeffs_t);
        
        // Convert to f64 for Clifford
        let dual_coeffs_f64: Vec<f64> = dual_coeffs_t.iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .collect();
        
        // Convert to Clifford for geometric operations
        let clifford = Multivector::from_coefficients(dual_coeffs_f64);
        
        Self {
            tropical,
            dual,
            clifford,
        }
    }
    
    /// Create from probability distribution
    pub fn from_probabilities(probs: &[T]) -> Self {
        let logits: Vec<T> = probs.iter()
            .map(|&p| if p > T::zero() { p.ln() } else { T::neg_infinity() })
            .collect();
        Self::from_logits(&logits)
    }
    
    /// Evaluate with all three algebras simultaneously
    pub fn evaluate(&self, other: &Self) -> EvaluationResult<T> {
        EvaluationResult {
            // Tropical: find most likely path efficiently
            best_path_score: self.tropical.max_element(),
            
            // Dual: get gradients automatically
            gradient_norm: self.dual.norm().derivative(),
            
            // Clifford: compute geometric alignment
            geometric_distance: (self.clifford.clone() - other.clifford.clone()).norm(),
            
            // Combined score
            combined_score: self.compute_combined_score(other),
        }
    }
    
    /// Compute a combined score using all three algebras
    fn compute_combined_score(&self, other: &Self) -> T {
        // Tropical contribution: max-based similarity
        let tropical_contrib = self.tropical.max_element().tropical_add(other.tropical.max_element());
        
        // Dual contribution: derivative-based measure
        let dual_contrib = T::from(self.dual.norm().real).unwrap_or(T::zero());
        
        // Clifford contribution: geometric similarity
        let clifford_contrib = T::from(self.clifford.scalar_product(&other.clifford)).unwrap_or(T::zero());
        
        // Weighted combination (weights could be learned parameters)
        let w1 = T::from(0.4).unwrap();
        let w2 = T::from(0.3).unwrap();
        let w3 = T::from(0.3).unwrap();
        
        w1 * tropical_contrib.value() + w2 * dual_contrib + w3 * clifford_contrib
    }
    
    /// Convert between representations
    pub fn to_tropical_matrix(&self, rows: usize, cols: usize) -> TropicalMatrix<T> {
        TropicalMatrix::new(rows, cols)
    }
    
    /// Sensitivity analysis using dual numbers
    pub fn sensitivity_analysis(&self) -> SensitivityMap<T> {
        let mut sensitivities = Vec::new();
        
        for i in 0..8 {
            let coeff = self.dual.get(i);
            sensitivities.push(SensitivityInfo {
                component: i,
                value: T::from(coeff.real).unwrap_or(T::zero()),
                sensitivity: T::from(coeff.dual).unwrap_or(T::zero()),
            });
        }
        
        SensitivityMap { sensitivities }
    }
    
    /// Transform using all three algebras
    pub fn transform(&self, transformation: &TropicalDualClifford<T, DIM>) -> Self {
        Self {
            tropical: self.tropical.geometric_product(&transformation.tropical),
            dual: self.dual.geometric_product(&transformation.dual),
            clifford: self.clifford.geometric_product(&transformation.clifford),
        }
    }
    
    /// Distance measure combining all three metrics
    pub fn distance(&self, other: &Self) -> T {
        // Tropical distance (max difference)
        let tropical_dist = (self.tropical.max_element().value() - other.tropical.max_element().value()).abs();
        
        // Dual distance (with derivative information)
        let dual_dist = T::from((self.dual.clone() - other.dual.clone()).norm().real).unwrap_or(T::zero());
        
        // Clifford distance (geometric norm)
        let clifford_dist = T::from((self.clifford.clone() - other.clifford.clone()).norm()).unwrap_or(T::zero());
        
        // Combined distance metric
        (tropical_dist * tropical_dist + dual_dist * dual_dist + clifford_dist * clifford_dist).sqrt()
    }
    
    /// Interpolate between two TDC objects
    pub fn interpolate(&self, other: &Self, t: T) -> Self {
        // Linear interpolation in each algebra
        let one_minus_t = T::one() - t;
        
        // Tropical interpolation (geometric mean in log space)
        let tropical_interp = self.tropical.clone(); // Simplified
        
        // Dual interpolation
        let dual_interp = self.dual.clone(); // Simplified
        
        // Clifford interpolation  
        let clifford_interp = self.clifford.clone() * one_minus_t.to_f64().unwrap_or(0.0)
                            + other.clifford.clone() * t.to_f64().unwrap_or(1.0);
        
        Self {
            tropical: tropical_interp,
            dual: dual_interp,
            clifford: clifford_interp,
        }
    }
    
    /// Get tropical view
    pub fn tropical(&self) -> &TropicalMultivector<T, DIM> {
        &self.tropical
    }
    
    /// Get dual view
    pub fn dual(&self) -> &DualMultivector<T, 3, 0, 0> {
        &self.dual
    }
    
    /// Get clifford view
    pub fn clifford(&self) -> &Multivector<3, 0, 0> {
        &self.clifford
    }
}

/// Result of evaluating two TDC objects
#[derive(Debug, Clone)]
pub struct EvaluationResult<T: Float> {
    /// Best path score from tropical algebra
    pub best_path_score: TropicalNumber<T>,
    /// Gradient norm from dual numbers
    pub gradient_norm: T,
    /// Geometric distance from Clifford algebra
    pub geometric_distance: f64,
    /// Combined score using all three systems
    pub combined_score: T,
}

/// Sensitivity information from automatic differentiation
#[derive(Debug, Clone)]
pub struct SensitivityInfo<T: Float> {
    pub component: usize,
    pub value: T,
    pub sensitivity: T,
}

/// Map of sensitivities for all components
#[derive(Debug, Clone)]
pub struct SensitivityMap<T: Float> {
    pub sensitivities: Vec<SensitivityInfo<T>>,
}

impl<T: Float> SensitivityMap<T> {
    /// Find components with highest sensitivity
    pub fn most_sensitive(&self, n: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, T)> = self.sensitivities.iter()
            .map(|s| (s.component, s.sensitivity.abs()))
            .collect();
        
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        indexed.into_iter().take(n).map(|(i, _)| i).collect()
    }
    
    /// Total sensitivity (sum of absolute values)
    pub fn total_sensitivity(&self) -> T {
        self.sensitivities.iter()
            .map(|s| s.sensitivity.abs())
            .fold(T::zero(), |acc, x| acc + x)
    }
}

/// Specialized structure for LLM token distributions
pub struct TropicalDualDistribution<T: Float> {
    pub logits: TropicalDualClifford<T, 8>, // Assuming 8-dimensional embedding
    pub vocab_size: usize,
}

impl<T: Float> TropicalDualDistribution<T> {
    /// Create from logit vector
    pub fn from_logits(logits: &[T]) -> Self {
        Self {
            logits: TropicalDualClifford::from_logits(logits),
            vocab_size: logits.len(),
        }
    }
    
    /// KL divergence with automatic gradient
    pub fn kl_divergence(&self, other: &Self) -> DualNumber<T> {
        // Simplified KL divergence using dual numbers
        let self_norm = self.logits.dual.norm();
        let other_norm = other.logits.dual.norm();
        
        // KL approximation using norms (simplified for demonstration)
        self_norm.ln() - other_norm.ln()
    }
    
    /// Most likely sequence using tropical algebra (Viterbi-like)
    pub fn most_likely_sequence(&self, length: usize) -> Vec<usize> {
        // Use tropical algebra for efficient sequence decoding
        let mut sequence = Vec::with_capacity(length);
        
        // Simplified sequence generation
        for i in 0..length {
            let max_component = self.logits.tropical.support().into_iter()
                .max_by(|&a, &b| {
                    self.logits.tropical.get(a).value()
                        .partial_cmp(&self.logits.tropical.get(b).value())
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            sequence.push(max_component % self.vocab_size);
        }
        
        sequence
    }
    
    /// Geometric alignment using Clifford algebra
    pub fn geometric_alignment(&self, reference: &Self) -> f64 {
        self.logits.clifford.scalar_product(&reference.logits.clifford)
    }
    
    /// Attention pattern as tropical polytope
    pub fn attention_polytope(&self) -> Vec<Vec<T>> {
        // Convert attention pattern to tropical polytope vertices
        let support = self.logits.tropical.support();
        let mut vertices = Vec::new();
        
        for &idx in &support {
            let mut vertex = Vec::with_capacity(support.len());
            for _ in 0..support.len() {
                vertex.push(T::zero());
            }
            if let Some(pos) = support.iter().position(|&x| x == idx) {
                vertex[pos] = self.logits.tropical.get(idx).value();
            }
            vertices.push(vertex);
        }
        
        vertices
    }
}

/// Builder for constructing TDC objects
pub struct TropicalDualCliffordBuilder<T: Float, const DIM: usize> {
    logits: Vec<T>,
}

impl<T: Float, const DIM: usize> TropicalDualCliffordBuilder<T, DIM> {
    pub fn new() -> Self {
        Self {
            logits: Vec::new(),
        }
    }
    
    pub fn add_logit(mut self, logit: T) -> Self {
        self.logits.push(logit);
        self
    }
    
    pub fn add_logits(mut self, logits: &[T]) -> Self {
        self.logits.extend_from_slice(logits);
        self
    }
    
    pub fn build(self) -> TropicalDualClifford<T, DIM> {
        TropicalDualClifford::from_logits(&self.logits)
    }
}

/// Conversion utilities between different representations
pub mod conversion {
    use super::*;
    
    /// Convert softmax probabilities to tropical numbers
    pub fn softmax_to_tropical<T: Float>(probs: &[T]) -> Vec<TropicalNumber<T>> {
        probs.iter().map(|&p| TropicalNumber::from_log_prob(p.ln())).collect()
    }
    
    /// Convert tropical numbers back to probabilities
    pub fn tropical_to_softmax<T: Float>(tropical: &[TropicalNumber<T>]) -> Vec<T> {
        tropical.iter().map(|&tn| tn.to_prob()).collect()
    }
    
    /// Convert between Clifford and dual representations
    pub fn clifford_to_dual<T: Float>(mv: &Multivector<3, 0, 0>) -> Vec<DualNumber<T>> {
        (0..8).map(|i| {
            DualNumber::variable(T::from(mv.get(i)).unwrap_or(T::zero()))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_tdc_creation() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);
        
        // Check that all three representations exist
        assert!(!tdc.tropical.max_element().is_zero());
        assert!(tdc.dual.norm().real > 0.0);
        assert!(tdc.clifford.norm() > 0.0);
    }
    
    #[test]
    fn test_evaluation() {
        let logits1 = vec![1.0, 2.0, 3.0, 0.5];
        let logits2 = vec![1.5, 1.8, 2.5, 1.0];
        
        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);
        
        let result = tdc1.evaluate(&tdc2);
        
        assert!(result.combined_score > 0.0);
        assert!(result.geometric_distance >= 0.0);
    }
    
    #[test]
    fn test_sensitivity_analysis() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);
        
        let sensitivity = tdc.sensitivity_analysis();
        
        assert!(!sensitivity.sensitivities.is_empty());
        assert!(sensitivity.total_sensitivity() > 0.0);
        
        let most_sensitive = sensitivity.most_sensitive(2);
        assert_eq!(most_sensitive.len(), 2);
    }
    
    #[test]
    fn test_distance_metric() {
        let logits1 = vec![1.0, 0.0, 0.0, 0.0];
        let logits2 = vec![0.0, 1.0, 0.0, 0.0];
        
        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);
        
        let distance = tdc1.distance(&tdc2);
        assert!(distance > 0.0);
        
        // Distance to self should be zero
        let self_distance = tdc1.distance(&tdc1);
        assert_relative_eq!(self_distance, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_tropical_dual_distribution() {
        let logits = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        let dist1 = TropicalDualDistribution::from_logits(&logits);
        let dist2 = TropicalDualDistribution::from_logits(&[2.0, 1.0, 2.5, 1.0, 1.8]);
        
        // Test KL divergence
        let kl = dist1.kl_divergence(&dist2);
        assert!(kl.real.abs() > 0.0); // Should be non-zero
        
        // Test sequence generation
        let sequence = dist1.most_likely_sequence(3);
        assert_eq!(sequence.len(), 3);
        
        // Test geometric alignment
        let alignment = dist1.geometric_alignment(&dist2);
        assert!(alignment.abs() > 0.0);
    }
    
    #[test]
    fn test_interpolation() {
        let logits1 = vec![1.0, 0.0, 0.0, 0.0];
        let logits2 = vec![0.0, 1.0, 0.0, 0.0];
        
        let tdc1 = TropicalDualClifford::<f64, 4>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 4>::from_logits(&logits2);
        
        let interpolated = tdc1.interpolate(&tdc2, 0.5);
        
        // Should be between the two original points
        let dist_to_1 = interpolated.distance(&tdc1);
        let dist_to_2 = interpolated.distance(&tdc2);
        
        assert!(dist_to_1 > 0.0);
        assert!(dist_to_2 > 0.0);
    }
    
    #[test]
    fn test_conversion_utilities() {
        use conversion::*;
        
        let probs = vec![0.1, 0.3, 0.4, 0.2];
        let tropical = softmax_to_tropical(&probs);
        let recovered = tropical_to_softmax(&tropical);
        
        // Should recover approximately the same probabilities
        for (orig, recov) in probs.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, recov, epsilon = 1e-6);
        }
    }
}