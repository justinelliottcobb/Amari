//! Fisher-Riemannian measures on statistical manifolds
//!
//! This module provides measure theory on manifolds of probability distributions,
//! using the Fisher information metric from information geometry.
//!
//! # Overview
//!
//! A statistical manifold is a smooth manifold where each point represents a
//! probability distribution. The **Fisher information metric** g_ij(θ) provides
//! a Riemannian structure:
//!
//! g_ij(θ) = E_θ[(∂/∂θ_i log p(x|θ))(∂/∂θ_j log p(x|θ))]
//!
//! # Volume Element
//!
//! The Fisher metric induces a volume form (measure) on the statistical manifold:
//!
//! dV = √det(g) dθ_1 ∧ ... ∧ dθ_n
//!
//! This volume element is invariant under reparametrization and respects
//! the geometry of the space of distributions.
//!
//! # Examples
//!
//! ```
//! use amari_measure::fisher_measure::FisherMeasure;
//!
//! // Create a Fisher measure on a 2-parameter exponential family
//! let fisher_measure: FisherMeasure = FisherMeasure::new(2);
//! ```

use crate::error::{MeasureError, Result};
use crate::parametric::ParametricDensity;

/// Fisher-Riemannian measure on a statistical manifold
///
/// This measure assigns volumes to sets of probability distributions
///using the Fisher information metric.
///
/// # Type Parameters
///
/// - `X`: Domain type for the random variable (typically `f64`)
///
/// # Mathematical Structure
///
/// The Fisher metric g_ij(θ) = E[(∂_i log p)(∂_j log p)] induces a volume form:
/// dV = √det(g(θ)) dθ
///
/// This provides a natural measure on the space of distributions.
pub struct FisherMeasure<X = f64> {
    /// Dimension of the parameter space
    num_params: usize,

    /// The parametric density family
    density: Option<ParametricDensity<X>>,
}

impl<X> FisherMeasure<X> {
    /// Create a new Fisher measure
    ///
    /// # Arguments
    ///
    /// * `num_params` - Dimension of the parameter space
    pub fn new(num_params: usize) -> Self {
        Self {
            num_params,
            density: None,
        }
    }

    /// Create a Fisher measure from a parametric density
    pub fn from_density(density: ParametricDensity<X>) -> Self {
        let num_params = density.num_params();
        Self {
            num_params,
            density: Some(density),
        }
    }

    /// Get the dimension of the parameter space
    pub fn dimension(&self) -> usize {
        self.num_params
    }

    /// Get the parametric density if available
    pub fn density(&self) -> Option<&ParametricDensity<X>> {
        self.density.as_ref()
    }
}

impl FisherMeasure<f64> {
    /// Compute the Fisher information matrix at parameters θ
    ///
    /// Returns g_ij(θ) = E[(∂_i log p)(∂_j log p)]
    ///
    /// # Arguments
    ///
    /// * `params` - The parameter vector θ
    /// * `data` - Sample data for empirical estimation
    pub fn fisher_information(&self, params: &[f64], data: &[f64]) -> Result<Vec<Vec<f64>>> {
        let density = self.density.as_ref().ok_or_else(|| {
            MeasureError::computation("No density specified for Fisher information".to_string())
        })?;

        density.fisher_information(data, params)
    }

    /// Compute the volume element at parameters θ
    ///
    /// Returns √det(g(θ)) where g is the Fisher metric
    ///
    /// # Arguments
    ///
    /// * `params` - The parameter vector θ
    /// * `data` - Sample data for empirical estimation
    pub fn volume_element(&self, params: &[f64], data: &[f64]) -> Result<f64> {
        let fisher = self.fisher_information(params, data)?;

        // Compute determinant using simple method for small matrices
        let det = Self::determinant(&fisher)?;

        if det < 0.0 {
            return Err(MeasureError::computation(
                "Fisher information matrix has negative determinant".to_string(),
            ));
        }

        Ok(det.sqrt())
    }

    /// Measure a parameter region (approximation via sampling)
    ///
    /// Computes ∫_R √det(g(θ)) dθ over region R by sampling
    ///
    /// # Arguments
    ///
    /// * `param_samples` - Sample points in parameter space
    /// * `data` - Data samples for estimating Fisher information
    pub fn measure_region(&self, param_samples: &[Vec<f64>], data: &[f64]) -> Result<f64> {
        if param_samples.is_empty() {
            return Ok(0.0);
        }

        let mut total_volume = 0.0;
        for params in param_samples {
            if params.len() != self.num_params {
                return Err(MeasureError::computation(format!(
                    "Parameter dimension mismatch: expected {}, got {}",
                    self.num_params,
                    params.len()
                )));
            }

            let vol_element = self.volume_element(params, data)?;
            total_volume += vol_element;
        }

        // Approximate integral by average value
        Ok(total_volume / param_samples.len() as f64)
    }

    /// Compute determinant of a square matrix (simple implementation)
    ///
    /// Uses cofactor expansion for small matrices
    fn determinant(matrix: &[Vec<f64>]) -> Result<f64> {
        let n = matrix.len();
        if n == 0 {
            return Ok(1.0);
        }

        // Check square matrix
        for row in matrix {
            if row.len() != n {
                return Err(MeasureError::computation(
                    "Matrix must be square".to_string(),
                ));
            }
        }

        match n {
            1 => Ok(matrix[0][0]),
            2 => {
                // 2x2: ad - bc
                Ok(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
            }
            3 => {
                // 3x3: Sarrus rule
                Ok(matrix[0][0] * matrix[1][1] * matrix[2][2]
                    + matrix[0][1] * matrix[1][2] * matrix[2][0]
                    + matrix[0][2] * matrix[1][0] * matrix[2][1]
                    - matrix[0][2] * matrix[1][1] * matrix[2][0]
                    - matrix[0][1] * matrix[1][0] * matrix[2][2]
                    - matrix[0][0] * matrix[1][2] * matrix[2][1])
            }
            _ => {
                // For larger matrices, use LU decomposition or external library
                // For now, return error
                Err(MeasureError::computation(
                    "Determinant computation only supported for matrices up to 3x3".to_string(),
                ))
            }
        }
    }
}

/// Statistical manifold with Fisher metric
///
/// Represents a smooth manifold M where each point θ ∈ M represents
/// a probability distribution p(·|θ).
///
/// # Examples
///
/// ```
/// use amari_measure::fisher_measure::FisherStatisticalManifold;
///
/// // Create a 2D exponential family manifold
/// let manifold: FisherStatisticalManifold = FisherStatisticalManifold::new(2);
/// ```
pub struct FisherStatisticalManifold<X = f64> {
    /// Dimension of the manifold
    dimension: usize,

    /// The Fisher measure
    fisher_measure: FisherMeasure<X>,
}

impl<X> FisherStatisticalManifold<X> {
    /// Create a new statistical manifold
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            fisher_measure: FisherMeasure::new(dimension),
        }
    }

    /// Create from a parametric density
    pub fn from_density(density: ParametricDensity<X>) -> Self {
        let dimension = density.num_params();
        Self {
            dimension,
            fisher_measure: FisherMeasure::from_density(density),
        }
    }

    /// Get the dimension of the manifold
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the Fisher measure
    pub fn fisher_measure(&self) -> &FisherMeasure<X> {
        &self.fisher_measure
    }
}

impl FisherStatisticalManifold<f64> {
    /// Compute the Fisher metric tensor at a point
    ///
    /// Returns g_ij(θ)
    pub fn metric_tensor(&self, params: &[f64], data: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.fisher_measure.fisher_information(params, data)
    }

    /// Compute the Riemannian volume form at a point
    ///
    /// Returns √det(g(θ))
    pub fn volume_form(&self, params: &[f64], data: &[f64]) -> Result<f64> {
        self.fisher_measure.volume_element(params, data)
    }

    /// Geodesic distance (placeholder - requires integration along geodesics)
    ///
    /// For now, returns Euclidean distance as approximation
    pub fn geodesic_distance(&self, theta1: &[f64], theta2: &[f64]) -> Result<f64> {
        if theta1.len() != self.dimension || theta2.len() != self.dimension {
            return Err(MeasureError::computation(
                "Parameter dimension mismatch".to_string(),
            ));
        }

        // Euclidean approximation (TODO: implement true geodesic distance)
        let mut sum_sq = 0.0;
        for i in 0..self.dimension {
            let diff = theta1[i] - theta2[i];
            sum_sq += diff * diff;
        }

        Ok(sum_sq.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parametric::families;

    #[test]
    fn test_fisher_measure_creation() {
        let measure = FisherMeasure::<f64>::new(2);
        assert_eq!(measure.dimension(), 2);
        assert!(measure.density().is_none());
    }

    #[test]
    fn test_fisher_measure_from_density() {
        let gaussian = families::gaussian();
        let measure = FisherMeasure::from_density(gaussian);

        assert_eq!(measure.dimension(), 2);
        assert!(measure.density().is_some());
    }

    #[test]
    fn test_fisher_information_gaussian() {
        let gaussian = families::gaussian();
        let measure = FisherMeasure::from_density(gaussian);

        // Test at μ=0, σ=1
        let params = vec![0.0, 1.0];
        let data = vec![-1.0, 0.0, 1.0]; // Small sample

        let fisher = measure.fisher_information(&params, &data);
        assert!(fisher.is_ok());

        let g = fisher.unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].len(), 2);
    }

    #[test]
    fn test_volume_element() {
        let gaussian = families::gaussian();
        let measure = FisherMeasure::from_density(gaussian);

        let params = vec![0.0, 1.0];
        let data = vec![-1.0, 0.0, 1.0];

        let vol = measure.volume_element(&params, &data);
        assert!(vol.is_ok());
        assert!(vol.unwrap() >= 0.0);
    }

    #[test]
    fn test_determinant_2x2() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let det = FisherMeasure::<f64>::determinant(&matrix).unwrap();
        assert_eq!(det, -2.0); // 1*4 - 2*3 = -2
    }

    #[test]
    fn test_determinant_3x3() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let det = FisherMeasure::<f64>::determinant(&matrix).unwrap();
        assert_eq!(det, 0.0); // Singular matrix
    }

    #[test]
    fn test_statistical_manifold_creation() {
        let manifold = FisherStatisticalManifold::<f64>::new(2);
        assert_eq!(manifold.dimension(), 2);
    }

    #[test]
    fn test_statistical_manifold_from_density() {
        let gaussian = families::gaussian();
        let manifold = FisherStatisticalManifold::from_density(gaussian);

        assert_eq!(manifold.dimension(), 2);
    }

    #[test]
    fn test_metric_tensor() {
        let gaussian = families::gaussian();
        let manifold = FisherStatisticalManifold::from_density(gaussian);

        let params = vec![0.0, 1.0];
        let data = vec![-1.0, 0.0, 1.0];

        let metric = manifold.metric_tensor(&params, &data);
        assert!(metric.is_ok());
    }

    #[test]
    fn test_geodesic_distance() {
        let manifold = FisherStatisticalManifold::<f64>::new(2);

        let theta1 = vec![0.0, 0.0];
        let theta2 = vec![3.0, 4.0];

        let dist = manifold.geodesic_distance(&theta1, &theta2).unwrap();
        assert_eq!(dist, 5.0); // Euclidean distance: sqrt(3² + 4²) = 5
    }
}
