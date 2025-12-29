//! Sobolev spaces for multivector-valued functions.
//!
//! This module provides Sobolev spaces W^{k,p}(Ω, Cl(P,Q,R)) of multivector-valued
//! functions with weak derivatives up to order k in L^p.
//!
//! # Mathematical Background
//!
//! The Sobolev space W^{k,p}(Ω, V) consists of functions f: Ω → V such that
//! all weak derivatives up to order k exist and belong to L^p.
//!
//! For p = 2, these are Hilbert spaces (denoted H^k) with inner product:
//!
//! ⟨f, g⟩_{H^k} = Σ_{|α| ≤ k} ∫_Ω ⟨D^α f, D^α g⟩ dx
//!
//! # Key Properties
//!
//! - **Sobolev embedding**: W^{k,p} ⊂ C^m for k > m + n/p
//! - **Poincaré inequality**: ||f||_{L^p} ≤ C ||∇f||_{L^p} for f with zero boundary
//! - **Trace theorem**: Boundary values are well-defined in W^{k-1/p,p}

use crate::phantom::HkRegularity;
use crate::space::Domain;
use amari_core::Multivector;
use std::marker::PhantomData;
use std::sync::Arc;

/// A function in a Sobolev space with weak derivatives.
///
/// Stores the function and its weak derivatives up to order k.
#[derive(Clone)]
pub struct SobolevFunction<const P: usize, const Q: usize, const R: usize, const K: usize> {
    /// The function f.
    func: Arc<dyn Fn(&[f64]) -> Multivector<P, Q, R> + Send + Sync>,
    /// Weak derivatives (indexed by multi-index).
    /// For K=1, this is just the gradient.
    derivatives: Vec<Arc<dyn Fn(&[f64]) -> Multivector<P, Q, R> + Send + Sync>>,
    /// Spatial dimension.
    spatial_dim: usize,
}

impl<const P: usize, const Q: usize, const R: usize, const K: usize> std::fmt::Debug
    for SobolevFunction<P, Q, R, K>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SobolevFunction")
            .field("regularity", &K)
            .field("spatial_dim", &self.spatial_dim)
            .finish()
    }
}

impl<const P: usize, const Q: usize, const R: usize, const K: usize> SobolevFunction<P, Q, R, K> {
    /// Create a new Sobolev function with its derivatives.
    pub fn new<F>(
        func: F,
        derivatives: Vec<Arc<dyn Fn(&[f64]) -> Multivector<P, Q, R> + Send + Sync>>,
        spatial_dim: usize,
    ) -> Self
    where
        F: Fn(&[f64]) -> Multivector<P, Q, R> + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(func),
            derivatives,
            spatial_dim,
        }
    }

    /// Evaluate the function at a point.
    pub fn eval(&self, point: &[f64]) -> Multivector<P, Q, R> {
        (self.func)(point)
    }

    /// Evaluate the i-th derivative at a point.
    pub fn eval_derivative(&self, i: usize, point: &[f64]) -> Option<Multivector<P, Q, R>> {
        self.derivatives.get(i).map(|d| d(point))
    }

    /// Get the number of derivatives stored.
    pub fn derivative_count(&self) -> usize {
        self.derivatives.len()
    }

    /// Get the spatial dimension.
    pub fn spatial_dimension(&self) -> usize {
        self.spatial_dim
    }
}

/// A Sobolev space W^{k,2}(Ω, Cl(P,Q,R)) = H^k(Ω, Cl(P,Q,R)).
///
/// This is the Hilbert space of multivector-valued functions with
/// k weak derivatives in L^2.
#[derive(Clone)]
pub struct SobolevSpace<const P: usize, const Q: usize, const R: usize, const K: usize> {
    /// The domain of the space.
    domain: Domain<f64>,
    /// Number of quadrature points for numerical integration.
    quadrature_points: usize,
    _phantom: PhantomData<HkRegularity<K>>,
}

impl<const P: usize, const Q: usize, const R: usize, const K: usize> std::fmt::Debug
    for SobolevSpace<P, Q, R, K>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SobolevSpace")
            .field("order", &K)
            .field("signature", &(P, Q, R))
            .field("domain", &self.domain)
            .finish()
    }
}

impl<const P: usize, const Q: usize, const R: usize, const K: usize> SobolevSpace<P, Q, R, K> {
    /// Create a new Sobolev space H^k over a domain.
    pub fn new(domain: Domain<f64>) -> Self {
        Self {
            domain,
            quadrature_points: 32,
            _phantom: PhantomData,
        }
    }

    /// Create H^k over the unit interval [0, 1].
    pub fn unit_interval() -> Self {
        Self::new(Domain::interval(0.0, 1.0))
    }

    /// Set the number of quadrature points.
    pub fn with_quadrature_points(mut self, n: usize) -> Self {
        self.quadrature_points = n;
        self
    }

    /// Get the Sobolev order k.
    pub fn order(&self) -> usize {
        K
    }

    /// Get the domain.
    pub fn domain(&self) -> &Domain<f64> {
        &self.domain
    }

    /// Compute the H^k inner product.
    ///
    /// ⟨f, g⟩_{H^k} = ⟨f, g⟩_{L^2} + Σ_{|α|=1}^k ⟨D^α f, D^α g⟩_{L^2}
    pub fn hk_inner_product(
        &self,
        f: &SobolevFunction<P, Q, R, K>,
        g: &SobolevFunction<P, Q, R, K>,
    ) -> f64 {
        let (a, b) = self.domain.bounds_1d().unwrap_or((0.0, 1.0));
        let h = (b - a) / self.quadrature_points as f64;

        let mut sum = 0.0;

        for i in 0..self.quadrature_points {
            let x = a + (i as f64 + 0.5) * h;

            // L² part: ⟨f(x), g(x)⟩
            let fx = f.eval(&[x]);
            let gx = g.eval(&[x]);
            let fx_coeffs = fx.to_vec();
            let gx_coeffs = gx.to_vec();
            let l2_part: f64 = fx_coeffs
                .iter()
                .zip(gx_coeffs.iter())
                .map(|(a, b)| a * b)
                .sum();
            sum += l2_part;

            // Derivative parts
            let num_derivs = f.derivative_count().min(g.derivative_count());
            for j in 0..num_derivs {
                if let (Some(df), Some(dg)) =
                    (f.eval_derivative(j, &[x]), g.eval_derivative(j, &[x]))
                {
                    let df_coeffs = df.to_vec();
                    let dg_coeffs = dg.to_vec();
                    let deriv_part: f64 = df_coeffs
                        .iter()
                        .zip(dg_coeffs.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    sum += deriv_part;
                }
            }
        }

        sum * h
    }

    /// Compute the H^k norm.
    pub fn hk_norm(&self, f: &SobolevFunction<P, Q, R, K>) -> f64 {
        self.hk_inner_product(f, f).sqrt()
    }

    /// Compute the H^k seminorm (only includes derivatives of order k).
    pub fn hk_seminorm(&self, f: &SobolevFunction<P, Q, R, K>) -> f64 {
        let (a, b) = self.domain.bounds_1d().unwrap_or((0.0, 1.0));
        let h = (b - a) / self.quadrature_points as f64;

        let mut sum = 0.0;

        // Only the k-th derivative (or derivatives of order k)
        if f.derivative_count() > 0 {
            let last_deriv_idx = f.derivative_count() - 1;
            for i in 0..self.quadrature_points {
                let x = a + (i as f64 + 0.5) * h;
                if let Some(df) = f.eval_derivative(last_deriv_idx, &[x]) {
                    let df_coeffs = df.to_vec();
                    let norm_sq: f64 = df_coeffs.iter().map(|c| c * c).sum();
                    sum += norm_sq;
                }
            }
        }

        (sum * h).sqrt()
    }
}

/// H^1 Sobolev space (first-order derivatives in L^2).
pub type H1Space<const P: usize, const Q: usize, const R: usize> = SobolevSpace<P, Q, R, 1>;

/// H^2 Sobolev space (second-order derivatives in L^2).
pub type H2Space<const P: usize, const Q: usize, const R: usize> = SobolevSpace<P, Q, R, 2>;

/// Compute the Poincaré constant estimate for a domain.
///
/// For a bounded domain Ω, there exists C > 0 such that:
/// ||f||_{L^2} ≤ C ||∇f||_{L^2} for f ∈ H^1_0(Ω)
pub fn poincare_constant_estimate(domain: &Domain<f64>) -> f64 {
    match domain {
        Domain::Interval { a, b } => {
            // For [a,b], Poincaré constant is (b-a)/π
            (b - a) / std::f64::consts::PI
        }
        Domain::Rectangle { bounds } => {
            // For rectangles, use the smallest dimension
            bounds
                .iter()
                .map(|(a, b)| (b - a) / std::f64::consts::PI)
                .fold(f64::MAX, f64::min)
        }
        Domain::RealLine => f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobolev_space_creation() {
        let h1: H1Space<2, 0, 0> = H1Space::unit_interval();
        assert_eq!(h1.order(), 1);
    }

    #[test]
    fn test_sobolev_function() {
        // f(x) = x, f'(x) = 1
        let f: SobolevFunction<2, 0, 0, 1> = SobolevFunction::new(
            |x| Multivector::<2, 0, 0>::scalar(x[0]),
            vec![Arc::new(|_x| Multivector::<2, 0, 0>::scalar(1.0))],
            1,
        );

        let val = f.eval(&[0.5]);
        assert!((val.scalar_part() - 0.5).abs() < 1e-10);

        let deriv = f.eval_derivative(0, &[0.5]).unwrap();
        assert!((deriv.scalar_part() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hk_norm() {
        let h1: H1Space<2, 0, 0> = H1Space::unit_interval().with_quadrature_points(64);

        // f(x) = 0, f'(x) = 0 -> ||f||_{H^1} = 0
        let zero: SobolevFunction<2, 0, 0, 1> = SobolevFunction::new(
            |_| Multivector::<2, 0, 0>::zero(),
            vec![Arc::new(|_| Multivector::<2, 0, 0>::zero())],
            1,
        );

        let norm = h1.hk_norm(&zero);
        assert!(norm < 1e-10);
    }

    #[test]
    fn test_h1_inner_product() {
        let h1: H1Space<2, 0, 0> = H1Space::unit_interval().with_quadrature_points(64);

        // f(x) = x, f'(x) = 1
        let f: SobolevFunction<2, 0, 0, 1> = SobolevFunction::new(
            |x| Multivector::<2, 0, 0>::scalar(x[0]),
            vec![Arc::new(|_| Multivector::<2, 0, 0>::scalar(1.0))],
            1,
        );

        // ||f||²_{H^1} = ||f||²_{L^2} + ||f'||²_{L^2}
        //             = ∫₀¹ x² dx + ∫₀¹ 1 dx
        //             = 1/3 + 1 = 4/3
        let norm_sq = h1.hk_inner_product(&f, &f);
        assert!((norm_sq - 4.0 / 3.0).abs() < 0.1);
    }

    #[test]
    fn test_poincare_constant() {
        let domain = Domain::interval(0.0, 1.0);
        let c = poincare_constant_estimate(&domain);

        // For [0,1], Poincaré constant is 1/π ≈ 0.318
        assert!((c - 1.0 / std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_h2_space() {
        let h2: H2Space<2, 0, 0> = H2Space::unit_interval();
        assert_eq!(h2.order(), 2);
    }

    #[test]
    fn test_seminorm() {
        let h1: H1Space<2, 0, 0> = H1Space::unit_interval().with_quadrature_points(64);

        // f(x) = x, f'(x) = 1
        let f: SobolevFunction<2, 0, 0, 1> = SobolevFunction::new(
            |x| Multivector::<2, 0, 0>::scalar(x[0]),
            vec![Arc::new(|_| Multivector::<2, 0, 0>::scalar(1.0))],
            1,
        );

        // |f|_{H^1}² = ||f'||²_{L^2} = ∫₀¹ 1 dx = 1
        let seminorm_sq = h1.hk_seminorm(&f).powi(2);
        assert!((seminorm_sq - 1.0).abs() < 0.1);
    }
}
