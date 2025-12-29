//! L² space of multivector-valued functions.
//!
//! This module provides the `MultivectorL2` type, representing the Hilbert space
//! of square-integrable functions with values in a Clifford algebra.
//!
//! # Mathematical Background
//!
//! L²(Ω, Cl(P,Q,R)) is the space of measurable functions f: Ω → Cl(P,Q,R) such that
//!
//! ∫_Ω ||f(x)||² dμ(x) < ∞
//!
//! where ||·|| is the norm on the Clifford algebra. This is a Hilbert space with
//! inner product:
//!
//! ⟨f, g⟩ = ∫_Ω ⟨f(x), g(x)⟩ dμ(x)

use crate::error::{FunctionalError, Result};
use crate::phantom::Complete;
use crate::space::traits::{
    BanachSpace, HilbertSpace, InnerProductSpace, NormedSpace, VectorSpace,
};
use amari_core::Multivector;
use core::marker::PhantomData;
use std::sync::Arc;

/// A domain of integration.
///
/// Represents the domain over which functions in L² are defined.
#[derive(Debug, Clone)]
pub enum Domain<T> {
    /// One-dimensional interval [a, b].
    Interval {
        /// Left endpoint of the interval.
        a: T,
        /// Right endpoint of the interval.
        b: T,
    },
    /// Rectangular domain in R^n.
    Rectangle {
        /// Bounds for each dimension as (min, max) pairs.
        bounds: Vec<(T, T)>,
    },
    /// The entire real line.
    RealLine,
}

impl Domain<f64> {
    /// Create an interval domain [a, b].
    pub fn interval(a: f64, b: f64) -> Self {
        Domain::Interval { a, b }
    }

    /// Create a rectangular domain from bounds.
    pub fn rectangle(bounds: Vec<(f64, f64)>) -> Self {
        Domain::Rectangle { bounds }
    }

    /// Get the bounds for a 1D domain.
    ///
    /// Returns `Some((a, b))` for an interval, or `None` for other domains.
    pub fn bounds_1d(&self) -> Option<(f64, f64)> {
        match self {
            Domain::Interval { a, b } => Some((*a, *b)),
            Domain::Rectangle { bounds } if bounds.len() == 1 => Some(bounds[0]),
            _ => None,
        }
    }

    /// Get the dimension of the domain.
    pub fn dimension(&self) -> Option<usize> {
        match self {
            Domain::Interval { .. } => Some(1),
            Domain::Rectangle { bounds } => Some(bounds.len()),
            Domain::RealLine => Some(1),
        }
    }
}

/// A square-integrable multivector-valued function.
///
/// This represents an element of L²(Ω, Cl(P,Q,R)) - a function from
/// domain Ω to the Clifford algebra that is square-integrable.
#[derive(Clone)]
pub struct L2Function<const P: usize, const Q: usize, const R: usize> {
    /// The function represented as a boxed closure.
    func: Arc<dyn Fn(&[f64]) -> Multivector<P, Q, R> + Send + Sync>,
    /// Cached L² norm (computed lazily).
    cached_norm: Option<f64>,
}

impl<const P: usize, const Q: usize, const R: usize> std::fmt::Debug for L2Function<P, Q, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("L2Function")
            .field("signature", &(P, Q, R))
            .field("cached_norm", &self.cached_norm)
            .finish()
    }
}

impl<const P: usize, const Q: usize, const R: usize> L2Function<P, Q, R> {
    /// Create a new L² function from a closure.
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&[f64]) -> Multivector<P, Q, R> + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(f),
            cached_norm: None,
        }
    }

    /// Evaluate the function at a point.
    pub fn eval(&self, point: &[f64]) -> Multivector<P, Q, R> {
        (self.func)(point)
    }

    /// Create the zero function.
    pub fn zero_function() -> Self {
        Self::new(|_| Multivector::<P, Q, R>::zero())
    }

    /// Create a constant function.
    pub fn constant(value: Multivector<P, Q, R>) -> Self {
        Self::new(move |_| value.clone())
    }
}

/// The L² space of multivector-valued functions.
///
/// This is a Hilbert space with the L² inner product.
///
/// # Type Parameters
///
/// * `P` - Number of positive signature basis vectors
/// * `Q` - Number of negative signature basis vectors
/// * `R` - Number of zero signature basis vectors
#[derive(Clone)]
pub struct MultivectorL2<const P: usize, const Q: usize, const R: usize> {
    /// The domain of integration.
    domain: Domain<f64>,
    /// Number of quadrature points per dimension for numerical integration.
    quadrature_points: usize,
    _phantom: PhantomData<Multivector<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize> std::fmt::Debug for MultivectorL2<P, Q, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultivectorL2")
            .field("signature", &(P, Q, R))
            .field("domain", &self.domain)
            .field("quadrature_points", &self.quadrature_points)
            .finish()
    }
}

impl<const P: usize, const Q: usize, const R: usize> MultivectorL2<P, Q, R> {
    /// The dimension of the Clifford algebra codomain.
    pub const CODOMAIN_DIM: usize = 1 << (P + Q + R);

    /// Create a new L² space over a domain.
    pub fn new(domain: Domain<f64>) -> Self {
        Self {
            domain,
            quadrature_points: 32, // Default quadrature resolution
            _phantom: PhantomData,
        }
    }

    /// Create an L² space over the unit interval [0, 1].
    pub fn unit_interval() -> Self {
        Self::new(Domain::interval(0.0, 1.0))
    }

    /// Create an L² space over [a, b].
    pub fn interval(a: f64, b: f64) -> Self {
        Self::new(Domain::interval(a, b))
    }

    /// Set the number of quadrature points for numerical integration.
    pub fn with_quadrature_points(mut self, n: usize) -> Self {
        self.quadrature_points = n;
        self
    }

    /// Get the domain of the space.
    pub fn domain(&self) -> &Domain<f64> {
        &self.domain
    }

    /// Get the signature of the Clifford algebra.
    pub fn signature(&self) -> (usize, usize, usize) {
        (P, Q, R)
    }

    /// Compute a numerical integral over the domain using Gauss-Legendre quadrature.
    fn integrate<F>(&self, f: F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        // For 1D domains, use Gauss-Legendre quadrature
        let (a, b) = self.domain.bounds_1d().unwrap_or((0.0, 1.0));
        gauss_legendre_integrate(&f, a, b, self.quadrature_points)
    }

    /// Compute the L² inner product of two functions.
    pub fn l2_inner_product(&self, f: &L2Function<P, Q, R>, g: &L2Function<P, Q, R>) -> f64 {
        self.integrate(|x| {
            let fx = f.eval(x);
            let gx = g.eval(x);
            // Inner product on Cl(P,Q,R) is the Euclidean inner product on coefficients
            fx.to_vec()
                .iter()
                .zip(gx.to_vec().iter())
                .map(|(a, b)| a * b)
                .sum()
        })
    }

    /// Compute the L² norm of a function.
    pub fn l2_norm(&self, f: &L2Function<P, Q, R>) -> f64 {
        self.l2_inner_product(f, f).sqrt()
    }
}

/// Gauss-Legendre quadrature for numerical integration.
fn gauss_legendre_integrate<F>(f: &F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    // Use simple trapezoidal rule for now - can be upgraded to true Gauss-Legendre
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(&[a]) + f(&[b]));
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += f(&[x]);
    }
    sum * h
}

impl<const P: usize, const Q: usize, const R: usize> VectorSpace<L2Function<P, Q, R>, f64>
    for MultivectorL2<P, Q, R>
{
    fn add(&self, f: &L2Function<P, Q, R>, g: &L2Function<P, Q, R>) -> L2Function<P, Q, R> {
        let f_clone = f.func.clone();
        let g_clone = g.func.clone();
        L2Function::new(move |x| f_clone(x).add(&g_clone(x)))
    }

    fn sub(&self, f: &L2Function<P, Q, R>, g: &L2Function<P, Q, R>) -> L2Function<P, Q, R> {
        let f_clone = f.func.clone();
        let g_clone = g.func.clone();
        L2Function::new(move |x| &f_clone(x) - &g_clone(x))
    }

    fn scale(&self, scalar: f64, f: &L2Function<P, Q, R>) -> L2Function<P, Q, R> {
        let f_clone = f.func.clone();
        L2Function::new(move |x| &f_clone(x) * scalar)
    }

    fn zero(&self) -> L2Function<P, Q, R> {
        L2Function::zero_function()
    }

    fn dimension(&self) -> Option<usize> {
        // L² is infinite-dimensional
        None
    }
}

impl<const P: usize, const Q: usize, const R: usize> NormedSpace<L2Function<P, Q, R>, f64>
    for MultivectorL2<P, Q, R>
{
    fn norm(&self, f: &L2Function<P, Q, R>) -> f64 {
        self.l2_norm(f)
    }

    fn normalize(&self, f: &L2Function<P, Q, R>) -> Option<L2Function<P, Q, R>> {
        let n = self.norm(f);
        if n < 1e-15 {
            None
        } else {
            Some(self.scale(1.0 / n, f))
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> InnerProductSpace<L2Function<P, Q, R>, f64>
    for MultivectorL2<P, Q, R>
{
    fn inner_product(&self, f: &L2Function<P, Q, R>, g: &L2Function<P, Q, R>) -> f64 {
        self.l2_inner_product(f, g)
    }

    fn project(&self, f: &L2Function<P, Q, R>, g: &L2Function<P, Q, R>) -> L2Function<P, Q, R> {
        let ip_fg = self.inner_product(f, g);
        let ip_gg = self.inner_product(g, g);
        if ip_gg.abs() < 1e-15 {
            self.zero()
        } else {
            self.scale(ip_fg / ip_gg, g)
        }
    }

    fn gram_schmidt(&self, functions: &[L2Function<P, Q, R>]) -> Vec<L2Function<P, Q, R>> {
        let mut orthonormal = Vec::new();
        for f in functions {
            let mut u = f.clone();
            for q in &orthonormal {
                let proj = self.project(&u, q);
                u = self.sub(&u, &proj);
            }
            if let Some(normalized) = self.normalize(&u) {
                orthonormal.push(normalized);
            }
        }
        orthonormal
    }
}

impl<const P: usize, const Q: usize, const R: usize> BanachSpace<L2Function<P, Q, R>, f64, Complete>
    for MultivectorL2<P, Q, R>
{
    fn is_cauchy_sequence(&self, sequence: &[L2Function<P, Q, R>], tolerance: f64) -> bool {
        if sequence.len() < 2 {
            return true;
        }

        let n = sequence.len();
        for i in (n.saturating_sub(5))..n {
            for j in (i + 1)..n {
                if self.distance(&sequence[i], &sequence[j]) > tolerance {
                    return false;
                }
            }
        }
        true
    }

    fn sequence_limit(
        &self,
        sequence: &[L2Function<P, Q, R>],
        tolerance: f64,
    ) -> Result<L2Function<P, Q, R>> {
        if sequence.is_empty() {
            return Err(FunctionalError::convergence_error(
                0,
                "Empty sequence has no limit",
            ));
        }

        if !self.is_cauchy_sequence(sequence, tolerance) {
            return Err(FunctionalError::convergence_error(
                sequence.len(),
                "Sequence is not Cauchy",
            ));
        }

        // Return the last element as the limit approximation
        Ok(sequence.last().unwrap().clone())
    }
}

impl<const P: usize, const Q: usize, const R: usize>
    HilbertSpace<L2Function<P, Q, R>, f64, Complete> for MultivectorL2<P, Q, R>
{
    fn riesz_representative<F>(&self, _functional: F) -> Result<L2Function<P, Q, R>>
    where
        F: Fn(&L2Function<P, Q, R>) -> f64,
    {
        // In infinite dimensions, finding the Riesz representative requires
        // solving an integral equation. For now, return an error.
        Err(FunctionalError::not_complete(
            "Riesz representative computation not implemented for infinite-dimensional L² spaces",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_space_creation() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();
        assert_eq!(l2.signature(), (2, 0, 0));
        assert_eq!(l2.domain().bounds_1d(), Some((0.0, 1.0)));
    }

    #[test]
    fn test_l2_function_evaluation() {
        let f = L2Function::<2, 0, 0>::new(|x| {
            Multivector::<2, 0, 0>::scalar(x[0]) // scalar part = x
        });

        let result = f.eval(&[0.5]);
        assert!((result.scalar_part() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_zero_function() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();
        let zero = l2.zero();

        let norm = l2.norm(&zero);
        assert!(norm < 1e-10);
    }

    #[test]
    fn test_constant_function() {
        let mv = Multivector::<2, 0, 0>::scalar(1.0);
        let f = L2Function::constant(mv);

        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();

        // ||1||² = ∫₀¹ 1 dx = 1, so ||1|| = 1
        let norm = l2.norm(&f);
        assert!((norm - 1.0).abs() < 0.01); // Numerical integration tolerance
    }

    #[test]
    fn test_vector_space_operations() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();

        let f = L2Function::new(|x| Multivector::<2, 0, 0>::scalar(x[0]));
        let g = L2Function::new(|x| Multivector::<2, 0, 0>::scalar(1.0 - x[0]));

        // f + g should be constantly 1
        let sum = l2.add(&f, &g);
        let result = sum.eval(&[0.5]);
        assert!((result.scalar_part() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inner_product_orthogonality() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval().with_quadrature_points(64);

        // sin(πx) and sin(2πx) should be orthogonal on [0,1]
        let f = L2Function::new(|x| {
            let val = (std::f64::consts::PI * x[0]).sin();
            Multivector::<2, 0, 0>::scalar(val)
        });
        let g = L2Function::new(|x| {
            let val = (2.0 * std::f64::consts::PI * x[0]).sin();
            Multivector::<2, 0, 0>::scalar(val)
        });

        let ip = l2.inner_product(&f, &g);
        assert!(ip.abs() < 0.1); // Numerical tolerance
    }

    #[test]
    fn test_l2_norm_squared() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval().with_quadrature_points(64);

        // ||sin(πx)||² = ∫₀¹ sin²(πx) dx = 1/2
        let f = L2Function::new(|x| {
            let val = (std::f64::consts::PI * x[0]).sin();
            Multivector::<2, 0, 0>::scalar(val)
        });

        let ip = l2.inner_product(&f, &f);
        assert!((ip - 0.5).abs() < 0.05); // Numerical tolerance
    }

    #[test]
    fn test_scaling() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();

        let f = L2Function::new(|x| Multivector::<2, 0, 0>::scalar(x[0]));

        let scaled = l2.scale(2.0, &f);
        let result = scaled.eval(&[0.5]);
        assert!((result.scalar_part() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalization() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval().with_quadrature_points(64);

        let f = L2Function::new(|x| Multivector::<2, 0, 0>::scalar(x[0]));

        let normalized = l2.normalize(&f).unwrap();
        let norm = l2.norm(&normalized);
        assert!((norm - 1.0).abs() < 0.05); // Numerical tolerance
    }

    #[test]
    fn test_infinite_dimension() {
        let l2: MultivectorL2<2, 0, 0> = MultivectorL2::unit_interval();
        assert_eq!(l2.dimension(), None);
    }
}
