//! General multivector field implementation

use amari_core::Multivector;

/// A general multivector field F: ℝⁿ → Cl(p,q,r)
///
/// Represents a function that maps points in n-dimensional space to general multivectors
/// (not necessarily grade-1).
#[derive(Clone)]
pub struct MultivectorField<const P: usize, const Q: usize, const R: usize> {
    /// The function defining the field
    function: fn(&[f64]) -> Multivector<P, Q, R>,
    /// Domain dimension
    dim: usize,
}

impl<const P: usize, const Q: usize, const R: usize> MultivectorField<P, Q, R> {
    /// Create a new multivector field from a function
    pub fn new(function: fn(&[f64]) -> Multivector<P, Q, R>) -> Self {
        Self {
            function,
            dim: P + Q + R,
        }
    }

    /// Create a multivector field with explicit dimension
    pub fn with_dimension(function: fn(&[f64]) -> Multivector<P, Q, R>, dim: usize) -> Self {
        Self { function, dim }
    }

    /// Evaluate the multivector field at a point
    pub fn evaluate(&self, coords: &[f64]) -> Multivector<P, Q, R> {
        (self.function)(coords)
    }

    /// Get the domain dimension
    pub fn dimension(&self) -> usize {
        self.dim
    }
}

impl<const P: usize, const Q: usize, const R: usize> std::fmt::Debug for MultivectorField<P, Q, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultivectorField")
            .field("dim", &self.dim)
            .field("signature", &format!("Cl({},{},{})", P, Q, R))
            .finish()
    }
}
