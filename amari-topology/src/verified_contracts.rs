//! Formal verification contracts for topological computations.
//!
//! This module provides Creusot-style contracts for formally verifying the correctness
//! of topological operations. The contracts specify mathematical properties that must
//! hold for all implementations.
//!
//! # Verification Focus
//!
//! - **Boundary Operator**: ∂∂ = 0 (fundamental property of chain complexes)
//! - **Simplex Validity**: No duplicate vertices, proper orientation
//! - **Chain Arithmetic**: Linearity, commutativity of addition
//! - **Homology Invariants**: Euler-Poincaré formula, Betti number properties
//! - **Filtration Properties**: Monotonicity, face ordering
//!
//! # Contract Syntax
//!
//! Contracts are documented in comments using Creusot-style notation:
//! - `requires(P)`: Precondition P must hold before the operation
//! - `ensures(Q)`: Postcondition Q will hold after the operation
//! - `invariant(I)`: Invariant I is maintained throughout

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use core::marker::PhantomData;

use crate::chain::{BoundaryMap, Chain, ChainGroup};
use crate::complex::SimplicialComplex;
use crate::simplex::Simplex;

/// Verification marker for topology contracts.
#[derive(Debug, Clone, Copy)]
pub struct TopologyVerified;

// ============================================================================
// Verified Simplex
// ============================================================================

/// A simplex with formal verification contracts.
///
/// This wrapper ensures that all simplex invariants are maintained.
#[derive(Clone, Debug)]
pub struct VerifiedSimplex {
    inner: Simplex,
    _verification: PhantomData<TopologyVerified>,
}

impl VerifiedSimplex {
    /// Create a verified simplex from vertices.
    ///
    /// # Contracts
    /// - `ensures(result.vertices().len() == vertices.iter().collect::<BTreeSet<_>>().len())`
    ///   (no duplicate vertices after deduplication)
    /// - `ensures(result.vertices().windows(2).all(|w| w[0] < w[1]))`
    ///   (vertices are sorted)
    /// - `ensures(result.orientation().abs() == 1)`
    ///   (orientation is ±1)
    pub fn new(vertices: Vec<usize>) -> Self {
        Self {
            inner: Simplex::new(vertices),
            _verification: PhantomData,
        }
    }

    /// Get the inner simplex.
    pub fn inner(&self) -> &Simplex {
        &self.inner
    }

    /// Get the dimension.
    ///
    /// # Contracts
    /// - `ensures(result == self.vertices().len().saturating_sub(1))`
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Get the vertices.
    ///
    /// # Contracts
    /// - `ensures(result.windows(2).all(|w| w[0] < w[1]))` (sorted)
    pub fn vertices(&self) -> &[usize] {
        self.inner.vertices()
    }

    /// Get the orientation.
    ///
    /// # Contracts
    /// - `ensures(result == 1 || result == -1)`
    pub fn orientation(&self) -> i8 {
        self.inner.orientation()
    }

    /// Check if vertex is contained.
    ///
    /// # Contracts
    /// - `ensures(result == self.vertices().contains(&vertex))`
    pub fn contains_vertex(&self, vertex: usize) -> bool {
        self.inner.contains_vertex(vertex)
    }

    /// Get boundary faces with verified ∂∂ = 0 property.
    ///
    /// # Contracts
    /// - `ensures(result.len() == self.vertices().len())`
    /// - `ensures(result.iter().all(|(f, s)| f.dimension() == self.dimension() - 1))`
    /// - `ensures(result.iter().map(|(_, s)| *s as i64).sum::<i64>() == 0 || self.dimension() == 0)`
    ///   (signs alternate, key for ∂∂ = 0)
    pub fn boundary_faces(&self) -> Vec<(VerifiedSimplex, i8)> {
        self.inner
            .boundary_faces()
            .into_iter()
            .map(|(s, sign)| {
                (
                    VerifiedSimplex {
                        inner: s,
                        _verification: PhantomData,
                    },
                    sign,
                )
            })
            .collect()
    }
}

// ============================================================================
// Verified Chain
// ============================================================================

/// A chain with formal verification contracts.
#[derive(Clone, Debug)]
pub struct VerifiedChain {
    inner: Chain,
    _verification: PhantomData<TopologyVerified>,
}

impl VerifiedChain {
    /// Create the zero chain.
    ///
    /// # Contracts
    /// - `ensures(result.is_zero())`
    /// - `ensures(result.support_size() == 0)`
    pub fn zero() -> Self {
        Self {
            inner: Chain::zero(),
            _verification: PhantomData,
        }
    }

    /// Create a chain from a single simplex.
    ///
    /// # Contracts
    /// - `ensures(result.coefficient(index) == 1)`
    /// - `ensures(result.support_size() == 1)`
    pub fn from_simplex(index: usize) -> Self {
        Self {
            inner: Chain::from_simplex(index),
            _verification: PhantomData,
        }
    }

    /// Get coefficient at index.
    pub fn coefficient(&self, index: usize) -> i64 {
        self.inner.coefficient(index)
    }

    /// Check if zero chain.
    ///
    /// # Contracts
    /// - `ensures(result == self.support_size() == 0)`
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Add two chains.
    ///
    /// # Contracts
    /// - `ensures(forall i: result.coefficient(i) == self.coefficient(i) + other.coefficient(i))`
    /// - `ensures(self.add(other) == other.add(self))` // Commutativity
    pub fn add(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.add(&other.inner),
            _verification: PhantomData,
        }
    }

    /// Subtract two chains.
    ///
    /// # Contracts
    /// - `ensures(forall i: result.coefficient(i) == self.coefficient(i) - other.coefficient(i))`
    /// - `ensures(self.sub(self).is_zero())` // x - x = 0
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            inner: self.inner.sub(&other.inner),
            _verification: PhantomData,
        }
    }

    /// Scale chain by scalar.
    ///
    /// # Contracts
    /// - `ensures(forall i: result.coefficient(i) == scalar * self.coefficient(i))`
    /// - `ensures(self.scale(0).is_zero())`
    /// - `ensures(self.scale(1) == self)`
    pub fn scale(&self, scalar: i64) -> Self {
        Self {
            inner: self.inner.scale(scalar),
            _verification: PhantomData,
        }
    }

    /// Number of non-zero terms.
    pub fn support_size(&self) -> usize {
        self.inner.support_size()
    }
}

// ============================================================================
// Verified Boundary Map
// ============================================================================

/// A boundary map with formal verification of ∂∂ = 0.
#[derive(Clone, Debug)]
pub struct VerifiedBoundaryMap {
    inner: BoundaryMap,
    _verification: PhantomData<TopologyVerified>,
}

impl VerifiedBoundaryMap {
    /// Create a verified boundary map from chain groups.
    ///
    /// # Contracts
    /// - `ensures(result.rows() == codomain.rank())`
    /// - `ensures(result.cols() == domain.rank())`
    pub fn from_chain_groups(domain: &ChainGroup, codomain: &ChainGroup) -> Self {
        Self {
            inner: BoundaryMap::from_chain_groups(domain, codomain),
            _verification: PhantomData,
        }
    }

    /// Apply boundary map to chain.
    ///
    /// # Contracts
    /// - `ensures(result.dimension() == self.codomain_dimension())`
    ///   (maps k-chains to (k-1)-chains)
    pub fn apply(&self, chain: &VerifiedChain) -> VerifiedChain {
        VerifiedChain {
            inner: self.inner.apply(&chain.inner),
            _verification: PhantomData,
        }
    }

    /// Get number of rows.
    pub fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Get number of columns.
    pub fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Compute the rank (dimension of image).
    ///
    /// # Contracts
    /// - `ensures(result <= self.cols().min(self.rows()))`
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Compute kernel dimension.
    ///
    /// # Contracts
    /// - `ensures(result == self.cols() - self.rank())`
    ///   (rank-nullity theorem)
    pub fn kernel_dim(&self) -> usize {
        self.inner.kernel_dim()
    }

    /// Compute image dimension.
    ///
    /// # Contracts
    /// - `ensures(result == self.rank())`
    pub fn image_dim(&self) -> usize {
        self.inner.image_dim()
    }
}

// ============================================================================
// Boundary-Boundary = 0 Verification
// ============================================================================

/// Verify the fundamental property ∂∂ = 0.
///
/// This function checks that applying the boundary operator twice yields zero.
///
/// # Mathematical Background
///
/// For any chain complex, the composition of consecutive boundary maps is zero:
/// ∂_{k-1} ∘ ∂_k = 0
///
/// This is because each (k-2)-face appears exactly twice in ∂∂σ with opposite signs.
///
/// # Contracts
/// - `ensures(result == true implies forall c: d_{k-1}.apply(d_k.apply(c)).is_zero())`
pub fn verify_boundary_squared_zero(
    d_k: &VerifiedBoundaryMap,
    d_k_minus_1: &VerifiedBoundaryMap,
) -> bool {
    // For each basis element in domain of d_k
    for col in 0..d_k.cols() {
        let chain = VerifiedChain::from_simplex(col);
        let boundary = d_k.apply(&chain);
        let double_boundary = d_k_minus_1.apply(&boundary);

        if !double_boundary.is_zero() {
            return false;
        }
    }
    true
}

/// Verify the Euler-Poincaré formula.
///
/// # Mathematical Background
///
/// For a simplicial complex K:
/// χ(K) = Σ (-1)^k |K_k| = Σ (-1)^k β_k
///
/// where |K_k| is the number of k-simplices and β_k is the k-th Betti number.
///
/// # Contracts
/// - `ensures(result == true implies chi_simplicial == chi_homological)`
pub fn verify_euler_poincare(complex: &SimplicialComplex, betti: &[usize]) -> bool {
    let chi_simplicial = complex.euler_characteristic();

    let chi_homological: i64 = betti
        .iter()
        .enumerate()
        .map(|(k, &b)| {
            let sign = if k % 2 == 0 { 1i64 } else { -1i64 };
            sign * (b as i64)
        })
        .sum();

    chi_simplicial == chi_homological
}

/// Verify Betti number non-negativity.
///
/// # Contracts
/// - `ensures(result == true implies betti.iter().all(|&b| b >= 0))`
///   (Betti numbers are dimensions, hence non-negative)
pub fn verify_betti_nonnegative(betti: &[usize]) -> bool {
    // usize is always non-negative, so this is trivially true
    // but we include it for documentation purposes
    betti.iter().all(|&_b| true)
}

/// Verify β₀ counts connected components.
///
/// # Contracts
/// - `ensures(result == true implies beta_0 == num_components)`
pub fn verify_beta_0_counts_components(complex: &SimplicialComplex, beta_0: usize) -> bool {
    complex.connected_components() == beta_0
}

// ============================================================================
// Morse Inequality Verification
// ============================================================================

/// Verify the weak Morse inequalities.
///
/// For a Morse function on a manifold:
/// c_k ≥ β_k for all k
///
/// where c_k is the number of critical points of index k.
///
/// # Contracts
/// - `ensures(result == true implies forall k: critical_counts[k] >= betti[k])`
pub fn verify_weak_morse_inequalities(critical_counts: &[usize], betti: &[usize]) -> bool {
    critical_counts
        .iter()
        .zip(betti.iter())
        .all(|(&c, &b)| c >= b)
}

/// Verify the strong Morse inequality (Euler characteristic equality).
///
/// Σ (-1)^k c_k = Σ (-1)^k β_k = χ
///
/// # Contracts
/// - `ensures(result == true implies alternating_sum(critical_counts) == alternating_sum(betti))`
pub fn verify_strong_morse_inequality(critical_counts: &[usize], betti: &[usize]) -> bool {
    let chi_morse: i64 = critical_counts
        .iter()
        .enumerate()
        .map(|(k, &c)| {
            let sign = if k % 2 == 0 { 1i64 } else { -1i64 };
            sign * (c as i64)
        })
        .sum();

    let chi_betti: i64 = betti
        .iter()
        .enumerate()
        .map(|(k, &b)| {
            let sign = if k % 2 == 0 { 1i64 } else { -1i64 };
            sign * (b as i64)
        })
        .sum();

    chi_morse == chi_betti
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verified_simplex() {
        let s = VerifiedSimplex::new(vec![2, 0, 1]);

        // Vertices should be sorted
        assert_eq!(s.vertices(), &[0, 1, 2]);

        // Orientation should be ±1
        assert!(s.orientation() == 1 || s.orientation() == -1);
    }

    #[test]
    fn test_verified_chain_arithmetic() {
        let c1 = VerifiedChain::from_simplex(0);
        let c2 = VerifiedChain::from_simplex(1);

        // Commutativity
        let sum1 = c1.add(&c2);
        let sum2 = c2.add(&c1);
        assert_eq!(sum1.coefficient(0), sum2.coefficient(0));
        assert_eq!(sum1.coefficient(1), sum2.coefficient(1));

        // x - x = 0
        let diff = c1.sub(&c1);
        assert!(diff.is_zero());

        // Scaling by 0
        let zero_scaled = c1.scale(0);
        assert!(zero_scaled.is_zero());
    }

    #[test]
    fn test_boundary_squared_zero_triangle() {
        // Build chain groups for a triangle
        let triangle = Simplex::new(vec![0, 1, 2]);
        let edges = triangle.faces(1);
        let vertices = triangle.faces(0);

        let c2 = ChainGroup::new(vec![triangle]);
        let c1 = ChainGroup::new(edges);
        let c0 = ChainGroup::new(vertices);

        let d2 = VerifiedBoundaryMap::from_chain_groups(&c2, &c1);
        let d1 = VerifiedBoundaryMap::from_chain_groups(&c1, &c0);

        // Verify ∂∂ = 0
        assert!(verify_boundary_squared_zero(&d2, &d1));
    }

    #[test]
    fn test_euler_poincare_triangle() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        // Filled triangle: β₀ = 1, β₁ = 0
        let betti = vec![1, 0];

        assert!(verify_euler_poincare(&complex, &betti));
    }

    #[test]
    fn test_morse_inequalities() {
        // Torus: β₀ = 1, β₁ = 2, β₂ = 1
        let betti = vec![1, 2, 1];

        // Standard Morse function on torus: 1 min, 2 saddles, 1 max
        let critical_counts = vec![1, 2, 1];

        assert!(verify_weak_morse_inequalities(&critical_counts, &betti));
        assert!(verify_strong_morse_inequality(&critical_counts, &betti));
    }
}
