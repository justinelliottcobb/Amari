//! Homology computation for simplicial complexes.
//!
//! Homology groups H_k measure "holes" in different dimensions:
//! - H_0 counts connected components
//! - H_1 counts 1-dimensional holes (loops)
//! - H_2 counts 2-dimensional holes (voids)
//! - etc.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::chain::BoundaryMap;
use crate::complex::SimplicialComplex;

/// Betti numbers β_k = dim(H_k).
///
/// Stored as a vector where betti[k] = β_k.
pub type BettiNumbers = Vec<usize>;

/// A homology group H_k.
#[derive(Clone, Debug)]
pub struct HomologyGroup {
    /// Dimension k of this homology group
    dimension: usize,
    /// Betti number (rank of the group)
    betti: usize,
    /// Generators (cycles that span the homology)
    generators: Vec<Vec<i64>>,
}

impl HomologyGroup {
    /// Create a new homology group.
    pub fn new(dimension: usize, betti: usize, generators: Vec<Vec<i64>>) -> Self {
        Self {
            dimension,
            betti,
            generators,
        }
    }

    /// Create a trivial homology group.
    pub fn trivial(dimension: usize) -> Self {
        Self {
            dimension,
            betti: 0,
            generators: Vec::new(),
        }
    }

    /// The dimension k of this group H_k.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// The Betti number (rank) of this group.
    pub fn betti(&self) -> usize {
        self.betti
    }

    /// Get the generators of this homology group.
    pub fn generators(&self) -> &[Vec<i64>] {
        &self.generators
    }

    /// Check if this group is trivial (H_k = 0).
    pub fn is_trivial(&self) -> bool {
        self.betti == 0
    }
}

/// Compute the homology groups of a simplicial complex.
///
/// Returns the Betti numbers β_0, β_1, ..., β_d where d is the dimension.
pub fn compute_homology(complex: &SimplicialComplex) -> BettiNumbers {
    let dim = complex.dimension();
    let mut betti = vec![0usize; dim + 1];

    // For each dimension k, compute β_k = dim(ker ∂_k) - dim(im ∂_{k+1})
    // = nullity(∂_k) - rank(∂_{k+1})

    // Precompute boundary maps
    let mut boundary_maps: Vec<BoundaryMap> = Vec::with_capacity(dim + 2);
    for k in 0..=dim + 1 {
        boundary_maps.push(complex.boundary_map(k));
    }

    for k in 0..=dim {
        let n_k = complex.simplex_count(k);
        if n_k == 0 {
            continue;
        }

        // dim(ker ∂_k) = n_k - rank(∂_k)
        let rank_dk = boundary_maps[k].rank();
        let kernel_dim = n_k.saturating_sub(rank_dk);

        // dim(im ∂_{k+1}) = rank(∂_{k+1})
        let image_dim = if k < dim {
            boundary_maps[k + 1].rank()
        } else {
            0
        };

        betti[k] = kernel_dim.saturating_sub(image_dim);
    }

    betti
}

/// Compute detailed homology information including generators.
pub fn compute_homology_groups(complex: &SimplicialComplex) -> Vec<HomologyGroup> {
    let betti = compute_homology(complex);

    betti
        .into_iter()
        .enumerate()
        .map(|(k, b)| {
            // TODO: Compute actual generators using Smith normal form
            // For now, just return the Betti number
            HomologyGroup::new(k, b, Vec::new())
        })
        .collect()
}

/// Verify ∂∂ = 0 for a simplicial complex.
///
/// This is a fundamental property of the boundary operator.
pub fn verify_boundary_squared_zero(complex: &SimplicialComplex) -> bool {
    let dim = complex.dimension();

    for k in 2..=dim {
        let d_k = complex.boundary_map(k);
        let d_k_minus_1 = complex.boundary_map(k - 1);

        // Check that ∂_{k-1} ∘ ∂_k = 0
        // For each basis element, apply both maps
        for col in 0..d_k.cols() {
            let mut chain = crate::chain::Chain::zero();
            chain.set_coefficient(col, 1);

            let once = d_k.apply(&chain);
            let twice = d_k_minus_1.apply(&once);

            if !twice.is_zero() {
                return false;
            }
        }
    }

    true
}

/// Compute the Euler characteristic using homology.
///
/// χ = Σ (-1)^k β_k
pub fn euler_characteristic_from_homology(betti: &BettiNumbers) -> i64 {
    betti
        .iter()
        .enumerate()
        .map(|(k, &b)| {
            let sign = if k % 2 == 0 { 1i64 } else { -1i64 };
            sign * (b as i64)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simplex::Simplex;

    #[test]
    fn test_point_homology() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0]));

        let betti = compute_homology(&complex);
        assert_eq!(betti[0], 1); // One connected component
    }

    #[test]
    fn test_edge_homology() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1]));

        let betti = compute_homology(&complex);
        assert_eq!(betti[0], 1); // One connected component
        assert_eq!(betti[1], 0); // No loops
    }

    #[test]
    fn test_triangle_filled_homology() {
        // Filled triangle
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        let betti = compute_homology(&complex);
        assert_eq!(betti[0], 1); // One connected component
        assert_eq!(betti[1], 0); // No holes (filled)
        assert_eq!(betti[2], 0); // No voids
    }

    #[test]
    fn test_triangle_boundary_homology() {
        // Triangle boundary (circle) - 3 edges, no fill
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1]));
        complex.add_simplex(Simplex::new(vec![1, 2]));
        complex.add_simplex(Simplex::new(vec![2, 0]));

        let betti = compute_homology(&complex);
        assert_eq!(betti[0], 1); // One connected component
        assert_eq!(betti[1], 1); // One hole
    }

    #[test]
    fn test_two_triangles_homology() {
        // Two separate triangles
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));
        complex.add_simplex(Simplex::new(vec![3, 4, 5]));

        let betti = compute_homology(&complex);
        assert_eq!(betti[0], 2); // Two connected components
    }

    #[test]
    fn test_tetrahedron_homology() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2, 3]));

        let betti = compute_homology(&complex);
        assert_eq!(betti[0], 1); // One connected component
        assert_eq!(betti[1], 0); // No loops
        assert_eq!(betti[2], 0); // No voids
        assert_eq!(betti[3], 0); // No 3-holes
    }

    #[test]
    fn test_hollow_tetrahedron_homology() {
        // Hollow tetrahedron (sphere) - 4 triangles, no 3-simplex
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));
        complex.add_simplex(Simplex::new(vec![0, 1, 3]));
        complex.add_simplex(Simplex::new(vec![0, 2, 3]));
        complex.add_simplex(Simplex::new(vec![1, 2, 3]));

        let betti = compute_homology(&complex);
        assert_eq!(betti[0], 1); // One connected component
        assert_eq!(betti[1], 0); // No loops
        assert_eq!(betti[2], 1); // One void (enclosed space)
    }

    #[test]
    fn test_boundary_squared_zero() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2, 3]));

        assert!(verify_boundary_squared_zero(&complex));
    }

    #[test]
    fn test_euler_poincare() {
        // For any complex, χ computed from simplices equals χ from homology
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));
        complex.add_simplex(Simplex::new(vec![1, 2, 3]));

        let chi_simplices = complex.euler_characteristic();
        let betti = compute_homology(&complex);
        let chi_homology = euler_characteristic_from_homology(&betti);

        assert_eq!(chi_simplices, chi_homology);
    }
}
