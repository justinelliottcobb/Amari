//! Simplicial complexes for algebraic topology.
//!
//! A simplicial complex is a collection of simplices closed under taking faces.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, collections::BTreeSet, vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::{BTreeMap, BTreeSet};

use crate::chain::{BoundaryMap, ChainGroup};
use crate::homology::{compute_homology, BettiNumbers};
use crate::simplex::Simplex;

/// An abstract simplicial complex.
///
/// A simplicial complex K is a set of simplices such that:
/// 1. Every face of a simplex in K is also in K
/// 2. The intersection of any two simplices in K is a face of both
///
/// This implementation maintains the closure property automatically.
#[derive(Clone, Debug, Default)]
pub struct SimplicialComplex {
    /// Simplices organized by dimension
    simplices_by_dim: BTreeMap<usize, BTreeSet<Simplex>>,
    /// Maximum dimension of any simplex
    max_dim: usize,
}

impl SimplicialComplex {
    /// Create an empty simplicial complex.
    pub fn new() -> Self {
        Self {
            simplices_by_dim: BTreeMap::new(),
            max_dim: 0,
        }
    }

    /// Add a simplex and all its faces to the complex.
    ///
    /// This maintains the closure property of simplicial complexes.
    pub fn add_simplex(&mut self, simplex: Simplex) {
        let dim = simplex.dimension();
        self.max_dim = self.max_dim.max(dim);

        // Add the simplex itself
        self.simplices_by_dim
            .entry(dim)
            .or_default()
            .insert(simplex.clone());

        // Add all faces (closure property)
        for k in 0..dim {
            for face in simplex.faces(k) {
                self.simplices_by_dim.entry(k).or_default().insert(face);
            }
        }
    }

    /// Remove a simplex and all simplices that have it as a face.
    ///
    /// This maintains the property that no simplex has a missing face.
    pub fn remove_simplex(&mut self, simplex: &Simplex) {
        let dim = simplex.dimension();

        // Remove from this dimension
        if let Some(set) = self.simplices_by_dim.get_mut(&dim) {
            set.remove(simplex);
        }

        // Remove all higher-dimensional simplices that contain this as a face
        for d in (dim + 1)..=self.max_dim {
            if let Some(set) = self.simplices_by_dim.get_mut(&d) {
                set.retain(|s| !simplex.is_face_of(s));
            }
        }

        // Update max_dim if needed
        while self.max_dim > 0 && self.simplex_count(self.max_dim) == 0 {
            self.max_dim -= 1;
        }
    }

    /// Check if the complex contains a simplex.
    pub fn contains(&self, simplex: &Simplex) -> bool {
        self.simplices_by_dim
            .get(&simplex.dimension())
            .map(|set| set.contains(simplex))
            .unwrap_or(false)
    }

    /// Get all simplices of a given dimension.
    pub fn simplices(&self, dim: usize) -> impl Iterator<Item = &Simplex> {
        self.simplices_by_dim
            .get(&dim)
            .into_iter()
            .flat_map(|set| set.iter())
    }

    /// Get all simplices in the complex.
    pub fn all_simplices(&self) -> impl Iterator<Item = &Simplex> {
        self.simplices_by_dim.values().flat_map(|set| set.iter())
    }

    /// Count simplices of a given dimension.
    pub fn simplex_count(&self, dim: usize) -> usize {
        self.simplices_by_dim
            .get(&dim)
            .map(|set| set.len())
            .unwrap_or(0)
    }

    /// Total number of simplices in the complex.
    pub fn total_simplex_count(&self) -> usize {
        self.simplices_by_dim.values().map(|set| set.len()).sum()
    }

    /// Maximum dimension of any simplex in the complex.
    pub fn dimension(&self) -> usize {
        self.max_dim
    }

    /// Number of vertices (0-simplices).
    pub fn vertex_count(&self) -> usize {
        self.simplex_count(0)
    }

    /// Number of edges (1-simplices).
    pub fn edge_count(&self) -> usize {
        self.simplex_count(1)
    }

    /// Get the chain group for dimension k.
    pub fn chain_group(&self, k: usize) -> ChainGroup {
        let simplices: Vec<Simplex> = self.simplices(k).cloned().collect();
        ChainGroup::new(simplices)
    }

    /// Get the boundary map ∂_k: C_k → C_{k-1}.
    pub fn boundary_map(&self, k: usize) -> BoundaryMap {
        if k == 0 {
            return BoundaryMap::zero(self.simplex_count(0), 0);
        }

        let domain = self.chain_group(k);
        let codomain = self.chain_group(k - 1);

        BoundaryMap::from_chain_groups(&domain, &codomain)
    }

    /// Compute the Betti numbers of the complex.
    ///
    /// β_k = dim(H_k) = dim(ker(∂_k)) - dim(im(∂_{k+1}))
    pub fn betti_numbers(&self) -> BettiNumbers {
        compute_homology(self)
    }

    /// Compute the Euler characteristic.
    ///
    /// χ = Σ (-1)^k · (number of k-simplices)
    ///   = β_0 - β_1 + β_2 - ... (alternating sum of Betti numbers)
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi = 0i64;
        for k in 0..=self.max_dim {
            let count = self.simplex_count(k) as i64;
            if k % 2 == 0 {
                chi += count;
            } else {
                chi -= count;
            }
        }
        chi
    }

    /// Get the f-vector (face count vector).
    ///
    /// f = (f_0, f_1, ..., f_d) where f_k is the number of k-simplices.
    pub fn f_vector(&self) -> Vec<usize> {
        (0..=self.max_dim).map(|k| self.simplex_count(k)).collect()
    }

    /// Total number of simplices across all dimensions.
    pub fn total_simplices(&self) -> usize {
        self.simplices_by_dim.values().map(|set| set.len()).sum()
    }

    /// Count the number of connected components.
    ///
    /// Uses union-find to determine connectivity via edges.
    /// Returns 0 for an empty complex.
    pub fn connected_components(&self) -> usize {
        if self.vertex_count() == 0 {
            return 0;
        }

        let vertices: Vec<usize> = self
            .simplices(0)
            .flat_map(|s| s.vertices().iter().copied())
            .collect();

        if vertices.is_empty() {
            return 0;
        }

        let max_vertex = *vertices.iter().max().unwrap_or(&0);
        let mut parent: Vec<usize> = (0..=max_vertex).collect();
        let mut rank = vec![0usize; max_vertex + 1];

        fn find(parent: &mut [usize], x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }

        fn union(parent: &mut [usize], rank: &mut [usize], x: usize, y: usize) {
            let px = find(parent, x);
            let py = find(parent, y);
            if px != py {
                if rank[px] < rank[py] {
                    parent[px] = py;
                } else if rank[px] > rank[py] {
                    parent[py] = px;
                } else {
                    parent[py] = px;
                    rank[px] += 1;
                }
            }
        }

        // Unite vertices connected by edges
        for edge in self.simplices(1) {
            let verts = edge.vertices();
            if verts.len() >= 2 {
                union(&mut parent, &mut rank, verts[0], verts[1]);
            }
        }

        // Count distinct roots among vertices
        let mut roots = BTreeSet::new();
        for &v in &vertices {
            roots.insert(find(&mut parent, v));
        }
        roots.len()
    }

    /// Check if the complex is connected.
    ///
    /// Uses union-find to determine connectivity via edges.
    pub fn is_connected(&self) -> bool {
        self.connected_components() <= 1
    }

    /// Create the 1-skeleton (vertices and edges only).
    pub fn skeleton_1(&self) -> SimplicialComplex {
        let mut skeleton = SimplicialComplex::new();
        for simplex in self.simplices(0) {
            skeleton.add_simplex(simplex.clone());
        }
        for simplex in self.simplices(1) {
            // Add edge without closure (vertices already added)
            skeleton
                .simplices_by_dim
                .entry(1)
                .or_default()
                .insert(simplex.clone());
        }
        skeleton.max_dim = 1.min(self.max_dim);
        skeleton
    }

    /// Create the k-skeleton (all simplices of dimension ≤ k).
    pub fn skeleton(&self, k: usize) -> SimplicialComplex {
        let mut skeleton = SimplicialComplex::new();
        for dim in 0..=k.min(self.max_dim) {
            for simplex in self.simplices(dim) {
                skeleton
                    .simplices_by_dim
                    .entry(dim)
                    .or_default()
                    .insert(simplex.clone());
            }
        }
        skeleton.max_dim = k.min(self.max_dim);
        skeleton
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_complex() {
        let complex = SimplicialComplex::new();
        assert_eq!(complex.total_simplex_count(), 0);
        assert_eq!(complex.dimension(), 0);
    }

    #[test]
    fn test_add_simplex_closure() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        // Should have triangle + 3 edges + 3 vertices
        assert_eq!(complex.simplex_count(2), 1);
        assert_eq!(complex.simplex_count(1), 3);
        assert_eq!(complex.simplex_count(0), 3);
    }

    #[test]
    fn test_euler_characteristic_triangle() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        // χ = V - E + F = 3 - 3 + 1 = 1
        assert_eq!(complex.euler_characteristic(), 1);
    }

    #[test]
    fn test_euler_characteristic_tetrahedron() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2, 3]));

        // χ = V - E + F - T = 4 - 6 + 4 - 1 = 1
        assert_eq!(complex.euler_characteristic(), 1);
    }

    #[test]
    fn test_connectivity() {
        // Connected: triangle
        let mut connected = SimplicialComplex::new();
        connected.add_simplex(Simplex::new(vec![0, 1, 2]));
        assert!(connected.is_connected());

        // Disconnected: two separate edges
        let mut disconnected = SimplicialComplex::new();
        disconnected.add_simplex(Simplex::new(vec![0, 1]));
        disconnected.add_simplex(Simplex::new(vec![2, 3]));
        assert!(!disconnected.is_connected());
    }

    #[test]
    fn test_f_vector() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        let f = complex.f_vector();
        assert_eq!(f, vec![3, 3, 1]); // 3 vertices, 3 edges, 1 triangle
    }

    #[test]
    fn test_skeleton() {
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2, 3])); // Tetrahedron

        let skel1 = complex.skeleton(1);
        assert_eq!(skel1.dimension(), 1);
        assert_eq!(skel1.simplex_count(0), 4);
        assert_eq!(skel1.simplex_count(1), 6);
        assert_eq!(skel1.simplex_count(2), 0);
    }
}
