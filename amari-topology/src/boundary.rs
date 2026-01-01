//! Manifold boundary detection and characterization.
//!
//! This module provides tools for detecting and analyzing the boundaries
//! of manifolds represented as simplicial complexes.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::complex::SimplicialComplex;
use crate::simplex::Simplex;

/// A component of a manifold boundary.
#[derive(Clone, Debug)]
pub struct BoundaryComponent {
    /// The simplices forming this boundary component
    pub simplices: Vec<Simplex>,
    /// Whether this component is oriented
    pub oriented: bool,
    /// Euler characteristic of this boundary component
    pub euler_characteristic: i64,
}

impl BoundaryComponent {
    /// Create a new boundary component.
    pub fn new(simplices: Vec<Simplex>) -> Self {
        let mut complex = SimplicialComplex::new();
        for s in &simplices {
            complex.add_simplex(s.clone());
        }
        let euler_characteristic = complex.euler_characteristic();

        Self {
            simplices,
            oriented: true,
            euler_characteristic,
        }
    }

    /// Number of simplices in this boundary component.
    pub fn len(&self) -> usize {
        self.simplices.len()
    }

    /// Check if this component is empty.
    pub fn is_empty(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Get the dimension of this boundary component.
    pub fn dimension(&self) -> usize {
        self.simplices
            .iter()
            .map(|s| s.dimension())
            .max()
            .unwrap_or(0)
    }
}

/// An oriented boundary with consistent orientation.
#[derive(Clone, Debug)]
pub struct OrientedBoundary {
    /// Simplices with their orientations
    pub simplices: Vec<(Simplex, i8)>,
}

impl OrientedBoundary {
    /// Create a new empty oriented boundary.
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
        }
    }

    /// Add a simplex with orientation.
    pub fn add(&mut self, simplex: Simplex, orientation: i8) {
        // Check if already present
        for (s, o) in &mut self.simplices {
            if s == &simplex {
                *o += orientation;
                return;
            }
        }
        self.simplices.push((simplex, orientation));
    }

    /// Remove zero-coefficient simplices.
    pub fn simplify(&mut self) {
        self.simplices.retain(|(_, o)| *o != 0);
    }

    /// Check if this is the zero boundary.
    pub fn is_zero(&self) -> bool {
        self.simplices.iter().all(|(_, o)| *o == 0)
    }

    /// Get the number of non-zero terms.
    pub fn len(&self) -> usize {
        self.simplices.iter().filter(|(_, o)| *o != 0).count()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for OrientedBoundary {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents the boundary structure of a manifold.
#[derive(Clone, Debug)]
pub struct ManifoldBoundary {
    /// Connected components of the boundary
    pub components: Vec<BoundaryComponent>,
    /// Dimension of the manifold
    pub manifold_dimension: usize,
}

impl ManifoldBoundary {
    /// Compute the boundary of a simplicial complex representing a manifold.
    ///
    /// The boundary consists of (n-1)-simplices that are faces of exactly
    /// one n-simplex.
    pub fn compute(complex: &SimplicialComplex) -> Self {
        let dim = complex.dimension();
        if dim == 0 {
            return Self {
                components: Vec::new(),
                manifold_dimension: 0,
            };
        }

        // Find boundary (n-1)-simplices
        // A simplex is on the boundary if it's a face of exactly one n-simplex
        let mut face_count = std::collections::HashMap::new();

        for top_simplex in complex.simplices(dim) {
            for (face, _) in top_simplex.boundary_faces() {
                *face_count.entry(face).or_insert(0) += 1;
            }
        }

        // Boundary simplices are those with count = 1
        let boundary_simplices: Vec<Simplex> = face_count
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(simplex, _)| simplex)
            .collect();

        if boundary_simplices.is_empty() {
            return Self {
                components: Vec::new(),
                manifold_dimension: dim,
            };
        }

        // Group into connected components
        let components = Self::find_connected_components(&boundary_simplices);

        Self {
            components,
            manifold_dimension: dim,
        }
    }

    /// Find connected components of boundary simplices.
    fn find_connected_components(simplices: &[Simplex]) -> Vec<BoundaryComponent> {
        if simplices.is_empty() {
            return Vec::new();
        }

        // Build adjacency based on shared vertices
        let n = simplices.len();
        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }

            // BFS to find connected component
            let mut component_simplices = Vec::new();
            let mut queue = vec![start];
            visited[start] = true;

            while let Some(idx) = queue.pop() {
                component_simplices.push(simplices[idx].clone());

                // Find adjacent simplices (share a vertex)
                for (other_idx, other) in simplices.iter().enumerate() {
                    if !visited[other_idx] {
                        let shares_vertex = simplices[idx]
                            .vertices()
                            .iter()
                            .any(|v| other.contains_vertex(*v));

                        if shares_vertex {
                            visited[other_idx] = true;
                            queue.push(other_idx);
                        }
                    }
                }
            }

            components.push(BoundaryComponent::new(component_simplices));
        }

        components
    }

    /// Check if the manifold is closed (has no boundary).
    pub fn is_closed(&self) -> bool {
        self.components.is_empty()
    }

    /// Total number of boundary simplices.
    pub fn total_simplices(&self) -> usize {
        self.components.iter().map(|c| c.len()).sum()
    }

    /// Number of boundary components.
    pub fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Get the Euler characteristic of the total boundary.
    pub fn euler_characteristic(&self) -> i64 {
        self.components.iter().map(|c| c.euler_characteristic).sum()
    }
}

/// Check if a simplicial complex represents a valid manifold.
///
/// A simplicial n-manifold requires:
/// - Every (n-1)-simplex is a face of exactly 1 or 2 n-simplices
/// - The link of every vertex is a (n-1)-sphere or (n-1)-ball
pub fn is_manifold(complex: &SimplicialComplex) -> bool {
    let dim = complex.dimension();
    if dim == 0 {
        return true;
    }

    // Check that every (n-1)-simplex is a face of 1 or 2 n-simplices
    let mut face_count = std::collections::HashMap::new();

    for top_simplex in complex.simplices(dim) {
        for (face, _) in top_simplex.boundary_faces() {
            *face_count.entry(face).or_insert(0) += 1;
        }
    }

    for count in face_count.values() {
        if *count != 1 && *count != 2 {
            return false;
        }
    }

    true
}

/// Check if a simplicial complex is orientable.
///
/// A manifold is orientable if we can consistently orient all top-dimensional
/// simplices such that shared faces have opposite orientations.
pub fn is_orientable(complex: &SimplicialComplex) -> bool {
    let dim = complex.dimension();
    if dim == 0 {
        return true;
    }

    let simplices: Vec<Simplex> = complex.simplices(dim).cloned().collect();
    if simplices.is_empty() {
        return true;
    }

    // Try to consistently orient all simplices
    let n = simplices.len();
    let mut orientations = vec![0i8; n]; // 0 = unassigned, 1 or -1
    let mut queue = vec![0usize];
    orientations[0] = 1;

    while let Some(idx) = queue.pop() {
        let current = &simplices[idx];
        let current_orient = orientations[idx];

        // Find neighbors (share a face)
        for (other_idx, other) in simplices.iter().enumerate() {
            if orientations[other_idx] != 0 {
                continue;
            }

            // Check if they share a face
            for (face, sign1) in current.boundary_faces() {
                for (other_face, sign2) in other.boundary_faces() {
                    if face == other_face {
                        // Shared face: orientations should induce opposite signs
                        let required_orient = -current_orient * sign1 * sign2;

                        if orientations[other_idx] == 0 {
                            orientations[other_idx] = required_orient;
                            queue.push(other_idx);
                        } else if orientations[other_idx] != required_orient {
                            return false;
                        }
                    }
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_boundary() {
        // Single triangle - boundary is the 3 edges
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));

        let boundary = ManifoldBoundary::compute(&complex);

        // Triangle has no boundary when considered as 2D
        // Wait - a single filled triangle has edges as boundary
        // Actually for a 2-manifold, we look at which 1-simplices are faces of exactly one 2-simplex
        // All 3 edges of the triangle are faces of exactly 1 triangle
        assert!(!boundary.is_closed());
        assert_eq!(boundary.total_simplices(), 3);
    }

    #[test]
    fn test_two_triangles_shared_edge() {
        // Two triangles sharing an edge
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));
        complex.add_simplex(Simplex::new(vec![0, 1, 3]));

        let boundary = ManifoldBoundary::compute(&complex);

        // The shared edge [0,1] is a face of 2 triangles (interior)
        // The other 4 edges are boundary
        assert_eq!(boundary.total_simplices(), 4);
    }

    #[test]
    fn test_is_manifold() {
        // Single triangle is a manifold with boundary
        let mut triangle = SimplicialComplex::new();
        triangle.add_simplex(Simplex::new(vec![0, 1, 2]));
        assert!(is_manifold(&triangle));

        // Tetrahedron is a closed 3-manifold
        let mut tetra = SimplicialComplex::new();
        tetra.add_simplex(Simplex::new(vec![0, 1, 2, 3]));
        assert!(is_manifold(&tetra));
    }

    #[test]
    fn test_closed_manifold() {
        // Hollow tetrahedron (sphere) is closed
        let mut sphere = SimplicialComplex::new();
        sphere.add_simplex(Simplex::new(vec![0, 1, 2]));
        sphere.add_simplex(Simplex::new(vec![0, 1, 3]));
        sphere.add_simplex(Simplex::new(vec![0, 2, 3]));
        sphere.add_simplex(Simplex::new(vec![1, 2, 3]));

        let boundary = ManifoldBoundary::compute(&sphere);
        assert!(boundary.is_closed());
    }

    #[test]
    fn test_boundary_components() {
        // Two separate triangles
        let mut complex = SimplicialComplex::new();
        complex.add_simplex(Simplex::new(vec![0, 1, 2]));
        complex.add_simplex(Simplex::new(vec![3, 4, 5]));

        let boundary = ManifoldBoundary::compute(&complex);
        assert_eq!(boundary.num_components(), 2);
    }

    #[test]
    fn test_oriented_boundary() {
        let mut boundary = OrientedBoundary::new();
        boundary.add(Simplex::new(vec![0, 1]), 1);
        boundary.add(Simplex::new(vec![1, 2]), 1);
        boundary.add(Simplex::new(vec![0, 1]), -1);
        boundary.simplify();

        // [0,1] should have canceled
        assert_eq!(boundary.len(), 1);
    }
}
