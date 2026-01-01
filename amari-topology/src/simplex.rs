//! Simplicial structures for algebraic topology.
//!
//! A simplex is the fundamental building block of simplicial complexes.
//! An n-simplex is the convex hull of n+1 affinely independent points.

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};

/// An abstract simplex represented by its vertex indices.
///
/// Vertices are stored in sorted order for canonical representation.
/// The dimension of a simplex is one less than the number of vertices:
/// - 0-simplex: point (1 vertex)
/// - 1-simplex: edge (2 vertices)
/// - 2-simplex: triangle (3 vertices)
/// - 3-simplex: tetrahedron (4 vertices)
#[derive(Clone, Debug)]
pub struct Simplex {
    /// Vertices in sorted order
    vertices: Vec<usize>,
    /// Optional orientation (+1 or -1)
    orientation: i8,
}

impl Simplex {
    /// Create a new simplex from vertex indices.
    ///
    /// Vertices are automatically sorted into canonical order.
    /// Orientation is determined by the parity of the sorting permutation.
    pub fn new(mut vertices: Vec<usize>) -> Self {
        // Count inversions to determine orientation
        let mut inversions = 0;
        for i in 0..vertices.len() {
            for j in i + 1..vertices.len() {
                if vertices[i] > vertices[j] {
                    inversions += 1;
                }
            }
        }

        vertices.sort_unstable();
        vertices.dedup();

        let orientation = if inversions % 2 == 0 { 1 } else { -1 };

        Self {
            vertices,
            orientation,
        }
    }

    /// Create a simplex with explicit orientation.
    pub fn with_orientation(vertices: Vec<usize>, orientation: i8) -> Self {
        let mut simplex = Self::new(vertices);
        simplex.orientation = orientation.signum();
        simplex
    }

    /// The dimension of the simplex (number of vertices - 1).
    #[inline]
    pub fn dimension(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.vertices.len() - 1
        }
    }

    /// Get the vertices of this simplex.
    #[inline]
    pub fn vertices(&self) -> &[usize] {
        &self.vertices
    }

    /// Get the orientation of this simplex (+1 or -1).
    #[inline]
    pub fn orientation(&self) -> i8 {
        self.orientation
    }

    /// Check if this simplex contains a vertex.
    #[inline]
    pub fn contains_vertex(&self, vertex: usize) -> bool {
        self.vertices.binary_search(&vertex).is_ok()
    }

    /// Check if this simplex is a face of another simplex.
    pub fn is_face_of(&self, other: &Simplex) -> bool {
        if self.dimension() >= other.dimension() {
            return false;
        }

        self.vertices.iter().all(|v| other.contains_vertex(*v))
    }

    /// Get all faces of dimension k.
    ///
    /// Returns empty vec if k >= self.dimension().
    pub fn faces(&self, k: usize) -> Vec<Simplex> {
        if k >= self.dimension() {
            return vec![];
        }

        let n = self.vertices.len();
        let choose_count = k + 1;

        // Generate all (k+1)-subsets of vertices
        let mut faces = Vec::new();
        let mut indices = (0..choose_count).collect::<Vec<_>>();

        loop {
            // Create face from current indices
            let face_vertices: Vec<usize> = indices.iter().map(|&i| self.vertices[i]).collect();
            faces.push(Simplex::new(face_vertices));

            // Generate next combination
            let mut found_next = false;
            let mut i = choose_count;
            while i > 0 {
                i -= 1;
                if indices[i] < n - choose_count + i {
                    indices[i] += 1;
                    for j in i + 1..choose_count {
                        indices[j] = indices[j - 1] + 1;
                    }
                    found_next = true;
                    break;
                }
            }

            if !found_next {
                break;
            }
        }

        faces
    }

    /// Get the boundary faces (codimension-1 faces) with induced orientations.
    ///
    /// The boundary of an n-simplex [v0, v1, ..., vn] is:
    /// ∂[v0, ..., vn] = Σ (-1)^i [v0, ..., v̂i, ..., vn]
    /// where v̂i means vi is omitted.
    pub fn boundary_faces(&self) -> Vec<(Simplex, i8)> {
        if self.vertices.is_empty() {
            return vec![];
        }

        let mut faces = Vec::with_capacity(self.vertices.len());

        for (i, _) in self.vertices.iter().enumerate() {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);

            let sign = if i % 2 == 0 { 1i8 } else { -1i8 };
            let face = Simplex::new(face_vertices);

            faces.push((face, sign * self.orientation));
        }

        faces
    }

    /// Return the opposite orientation of this simplex.
    pub fn negate(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            orientation: -self.orientation,
        }
    }
}

impl PartialEq for Simplex {
    fn eq(&self, other: &Self) -> bool {
        self.vertices == other.vertices
    }
}

impl Eq for Simplex {}

impl Hash for Simplex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertices.hash(state);
    }
}

impl PartialOrd for Simplex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Simplex {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare by dimension first, then lexicographically by vertices
        match self.dimension().cmp(&other.dimension()) {
            Ordering::Equal => self.vertices.cmp(&other.vertices),
            ord => ord,
        }
    }
}

impl fmt::Display for Simplex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.orientation >= 0 { '+' } else { '-' };
        write!(f, "{}[", sign)?;
        for (i, v) in self.vertices.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_dimension() {
        assert_eq!(Simplex::new(vec![0]).dimension(), 0);
        assert_eq!(Simplex::new(vec![0, 1]).dimension(), 1);
        assert_eq!(Simplex::new(vec![0, 1, 2]).dimension(), 2);
        assert_eq!(Simplex::new(vec![0, 1, 2, 3]).dimension(), 3);
    }

    #[test]
    fn test_simplex_canonical_order() {
        let s1 = Simplex::new(vec![2, 0, 1]);
        let s2 = Simplex::new(vec![0, 1, 2]);
        assert_eq!(s1.vertices(), s2.vertices());
    }

    #[test]
    fn test_simplex_orientation() {
        // Even permutation -> +1
        let s1 = Simplex::new(vec![0, 1, 2]);
        assert_eq!(s1.orientation(), 1);

        // Odd permutation -> -1
        let s2 = Simplex::new(vec![1, 0, 2]);
        assert_eq!(s2.orientation(), -1);
    }

    #[test]
    fn test_faces() {
        let triangle = Simplex::new(vec![0, 1, 2]);

        // Vertices (0-faces)
        let vertices = triangle.faces(0);
        assert_eq!(vertices.len(), 3);

        // Edges (1-faces)
        let edges = triangle.faces(1);
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_boundary_faces() {
        let edge = Simplex::new(vec![0, 1]);
        let boundary = edge.boundary_faces();

        assert_eq!(boundary.len(), 2);
        // ∂[0,1] = [1] - [0]
        assert_eq!(boundary[0].0.vertices(), &[1]);
        assert_eq!(boundary[0].1, 1);
        assert_eq!(boundary[1].0.vertices(), &[0]);
        assert_eq!(boundary[1].1, -1);
    }

    #[test]
    fn test_is_face_of() {
        let triangle = Simplex::new(vec![0, 1, 2]);
        let edge = Simplex::new(vec![0, 1]);
        let vertex = Simplex::new(vec![0]);

        assert!(edge.is_face_of(&triangle));
        assert!(vertex.is_face_of(&triangle));
        assert!(vertex.is_face_of(&edge));
        assert!(!triangle.is_face_of(&edge));
    }
}
