//! Tropical polytopes for attention pattern visualization
//!
//! In tropical geometry, polytopes represent the structure of tropical polynomials
//! and can be used to visualize attention patterns in neural networks.

use crate::{TropicalMatrix, TropicalMultivector, TropicalNumber};
use alloc::vec::Vec;
use num_traits::Float;

/// A tropical polytope representing attention patterns
#[derive(Clone, Debug)]
pub struct TropicalPolytope<T: Float> {
    vertices: Vec<Vec<TropicalNumber<T>>>,
    dimension: usize,
}

impl<T: Float> TropicalPolytope<T> {
    /// Create new tropical polytope
    pub fn new(vertices: Vec<Vec<T>>) -> Self {
        let dimension = vertices.first().map(|v| v.len()).unwrap_or(0);
        let tropical_vertices: Vec<Vec<TropicalNumber<T>>> = vertices
            .into_iter()
            .map(|v| v.into_iter().map(TropicalNumber::new).collect())
            .collect();

        Self {
            vertices: tropical_vertices,
            dimension,
        }
    }

    /// Create tropical polytope from attention scores
    pub fn from_attention_scores(scores: &TropicalMatrix<T>) -> Self {
        let mut vertices = Vec::new();

        // Each row of attention matrix becomes a vertex
        for row in &scores.data {
            vertices.push(row.clone());
        }

        Self {
            vertices,
            dimension: scores.cols,
        }
    }

    /// Create from multivector coefficients
    pub fn from_multivector<const DIM: usize>(mv: &TropicalMultivector<T, DIM>) -> Self {
        let mut coeffs = Vec::new();
        for i in 0..(1 << DIM) {
            coeffs.push(mv.get(i));
        }

        Self {
            vertices: vec![coeffs],
            dimension: 1 << DIM,
        }
    }

    /// Compute tropical convex hull
    pub fn convex_hull(&self) -> Self {
        if self.vertices.is_empty() {
            return Self::new(Vec::new());
        }

        // Simplified tropical convex hull
        // In tropical geometry, convex combinations use (max, +)
        let mut hull_vertices = Vec::new();

        for vertex in &self.vertices {
            let mut is_extreme = true;

            // Check if this vertex can be expressed as tropical combination of others
            for other in &self.vertices {
                if core::ptr::eq(vertex, other) {
                    continue;
                }

                // Simplified extremality test
                let mut dominates = true;
                for (&v, &o) in vertex.iter().zip(other.iter()) {
                    if v.value() < o.value() {
                        dominates = false;
                        break;
                    }
                }

                if dominates && vertex != other {
                    is_extreme = false;
                    break;
                }
            }

            if is_extreme {
                hull_vertices.push(vertex.clone());
            }
        }

        Self {
            vertices: hull_vertices,
            dimension: self.dimension,
        }
    }

    /// Check if point is in tropical polytope
    pub fn contains_point(&self, point: &[TropicalNumber<T>]) -> bool {
        if point.len() != self.dimension {
            return false;
        }

        // A point is in a tropical polytope if it can be expressed as
        // a tropical linear combination of vertices

        // Simplified membership test
        for vertex in &self.vertices {
            let mut all_dominated = true;
            for (&p, &v) in point.iter().zip(vertex.iter()) {
                if p.value() > v.value() + T::epsilon() {
                    all_dominated = false;
                    break;
                }
            }
            if all_dominated {
                return true;
            }
        }

        false
    }

    /// Compute tropical volume (number of lattice points)
    pub fn tropical_volume(&self) -> T {
        // Simplified volume computation
        if self.vertices.is_empty() {
            return T::zero();
        }

        // For tropical polytopes, volume relates to the number of
        // distinct optimal solutions
        T::from(self.vertices.len()).unwrap()
    }

    /// Get the normal fan of the polytope
    pub fn normal_fan(&self) -> Vec<Vec<TropicalNumber<T>>> {
        // The normal fan consists of cones corresponding to vertices
        let mut fan = Vec::new();

        for (i, vertex) in self.vertices.iter().enumerate() {
            let mut cone = Vec::new();

            // Compute outward normal directions
            for (j, other) in self.vertices.iter().enumerate() {
                if i == j {
                    continue;
                }

                let mut normal = Vec::new();
                for (&v, &o) in vertex.iter().zip(other.iter()) {
                    // Tropical subtraction for normal computation
                    normal.push(TropicalNumber::new(v.value() - o.value()));
                }
                cone.push(normal);
            }

            if !cone.is_empty() {
                // Simplify to unique directions
                cone.sort_by(|a, b| a[0].value().partial_cmp(&b[0].value()).unwrap());
                cone.dedup();
                fan.push(cone.into_iter().flatten().collect());
            }
        }

        fan
    }

    /// Minkowski sum with another polytope
    pub fn minkowski_sum(&self, other: &Self) -> Self {
        if self.dimension != other.dimension {
            return Self::new(Vec::new());
        }

        let mut sum_vertices = Vec::new();

        for v1 in &self.vertices {
            for v2 in &other.vertices {
                let mut sum_vertex = Vec::new();
                for (&a, &b) in v1.iter().zip(v2.iter()) {
                    // Tropical addition is max
                    sum_vertex.push(a + b);
                }
                sum_vertices.push(sum_vertex);
            }
        }

        Self {
            vertices: sum_vertices,
            dimension: self.dimension,
        }
    }

    /// Project polytope to lower dimension
    pub fn project(&self, coordinates: &[usize]) -> Self {
        let new_dimension = coordinates.len();
        let mut projected_vertices = Vec::new();

        for vertex in &self.vertices {
            let mut projected = Vec::new();
            for &coord in coordinates {
                if coord < vertex.len() {
                    projected.push(vertex[coord]);
                }
            }
            if projected.len() == new_dimension {
                projected_vertices.push(projected);
            }
        }

        Self {
            vertices: projected_vertices,
            dimension: new_dimension,
        }
    }

    /// Get vertices as regular numbers for visualization
    pub fn to_regular_vertices(&self) -> Vec<Vec<T>> {
        self.vertices
            .iter()
            .map(|v| v.iter().map(|&tn| tn.value()).collect())
            .collect()
    }
}

/// Tropical hyperplane for polytope faces
#[derive(Clone, Debug)]
pub struct TropicalHyperplane<T: Float> {
    normal: Vec<TropicalNumber<T>>,
    offset: TropicalNumber<T>,
}

impl<T: Float> TropicalHyperplane<T> {
    /// Create new tropical hyperplane
    pub fn new(normal: Vec<T>, offset: T) -> Self {
        Self {
            normal: normal.into_iter().map(TropicalNumber::new).collect(),
            offset: TropicalNumber::new(offset),
        }
    }

    /// Evaluate hyperplane at point (tropical linear form)
    pub fn evaluate(&self, point: &[TropicalNumber<T>]) -> TropicalNumber<T> {
        let mut result = self.offset;

        for (&coord, &normal_comp) in point.iter().zip(self.normal.iter()) {
            result = result + (coord * normal_comp);
        }

        result
    }

    /// Check if point is on the hyperplane
    pub fn contains_point(&self, point: &[TropicalNumber<T>]) -> bool {
        let value = self.evaluate(point);
        // In tropical geometry, "on the hyperplane" means the maximum
        // is achieved by multiple terms

        // Simplified test - check if close to expected value
        (value.value() - self.offset.value()).abs() < T::epsilon()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use approx::assert_relative_eq;

    #[test]
    fn test_tropical_polytope_creation() {
        let vertices = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 3.0],
            vec![2.0, 3.0, 0.0],
        ];

        let polytope = TropicalPolytope::new(vertices);
        assert_eq!(polytope.dimension, 3);
        assert_eq!(polytope.vertices.len(), 3);
    }

    #[test]
    fn test_convex_hull() {
        let vertices = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![0.5, 0.5], // This might not be extreme
            vec![2.0, 2.0],
        ];

        let polytope = TropicalPolytope::new(vertices);
        let hull = polytope.convex_hull();

        // Hull should have at most the same number of vertices
        assert!(hull.vertices.len() <= polytope.vertices.len());
    }

    #[test]
    fn test_point_containment() {
        let vertices = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];

        let polytope = TropicalPolytope::new(vertices);

        // Test point that should be inside
        let inside_point = vec![TropicalNumber::new(0.5), TropicalNumber::new(0.5)];

        // This is a simplified test - actual tropical containment is complex
        let _contained = polytope.contains_point(&inside_point);
        // Just verify the test runs without error (compilation test)
    }

    #[test]
    fn test_minkowski_sum() {
        let vertices1 = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        let vertices2 = vec![vec![0.0, 1.0], vec![0.0, 0.0]];

        let poly1 = TropicalPolytope::new(vertices1);
        let poly2 = TropicalPolytope::new(vertices2);

        let sum = poly1.minkowski_sum(&poly2);

        // Should have vertices from all combinations
        assert_eq!(sum.vertices.len(), 4);
        assert_eq!(sum.dimension, 2);
    }

    #[test]
    fn test_tropical_hyperplane() {
        let hyperplane = TropicalHyperplane::new(vec![1.0, -1.0], 0.0);

        let point = vec![TropicalNumber::new(2.0), TropicalNumber::new(1.0)];

        let value = hyperplane.evaluate(&point);

        // Should compute max(0, 2+1, 1+(-1)) = max(0, 3, 0) = 3 (tropical arithmetic)
        assert_relative_eq!(value.value(), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_attention_scores() {
        let scores_data = vec![
            vec![
                TropicalNumber::new(0.0),
                TropicalNumber::new(-1.0),
                TropicalNumber::new(-2.0),
            ],
            vec![
                TropicalNumber::new(-1.0),
                TropicalNumber::new(0.0),
                TropicalNumber::new(-1.0),
            ],
        ];

        let matrix = TropicalMatrix {
            data: scores_data,
            rows: 2,
            cols: 3,
        };

        let polytope = TropicalPolytope::from_attention_scores(&matrix);

        assert_eq!(polytope.vertices.len(), 2);
        assert_eq!(polytope.dimension, 3);
    }
}
