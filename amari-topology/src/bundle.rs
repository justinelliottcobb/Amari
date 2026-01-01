//! Fiber bundles over geometric spaces.
//!
//! A fiber bundle is a space E (total space) that locally looks like a product
//! B × F of a base space B and a fiber F, but may have global twisting.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec, vec::Vec};

use crate::{Result, TopologyError};

/// A fiber bundle over a base space with typical fiber F.
///
/// E → B with fiber F
#[derive(Clone, Debug)]
pub struct FiberBundle<B, F> {
    /// Base space
    pub base: B,
    /// Typical fiber
    pub fiber: F,
    /// Local trivializations (open cover with bundle isomorphisms)
    pub trivializations: Vec<Trivialization<B, F>>,
}

/// A local trivialization of a fiber bundle.
#[derive(Clone, Debug)]
pub struct Trivialization<B, F> {
    /// Open set in base space (represented as indices)
    pub open_set: Vec<usize>,
    /// The local product structure
    pub local_product: LocalProduct<B, F>,
}

/// Local product structure B × F.
#[derive(Clone, Debug)]
pub struct LocalProduct<B, F> {
    pub base_component: B,
    pub fiber_component: F,
}

impl<B: Clone, F: Clone> FiberBundle<B, F> {
    /// Create a trivial bundle (global product).
    pub fn trivial(base: B, fiber: F) -> Self {
        Self {
            base,
            fiber,
            trivializations: Vec::new(),
        }
    }

    /// Check if the bundle is trivial (globally a product).
    pub fn is_trivial(&self) -> bool {
        self.trivializations.len() <= 1
    }
}

/// A section of a fiber bundle assigns a point in the fiber to each point in the base.
///
/// s: B → E such that π ∘ s = id_B
#[derive(Clone)]
pub struct Section<F> {
    /// Values of the section at base points
    values: Vec<F>,
}

impl<F: Clone> Section<F> {
    /// Create a constant section.
    pub fn constant(value: F, size: usize) -> Self {
        Self {
            values: vec![value; size],
        }
    }

    /// Create a section from values.
    pub fn new(values: Vec<F>) -> Self {
        Self { values }
    }

    /// Get the value at a base point.
    pub fn at(&self, index: usize) -> Option<&F> {
        self.values.get(index)
    }

    /// Set the value at a base point.
    pub fn set(&mut self, index: usize, value: F) -> Result<()> {
        if index >= self.values.len() {
            return Err(TopologyError::InvalidManifold(format!(
                "Index {} out of bounds",
                index
            )));
        }
        self.values[index] = value;
        Ok(())
    }

    /// Number of base points.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the section is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate over all values.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.values.iter()
    }
}

/// A connection on a fiber bundle defines parallel transport.
///
/// Connections allow us to compare fibers at different points.
#[derive(Clone, Debug)]
pub struct Connection<F> {
    /// Connection coefficients (simplified representation)
    coefficients: Vec<Vec<F>>,
}

impl<F: Clone + Default> Connection<F> {
    /// Create a flat (trivial) connection.
    pub fn flat(dimension: usize) -> Self {
        Self {
            coefficients: vec![vec![F::default(); dimension]; dimension],
        }
    }

    /// Get the connection coefficient.
    pub fn coefficient(&self, i: usize, j: usize) -> Option<&F> {
        self.coefficients.get(i).and_then(|row| row.get(j))
    }
}

/// A vector bundle is a fiber bundle where the fiber is a vector space.
#[derive(Clone, Debug)]
pub struct VectorBundle {
    /// Dimension of base manifold
    pub base_dimension: usize,
    /// Dimension of fiber (rank of the bundle)
    pub rank: usize,
    /// Transition functions between local trivializations
    pub transition_functions: Vec<TransitionFunction>,
}

/// A transition function between overlapping trivializations.
#[derive(Clone, Debug)]
pub struct TransitionFunction {
    /// From trivialization index
    pub from: usize,
    /// To trivialization index
    pub to: usize,
    /// The matrix-valued function (simplified as constant matrices here)
    pub matrix: Vec<Vec<f64>>,
}

impl VectorBundle {
    /// Create a trivial vector bundle.
    pub fn trivial(base_dimension: usize, rank: usize) -> Self {
        Self {
            base_dimension,
            rank,
            transition_functions: Vec::new(),
        }
    }

    /// Create the tangent bundle of an n-manifold.
    pub fn tangent_bundle(manifold_dimension: usize) -> Self {
        Self::trivial(manifold_dimension, manifold_dimension)
    }

    /// Create the cotangent bundle of an n-manifold.
    pub fn cotangent_bundle(manifold_dimension: usize) -> Self {
        Self::trivial(manifold_dimension, manifold_dimension)
    }

    /// Check if this is a line bundle (rank 1).
    pub fn is_line_bundle(&self) -> bool {
        self.rank == 1
    }

    /// Compute the first Chern number for a complex line bundle over S².
    ///
    /// For the Möbius bundle (twisted): c₁ = 1
    /// For the trivial bundle: c₁ = 0
    pub fn first_chern_number(&self) -> i64 {
        if !self.is_line_bundle() {
            return 0;
        }

        // Count winding numbers from transition functions
        // Simplified: count how many "twists" in the transition functions
        let mut winding = 0i64;
        for tf in &self.transition_functions {
            // For a line bundle, the matrix is 1x1
            // Check if it represents a twist
            if let Some(row) = tf.matrix.first() {
                if let Some(&val) = row.first() {
                    if val < 0.0 {
                        winding += 1;
                    }
                }
            }
        }
        winding
    }
}

/// Principal bundle with structure group G.
#[derive(Clone, Debug)]
pub struct PrincipalBundle {
    /// Dimension of base
    pub base_dimension: usize,
    /// Name of the structure group
    pub structure_group: String,
    /// Whether the bundle is trivial
    pub is_trivial: bool,
}

impl PrincipalBundle {
    /// Create a trivial principal bundle.
    pub fn trivial(base_dimension: usize, group: &str) -> Self {
        Self {
            base_dimension,
            structure_group: group.to_string(),
            is_trivial: true,
        }
    }

    /// Create the frame bundle of a manifold.
    ///
    /// The frame bundle has structure group GL(n).
    pub fn frame_bundle(manifold_dimension: usize) -> Self {
        Self {
            base_dimension: manifold_dimension,
            structure_group: format!("GL({})", manifold_dimension),
            is_trivial: false, // Generally not trivial
        }
    }

    /// Create the orthonormal frame bundle.
    ///
    /// The orthonormal frame bundle has structure group O(n).
    pub fn orthonormal_frame_bundle(manifold_dimension: usize) -> Self {
        Self {
            base_dimension: manifold_dimension,
            structure_group: format!("O({})", manifold_dimension),
            is_trivial: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_bundle() {
        let bundle: FiberBundle<usize, f64> = FiberBundle::trivial(3, 1.0);
        assert!(bundle.is_trivial());
    }

    #[test]
    fn test_section() {
        let mut section = Section::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(section.len(), 3);
        assert_eq!(section.at(1), Some(&2.0));

        section.set(1, 5.0).unwrap();
        assert_eq!(section.at(1), Some(&5.0));
    }

    #[test]
    fn test_constant_section() {
        let section = Section::constant(42.0, 5);
        assert_eq!(section.len(), 5);
        for val in section.iter() {
            assert_eq!(*val, 42.0);
        }
    }

    #[test]
    fn test_vector_bundle() {
        let tangent = VectorBundle::tangent_bundle(3);
        assert_eq!(tangent.base_dimension, 3);
        assert_eq!(tangent.rank, 3);
        assert!(!tangent.is_line_bundle());

        let line = VectorBundle::trivial(2, 1);
        assert!(line.is_line_bundle());
    }

    #[test]
    fn test_principal_bundle() {
        let frame = PrincipalBundle::frame_bundle(3);
        assert_eq!(frame.structure_group, "GL(3)");

        let ortho = PrincipalBundle::orthonormal_frame_bundle(3);
        assert_eq!(ortho.structure_group, "O(3)");
    }

    #[test]
    fn test_connection() {
        let conn: Connection<f64> = Connection::flat(3);
        assert_eq!(conn.coefficient(0, 0), Some(&0.0));
    }
}
