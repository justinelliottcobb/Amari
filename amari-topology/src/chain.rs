//! Chain groups and boundary maps for homology computation.
//!
//! A k-chain is a formal sum of k-simplices with integer coefficients.
//! The boundary operator ∂_k maps k-chains to (k-1)-chains.

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap, vec, vec::Vec};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

use core::fmt;

use crate::simplex::Simplex;

/// A chain is a formal sum of simplices with integer coefficients.
///
/// Represented as a sparse map from simplex index to coefficient.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Chain {
    /// Coefficients indexed by simplex position in the chain group
    coefficients: BTreeMap<usize, i64>,
}

impl Chain {
    /// Create the zero chain.
    pub fn zero() -> Self {
        Self {
            coefficients: BTreeMap::new(),
        }
    }

    /// Create a chain from a single simplex with coefficient 1.
    pub fn from_simplex(index: usize) -> Self {
        let mut coefficients = BTreeMap::new();
        coefficients.insert(index, 1);
        Self { coefficients }
    }

    /// Create a chain from a single simplex with given coefficient.
    pub fn from_simplex_with_coeff(index: usize, coeff: i64) -> Self {
        if coeff == 0 {
            return Self::zero();
        }
        let mut coefficients = BTreeMap::new();
        coefficients.insert(index, coeff);
        Self { coefficients }
    }

    /// Get the coefficient of a simplex.
    pub fn coefficient(&self, index: usize) -> i64 {
        self.coefficients.get(&index).copied().unwrap_or(0)
    }

    /// Set the coefficient of a simplex.
    pub fn set_coefficient(&mut self, index: usize, coeff: i64) {
        if coeff == 0 {
            self.coefficients.remove(&index);
        } else {
            self.coefficients.insert(index, coeff);
        }
    }

    /// Add another chain to this one.
    pub fn add(&self, other: &Chain) -> Chain {
        let mut result = self.clone();
        for (&idx, &coeff) in &other.coefficients {
            let new_coeff = result.coefficient(idx) + coeff;
            result.set_coefficient(idx, new_coeff);
        }
        result
    }

    /// Subtract another chain from this one.
    pub fn sub(&self, other: &Chain) -> Chain {
        let mut result = self.clone();
        for (&idx, &coeff) in &other.coefficients {
            let new_coeff = result.coefficient(idx) - coeff;
            result.set_coefficient(idx, new_coeff);
        }
        result
    }

    /// Multiply the chain by a scalar.
    pub fn scale(&self, scalar: i64) -> Chain {
        if scalar == 0 {
            return Chain::zero();
        }
        let coefficients = self
            .coefficients
            .iter()
            .map(|(&idx, &coeff)| (idx, coeff * scalar))
            .filter(|(_, c)| *c != 0)
            .collect();
        Chain { coefficients }
    }

    /// Check if this is the zero chain.
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Get all non-zero terms as (index, coefficient) pairs.
    pub fn terms(&self) -> impl Iterator<Item = (usize, i64)> + '_ {
        self.coefficients.iter().map(|(&i, &c)| (i, c))
    }

    /// Number of non-zero terms.
    pub fn support_size(&self) -> usize {
        self.coefficients.len()
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }

        let mut first = true;
        for (&idx, &coeff) in &self.coefficients {
            if !first {
                if coeff >= 0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
            } else if coeff < 0 {
                write!(f, "-")?;
            }

            let abs_coeff = coeff.abs();
            if abs_coeff != 1 {
                write!(f, "{}·σ_{}", abs_coeff, idx)?;
            } else {
                write!(f, "σ_{}", idx)?;
            }
            first = false;
        }
        Ok(())
    }
}

/// A chain group C_k consists of all k-chains.
///
/// Represented by the set of k-simplices that form its basis.
#[derive(Clone, Debug)]
pub struct ChainGroup {
    /// Basis simplices for this chain group
    simplices: Vec<Simplex>,
    /// Map from simplex to its index for fast lookup
    simplex_to_index: BTreeMap<Simplex, usize>,
}

impl ChainGroup {
    /// Create a chain group from a list of simplices.
    pub fn new(simplices: Vec<Simplex>) -> Self {
        let simplex_to_index = simplices
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();
        Self {
            simplices,
            simplex_to_index,
        }
    }

    /// Create an empty chain group.
    pub fn empty() -> Self {
        Self {
            simplices: Vec::new(),
            simplex_to_index: BTreeMap::new(),
        }
    }

    /// Get the dimension (rank) of this chain group.
    pub fn rank(&self) -> usize {
        self.simplices.len()
    }

    /// Check if the group is trivial (rank 0).
    pub fn is_trivial(&self) -> bool {
        self.simplices.is_empty()
    }

    /// Get the index of a simplex in this group.
    pub fn index_of(&self, simplex: &Simplex) -> Option<usize> {
        self.simplex_to_index.get(simplex).copied()
    }

    /// Get the simplex at a given index.
    pub fn simplex_at(&self, index: usize) -> Option<&Simplex> {
        self.simplices.get(index)
    }

    /// Iterate over all simplices in the group.
    pub fn simplices(&self) -> impl Iterator<Item = &Simplex> {
        self.simplices.iter()
    }

    /// Create a chain from a simplex in this group.
    pub fn chain_from_simplex(&self, simplex: &Simplex) -> Option<Chain> {
        self.index_of(simplex).map(Chain::from_simplex)
    }
}

impl fmt::Display for ChainGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "C_{} (rank {})",
            self.simplices.first().map_or(0, |s| s.dimension()),
            self.rank()
        )
    }
}

/// The boundary map ∂_k: C_k → C_{k-1}.
///
/// Represented as a sparse matrix.
#[derive(Clone, Debug)]
pub struct BoundaryMap {
    /// Number of rows (dimension of codomain)
    rows: usize,
    /// Number of columns (dimension of domain)
    cols: usize,
    /// Sparse matrix entries: (row, col) -> coefficient
    entries: BTreeMap<(usize, usize), i64>,
}

impl BoundaryMap {
    /// Create a zero boundary map.
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            entries: BTreeMap::new(),
        }
    }

    /// Create a boundary map from chain groups.
    ///
    /// domain is C_k, codomain is C_{k-1}.
    pub fn from_chain_groups(domain: &ChainGroup, codomain: &ChainGroup) -> Self {
        let rows = codomain.rank();
        let cols = domain.rank();
        let mut entries = BTreeMap::new();

        for (col, simplex) in domain.simplices().enumerate() {
            for (face, sign) in simplex.boundary_faces() {
                if let Some(row) = codomain.index_of(&face) {
                    let coeff = sign as i64;
                    if coeff != 0 {
                        entries.insert((row, col), coeff);
                    }
                }
            }
        }

        Self {
            rows,
            cols,
            entries,
        }
    }

    /// Apply the boundary map to a chain.
    pub fn apply(&self, chain: &Chain) -> Chain {
        let mut result = Chain::zero();

        for (col, coeff) in chain.terms() {
            for (&(row, c), &val) in &self.entries {
                if c == col {
                    let new_coeff = result.coefficient(row) + coeff * val;
                    result.set_coefficient(row, new_coeff);
                }
            }
        }

        result
    }

    /// Get the matrix entry at (row, col).
    pub fn get(&self, row: usize, col: usize) -> i64 {
        self.entries.get(&(row, col)).copied().unwrap_or(0)
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Convert to dense matrix representation.
    pub fn to_dense(&self) -> Vec<Vec<i64>> {
        let mut matrix = vec![vec![0i64; self.cols]; self.rows];
        for (&(row, col), &val) in &self.entries {
            matrix[row][col] = val;
        }
        matrix
    }

    /// Compute the rank of this matrix using Gaussian elimination.
    pub fn rank(&self) -> usize {
        if self.rows == 0 || self.cols == 0 {
            return 0;
        }

        let mut matrix = self.to_dense();
        gaussian_elimination_rank(&mut matrix)
    }

    /// Compute the dimension of the kernel (null space).
    pub fn kernel_dim(&self) -> usize {
        self.cols.saturating_sub(self.rank())
    }

    /// Compute the dimension of the image.
    pub fn image_dim(&self) -> usize {
        self.rank()
    }
}

impl fmt::Display for BoundaryMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "∂: {} × {} (rank {}, nullity {})",
            self.rows,
            self.cols,
            self.rank(),
            self.kernel_dim()
        )
    }
}

/// Compute the rank of a matrix using Gaussian elimination over integers.
fn gaussian_elimination_rank(matrix: &mut [Vec<i64>]) -> usize {
    if matrix.is_empty() || matrix[0].is_empty() {
        return 0;
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut rank = 0;
    let mut pivot_col = 0;

    for row in 0..rows {
        if pivot_col >= cols {
            break;
        }

        // Find pivot (need index to store as pivot_row)
        let mut pivot_row = None;
        #[allow(clippy::needless_range_loop)]
        for r in row..rows {
            if matrix[r][pivot_col] != 0 {
                pivot_row = Some(r);
                break;
            }
        }

        match pivot_row {
            Some(pr) => {
                // Swap rows
                matrix.swap(row, pr);
                rank += 1;

                // Eliminate below (need index r to access both matrix[r] and matrix[row])
                let pivot = matrix[row][pivot_col];
                for r in (row + 1)..rows {
                    if matrix[r][pivot_col] != 0 {
                        let factor = matrix[r][pivot_col];
                        #[allow(clippy::needless_range_loop)]
                        for c in pivot_col..cols {
                            matrix[r][c] = matrix[r][c] * pivot - matrix[row][c] * factor;
                        }
                    }
                }
                pivot_col += 1;
            }
            None => {
                pivot_col += 1;
            }
        }
    }

    rank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_arithmetic() {
        let c1 = Chain::from_simplex(0);
        let c2 = Chain::from_simplex(1);

        let sum = c1.add(&c2);
        assert_eq!(sum.coefficient(0), 1);
        assert_eq!(sum.coefficient(1), 1);

        let diff = c1.sub(&c2);
        assert_eq!(diff.coefficient(0), 1);
        assert_eq!(diff.coefficient(1), -1);
    }

    #[test]
    fn test_chain_group() {
        let simplices = vec![
            Simplex::new(vec![0]),
            Simplex::new(vec![1]),
            Simplex::new(vec![2]),
        ];
        let group = ChainGroup::new(simplices);

        assert_eq!(group.rank(), 3);
        assert_eq!(group.index_of(&Simplex::new(vec![1])), Some(1));
    }

    #[test]
    fn test_boundary_map_edge() {
        // Edge [0,1] has boundary [1] - [0]
        let domain = ChainGroup::new(vec![Simplex::new(vec![0, 1])]);
        let codomain = ChainGroup::new(vec![Simplex::new(vec![0]), Simplex::new(vec![1])]);

        let boundary = BoundaryMap::from_chain_groups(&domain, &codomain);

        // Apply to the edge
        let edge_chain = Chain::from_simplex(0);
        let result = boundary.apply(&edge_chain);

        // Should be [1] - [0], i.e., coeff[1] = 1, coeff[0] = -1
        assert_eq!(result.coefficient(0), -1); // [0] vertex
        assert_eq!(result.coefficient(1), 1); // [1] vertex
    }

    #[test]
    fn test_boundary_squared_is_zero() {
        // ∂∂ = 0 for triangle
        let triangle = Simplex::new(vec![0, 1, 2]);

        let c2 = ChainGroup::new(vec![triangle.clone()]);
        let c1 = ChainGroup::new(triangle.faces(1));
        let c0 = ChainGroup::new(triangle.faces(0));

        let d2 = BoundaryMap::from_chain_groups(&c2, &c1);
        let d1 = BoundaryMap::from_chain_groups(&c1, &c0);

        // Apply ∂₂ then ∂₁
        let chain = Chain::from_simplex(0);
        let boundary1 = d2.apply(&chain);
        let boundary2 = d1.apply(&boundary1);

        assert!(boundary2.is_zero(), "∂∂ should be zero");
    }

    #[test]
    fn test_matrix_rank() {
        // Identity 3x3 has rank 3
        let mut identity = vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]];
        assert_eq!(gaussian_elimination_rank(&mut identity), 3);

        // Zero matrix has rank 0
        let mut zero = vec![vec![0, 0], vec![0, 0]];
        assert_eq!(gaussian_elimination_rank(&mut zero), 0);

        // Rank 2 matrix
        let mut rank2 = vec![vec![1, 2, 3], vec![2, 4, 6], vec![0, 1, 1]];
        assert_eq!(gaussian_elimination_rank(&mut rank2), 2);
    }
}
