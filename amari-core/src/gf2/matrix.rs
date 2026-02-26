//! GF(2) matrices with Gaussian elimination, rank, null space, and linear system solving.

use super::scalar::GF2;
use super::vector::GF2Vector;
use crate::error::{CoreError, CoreResult};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

/// A matrix over GF(2), stored as row vectors.
///
/// Supports Gaussian elimination, rank computation, null space extraction,
/// and matrix-vector multiplication — all via bitwise operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GF2Matrix {
    rows: Vec<GF2Vector>,
    nrows: usize,
    ncols: usize,
}

impl GF2Matrix {
    /// Create a zero matrix.
    #[must_use]
    pub fn zero(nrows: usize, ncols: usize) -> Self {
        let rows = (0..nrows).map(|_| GF2Vector::zero(ncols)).collect();
        Self { rows, nrows, ncols }
    }

    /// Create an identity matrix.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zero(n, n);
        for i in 0..n {
            m.set(i, i, GF2::ONE);
        }
        m
    }

    /// Create from row vectors. All rows must have the same dimension.
    #[must_use]
    pub fn from_rows(rows: Vec<GF2Vector>) -> Self {
        let nrows = rows.len();
        let ncols = if nrows > 0 { rows[0].dim() } else { 0 };
        debug_assert!(rows.iter().all(|r| r.dim() == ncols));
        Self { rows, nrows, ncols }
    }

    /// Number of rows.
    #[inline]
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Get element at (row, col).
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> GF2 {
        self.rows[row].get(col)
    }

    /// Set element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: GF2) {
        self.rows[row].set(col, value);
    }

    /// Get a reference to row i.
    #[must_use]
    pub fn row(&self, i: usize) -> &GF2Vector {
        &self.rows[i]
    }

    /// Matrix-vector product over GF(2).
    #[must_use]
    pub fn mul_vec(&self, v: &GF2Vector) -> GF2Vector {
        assert_eq!(self.ncols, v.dim(), "dimension mismatch");
        let bits: Vec<u8> = self.rows.iter().map(|row| row.dot(v).value()).collect();
        GF2Vector::from_bits(&bits)
    }

    /// Matrix-matrix product over GF(2).
    #[must_use]
    pub fn mul_mat(&self, other: &Self) -> Self {
        assert_eq!(self.ncols, other.nrows, "dimension mismatch");
        let other_t = other.transpose();
        let rows: Vec<GF2Vector> = self
            .rows
            .iter()
            .map(|row| {
                let bits: Vec<u8> = other_t
                    .rows
                    .iter()
                    .map(|col| row.dot(col).value())
                    .collect();
                GF2Vector::from_bits(&bits)
            })
            .collect();
        Self::from_rows(rows)
    }

    /// Transpose.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut t = Self::zero(self.ncols, self.nrows);
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Reduced row echelon form (in-place). Returns pivot column indices.
    pub fn reduced_row_echelon(&mut self) -> Vec<usize> {
        let mut pivots = Vec::new();
        let mut pivot_row = 0;

        for col in 0..self.ncols {
            // Find a row with a 1 in this column at or below pivot_row.
            let found = (pivot_row..self.nrows).find(|&r| self.get(r, col).is_one());

            if let Some(swap_row) = found {
                self.rows.swap(pivot_row, swap_row);

                // Eliminate all other rows with a 1 in this column.
                for r in 0..self.nrows {
                    if r != pivot_row && self.get(r, col).is_one() {
                        let pivot = self.rows[pivot_row].clone();
                        self.rows[r] = self.rows[r].add(&pivot);
                    }
                }

                pivots.push(col);
                pivot_row += 1;
            }
        }
        pivots
    }

    /// Row echelon form (in-place). Returns pivot column indices.
    ///
    /// Over GF(2), this produces the same result as `reduced_row_echelon` since
    /// the elimination above and below is equivalent when the only nonzero scalar is 1.
    pub fn row_echelon(&mut self) -> Vec<usize> {
        self.reduced_row_echelon()
    }

    /// Rank = number of pivots.
    #[must_use]
    pub fn rank(&self) -> usize {
        let mut copy = self.clone();
        copy.reduced_row_echelon().len()
    }

    /// Null space basis vectors (kernel of the matrix).
    #[must_use]
    pub fn null_space(&self) -> Vec<GF2Vector> {
        let mut rref = self.clone();
        let pivots = rref.reduced_row_echelon();

        let pivot_set: Vec<bool> = (0..self.ncols).map(|c| pivots.contains(&c)).collect();

        // Map pivot columns to their row index.
        let mut pivot_row_for_col = vec![usize::MAX; self.ncols];
        for (row, &col) in pivots.iter().enumerate() {
            pivot_row_for_col[col] = row;
        }

        let free_cols: Vec<usize> = (0..self.ncols).filter(|c| !pivot_set[*c]).collect();

        let mut basis = Vec::new();
        for &fc in &free_cols {
            let mut v = GF2Vector::zero(self.ncols);
            v.set(fc, GF2::ONE);
            // For each pivot column, read the entry in the RREF at (pivot_row, fc).
            for &pc in &pivots {
                let pr = pivot_row_for_col[pc];
                v.set(pc, rref.get(pr, fc));
            }
            basis.push(v);
        }
        basis
    }

    /// Determinant (only for square matrices).
    pub fn determinant(&self) -> CoreResult<GF2> {
        if self.nrows != self.ncols {
            return Err(CoreError::GF2NotSquare {
                rows: self.nrows,
                cols: self.ncols,
            });
        }
        let r = self.rank();
        Ok(if r == self.nrows { GF2::ONE } else { GF2::ZERO })
    }

    /// Column space basis vectors (image of the matrix).
    #[must_use]
    pub fn column_space(&self) -> Vec<GF2Vector> {
        let t = self.transpose();
        let mut rref = t.clone();
        let pivots = rref.reduced_row_echelon();
        pivots.iter().map(|&c| t.row(c).clone()).collect()
    }

    /// Check if a vector is in the column space.
    #[must_use]
    pub fn in_column_space(&self, v: &GF2Vector) -> bool {
        self.solve(v).is_some()
    }

    /// Solve Ax = b over GF(2). Returns None if no solution exists.
    #[must_use]
    pub fn solve(&self, b: &GF2Vector) -> Option<GF2Vector> {
        assert_eq!(self.nrows, b.dim(), "dimension mismatch");
        let mut aug = self.augment(b);
        let pivots = aug.reduced_row_echelon();

        // Check for inconsistency: pivot in the augmented column (last column).
        let aug_col = self.ncols;
        if pivots.contains(&aug_col) {
            return None;
        }

        // Extract solution: for each pivot column, read the value from the augmented column.
        let mut x = GF2Vector::zero(self.ncols);
        for (row, &col) in pivots.iter().enumerate() {
            x.set(col, aug.get(row, aug_col));
        }
        Some(x)
    }

    /// Augmented matrix [A | b].
    #[must_use]
    pub fn augment(&self, b: &GF2Vector) -> Self {
        assert_eq!(self.nrows, b.dim(), "dimension mismatch");
        let new_ncols = self.ncols + 1;
        let rows: Vec<GF2Vector> = self
            .rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut new_row = GF2Vector::zero(new_ncols);
                for j in 0..self.ncols {
                    new_row.set(j, row.get(j));
                }
                new_row.set(self.ncols, b.get(i));
                new_row
            })
            .collect();
        Self {
            rows,
            nrows: self.nrows,
            ncols: new_ncols,
        }
    }

    /// Horizontal concatenation [A | B].
    #[must_use]
    pub fn hcat(&self, other: &Self) -> Self {
        assert_eq!(self.nrows, other.nrows, "row count mismatch");
        let new_ncols = self.ncols + other.ncols;
        let rows: Vec<GF2Vector> = self
            .rows
            .iter()
            .zip(other.rows.iter())
            .map(|(a, b)| {
                let mut new_row = GF2Vector::zero(new_ncols);
                for j in 0..self.ncols {
                    new_row.set(j, a.get(j));
                }
                for j in 0..other.ncols {
                    new_row.set(self.ncols + j, b.get(j));
                }
                new_row
            })
            .collect();
        Self {
            rows,
            nrows: self.nrows,
            ncols: new_ncols,
        }
    }
}

impl fmt::Display for GF2Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", row)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_properties() {
        let id = GF2Matrix::identity(3);
        assert_eq!(id.rank(), 3);
        assert_eq!(id.determinant().unwrap(), GF2::ONE);

        let v = GF2Vector::from_bits(&[1, 0, 1]);
        assert_eq!(id.mul_vec(&v), v);
    }

    #[test]
    fn test_matrix_vector_product() {
        // [[1,0,1],[0,1,1]] * [1,1,0] = [1, 1]
        let m = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 0, 1]),
            GF2Vector::from_bits(&[0, 1, 1]),
        ]);
        let v = GF2Vector::from_bits(&[1, 1, 0]);
        let result = m.mul_vec(&v);
        assert_eq!(result, GF2Vector::from_bits(&[1, 1]));
    }

    #[test]
    fn test_row_echelon_and_rank() {
        let mut m = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 0, 1, 0]),
            GF2Vector::from_bits(&[0, 1, 1, 0]),
            GF2Vector::from_bits(&[1, 1, 0, 0]),
        ]);
        let pivots = m.reduced_row_echelon();
        assert_eq!(pivots.len(), 2); // rank 2 (third row is sum of first two over GF(2))
    }

    #[test]
    fn test_full_rank() {
        let m = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 0, 0]),
            GF2Vector::from_bits(&[0, 1, 0]),
            GF2Vector::from_bits(&[0, 0, 1]),
        ]);
        assert_eq!(m.rank(), 3);
        assert_eq!(m.determinant().unwrap(), GF2::ONE);
    }

    #[test]
    fn test_rank_deficient() {
        let m = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 1, 0]),
            GF2Vector::from_bits(&[0, 0, 1]),
            GF2Vector::from_bits(&[1, 1, 1]),
        ]);
        assert_eq!(m.rank(), 2);
        assert_eq!(m.determinant().unwrap(), GF2::ZERO);
    }

    #[test]
    fn test_null_space() {
        // [[1,0,1],[0,1,1]] — null space should be [1,1,1]
        let m = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 0, 1]),
            GF2Vector::from_bits(&[0, 1, 1]),
        ]);
        let ns = m.null_space();
        assert_eq!(ns.len(), 1);
        // Verify Ax = 0 for each null space vector.
        for v in &ns {
            let product = m.mul_vec(v);
            assert!(product.is_zero(), "null space vector not in kernel");
        }
    }

    #[test]
    fn test_determinant_non_square() {
        let m = GF2Matrix::zero(2, 3);
        assert!(m.determinant().is_err());
    }

    #[test]
    fn test_solve() {
        // A = [[1,0],[0,1]], b = [1,1] => x = [1,1]
        let a = GF2Matrix::identity(2);
        let b = GF2Vector::from_bits(&[1, 1]);
        let x = a.solve(&b).unwrap();
        assert_eq!(a.mul_vec(&x), b);
    }

    #[test]
    fn test_solve_inconsistent() {
        // A = [[1,0],[1,0]], b = [1,0] => inconsistent for b=[0,1]
        let a = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 0]),
            GF2Vector::from_bits(&[1, 0]),
        ]);
        let b = GF2Vector::from_bits(&[0, 1]);
        assert!(a.solve(&b).is_none());
    }

    #[test]
    fn test_transpose_roundtrip() {
        let m = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 0, 1]),
            GF2Vector::from_bits(&[0, 1, 0]),
        ]);
        let tt = m.transpose().transpose();
        assert_eq!(m, tt);
    }

    #[test]
    fn test_matrix_product() {
        let a = GF2Matrix::identity(3);
        let b = GF2Matrix::from_rows(vec![
            GF2Vector::from_bits(&[1, 1, 0]),
            GF2Vector::from_bits(&[0, 1, 1]),
            GF2Vector::from_bits(&[1, 0, 1]),
        ]);
        assert_eq!(a.mul_mat(&b), b);
    }
}
