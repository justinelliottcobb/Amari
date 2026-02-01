//! Littlewood-Richardson coefficient computation
//!
//! Implements the combinatorial algorithm for computing LR coefficients
//! via enumeration of valid skew tableaux.
//!
//! The Littlewood-Richardson coefficient `c^ν_{λμ}` appears in the product of Schubert classes:
//!
//! ```text
//! σ_λ · σ_μ = Σ_ν c^ν_{λμ} σ_ν
//! ```
//!
//! These coefficients count Young tableaux of skew shape `ν/λ` with content `μ` satisfying
//! the *lattice word condition*: reading the tableau right-to-left, top-to-bottom, at every
//! point the number of `i`s seen is ≥ the number of `(i+1)`s seen.
//!
//! # Contracts
//!
//! Functions in this module satisfy the following contracts:
//! - `lr_coefficient`: Non-negative, symmetric in λ and μ
//! - `schubert_product`: Coefficients sum correctly, respects box constraints

use std::collections::BTreeMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A partition (Young diagram) represented as weakly decreasing sequence.
///
/// # Invariants
///
/// - Parts are stored in weakly decreasing order: `parts[i] >= parts[i+1]`
/// - All parts are positive: `parts[i] > 0`
///
/// # Contracts
///
/// - `new`: `ensures(result.is_valid())`
/// - `contains`: `ensures(result == (self.part(i) >= other.part(i) for all i))`
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Partition {
    /// Parts of the partition, weakly decreasing: λ_1 ≥ λ_2 ≥ ... ≥ λ_k > 0
    pub parts: Vec<usize>,
}

impl Default for Partition {
    fn default() -> Self {
        Self::empty()
    }
}

impl From<Vec<usize>> for Partition {
    fn from(parts: Vec<usize>) -> Self {
        Self::new(parts)
    }
}

impl FromIterator<usize> for Partition {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self::new(iter.into_iter().collect())
    }
}

impl Partition {
    /// Create a new partition, automatically sorting and removing zeros.
    ///
    /// # Contracts
    ///
    /// - `ensures(result.parts.is_sorted_by(|a, b| b.cmp(a)))`
    /// - `ensures(result.parts.iter().all(|&p| p > 0))`
    #[must_use]
    pub fn new(mut parts: Vec<usize>) -> Self {
        parts.sort_unstable_by(|a, b| b.cmp(a)); // Descending order
        parts.retain(|&x| x > 0);
        Self { parts }
    }

    /// The empty partition.
    ///
    /// # Contracts
    ///
    /// - `ensures(result.size() == 0)`
    /// - `ensures(result.length() == 0)`
    #[must_use]
    pub const fn empty() -> Self {
        Self { parts: Vec::new() }
    }

    /// Size (sum of parts).
    ///
    /// # Contracts
    ///
    /// - `ensures(result == self.parts.iter().sum())`
    #[must_use]
    pub fn size(&self) -> usize {
        self.parts.iter().sum()
    }

    /// Length (number of nonzero parts).
    ///
    /// # Contracts
    ///
    /// - `ensures(result == self.parts.len())`
    #[must_use]
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    /// Check if this is the empty partition.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Get part at index (0 if index out of bounds).
    ///
    /// # Contracts
    ///
    /// - `ensures(i < self.length() ==> result == self.parts[i])`
    /// - `ensures(i >= self.length() ==> result == 0)`
    #[must_use]
    pub fn part(&self, i: usize) -> usize {
        self.parts.get(i).copied().unwrap_or(0)
    }

    /// Check if this partition contains another (ν ⊇ λ iff ν_i ≥ λ_i for all i).
    ///
    /// # Contracts
    ///
    /// - `ensures(result == (0..max(self.length(), other.length())).all(|i| self.part(i) >= other.part(i)))`
    #[must_use]
    pub fn contains(&self, other: &Partition) -> bool {
        let max_len = self.length().max(other.length());
        (0..max_len).all(|i| self.part(i) >= other.part(i))
    }

    /// Check if partition fits in a k × m box.
    ///
    /// # Contracts
    ///
    /// - `ensures(result == (self.length() <= k && self.parts.iter().all(|&p| p <= m)))`
    #[must_use]
    pub fn fits_in_box(&self, k: usize, m: usize) -> bool {
        self.length() <= k && self.parts.iter().all(|&p| p <= m)
    }

    /// Conjugate (transpose) partition.
    ///
    /// # Contracts
    ///
    /// - `ensures(result.conjugate() == self)` (involution)
    /// - `ensures(result.size() == self.size())`
    #[must_use]
    pub fn conjugate(&self) -> Partition {
        if self.parts.is_empty() {
            return Partition::empty();
        }

        let max_part = self.parts[0];
        let conj: Vec<usize> = (1..=max_part)
            .map(|j| self.parts.iter().filter(|&&p| p >= j).count())
            .filter(|&c| c > 0)
            .collect();

        Partition { parts: conj }
    }

    /// Check if partition is valid (satisfies invariants).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.parts.windows(2).all(|w| w[0] >= w[1]) && self.parts.iter().all(|&p| p > 0)
    }
}

/// A skew shape ν/λ (the cells in ν but not in λ).
///
/// # Invariants
///
/// - `outer.contains(&inner)` is true
/// - `cells` contains exactly the cells in outer but not in inner
#[derive(Debug, Clone)]
pub struct SkewShape {
    /// Outer partition
    pub outer: Partition,
    /// Inner partition (must be contained in outer)
    pub inner: Partition,
    /// Cells of the skew shape as (row, col) pairs, row-major order
    pub cells: Vec<(usize, usize)>,
}

impl SkewShape {
    /// Create a new skew shape.
    ///
    /// # Contracts
    ///
    /// - `requires(outer.contains(&inner))`
    /// - `ensures(result.is_some() ==> result.unwrap().size() == outer.size() - inner.size())`
    #[must_use]
    pub fn new(outer: Partition, inner: Partition) -> Option<Self> {
        if !outer.contains(&inner) {
            return None;
        }

        let cells: Vec<(usize, usize)> = (0..outer.length())
            .flat_map(|row| {
                let start_col = inner.part(row);
                let end_col = outer.part(row);
                (start_col..end_col).map(move |col| (row, col))
            })
            .collect();

        Some(Self {
            outer,
            inner,
            cells,
        })
    }

    /// Number of cells.
    #[must_use]
    pub fn size(&self) -> usize {
        self.cells.len()
    }

    /// Check if shape is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Get cell by index in reading order.
    #[must_use]
    pub fn cell(&self, index: usize) -> Option<(usize, usize)> {
        self.cells.get(index).copied()
    }

    /// Find index of cell above the given cell index (if exists and in skew shape).
    #[must_use]
    pub fn cell_above(&self, index: usize) -> Option<usize> {
        let (row, col) = self.cells.get(index)?;
        if *row == 0 {
            return None;
        }
        let above = (row - 1, *col);
        self.cells.iter().position(|&c| c == above)
    }

    /// Check if two cell indices are in the same row.
    #[must_use]
    pub fn same_row(&self, i: usize, j: usize) -> bool {
        matches!((self.cells.get(i), self.cells.get(j)), (Some((r1, _)), Some((r2, _))) if r1 == r2)
    }
}

/// A semistandard Young tableau of skew shape.
///
/// # Invariants (when valid)
///
/// - Rows are weakly increasing
/// - Columns are strictly increasing
#[derive(Debug, Clone)]
pub struct SkewTableau {
    /// The skew shape
    pub shape: SkewShape,
    /// Filling: label at each cell (indexed same as shape.cells)
    pub filling: Vec<u8>,
}

impl SkewTableau {
    /// Create a new tableau (unchecked).
    #[must_use]
    pub fn new(shape: SkewShape, filling: Vec<u8>) -> Self {
        Self { shape, filling }
    }

    /// Check if the filling is semistandard (rows weakly increasing, columns strictly increasing).
    ///
    /// # Contracts
    ///
    /// - `ensures(result ==> rows are weakly increasing)`
    /// - `ensures(result ==> columns are strictly increasing)`
    #[must_use]
    pub fn is_semistandard(&self) -> bool {
        self.shape
            .cells
            .iter()
            .enumerate()
            .all(|(idx, &(row, col))| {
                let label = self.filling[idx];

                // Check cell to the left (if in skew shape)
                let left_ok = if col > self.shape.inner.part(row) {
                    self.shape
                        .cells
                        .iter()
                        .position(|&c| c == (row, col - 1))
                        .is_none_or(|left_idx| self.filling[left_idx] <= label)
                } else {
                    true
                };

                // Check cell above (if exists)
                let above_ok = self
                    .shape
                    .cell_above(idx)
                    .is_none_or(|above_idx| self.filling[above_idx] < label);

                left_ok && above_ok
            })
    }

    /// Get the content (μ) of this tableau: μ_i = count of label i.
    ///
    /// # Contracts
    ///
    /// - `ensures(result.size() == self.shape.size())`
    #[must_use]
    pub fn content(&self) -> Partition {
        let max_label = self.filling.iter().max().copied().unwrap_or(0) as usize;
        if max_label == 0 {
            return Partition::empty();
        }

        let counts: Vec<usize> = (1..=max_label)
            .map(|label| self.filling.iter().filter(|&&l| l == label as u8).count())
            .collect();

        Partition::new(counts)
    }

    /// Read the tableau in reverse reading order (right-to-left, top-to-bottom).
    #[must_use]
    pub fn reverse_reading_word(&self) -> Vec<u8> {
        let max_row = self.shape.outer.length();
        let mut word = Vec::with_capacity(self.filling.len());

        for row in 0..max_row {
            let mut row_cells: Vec<(usize, u8)> = self
                .shape
                .cells
                .iter()
                .enumerate()
                .filter(|(_, &(r, _))| r == row)
                .map(|(idx, &(_, col))| (col, self.filling[idx]))
                .collect();

            row_cells.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            word.extend(row_cells.into_iter().map(|(_, label)| label));
        }

        word
    }

    /// Check the lattice word condition on reverse reading word.
    ///
    /// # Contracts
    ///
    /// - `ensures(result ==> for all prefixes: count(i) >= count(i+1))`
    #[must_use]
    pub fn satisfies_lattice_condition(&self) -> bool {
        let word = self.reverse_reading_word();
        let max_label = word.iter().max().copied().unwrap_or(0) as usize;

        let mut counts = vec![0i32; max_label + 1];

        for &label in &word {
            counts[label as usize] += 1;

            // Check: count of i must be >= count of i+1 at every prefix
            if (1..max_label).any(|i| counts[i] < counts[i + 1]) {
                return false;
            }
        }

        true
    }
}

/// Compute the Littlewood-Richardson coefficient c^ν_{λμ}.
///
/// # Contracts
///
/// - `ensures(result >= 0)`
/// - `ensures(lr_coefficient(lambda, mu, nu) == lr_coefficient(mu, lambda, nu))` (symmetry)
/// - `ensures(nu.contains(lambda) || result == 0)`
/// - `ensures(skew_size == mu.size() || result == 0)`
#[must_use]
pub fn lr_coefficient(lambda: &Partition, mu: &Partition, nu: &Partition) -> u64 {
    // Quick checks
    if !nu.contains(lambda) {
        return 0;
    }

    let skew = match SkewShape::new(nu.clone(), lambda.clone()) {
        Some(s) => s,
        None => return 0,
    };

    if skew.size() != mu.size() {
        return 0;
    }

    if skew.is_empty() && mu.is_empty() {
        return 1; // Empty tableau contributes 1
    }

    // Enumerate valid tableaux
    let mut count = 0u64;
    enumerate_lr_tableaux(&skew, mu, &mut vec![], &mut count);
    count
}

/// Recursively enumerate LR tableaux.
fn enumerate_lr_tableaux(
    skew: &SkewShape,
    content: &Partition,
    partial: &mut Vec<u8>,
    count: &mut u64,
) {
    if partial.len() == skew.size() {
        // Complete filling - check lattice condition
        let tableau = SkewTableau::new(skew.clone(), partial.clone());
        if tableau.satisfies_lattice_condition() {
            *count += 1;
        }
        return;
    }

    let idx = partial.len();
    let (row, col) = skew.cell(idx).unwrap();
    let max_label = content.length() as u8;

    for label in 1..=max_label {
        // Check content constraint
        let used = partial.iter().filter(|&&l| l == label).count();
        if used >= content.part(label as usize - 1) {
            continue;
        }

        // Check row constraint (weakly increasing)
        if col > skew.inner.part(row) {
            if let Some(left_idx) = skew.cells.iter().position(|&c| c == (row, col - 1)) {
                if left_idx < partial.len() && partial[left_idx] > label {
                    continue;
                }
            }
        }

        // Check column constraint (strictly increasing)
        if let Some(above_idx) = skew.cell_above(idx) {
            if partial[above_idx] >= label {
                continue;
            }
        }

        partial.push(label);
        enumerate_lr_tableaux(skew, content, partial, count);
        partial.pop();
    }
}

/// Expand a Schubert product σ_λ · σ_μ as a sum of Schubert classes.
///
/// # Contracts
///
/// - `ensures(result.values().all(|&c| c > 0))`
/// - `ensures(result.keys().all(|p| p.fits_in_box(k, n - k)))`
#[must_use]
pub fn schubert_product(
    lambda: &Partition,
    mu: &Partition,
    grassmannian: (usize, usize),
) -> BTreeMap<Partition, u64> {
    let (k, n) = grassmannian;
    let max_part = n - k;
    let target_size = lambda.size() + mu.size();

    #[cfg(feature = "parallel")]
    {
        schubert_product_parallel(lambda, mu, k, max_part, target_size)
    }

    #[cfg(not(feature = "parallel"))]
    {
        schubert_product_sequential(lambda, mu, k, max_part, target_size)
    }
}

/// Sequential implementation of Schubert product.
#[cfg(not(feature = "parallel"))]
fn schubert_product_sequential(
    lambda: &Partition,
    mu: &Partition,
    k: usize,
    max_part: usize,
    target_size: usize,
) -> BTreeMap<Partition, u64> {
    let mut result = BTreeMap::new();

    enumerate_partitions_in_box(k, max_part, target_size, &mut |nu| {
        let coeff = lr_coefficient(lambda, mu, &nu);
        if coeff > 0 {
            result.insert(nu, coeff);
        }
    });

    result
}

/// Parallel implementation of Schubert product using Rayon.
#[cfg(feature = "parallel")]
fn schubert_product_parallel(
    lambda: &Partition,
    mu: &Partition,
    k: usize,
    max_part: usize,
    target_size: usize,
) -> BTreeMap<Partition, u64> {
    // Collect all candidate partitions first
    let mut candidates = Vec::new();
    enumerate_partitions_in_box(k, max_part, target_size, &mut |nu| {
        candidates.push(nu);
    });

    // Only parallelize if there are enough candidates
    if candidates.len() < 4 {
        return candidates
            .into_iter()
            .filter_map(|nu| {
                let coeff = lr_coefficient(lambda, mu, &nu);
                if coeff > 0 {
                    Some((nu, coeff))
                } else {
                    None
                }
            })
            .collect();
    }

    // Compute LR coefficients in parallel
    candidates
        .into_par_iter()
        .filter_map(|nu| {
            let coeff = lr_coefficient(lambda, mu, &nu);
            if coeff > 0 {
                Some((nu, coeff))
            } else {
                None
            }
        })
        .collect()
}

/// Enumerate partitions fitting in a k × m box with given size.
fn enumerate_partitions_in_box<F>(k: usize, m: usize, size: usize, callback: &mut F)
where
    F: FnMut(Partition),
{
    fn enumerate_rec<F>(
        parts: &mut Vec<usize>,
        k: usize,
        m: usize,
        remaining: usize,
        max_part: usize,
        callback: &mut F,
    ) where
        F: FnMut(Partition),
    {
        if remaining == 0 {
            callback(Partition::new(parts.clone()));
            return;
        }

        if parts.len() >= k {
            return;
        }

        let upper = remaining.min(m).min(max_part);
        for part in (1..=upper).rev() {
            parts.push(part);
            enumerate_rec(parts, k, m, remaining - part, part, callback);
            parts.pop();
        }
    }

    let mut parts = Vec::new();
    enumerate_rec(&mut parts, k, m, size, m, callback);
}

/// Batch compute multiple LR coefficients in parallel.
///
/// Given a list of (λ, μ, ν) triples, compute all coefficients.
#[cfg(feature = "parallel")]
#[must_use]
pub fn lr_coefficients_batch(triples: &[(Partition, Partition, Partition)]) -> Vec<u64> {
    triples
        .par_iter()
        .map(|(lambda, mu, nu)| lr_coefficient(lambda, mu, nu))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_new() {
        let p = Partition::new(vec![1, 3, 2]);
        assert_eq!(p.parts, vec![3, 2, 1]);

        let p2 = Partition::new(vec![2, 0, 1, 0]);
        assert_eq!(p2.parts, vec![2, 1]);
    }

    #[test]
    fn test_partition_from_iterator() {
        let p: Partition = vec![1, 3, 2].into_iter().collect();
        assert_eq!(p.parts, vec![3, 2, 1]);
    }

    #[test]
    fn test_partition_default() {
        let p: Partition = Default::default();
        assert!(p.is_empty());
    }

    #[test]
    fn test_partition_size_length() {
        let p = Partition::new(vec![3, 2, 1]);
        assert_eq!(p.size(), 6);
        assert_eq!(p.length(), 3);
    }

    #[test]
    fn test_partition_contains() {
        let nu = Partition::new(vec![3, 2, 1]);
        let lambda = Partition::new(vec![2, 1]);
        let mu = Partition::new(vec![3, 3]);

        assert!(nu.contains(&lambda));
        assert!(!nu.contains(&mu));
    }

    #[test]
    fn test_partition_fits_in_box() {
        let p = Partition::new(vec![2, 1]);
        assert!(p.fits_in_box(2, 2));
        assert!(p.fits_in_box(3, 3));
        assert!(!p.fits_in_box(1, 2)); // Too many parts
        assert!(!p.fits_in_box(2, 1)); // Part too large
    }

    #[test]
    fn test_partition_conjugate() {
        let p = Partition::new(vec![3, 2, 1]);
        let conj = p.conjugate();
        assert_eq!(conj.parts, vec![3, 2, 1]); // Self-conjugate!

        let p2 = Partition::new(vec![4, 2, 1]);
        let conj2 = p2.conjugate();
        assert_eq!(conj2.parts, vec![3, 2, 1, 1]);

        // Check involution property
        assert_eq!(p2.conjugate().conjugate(), p2);
    }

    #[test]
    fn test_skew_shape() {
        let outer = Partition::new(vec![3, 2]);
        let inner = Partition::new(vec![1]);
        let skew = SkewShape::new(outer, inner).unwrap();

        assert_eq!(skew.size(), 4);
        assert_eq!(skew.cells, vec![(0, 1), (0, 2), (1, 0), (1, 1)]);
    }

    #[test]
    fn test_skew_shape_invalid() {
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![3]);
        assert!(SkewShape::new(outer, inner).is_none());
    }

    #[test]
    fn test_lr_coefficient_simple() {
        let lambda = Partition::new(vec![1]);
        let mu = Partition::new(vec![1]);

        let nu_2 = Partition::new(vec![2]);
        let nu_11 = Partition::new(vec![1, 1]);

        assert_eq!(lr_coefficient(&lambda, &mu, &nu_2), 1);
        assert_eq!(lr_coefficient(&lambda, &mu, &nu_11), 1);
    }

    #[test]
    fn test_lr_coefficient_symmetry() {
        let lambda = Partition::new(vec![2, 1]);
        let mu = Partition::new(vec![1, 1]);
        let nu = Partition::new(vec![3, 2]);

        // LR coefficients are symmetric in λ and μ
        assert_eq!(
            lr_coefficient(&lambda, &mu, &nu),
            lr_coefficient(&mu, &lambda, &nu)
        );
    }

    #[test]
    fn test_lr_coefficient_zero() {
        let lambda = Partition::new(vec![1]);
        let mu = Partition::new(vec![1]);
        let nu = Partition::new(vec![4]);

        assert_eq!(lr_coefficient(&lambda, &mu, &nu), 0);
    }

    #[test]
    fn test_schubert_product() {
        let lambda = Partition::new(vec![1]);
        let mu = Partition::new(vec![1]);

        let products = schubert_product(&lambda, &mu, (2, 4));

        assert_eq!(products.len(), 2);
        assert_eq!(products.get(&Partition::new(vec![2])), Some(&1));
        assert_eq!(products.get(&Partition::new(vec![1, 1])), Some(&1));
    }

    #[test]
    fn test_schubert_product_larger() {
        let lambda = Partition::new(vec![2]);
        let mu = Partition::new(vec![1]);

        let products = schubert_product(&lambda, &mu, (2, 5));

        assert_eq!(products.len(), 2);
        assert_eq!(products.get(&Partition::new(vec![3])), Some(&1));
        assert_eq!(products.get(&Partition::new(vec![2, 1])), Some(&1));
    }

    #[test]
    fn test_lattice_condition() {
        let outer = Partition::new(vec![2, 1]);
        let inner = Partition::new(vec![1]);
        let skew = SkewShape::new(outer, inner).unwrap();

        let tableau = SkewTableau::new(skew, vec![1, 2]);

        assert!(tableau.is_semistandard());
        assert!(tableau.satisfies_lattice_condition());
    }
}

// ============================================================================
// Parallel Batch Operation Tests
// ============================================================================

#[cfg(all(test, feature = "parallel"))]
mod parallel_tests {
    use super::*;

    #[test]
    fn test_lr_coefficients_batch() {
        let triples = vec![
            (
                Partition::new(vec![2, 1]),
                Partition::new(vec![1]),
                Partition::new(vec![3, 1]),
            ),
            (
                Partition::new(vec![1]),
                Partition::new(vec![1]),
                Partition::new(vec![2]),
            ),
            (
                Partition::new(vec![1]),
                Partition::new(vec![1]),
                Partition::new(vec![1, 1]),
            ),
            // Zero case
            (
                Partition::new(vec![3, 2]),
                Partition::new(vec![1]),
                Partition::new(vec![2, 1]),
            ),
        ];

        let results = lr_coefficients_batch(&triples);

        assert_eq!(results.len(), 4);
        assert!(results[0] > 0); // c^{3,1}_{2,1; 1} > 0
        assert_eq!(results[1], 1); // c^{2}_{1,1} = 1
        assert_eq!(results[2], 1); // c^{1,1}_{1,1} = 1
        assert_eq!(results[3], 0); // Size mismatch
    }

    #[test]
    fn test_lr_coefficients_batch_empty() {
        let triples: Vec<(Partition, Partition, Partition)> = vec![];
        let results = lr_coefficients_batch(&triples);
        assert!(results.is_empty());
    }

    #[test]
    fn test_lr_coefficients_batch_single() {
        let triples = vec![(
            Partition::new(vec![1]),
            Partition::empty(),
            Partition::new(vec![1]),
        )];

        let results = lr_coefficients_batch(&triples);
        assert_eq!(results, vec![1]); // c^λ_{λ,∅} = 1
    }
}
