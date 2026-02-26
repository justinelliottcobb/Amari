//! Matroid representability over GF(2), GF(3), and GF(q).
//!
//! Checks whether a matroid can be represented as the column matroid of a matrix
//! over a finite field. Provides excluded minor characterizations, exhaustive
//! search for small matroids, and standard named matroids (Fano, dual Fano).

use crate::matroid::Matroid;
use amari_core::gf2::{GF2Matrix, GF2};
use std::collections::BTreeSet;

/// Result of a representability check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepresentabilityResult {
    /// The matroid is representable; a witnessing matrix is provided.
    Representable(GF2Matrix),
    /// The matroid is not representable over this field.
    NotRepresentable,
    /// The check was inconclusive (search space too large).
    Inconclusive { reason: String },
}

/// Check if a matroid is representable over GF(2).
///
/// For small matroids (k*(n-k) ≤ 24), performs exhaustive search.
/// Otherwise checks for U_{2,4} minor first.
pub fn is_binary(matroid: &Matroid) -> RepresentabilityResult {
    // Quick excluded minor check.
    if matroid.has_excluded_minor("U24") {
        return RepresentabilityResult::NotRepresentable;
    }

    // Try exhaustive search.
    match find_gf2_representation(matroid) {
        Some(matrix) => RepresentabilityResult::Representable(matrix),
        None => {
            let k = matroid.rank;
            let n = matroid.ground_set_size;
            let free = k * (n - k);
            if free > 24 {
                RepresentabilityResult::Inconclusive {
                    reason: format!("search space 2^{} too large", free),
                }
            } else {
                RepresentabilityResult::NotRepresentable
            }
        }
    }
}

/// Check if a matroid is representable over GF(3).
///
/// Uses local GF(3) arithmetic. Checks excluded minors first.
pub fn is_ternary(matroid: &Matroid) -> RepresentabilityResult {
    // GF(3) excluded minors: U_{2,5}, U_{3,5}, F_7^-, F_7^-* (the non-Fano and its dual)
    // But the classical result is: M is ternary iff it has no U_{2,5} or U_{3,5} minor...
    // Actually that's not quite right. The excluded minors for GF(3) are: U_{2,5}, U_{3,5}, F_7, F_7*.
    // Check the two standard ones.
    if matroid.has_excluded_minor("F7") {
        return RepresentabilityResult::NotRepresentable;
    }
    if matroid.has_excluded_minor("F7*") {
        return RepresentabilityResult::NotRepresentable;
    }

    // For small matroids, try exhaustive GF(3) search.
    let k = matroid.rank;
    let n = matroid.ground_set_size;
    let free = k * (n - k);

    if free > 16 {
        return RepresentabilityResult::Inconclusive {
            reason: format!("GF(3) search space 3^{} too large", free),
        };
    }

    match find_gf3_representation(matroid) {
        Some(matrix) => {
            // Convert to GF2Matrix representation (just for the type; entries are mod 3).
            // We'll store the witness as a GF2 matrix with a note that it's really GF(3).
            // For API simplicity, we return a GF2Matrix with the same shape.
            RepresentabilityResult::Representable(matrix)
        }
        None => RepresentabilityResult::NotRepresentable,
    }
}

/// Check if a matroid is representable over GF(q) for prime q.
pub fn is_representable_over_gfq(matroid: &Matroid, q: u64) -> RepresentabilityResult {
    match q {
        2 => is_binary(matroid),
        3 => is_ternary(matroid),
        _ => {
            let k = matroid.rank;
            let n = matroid.ground_set_size;
            let free = k * (n - k);
            if free > 12 {
                RepresentabilityResult::Inconclusive {
                    reason: format!("GF({}) search space {}^{} too large", q, q, free),
                }
            } else {
                // Exhaustive search over GF(q).
                match find_gfq_representation(matroid, q) {
                    Some(matrix) => RepresentabilityResult::Representable(matrix),
                    None => RepresentabilityResult::NotRepresentable,
                }
            }
        }
    }
}

/// Check if a matroid is regular (representable over every field).
///
/// A matroid is regular iff representable over both GF(2) and GF(3).
pub fn is_regular(matroid: &Matroid) -> bool {
    // Regular iff no U_{2,4}, F_7, or F_7* minor.
    !matroid.has_excluded_minor("U24")
        && !matroid.has_excluded_minor("F7")
        && !matroid.has_excluded_minor("F7*")
}

/// Check if a matroid has another matroid as a minor.
///
/// Delegates to the internal has_minor_check in matroid.rs via excluded minor interface.
/// For arbitrary minors, this performs exhaustive search.
pub fn has_minor(matroid: &Matroid, minor: &Matroid) -> bool {
    // Use the matroid module's internal minor check.
    // We check by trying all contraction/deletion sequences.
    check_has_minor(matroid, minor)
}

fn check_has_minor(matroid: &Matroid, minor: &Matroid) -> bool {
    let n = matroid.ground_set_size;
    let m_n = minor.ground_set_size;
    let m_r = minor.rank;

    if n < m_n || matroid.rank < m_r {
        return false;
    }
    if n == m_n {
        return matroid.bases == minor.bases;
    }

    // Try deleting or contracting each element and recurse.
    for e in 0..n {
        let deleted = matroid.delete(e);
        if check_has_minor(&deleted, minor) {
            return true;
        }
        let contracted = matroid.contract(e);
        if check_has_minor(&contracted, minor) {
            return true;
        }
    }
    false
}

/// The Fano matroid F_7 (matroid of the Fano plane PG(2,2)).
///
/// Rank 3 on 7 elements. Representable over GF(2) but not GF(3).
#[must_use]
pub fn fano_matroid() -> Matroid {
    // Lines of the Fano plane: {0,1,3}, {1,2,4}, {2,3,5}, {3,4,6}, {0,4,5}, {1,5,6}, {0,2,6}
    let n = 7;
    let k = 3;
    let lines: Vec<BTreeSet<usize>> = vec![
        [0, 1, 3].iter().copied().collect(),
        [1, 2, 4].iter().copied().collect(),
        [2, 3, 5].iter().copied().collect(),
        [3, 4, 6].iter().copied().collect(),
        [0, 4, 5].iter().copied().collect(),
        [1, 5, 6].iter().copied().collect(),
        [0, 2, 6].iter().copied().collect(),
    ];

    let all_triples: Vec<BTreeSet<usize>> = k_subsets_vec(n, k)
        .into_iter()
        .map(|v| v.into_iter().collect())
        .collect();
    let bases: BTreeSet<BTreeSet<usize>> = all_triples
        .into_iter()
        .filter(|t| !lines.contains(t))
        .collect();

    Matroid {
        ground_set_size: n,
        bases,
        rank: k,
    }
}

/// The dual Fano matroid F_7*.
///
/// Rank 4 on 7 elements.
#[must_use]
pub fn dual_fano_matroid() -> Matroid {
    fano_matroid().dual()
}

/// Extract the GF(2) representation matrix in standard form [I_k | A].
///
/// Returns None if the matroid is not binary.
pub fn standard_representation(matroid: &Matroid) -> Option<GF2Matrix> {
    match is_binary(matroid) {
        RepresentabilityResult::Representable(m) => Some(m),
        _ => None,
    }
}

/// Given a GF(2) matrix, extract its column matroid.
///
/// The bases are k-element subsets of columns that are linearly independent.
#[must_use]
pub fn column_matroid(matrix: &GF2Matrix) -> Matroid {
    let k = matrix.nrows();
    let n = matrix.ncols();

    let mut bases = BTreeSet::new();

    for subset in k_subsets_vec(n, k) {
        let sub = extract_columns(matrix, &subset);
        if sub.rank() == k {
            let basis: BTreeSet<usize> = subset.into_iter().collect();
            bases.insert(basis);
        }
    }

    Matroid {
        ground_set_size: n,
        bases,
        rank: k,
    }
}

// --- Internal helpers ---

fn find_gf2_representation(matroid: &Matroid) -> Option<GF2Matrix> {
    let k = matroid.rank;
    let n = matroid.ground_set_size;
    if k == 0 {
        return Some(GF2Matrix::zero(0, n));
    }

    let free_entries = k * (n - k);
    if free_entries > 24 {
        return None;
    }

    for bits in 0..(1u64 << free_entries) {
        let matrix = build_rref_matrix_gf2(k, n, bits);
        let candidate = column_matroid(&matrix);
        if candidate.bases == matroid.bases {
            return Some(matrix);
        }
    }
    None
}

fn build_rref_matrix_gf2(k: usize, n: usize, bits: u64) -> GF2Matrix {
    // Build k×n matrix in RREF: identity in first k columns, free entries elsewhere.
    let mut m = GF2Matrix::zero(k, n);
    for i in 0..k {
        m.set(i, i, GF2::ONE);
    }

    let mut bit_idx = 0;
    for i in 0..k {
        for j in k..n {
            if (bits >> bit_idx) & 1 == 1 {
                m.set(i, j, GF2::ONE);
            }
            bit_idx += 1;
        }
    }
    m
}

fn extract_columns(matrix: &GF2Matrix, cols: &[usize]) -> GF2Matrix {
    let k = matrix.nrows();
    let new_n = cols.len();
    let mut sub = GF2Matrix::zero(k, new_n);
    for (new_j, &old_j) in cols.iter().enumerate() {
        for i in 0..k {
            sub.set(i, new_j, matrix.get(i, old_j));
        }
    }
    sub
}

/// GF(3) representation search.
fn find_gf3_representation(matroid: &Matroid) -> Option<GF2Matrix> {
    let k = matroid.rank;
    let n = matroid.ground_set_size;
    if k == 0 {
        return Some(GF2Matrix::zero(0, n));
    }

    let free_entries = k * (n - k);
    if free_entries > 16 {
        return None;
    }

    // 3^free_entries possibilities.
    let total = 3u64.pow(free_entries as u32);
    for pattern in 0..total {
        let matrix_gf3 = build_rref_matrix_gf3(k, n, pattern);
        // Check if column matroid over GF(3) matches.
        let candidate = column_matroid_gf3(&matrix_gf3, k, n);
        if candidate.bases == matroid.bases {
            // Return as GF2Matrix (shape only; caller knows it's GF(3)).
            let mut result = GF2Matrix::zero(k, n);
            for i in 0..k {
                for j in 0..n {
                    if matrix_gf3[i * n + j] != 0 {
                        result.set(i, j, GF2::ONE);
                    }
                }
            }
            return Some(result);
        }
    }
    None
}

fn build_rref_matrix_gf3(k: usize, n: usize, pattern: u64) -> Vec<u8> {
    let mut m = vec![0u8; k * n];
    // Identity in first k columns.
    for i in 0..k {
        m[i * n + i] = 1;
    }
    // Free entries from pattern (base-3 digits).
    let mut p = pattern;
    for i in 0..k {
        for j in k..n {
            m[i * n + j] = (p % 3) as u8;
            p /= 3;
        }
    }
    m
}

fn column_matroid_gf3(matrix: &[u8], k: usize, n: usize) -> Matroid {
    let mut bases = BTreeSet::new();

    for subset in k_subsets_vec(n, k) {
        if gf3_columns_independent(matrix, k, n, &subset) {
            let basis: BTreeSet<usize> = subset.into_iter().collect();
            bases.insert(basis);
        }
    }

    Matroid {
        ground_set_size: n,
        bases,
        rank: k,
    }
}

fn gf3_columns_independent(matrix: &[u8], k: usize, n: usize, cols: &[usize]) -> bool {
    // Extract submatrix and compute rank via Gaussian elimination mod 3.
    let m = cols.len();
    let mut sub = vec![0u8; k * m];
    for (new_j, &old_j) in cols.iter().enumerate() {
        for i in 0..k {
            sub[i * m + new_j] = matrix[i * n + old_j];
        }
    }

    // Gaussian elimination mod 3.
    let mut pivot_row = 0;
    for col in 0..m {
        // Find pivot.
        let mut found = None;
        for row in pivot_row..k {
            if sub[row * m + col] != 0 {
                found = Some(row);
                break;
            }
        }
        let Some(pr) = found else { continue };

        // Swap rows.
        if pr != pivot_row {
            for c in 0..m {
                sub.swap(pr * m + c, pivot_row * m + c);
            }
        }

        // Scale pivot to 1.
        let inv = gf3_inv(sub[pivot_row * m + col]);
        for c in 0..m {
            sub[pivot_row * m + c] = (sub[pivot_row * m + c] * inv) % 3;
        }

        // Eliminate.
        for row in 0..k {
            if row != pivot_row && sub[row * m + col] != 0 {
                let factor = sub[row * m + col];
                for c in 0..m {
                    sub[row * m + c] =
                        (sub[row * m + c] + 3 - (factor * sub[pivot_row * m + c]) % 3) % 3;
                }
            }
        }

        pivot_row += 1;
    }

    pivot_row == k
}

fn gf3_inv(x: u8) -> u8 {
    match x % 3 {
        1 => 1,
        2 => 2,
        _ => 0, // undefined for 0
    }
}

/// GF(q) representation search for prime q.
fn find_gfq_representation(matroid: &Matroid, q: u64) -> Option<GF2Matrix> {
    let k = matroid.rank;
    let n = matroid.ground_set_size;
    if k == 0 {
        return Some(GF2Matrix::zero(0, n));
    }

    let free_entries = k * (n - k);
    let total = q.checked_pow(free_entries as u32)?;

    if total > 1_000_000 {
        return None;
    }

    for pattern in 0..total {
        let matrix = build_rref_matrix_gfq(k, n, pattern, q);
        let candidate = column_matroid_gfq(&matrix, k, n, q);
        if candidate.bases == matroid.bases {
            let mut result = GF2Matrix::zero(k, n);
            for i in 0..k {
                for j in 0..n {
                    if matrix[i * n + j] != 0 {
                        result.set(i, j, GF2::ONE);
                    }
                }
            }
            return Some(result);
        }
    }
    None
}

fn build_rref_matrix_gfq(k: usize, n: usize, pattern: u64, q: u64) -> Vec<u64> {
    let mut m = vec![0u64; k * n];
    for i in 0..k {
        m[i * n + i] = 1;
    }
    let mut p = pattern;
    for i in 0..k {
        for j in k..n {
            m[i * n + j] = p % q;
            p /= q;
        }
    }
    m
}

fn column_matroid_gfq(matrix: &[u64], k: usize, n: usize, q: u64) -> Matroid {
    let mut bases = BTreeSet::new();
    for subset in k_subsets_vec(n, k) {
        if gfq_columns_independent(matrix, k, n, &subset, q) {
            let basis: BTreeSet<usize> = subset.into_iter().collect();
            bases.insert(basis);
        }
    }
    Matroid {
        ground_set_size: n,
        bases,
        rank: k,
    }
}

fn gfq_columns_independent(matrix: &[u64], k: usize, n: usize, cols: &[usize], q: u64) -> bool {
    let m = cols.len();
    let mut sub: Vec<u64> = vec![0; k * m];
    for (new_j, &old_j) in cols.iter().enumerate() {
        for i in 0..k {
            sub[i * m + new_j] = matrix[i * n + old_j];
        }
    }

    let mut pivot_row = 0;
    for col in 0..m {
        let mut found = None;
        for row in pivot_row..k {
            if sub[row * m + col] != 0 {
                found = Some(row);
                break;
            }
        }
        let Some(pr) = found else { continue };

        if pr != pivot_row {
            for c in 0..m {
                sub.swap(pr * m + c, pivot_row * m + c);
            }
        }

        let inv = gfq_inv(sub[pivot_row * m + col], q);
        for c in 0..m {
            sub[pivot_row * m + c] = (sub[pivot_row * m + c] * inv) % q;
        }

        for row in 0..k {
            if row != pivot_row && sub[row * m + col] != 0 {
                let factor = sub[row * m + col];
                for c in 0..m {
                    sub[row * m + c] =
                        (sub[row * m + c] + q - (factor * sub[pivot_row * m + c]) % q) % q;
                }
            }
        }

        pivot_row += 1;
    }

    pivot_row == k
}

fn gfq_inv(x: u64, q: u64) -> u64 {
    // Extended Euclidean algorithm for modular inverse.
    if x == 0 {
        return 0;
    }
    let mut a = x as i64;
    let mut b = q as i64;
    let mut x0 = 0i64;
    let mut x1 = 1i64;
    while a > 1 {
        let quotient = a / b;
        let temp = b;
        b = a % b;
        a = temp;
        let temp = x0;
        x0 = x1 - quotient * x0;
        x1 = temp;
    }
    ((x1 % q as i64 + q as i64) % q as i64) as u64
}

fn k_subsets_vec(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(k);
    gen_subsets_vec(n, k, 0, &mut current, &mut result);
    result
}

fn gen_subsets_vec(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    let remaining = k - current.len();
    if start + remaining > n {
        return;
    }
    for i in start..=(n - remaining) {
        current.push(i);
        gen_subsets_vec(n, k, i + 1, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u24_not_binary() {
        let m = Matroid::uniform(2, 4);
        assert_eq!(is_binary(&m), RepresentabilityResult::NotRepresentable);
    }

    #[test]
    fn test_u23_is_binary() {
        let m = Matroid::uniform(2, 3);
        match is_binary(&m) {
            RepresentabilityResult::Representable(matrix) => {
                assert_eq!(matrix.nrows(), 2);
                assert_eq!(matrix.ncols(), 3);
            }
            other => panic!("Expected Representable, got {:?}", other),
        }
    }

    #[test]
    fn test_fano_is_binary() {
        let f7 = fano_matroid();
        match is_binary(&f7) {
            RepresentabilityResult::Representable(_) => {}
            other => panic!("F7 should be binary, got {:?}", other),
        }
    }

    #[test]
    fn test_fano_not_ternary() {
        let f7 = fano_matroid();
        match is_ternary(&f7) {
            RepresentabilityResult::NotRepresentable => {}
            other => panic!("F7 should not be ternary, got {:?}", other),
        }
    }

    #[test]
    fn test_column_matroid_roundtrip() {
        let m = Matroid::uniform(2, 3);
        if let RepresentabilityResult::Representable(matrix) = is_binary(&m) {
            let recovered = column_matroid(&matrix);
            assert_eq!(recovered.bases, m.bases);
        }
    }

    #[test]
    fn test_is_regular_u23() {
        let m = Matroid::uniform(2, 3);
        assert!(is_regular(&m));
    }

    #[test]
    fn test_is_not_regular_u24() {
        let m = Matroid::uniform(2, 4);
        assert!(!is_regular(&m));
    }

    #[test]
    fn test_dual_fano() {
        let f7star = dual_fano_matroid();
        assert_eq!(f7star.rank, 4);
        assert_eq!(f7star.ground_set_size, 7);
    }

    #[test]
    fn test_fano_matroid_properties() {
        let f7 = fano_matroid();
        assert_eq!(f7.rank, 3);
        assert_eq!(f7.ground_set_size, 7);
        // F7 should have C(7,3) - 7 = 28 bases (35 total triples - 7 lines)
        assert_eq!(f7.bases.len(), 28);
    }
}
