//! F_q point counting on Grassmannians and Schubert varieties.
//!
//! Provides:
//! - Point counting on Grassmannians over finite fields
//! - Schubert cell and variety point counting
//! - Poincaré polynomials and Weil zeta function data
//! - Integration with amari-core's GF(2) module

use crate::littlewood_richardson::Partition;
use crate::schubert::SchubertClass;
use crate::{EnumerativeError, EnumerativeResult};
use amari_core::gf2::gaussian_binomial;

/// Count F_q-rational points on Gr(k, n).
///
/// Uses the Gaussian binomial coefficient [n choose k]_q.
#[must_use]
pub fn grassmannian_points(k: usize, n: usize, q: u64) -> u64 {
    gaussian_binomial(n, k, q)
}

/// Count F_q-rational points in a Schubert cell C_λ ⊂ Gr(k, n).
///
/// |C_λ(F_q)| = q^|λ| where |λ| is the size of the partition.
#[must_use]
pub fn schubert_cell_points(partition: &Partition, q: u64) -> u64 {
    q.pow(partition.size() as u32)
}

/// Count F_q-rational points on a Schubert variety X_λ ⊂ Gr(k, n).
///
/// |X_λ(F_q)| = Σ_{μ ≤ λ} q^|μ| where the sum is over all partitions
/// contained in λ (containment/Bruhat order).
#[must_use]
pub fn schubert_variety_points(partition: &Partition, grassmannian: (usize, usize), q: u64) -> u64 {
    let (k, n) = grassmannian;
    let m = n - k;
    let mut count = 0u64;

    for mu in partitions_in_box(k, m) {
        if partition.contains(&mu) {
            count += q.pow(mu.size() as u32);
        }
    }
    count
}

/// Count F_q-rational points in the intersection of Schubert varieties.
///
/// When the intersection is transverse (codimensions sum to dim Gr),
/// this equals the Schubert intersection number (independent of q).
pub fn schubert_intersection_points(
    classes: &[SchubertClass],
    grassmannian: (usize, usize),
    q: u64,
) -> EnumerativeResult<u64> {
    let (k, n) = grassmannian;
    let gr_dim = k * (n - k);

    let total_codim: usize = classes.iter().map(|c| c.codimension()).sum();

    if total_codim > gr_dim {
        return Ok(0); // overdetermined
    }

    if total_codim == gr_dim {
        // Transverse: use LR coefficients (independent of q).
        use crate::schubert::{IntersectionResult, SchubertCalculus};
        let mut calc = SchubertCalculus::new(grassmannian);
        match calc.multi_intersect(classes) {
            IntersectionResult::Finite(n) => Ok(n),
            IntersectionResult::Empty => Ok(0),
            _ => Err(EnumerativeError::ComputationError(
                "unexpected non-transverse result".to_string(),
            )),
        }
    } else {
        // Non-transverse: the count depends on q.
        // For a single Schubert condition, this is the variety point count.
        if classes.len() == 1 {
            let partition = Partition::new(classes[0].partition.clone());
            Ok(schubert_variety_points(&partition, grassmannian, q))
        } else {
            // General non-transverse case: punt to computation error for now.
            Err(EnumerativeError::ComputationError(
                "non-transverse multi-class intersection counting not yet implemented".to_string(),
            ))
        }
    }
}

/// Poincaré polynomial of a Schubert variety X_λ.
///
/// P_{X_λ}(t) = Σ_{μ ≤ λ} t^|μ|.
/// Coefficients are Betti numbers b_{2i} = #{μ ≤ λ : |μ| = i}.
#[must_use]
pub fn schubert_poincare_polynomial(
    partition: &Partition,
    grassmannian: (usize, usize),
) -> Vec<u64> {
    let (k, n) = grassmannian;
    let m = n - k;
    let max_size = partition.size();
    let mut coeffs = vec![0u64; max_size + 1];

    for mu in partitions_in_box(k, m) {
        if partition.contains(&mu) {
            let s = mu.size();
            coeffs[s] += 1;
        }
    }

    // Trim trailing zeros.
    while coeffs.len() > 1 && coeffs.last() == Some(&0) {
        coeffs.pop();
    }
    coeffs
}

/// Poincaré polynomial of Gr(k, n).
///
/// P_{Gr(k,n)}(t) = [n choose k]_t (Gaussian binomial as a polynomial in t).
#[must_use]
pub fn grassmannian_poincare_polynomial(k: usize, n: usize) -> Vec<u64> {
    let m = n - k;
    let max_size = k * m;
    let mut coeffs = vec![0u64; max_size + 1];

    // The Gaussian binomial as a polynomial: sum over all partitions in k × m box.
    for mu in partitions_in_box(k, m) {
        let s = mu.size();
        coeffs[s] += 1;
    }

    while coeffs.len() > 1 && coeffs.last() == Some(&0) {
        coeffs.pop();
    }
    coeffs
}

/// Weil zeta function exponents for a Schubert variety.
///
/// Returns the exponents {|μ| : μ ≤ λ} so that:
/// Z(X_λ, t) = ∏_i 1/(1 − q^{e_i} · t)
#[must_use]
pub fn schubert_zeta_exponents(partition: &Partition, grassmannian: (usize, usize)) -> Vec<usize> {
    let (k, n) = grassmannian;
    let m = n - k;
    let mut exponents = Vec::new();

    for mu in partitions_in_box(k, m) {
        if partition.contains(&mu) {
            exponents.push(mu.size());
        }
    }
    exponents.sort_unstable();
    exponents
}

/// Count F_{q^n}-rational points for n = 1, 2, ..., max_n.
///
/// Returns |X(F_{q^n})| for each extension degree.
#[must_use]
pub fn point_counts_over_extensions(
    partition: &Partition,
    grassmannian: (usize, usize),
    q: u64,
    max_n: usize,
) -> Vec<u64> {
    (1..=max_n)
        .map(|ext| {
            let q_n = q.pow(ext as u32);
            schubert_variety_points(partition, grassmannian, q_n)
        })
        .collect()
}

/// Enumerate all partitions fitting in a k × m box.
///
/// Returns partitions with at most k parts, each ≤ m.
fn partitions_in_box(k: usize, m: usize) -> Vec<Partition> {
    let mut result = Vec::new();
    let mut current = Vec::with_capacity(k);
    gen_partitions(k, m, m, &mut current, &mut result);
    result
}

fn gen_partitions(
    remaining_parts: usize,
    max_part: usize,
    _box_width: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Partition>,
) {
    // Add current partition (including empty).
    result.push(Partition::new(current.clone()));

    if remaining_parts == 0 {
        return;
    }

    for part in 1..=max_part {
        current.push(part);
        gen_partitions(remaining_parts - 1, part, _box_width, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grassmannian_points_gr13() {
        assert_eq!(grassmannian_points(1, 3, 2), 7);
    }

    #[test]
    fn test_grassmannian_points_gr24() {
        assert_eq!(grassmannian_points(2, 4, 2), 35);
    }

    #[test]
    fn test_schubert_cell_points() {
        let lambda = Partition::new(vec![2, 1]);
        assert_eq!(schubert_cell_points(&lambda, 2), 8); // 2^3
    }

    #[test]
    fn test_schubert_variety_sum_equals_grassmannian() {
        // Σ_λ |C_λ(F_q)| = |Gr(k,n; F_q)|
        let (k, n, q) = (2, 4, 2u64);
        let m = n - k;
        let mut total = 0u64;
        for mu in partitions_in_box(k, m) {
            total += q.pow(mu.size() as u32);
        }
        assert_eq!(total, grassmannian_points(k, n, q));
    }

    #[test]
    fn test_poincare_polynomial_gr24_at_q() {
        let poincare = grassmannian_poincare_polynomial(2, 4);
        // Evaluate at t=2: should give |Gr(2,4; F_2)| = 35
        let mut val = 0u64;
        let mut t_power = 1u64;
        for &c in &poincare {
            val += c * t_power;
            t_power *= 2;
        }
        assert_eq!(val, 35);
    }

    #[test]
    fn test_zeta_exponents() {
        let lambda = Partition::new(vec![2, 1]);
        let exps = schubert_zeta_exponents(&lambda, (2, 4));
        // Partitions ≤ [2,1] in 2×2 box: [], [1], [1,1], [2], [2,1]
        // sizes: 0, 1, 2, 2, 3
        assert!(exps.contains(&0));
        assert!(exps.contains(&3));
    }

    #[test]
    fn test_point_counts_extensions() {
        // |Gr(1,2; F_{2^n})| = 2^n + 1
        let lambda = Partition::new(vec![1]); // full Grassmannian Gr(1,2) = P^1
        let counts = point_counts_over_extensions(&lambda, (1, 2), 2, 3);
        assert_eq!(counts[0], 3); // 2^1 + 1
        assert_eq!(counts[1], 5); // 2^2 + 1
        assert_eq!(counts[2], 9); // 2^3 + 1
    }

    #[test]
    fn test_transverse_intersection() {
        // σ_1^4 on Gr(2,4) = 2
        let sigma1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
        let classes = vec![sigma1.clone(), sigma1.clone(), sigma1.clone(), sigma1];
        let count = schubert_intersection_points(&classes, (2, 4), 2).unwrap();
        assert_eq!(count, 2);
    }
}
