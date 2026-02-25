//! Weight enumerators for binary linear codes.
//!
//! Provides the `BinaryCode` type for [n, k, d] binary linear codes,
//! weight enumerator computation, MacWilliams identity, and standard codes
//! (Hamming, simplex, Reed-Muller).

use crate::matroid::Matroid;
use crate::EnumerativeError;
use amari_core::gf2::{GF2Matrix, GF2Vector, GF2};
use num_rational::Rational64;

/// A binary linear code [n, k, d].
///
/// k-dimensional subspace of GF(2)^n with minimum distance d.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryCode {
    /// Generator matrix (k × n), in systematic form if possible.
    generator: GF2Matrix,
    /// Code length.
    n: usize,
    /// Code dimension.
    k: usize,
}

impl BinaryCode {
    /// Create a code from its generator matrix.
    ///
    /// The matrix is reduced to RREF. Its rank determines k.
    pub fn from_generator(matrix: GF2Matrix) -> Result<Self, EnumerativeError> {
        let n = matrix.ncols();
        let mut gen = matrix;
        gen.reduced_row_echelon();
        let k = gen.rank();
        if k == 0 {
            return Err(EnumerativeError::CodeError(
                "generator matrix has rank 0".to_string(),
            ));
        }
        // Keep only the k nonzero rows.
        let rows: Vec<GF2Vector> = (0..k).map(|i| gen.row(i).clone()).collect();
        let gen = GF2Matrix::from_rows(rows);
        Ok(Self {
            generator: gen,
            n,
            k,
        })
    }

    /// Create a code from its parity check matrix H.
    ///
    /// The code is C = ker(H).
    pub fn from_parity_check(h: GF2Matrix) -> Result<Self, EnumerativeError> {
        let null = h.null_space();
        if null.is_empty() {
            return Err(EnumerativeError::CodeError(
                "parity check matrix has trivial null space".to_string(),
            ));
        }
        let gen = GF2Matrix::from_rows(null);
        Self::from_generator(gen)
    }

    /// Code parameters [n, k, d].
    #[must_use]
    pub fn parameters(&self) -> (usize, usize, usize) {
        (self.n, self.k, self.minimum_distance())
    }

    /// Code length n.
    #[must_use]
    pub fn length(&self) -> usize {
        self.n
    }

    /// Code dimension k.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.k
    }

    /// Minimum distance d (minimum Hamming weight of nonzero codewords).
    #[must_use]
    pub fn minimum_distance(&self) -> usize {
        let dist = self.weight_distribution();
        for (i, &count) in dist.iter().enumerate() {
            if i > 0 && count > 0 {
                return i;
            }
        }
        self.n // shouldn't happen for a valid code
    }

    /// Generator matrix.
    #[must_use]
    pub fn generator_matrix(&self) -> &GF2Matrix {
        &self.generator
    }

    /// Parity check matrix H such that H * c = 0 for all codewords c.
    #[must_use]
    pub fn parity_check_matrix(&self) -> GF2Matrix {
        // We need H such that for all codewords c, H * c = 0.
        // Equivalently, the rows of H span the orthogonal complement of the row space of G.
        // If G is in RREF [I_k | P], then H = [P^T | I_{n-k}] (columns reordered to match pivots).

        // Get RREF and pivots.
        let mut rref = self.generator.clone();
        let pivots = rref.reduced_row_echelon();
        let n_k = self.n - self.k;

        if n_k == 0 {
            return GF2Matrix::zero(0, self.n);
        }

        let free_cols: Vec<usize> = (0..self.n).filter(|c| !pivots.contains(c)).collect();

        // Build H: (n-k) × n matrix.
        // For each free column f_j (j=0..n-k-1):
        //   Row j of H has 1 at position f_j (identity part)
        //   and for each pivot column p_i, H[j][p_i] = RREF[i][f_j]
        let mut h = GF2Matrix::zero(n_k, self.n);
        for (j, &fc) in free_cols.iter().enumerate() {
            h.set(j, fc, GF2::ONE);
            for (i, &pc) in pivots.iter().enumerate() {
                let val = rref.get(i, fc);
                h.set(j, pc, val);
            }
        }

        h
    }

    /// The dual code C^⊥.
    #[must_use]
    pub fn dual(&self) -> BinaryCode {
        let h = self.parity_check_matrix();
        BinaryCode {
            n: self.n,
            k: self.n - self.k,
            generator: h,
        }
    }

    /// Whether the code is self-dual (C = C^⊥).
    #[must_use]
    pub fn is_self_dual(&self) -> bool {
        if self.k != self.n - self.k {
            return false;
        }
        // C is self-dual iff G·G^T = 0 (all codewords are mutually orthogonal)
        let gt = self.generator.transpose();
        let product = self.generator.mul_mat(&gt);
        // Check all entries are zero
        for i in 0..product.nrows() {
            for j in 0..product.ncols() {
                if product.get(i, j).value() != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Enumerate all 2^k codewords.
    pub fn codewords(&self) -> Vec<GF2Vector> {
        assert!(self.k <= 20, "codeword enumeration limited to k ≤ 20");
        let gt = self.generator.transpose(); // n × k
        let mut words = Vec::with_capacity(1 << self.k);
        for msg in 0..(1u64 << self.k) {
            let message = GF2Vector::from_u64(self.k, msg);
            words.push(gt.mul_vec(&message)); // n × k times k-vec = n-vec
        }
        words
    }

    /// Encode a k-bit message into an n-bit codeword.
    #[must_use]
    pub fn encode(&self, message: &GF2Vector) -> GF2Vector {
        assert_eq!(message.dim(), self.k);
        let gt = self.generator.transpose(); // n × k
        gt.mul_vec(message) // n × k times k-vec = n-vec
    }

    /// Syndrome of a received vector.
    #[must_use]
    pub fn syndrome(&self, received: &GF2Vector) -> GF2Vector {
        let h = self.parity_check_matrix();
        // Syndrome = H * r (treating r as column vector)
        // But our mul_vec does M * v, so we need H * received.
        // Since H is (n-k) × n and received is n-dimensional, this works.
        h.mul_vec(received)
    }

    /// The column matroid of the generator matrix.
    #[must_use]
    pub fn matroid(&self) -> Matroid {
        crate::representability::column_matroid(&self.generator)
    }

    /// Weight enumerator coefficients A_0, A_1, ..., A_n.
    ///
    /// A_i = number of codewords of Hamming weight i.
    #[must_use]
    pub fn weight_enumerator(&self) -> Vec<u64> {
        self.weight_distribution()
    }

    /// Weight distribution: A_i = number of codewords of Hamming weight i.
    #[must_use]
    pub fn weight_distribution(&self) -> Vec<u64> {
        let mut dist = vec![0u64; self.n + 1];
        for codeword in self.codewords() {
            dist[codeword.weight()] += 1;
        }
        dist
    }

    /// MacWilliams identity: dual weight enumerator computed from transform.
    ///
    /// W_{C⊥}[j] = (1/2^k) · Σ_i W_C[i] · K_j(i; n)
    /// where K_j(i; n) is the Krawtchouk polynomial.
    #[must_use]
    pub fn dual_weight_enumerator(&self) -> Vec<Rational64> {
        let w = self.weight_distribution();
        let n = self.n;
        let size = Rational64::from(1i64 << self.k);

        let mut dual_w = vec![Rational64::from(0); n + 1];
        for (j, dw_j) in dual_w.iter_mut().enumerate() {
            let mut sum = Rational64::from(0);
            for (i, &wi) in w.iter().enumerate() {
                let k_val = krawtchouk(j, i, n);
                sum += Rational64::from(wi as i64) * Rational64::from(k_val);
            }
            *dw_j = sum / size;
        }
        dual_w
    }
}

/// Krawtchouk polynomial K_j(i; n) = Σ_s (-1)^s C(i,s) C(n-i, j-s).
fn krawtchouk(j: usize, i: usize, n: usize) -> i64 {
    let mut sum = 0i64;
    for s in 0..=j.min(i) {
        if j >= s && n >= i && (n - i) >= (j - s) {
            let sign = if s % 2 == 0 { 1i64 } else { -1 };
            let c1 = binomial(i, s) as i64;
            let c2 = binomial(n - i, j - s) as i64;
            sum += sign * c1 * c2;
        }
    }
    sum
}

fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: u64 = 1;
    for i in 0..k {
        result = result * (n - i) as u64 / (i + 1) as u64;
    }
    result
}

/// Construct the Hamming code Ham(r, 2).
///
/// Parameters: [2^r − 1, 2^r − 1 − r, 3].
#[must_use]
pub fn hamming_code(r: usize) -> BinaryCode {
    assert!((2..=10).contains(&r), "Hamming code r must be in [2, 10]");
    let n = (1 << r) - 1;
    let n_minus_k = r;

    // Parity check matrix: all nonzero r-bit vectors as columns.
    let mut h = GF2Matrix::zero(n_minus_k, n);
    for col in 0..n {
        let val = col + 1; // nonzero r-bit value
        for bit in 0..r {
            if (val >> bit) & 1 == 1 {
                h.set(bit, col, GF2::ONE);
            }
        }
    }

    BinaryCode::from_parity_check(h).expect("Hamming code construction should not fail")
}

/// Construct the simplex code (dual of Hamming code).
///
/// Parameters: [2^r − 1, r, 2^(r−1)].
#[must_use]
pub fn simplex_code(r: usize) -> BinaryCode {
    hamming_code(r).dual()
}

/// Construct the Reed-Muller code RM(r, m).
///
/// Parameters: [2^m, Σ_{i=0}^{r} C(m,i), 2^(m−r)].
pub fn reed_muller_code(r: usize, m: usize) -> Result<BinaryCode, EnumerativeError> {
    if r > m {
        return Err(EnumerativeError::CodeError(format!(
            "Reed-Muller order r={} must be <= m={}",
            r, m
        )));
    }
    if m > 10 {
        return Err(EnumerativeError::CodeError(format!(
            "Reed-Muller m={} too large (max 10)",
            m
        )));
    }

    let n = 1usize << m;

    // Evaluation points: all m-bit vectors.
    let points: Vec<Vec<u8>> = (0..n)
        .map(|v| (0..m).map(|i| ((v >> i) & 1) as u8).collect())
        .collect();

    // Monomials of degree ≤ r: subsets of {0,...,m-1} of size ≤ r.
    let mut generator_rows = Vec::new();
    for deg in 0..=r {
        for subset in k_subsets_vec(m, deg) {
            // Evaluate monomial x_{i1} · x_{i2} · ... at each point.
            let mut row = GF2Vector::zero(n);
            for (j, point) in points.iter().enumerate() {
                let val: u8 = subset.iter().map(|&i| point[i]).product();
                if val == 1 {
                    row.set(j, GF2::ONE);
                }
            }
            generator_rows.push(row);
        }
    }

    let gen = GF2Matrix::from_rows(generator_rows);
    BinaryCode::from_generator(gen)
}

/// Construct the extended Golay code [24, 12, 8].
#[must_use]
pub fn extended_golay_code() -> BinaryCode {
    // The extended binary Golay code generator matrix.
    // P matrix (12×12) from the standard construction.
    #[rustfmt::skip]
    let p: [[u8; 12]; 12] = [
        [1,1,0,1,1,1,0,0,0,1,0,1],
        [1,0,1,1,1,0,0,0,1,0,1,1],
        [0,1,1,1,0,0,0,1,0,1,1,1],
        [1,1,1,0,0,0,1,0,1,1,0,1],
        [1,1,0,0,0,1,0,1,1,0,1,1],
        [1,0,0,0,1,0,1,1,0,1,1,1],
        [0,0,0,1,0,1,1,0,1,1,1,1],
        [0,0,1,0,1,1,0,1,1,1,0,1],
        [0,1,0,1,1,0,1,1,1,0,0,1],
        [1,0,1,1,0,1,1,1,0,0,0,1],
        [0,1,1,0,1,1,1,0,0,0,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,0],
    ];

    let k = 12;
    let n = 24;
    let mut gen = GF2Matrix::zero(k, n);

    // [I_12 | P]
    for (i, p_row) in p.iter().enumerate().take(k) {
        gen.set(i, i, GF2::ONE);
        for (j, &p_val) in p_row.iter().enumerate() {
            if p_val == 1 {
                gen.set(i, k + j, GF2::ONE);
            }
        }
    }

    BinaryCode {
        generator: gen,
        n,
        k,
    }
}

/// Singleton bound: d ≤ n − k + 1.
#[must_use]
pub fn singleton_bound(n: usize, k: usize) -> usize {
    n - k + 1
}

/// Hamming bound: maximum number of correctable errors for [n, k] code.
///
/// t such that Σ_{i=0}^{t} C(n, i) ≤ 2^(n−k).
#[must_use]
pub fn hamming_bound(n: usize, k: usize) -> usize {
    let sphere_cap = 1u64 << (n - k);
    let mut sum = 0u64;
    for t in 0..=n {
        sum += binomial(n, t);
        if sum > sphere_cap {
            return if t > 0 { t - 1 } else { 0 };
        }
    }
    n
}

/// Plotkin bound: d ≤ 2^(k−1) · n / (2^k − 1) for d even.
///
/// Returns None if the bound doesn't apply.
#[must_use]
pub fn plotkin_bound(n: usize, k: usize) -> Option<usize> {
    if k == 0 {
        return Some(n);
    }
    let two_k = 1u64 << k;
    let two_k_minus_1 = 1u64 << (k - 1);
    let bound = two_k_minus_1 * n as u64 / (two_k - 1);
    Some(bound as usize)
}

/// Gilbert-Varshamov bound: maximum guaranteed minimum distance.
///
/// Largest d such that Σ_{i=0}^{d−2} C(n−1, i) < 2^(n−k).
#[must_use]
pub fn gilbert_varshamov_bound(n: usize, k: usize) -> usize {
    if n <= k {
        return 1;
    }
    let target = 1u64 << (n - k);
    let mut sum = 0u64;
    for d in 1..=n {
        if d >= 2 {
            sum += binomial(n - 1, d - 2);
        }
        if sum >= target {
            return d - 1;
        }
    }
    n
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
    fn test_hamming_code_parameters() {
        let ham = hamming_code(3);
        let (n, k, d) = ham.parameters();
        assert_eq!(n, 7);
        assert_eq!(k, 4);
        assert_eq!(d, 3);
    }

    #[test]
    fn test_hamming_weight_distribution() {
        let ham = hamming_code(3);
        let dist = ham.weight_distribution();
        // [7,4,3] Hamming code: A_0=1, A_3=7, A_4=7, A_7=1
        assert_eq!(dist[0], 1);
        assert_eq!(dist[3], 7);
        assert_eq!(dist[4], 7);
        assert_eq!(dist[7], 1);
    }

    #[test]
    fn test_simplex_is_dual_of_hamming() {
        let _ham = hamming_code(3);
        let simp = simplex_code(3);
        assert_eq!(simp.n, 7);
        assert_eq!(simp.k, 3);
        // Simplex code minimum distance = 2^(r-1) = 4
        assert_eq!(simp.minimum_distance(), 4);
    }

    #[test]
    fn test_macwilliams_consistency() {
        let ham = hamming_code(3);
        // Compute dual weight enumerator via MacWilliams.
        let dual_we = ham.dual_weight_enumerator();
        // Compute dual weight enumerator directly.
        let dual = ham.dual();
        let direct_we = dual.weight_distribution();

        for (j, &d) in direct_we.iter().enumerate() {
            let from_transform = dual_we[j];
            assert_eq!(
                from_transform,
                Rational64::from(d as i64),
                "MacWilliams mismatch at weight {}",
                j
            );
        }
    }

    #[test]
    fn test_reed_muller_rm1_3() {
        let rm = reed_muller_code(1, 3).unwrap();
        let (n, k, d) = rm.parameters();
        assert_eq!(n, 8);
        assert_eq!(k, 4);
        assert_eq!(d, 4);
    }

    #[test]
    fn test_singleton_bound() {
        assert_eq!(singleton_bound(7, 4), 4);
    }

    #[test]
    fn test_encode_syndrome_roundtrip() {
        let ham = hamming_code(3);
        let msg = GF2Vector::from_bits(&[1, 0, 1, 1]);
        let codeword = ham.encode(&msg);
        let syn = ham.syndrome(&codeword);
        assert!(syn.is_zero(), "syndrome of valid codeword should be zero");
    }

    #[test]
    fn test_extended_golay_code() {
        let golay = extended_golay_code();
        assert_eq!(golay.length(), 24);
        assert_eq!(golay.dimension(), 12);
        // Minimum distance should be 8.
        assert_eq!(golay.minimum_distance(), 8);
    }
}
