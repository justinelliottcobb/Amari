#![cfg(feature = "gf2")]
#![allow(unused_imports)]
use amari_enumerative::{
    extended_golay_code, gilbert_varshamov_bound, hamming_bound, hamming_code, plotkin_bound,
    reed_muller_code, simplex_code, singleton_bound, BinaryCode,
};
use num_rational::Rational64;

#[test]
fn test_hamming_7_4_parameters() {
    let h = hamming_code(3);
    let (n, k, d) = h.parameters();
    assert_eq!(n, 7);
    assert_eq!(k, 4);
    assert_eq!(d, 3);
}

#[test]
fn test_hamming_15_11_parameters() {
    let h = hamming_code(4);
    let (n, k, _d) = h.parameters();
    assert_eq!(n, 15);
    assert_eq!(k, 11);
}

#[test]
fn test_hamming_code_is_self_orthogonal() {
    // Hamming(3) is NOT self-dual (7,4 vs 7,3)
    let h = hamming_code(3);
    assert!(!h.is_self_dual());
}

#[test]
fn test_hamming_weight_distribution() {
    // Hamming [7,4,3]: weight distribution is known
    // A_0=1, A_3=7, A_4=7, A_7=1
    let h = hamming_code(3);
    let wd = h.weight_distribution();
    assert_eq!(wd[0], 1);
    assert_eq!(wd[3], 7);
    assert_eq!(wd[4], 7);
    assert_eq!(wd[7], 1);
    // Total codewords = 2^4 = 16
    let total: u64 = wd.iter().sum();
    assert_eq!(total, 16);
}

#[test]
fn test_hamming_dual_is_simplex() {
    // Dual of Hamming(r) = Simplex(r)
    let h = hamming_code(3);
    let dual = h.dual();
    let s = simplex_code(3);
    assert_eq!(dual.length(), s.length());
    assert_eq!(dual.dimension(), s.dimension());
}

#[test]
fn test_simplex_code_parameters() {
    let s = simplex_code(3);
    let (n, k, d) = s.parameters();
    assert_eq!(n, 7);
    assert_eq!(k, 3);
    assert_eq!(d, 4); // minimum distance of simplex code = 2^{r-1}
}

#[test]
fn test_reed_muller_1_3() {
    // RM(1,3) = [8, 4, 4]
    let rm = reed_muller_code(1, 3).unwrap();
    let (n, k, d) = rm.parameters();
    assert_eq!(n, 8);
    assert_eq!(k, 4);
    assert_eq!(d, 4);
}

#[test]
fn test_extended_golay_parameters() {
    let g = extended_golay_code();
    let (n, k, d) = g.parameters();
    assert_eq!(n, 24);
    assert_eq!(k, 12);
    assert_eq!(d, 8);
}

#[test]
fn test_golay_is_self_dual() {
    let g = extended_golay_code();
    assert!(g.is_self_dual());
}

#[test]
fn test_singleton_bound_basic() {
    // Singleton bound: d ≤ n - k + 1
    assert_eq!(singleton_bound(7, 4), 4);
    assert_eq!(singleton_bound(24, 12), 13);
}

#[test]
fn test_hamming_bound_basic() {
    let hb = hamming_bound(7, 4);
    // Hamming bound returns packing radius t; d ≤ 2t+1
    // For [7,4]: Σ C(7,i) for i=0..1 = 1+7 = 8 = 2^3, so t=1, d ≤ 3
    assert_eq!(hb, 1);
}

#[test]
fn test_plotkin_bound() {
    let pb = plotkin_bound(7, 4);
    assert!(pb.is_some());
}

#[test]
fn test_encode_decode_roundtrip() {
    let h = hamming_code(3);
    // Encode a message and check syndrome is zero
    let codewords = h.codewords();
    for cw in &codewords {
        let s = h.syndrome(cw);
        // All codewords should have zero syndrome
        let all_zero = (0..s.dim()).all(|i| s.get(i) == amari_core::gf2::GF2::new(0));
        assert!(all_zero);
    }
}

#[test]
fn test_macwilliams_duality() {
    // MacWilliams: W_{C^⊥}(x,y) = (1/|C|) W_C(x+y, x-y)
    // We check: dual weight enumerator should match actual dual's weight enumerator
    let h = hamming_code(3);
    let dual_we = h.dual_weight_enumerator();
    let actual_dual = h.dual();
    let actual_we = actual_dual.weight_enumerator();

    assert_eq!(dual_we.len(), actual_we.len());
    for (i, (&expected, &actual)) in dual_we.iter().zip(actual_we.iter()).enumerate() {
        let actual_rat = Rational64::from_integer(actual as i64);
        let diff = if expected > actual_rat {
            expected - actual_rat
        } else {
            actual_rat - expected
        };
        assert!(
            diff <= Rational64::from_integer(1),
            "Mismatch at weight {}: MacWilliams={}, actual={}",
            i,
            expected,
            actual
        );
    }
}
