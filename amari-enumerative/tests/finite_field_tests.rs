#![cfg(feature = "gf2")]
#![allow(unused_imports)]
use amari_enumerative::{
    grassmannian_poincare_polynomial, grassmannian_points, point_counts_over_extensions,
    schubert_cell_points, schubert_poincare_polynomial, schubert_variety_points,
    schubert_zeta_exponents, Partition,
};

#[test]
fn test_grassmannian_gr14_over_f2() {
    // Gr(1,4;F_2) = P^3(F_2) = 2^3 + 2^2 + 2 + 1 = 15
    assert_eq!(grassmannian_points(1, 4, 2), 15);
}

#[test]
fn test_grassmannian_gr24_over_f2() {
    // Gr(2,4;F_2) = [4,2]_2 = (2^4-1)(2^3-1)/((2^2-1)(2^1-1)) = 15*7/(3*1) = 35
    assert_eq!(grassmannian_points(2, 4, 2), 35);
}

#[test]
fn test_grassmannian_gr24_over_f3() {
    // Gr(2,4;F_3) = [4,2]_3 = (3^4-1)(3^3-1)/((3^2-1)(3^1-1))
    // = 80*26/(8*2) = 2080/16 = 130
    assert_eq!(grassmannian_points(2, 4, 3), 130);
}

#[test]
fn test_schubert_cell_points() {
    // Cell dimension = |λ|, points = q^|λ|
    let lambda = Partition::new(vec![2, 1]);
    assert_eq!(schubert_cell_points(&lambda, 2), 8); // 2^3
    assert_eq!(schubert_cell_points(&lambda, 3), 27); // 3^3
}

#[test]
fn test_schubert_variety_sum_equals_grassmannian() {
    // Sum over all Schubert cells = total Grassmannian points
    let q = 2u64;
    let (k, n) = (2, 4);
    let total = grassmannian_points(k, n, q);

    // Partitions fitting in k × (n-k) = 2 × 2 box
    let partitions = [
        Partition::new(vec![]),
        Partition::new(vec![1]),
        Partition::new(vec![1, 1]),
        Partition::new(vec![2]),
        Partition::new(vec![2, 1]),
        Partition::new(vec![2, 2]),
    ];

    let cell_sum: u64 = partitions.iter().map(|p| schubert_cell_points(p, q)).sum();
    assert_eq!(cell_sum, total);
}

#[test]
fn test_grassmannian_poincare_polynomial_gr13() {
    // Gr(1,3) = P^2: Poincaré = 1 + t^2 + t^4 → coefficients [1, 0, 1, 0, 1]
    let p = grassmannian_poincare_polynomial(1, 3);
    // Evaluating at t^2=q=2: 1+2+4=7 = |Gr(1,3;F_2)|
    let eval: u64 = p
        .iter()
        .enumerate()
        .map(|(i, &c)| c * 2u64.pow(i as u32))
        .sum();
    assert_eq!(eval, 7);
}

#[test]
fn test_schubert_poincare_polynomial() {
    // Point class (empty partition): Poincaré = [1]
    let empty = Partition::new(vec![]);
    let p = schubert_poincare_polynomial(&empty, (2, 4));
    assert_eq!(p[0], 1);
}

#[test]
fn test_point_counts_over_extensions() {
    // Point counts of Schubert variety over F_{q^i} for i=1..max_n
    let lambda = Partition::new(vec![1]);
    let counts = point_counts_over_extensions(&lambda, (2, 4), 2, 3);
    assert_eq!(counts.len(), 3);
    // First count should be schubert_variety_points at q=2
    assert_eq!(counts[0], schubert_variety_points(&lambda, (2, 4), 2));
}

#[test]
fn test_schubert_zeta_exponents() {
    let lambda = Partition::new(vec![1, 1]);
    let exps = schubert_zeta_exponents(&lambda, (2, 4));
    assert!(!exps.is_empty());
}
