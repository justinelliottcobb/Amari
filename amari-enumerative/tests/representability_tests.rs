#![cfg(feature = "gf2")]
#![allow(unused_imports)]
use amari_enumerative::{
    column_matroid, dual_fano_matroid, fano_matroid, has_minor, is_binary, is_regular, is_ternary,
    standard_representation, Matroid, RepresentabilityResult,
};
use std::collections::BTreeSet;

fn uniform_matroid(r: usize, n: usize) -> Matroid {
    let ground: Vec<usize> = (0..n).collect();
    let bases: BTreeSet<BTreeSet<usize>> = subsets_of_size(&ground, r)
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect();
    Matroid::from_bases(n, bases).unwrap()
}

fn subsets_of_size(elements: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if elements.len() < k {
        return vec![];
    }
    let mut result = Vec::new();
    for (i, &e) in elements.iter().enumerate() {
        for mut sub in subsets_of_size(&elements[i + 1..], k - 1) {
            sub.insert(0, e);
            result.push(sub);
        }
    }
    result
}

#[test]
fn test_fano_matroid_is_binary() {
    let f7 = fano_matroid();
    assert_eq!(f7.ground_set_size, 7);
    assert_eq!(f7.rank, 3);
    match is_binary(&f7) {
        RepresentabilityResult::Representable(_) => {}
        other => panic!("Fano matroid should be binary, got {:?}", other),
    }
}

#[test]
fn test_fano_not_ternary() {
    // Fano matroid is not representable over GF(3)
    let f7 = fano_matroid();
    match is_ternary(&f7) {
        RepresentabilityResult::NotRepresentable => {}
        other => panic!("Fano matroid should NOT be ternary, got {:?}", other),
    }
}

#[test]
fn test_fano_not_regular() {
    // Regular = binary AND ternary. Fano is binary but not ternary.
    let f7 = fano_matroid();
    assert!(!is_regular(&f7));
}

#[test]
fn test_u24_not_binary() {
    // U(2,4) is the excluded minor for binary matroids
    let u24 = uniform_matroid(2, 4);
    match is_binary(&u24) {
        RepresentabilityResult::NotRepresentable => {}
        other => panic!("U(2,4) should NOT be binary, got {:?}", other),
    }
}

#[test]
fn test_u23_is_binary() {
    // U(2,3) is representable over every field
    let u23 = uniform_matroid(2, 3);
    match is_binary(&u23) {
        RepresentabilityResult::Representable(_) => {}
        other => panic!("U(2,3) should be binary, got {:?}", other),
    }
}

#[test]
fn test_column_matroid_roundtrip() {
    // column_matroid of a representation should give back a matroid
    // with the same bases (up to isomorphism)
    let f7 = fano_matroid();
    if let RepresentabilityResult::Representable(matrix) = is_binary(&f7) {
        let m = column_matroid(&matrix);
        assert_eq!(m.ground_set_size, f7.ground_set_size);
        assert_eq!(m.rank, f7.rank);
        assert_eq!(m.bases.len(), f7.bases.len());
    }
}

#[test]
fn test_standard_representation() {
    let f7 = fano_matroid();
    let rep = standard_representation(&f7);
    assert!(rep.is_some());
}

#[test]
fn test_dual_fano_matroid() {
    let f7_star = dual_fano_matroid();
    assert_eq!(f7_star.ground_set_size, 7);
    assert_eq!(f7_star.rank, 4); // rank of dual = n - rank
}

#[test]
fn test_has_minor_u24_in_fano() {
    // Fano matroid does NOT contain U(2,4) as a minor (it's binary)
    let f7 = fano_matroid();
    let u24 = uniform_matroid(2, 4);
    assert!(!has_minor(&f7, &u24));
}
