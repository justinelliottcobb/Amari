#![allow(unused_imports)]
use amari_enumerative::{
    flag_f_vector, inverse_kl_polynomial, kl_is_non_negative, kl_polynomial, z_polynomial, Matroid,
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
fn test_kl_polynomial_boolean_matroid() {
    // Boolean matroid U(n,n) has trivial KL polynomial = [1]
    let m = uniform_matroid(3, 3);
    let p = kl_polynomial(&m);
    assert_eq!(p, vec![1]);
}

#[test]
fn test_kl_polynomial_uniform_u24() {
    let m = uniform_matroid(2, 4);
    let p = kl_polynomial(&m);
    // KL polynomial of U(2,4) is known: P = 1
    assert!(!p.is_empty());
    assert_eq!(p[0], 1);
}

#[test]
fn test_kl_non_negativity() {
    // KL non-negativity conjecture (proved by Elias-Williamson for Coxeter matroids)
    let m = uniform_matroid(2, 5);
    assert!(kl_is_non_negative(&m));
}

#[test]
fn test_z_polynomial_rank_one() {
    // Rank-1 matroid: Z-polynomial should be simple
    let m = uniform_matroid(1, 3);
    let z = z_polynomial(&m);
    assert!(!z.is_empty());
}

#[test]
fn test_inverse_kl_polynomial_rank_one() {
    // Rank-1 matroids have trivial inverse KL polynomial
    let m = uniform_matroid(1, 3);
    let q = inverse_kl_polynomial(&m);
    assert_eq!(q, vec![1]);
}

#[test]
fn test_flag_f_vector_u23() {
    let m = uniform_matroid(2, 3);
    let ffv = flag_f_vector(&m);
    // The empty set flag should always have count 1
    let empty: BTreeSet<usize> = BTreeSet::new();
    assert_eq!(ffv.get(&empty), Some(&1));
}

#[test]
fn test_kl_polynomial_consistency() {
    // For any matroid, degree of P_M should be < floor(rank/2)
    let m = uniform_matroid(3, 6);
    let p = kl_polynomial(&m);
    let max_degree = m.rank / 2;
    assert!(p.len() <= max_degree + 1);
}
