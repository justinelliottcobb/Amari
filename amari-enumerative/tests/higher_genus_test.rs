use amari_enumerative::{
    HigherGenusCurve, PTInvariant, DTInvariant, AdvancedCurveCounting,
    JacobianData, ChowClass
};
use amari_enumerative::gromov_witten::CurveClass;
use num_rational::Rational64;
use std::collections::BTreeMap;

#[test]
fn test_higher_genus_curve_creation() {
    let curve = HigherGenusCurve::new(3, 5);

    assert_eq!(curve.genus, 3);
    assert_eq!(curve.degree, 5);
    assert_eq!(curve.canonical_degree, 4); // 2g - 2 = 2*3 - 2 = 4
    assert_eq!(curve.jacobian.dimension, 3);
}

#[test]
fn test_riemann_roch_dimension() {
    let curve = HigherGenusCurve::new(2, 3);

    // For genus 2, canonical degree is 2
    // Test various divisor degrees
    assert_eq!(curve.riemann_roch_dimension(0), 0);
    assert_eq!(curve.riemann_roch_dimension(1), 0);
    assert_eq!(curve.riemann_roch_dimension(2), 0); // Canonical divisor
    assert!(curve.riemann_roch_dimension(4) > 0); // High degree
    assert!(curve.riemann_roch_dimension(10) > 0); // Very high degree
}

#[test]
fn test_brill_noether_theory() {
    let curve = HigherGenusCurve::new(3, 5);

    // Test Brill-Noether number ρ(g,r,d) = g - (r+1)(g-d+r)
    let rho = curve.brill_noether_number(1, 4); // Linear series g¹₄ on genus 3 curve
    assert_eq!(rho, 3 - 2 * (3 - 4 + 1)); // 3 - 2*0 = 3

    assert!(curve.is_brill_noether_general(1, 4)); // ρ ≥ 0
    assert!(!curve.is_brill_noether_general(1, 1)); // ρ < 0
}

#[test]
fn test_clifford_index() {
    let curve = HigherGenusCurve::new(4, 6);

    // Clifford index for special divisors
    let clifford = curve.clifford_index(3);
    assert!(clifford.is_some());

    // Outside valid range
    assert!(curve.clifford_index(0).is_none());
    assert!(curve.clifford_index(8).is_none()); // 2g = 8 for g=4
}

#[test]
fn test_gieseker_petri_defect() {
    let curve = HigherGenusCurve::new(3, 4);

    let defect = curve.gieseker_petri_defect(2, 2);
    assert!(defect >= 0); // Defect is always non-negative
}

#[test]
fn test_virtual_gw_invariant() {
    let curve = HigherGenusCurve::new(1, 2);

    let insertion_classes = vec![
        ChowClass::new(1, Rational64::from(1)),
        ChowClass::new(1, Rational64::from(1))
    ];

    let gw_result = curve.virtual_gw_invariant("P2", &insertion_classes);
    assert!(gw_result.is_ok());

    let gw_value = gw_result.unwrap();
    assert!(gw_value >= Rational64::from(0));
}

#[test]
fn test_moduli_stack_data() {
    let curve = HigherGenusCurve::new(2, 3);

    // M₂ has dimension 3*2 - 3 = 3
    assert_eq!(curve.moduli_stack.dimension, 3);
    assert_eq!(curve.moduli_stack.genus, 2);
    assert!(curve.moduli_stack.tautological_classes.len() > 0);

    // Test κ classes
    assert!(curve.moduli_stack.tautological_classes.contains_key("kappa_1"));
    assert!(curve.moduli_stack.tautological_classes.contains_key("kappa_2"));
}

#[test]
fn test_moduli_space_intersection_numbers() {
    let curve = HigherGenusCurve::new(2, 3);

    let classes = vec!["kappa_1".to_string()];
    let intersection = curve.moduli_stack.intersection_number(&classes);
    assert!(intersection.is_ok());

    // For wrong dimensional classes, should get 0
    let wrong_classes = vec!["kappa_1".to_string(), "kappa_1".to_string(), "kappa_1".to_string()];
    let zero_result = curve.moduli_stack.intersection_number(&wrong_classes);
    assert_eq!(zero_result.unwrap(), Rational64::from(0));
}

#[test]
fn test_jacobian_data() {
    let curve = HigherGenusCurve::new(3, 4);

    assert_eq!(curve.jacobian.dimension, 3);
    assert!(curve.jacobian.is_principally_polarized);
    assert_eq!(curve.jacobian.theta_divisor.ambient_dimension, 3);

    // Test Abel-Jacobi map
    let aj_result = curve.jacobian.abel_jacobi_map(2);
    assert!(aj_result.is_ok());

    let aj_element = aj_result.unwrap();
    assert_eq!(aj_element.degree, 2);
    assert_eq!(aj_element.jacobian_coordinates.len(), 3);
}

#[test]
fn test_theta_divisor() {
    let curve = HigherGenusCurve::new(2, 3);

    let zeroes = curve.jacobian.theta_divisor.compute_zeroes();
    let expected_count = 2_usize.pow(2 - 1); // 2^(g-1)
    assert_eq!(zeroes.len(), expected_count);

    // Test theta function
    let characteristic = vec![Rational64::from(0); 4]; // 2g = 4 for g=2
    let theta_result = curve.jacobian.theta_function(&characteristic);
    assert!(theta_result.is_ok());
}

#[test]
fn test_torelli_map() {
    let curve = HigherGenusCurve::new(3, 4);

    assert!(curve.jacobian.torelli_map.is_torelli_injective()); // g ≥ 2
    assert_eq!(curve.jacobian.torelli_map.genus, 3);
    assert_eq!(curve.jacobian.torelli_map.jacobian_locus_dimension, 6); // 3g - 3

    // Test genus 1 case - Torelli theorem is false for g=1
    let genus1_curve = HigherGenusCurve::new(1, 2);
    assert!(!genus1_curve.jacobian.torelli_map.is_torelli_injective()); // False for g=1 by Torelli theorem
}

#[test]
fn test_pt_invariant() {
    let curve_class = CurveClass::new(3);
    let mut pt = PTInvariant::new(curve_class, 1);

    let result = pt.compute_virtual();
    assert!(result.is_ok());
    assert!(pt.pt_number >= Rational64::from(0));
}

#[test]
fn test_dt_invariant() {
    let mut chern_character = BTreeMap::new();
    chern_character.insert(0, Rational64::from(1));
    chern_character.insert(1, Rational64::from(3));
    chern_character.insert(2, Rational64::from(1));

    let mut dt = DTInvariant::new(chern_character);

    let result = dt.compute_localization();
    assert!(result.is_ok());
    assert_eq!(dt.dt_number, Rational64::from(5)); // 1 + 3 + 1 = 5
}

#[test]
fn test_mnop_correspondence() {
    let mut chern_character = BTreeMap::new();
    chern_character.insert(0, Rational64::from(1));
    chern_character.insert(1, Rational64::from(2));

    let dt = DTInvariant::new(chern_character);

    let gw_invariants = vec![
        Rational64::from(1), // genus 0
        Rational64::from(2), // genus 1
        Rational64::from(1), // genus 2
    ];

    let mnop_result = dt.mnop_correspondence(&gw_invariants);
    assert!(mnop_result.is_ok());

    let mnop_value = mnop_result.unwrap();
    assert!(mnop_value > Rational64::from(0));
}

#[test]
fn test_advanced_curve_counting() {
    let mut counting = AdvancedCurveCounting::new("P2".to_string(), 2);

    let result = counting.compute_all_invariants(3);
    assert!(result.is_ok());

    // Check that invariants were computed
    assert!(!counting.gw_invariants.is_empty());
    assert!(!counting.pt_invariants.is_empty());
    assert!(!counting.dt_invariants.is_empty());

    // Test MNOP verification
    let mnop_valid = counting.verify_mnop_correspondence();
    assert!(mnop_valid.is_ok());

    // Test summary
    let summary = counting.summary();
    assert!(summary.contains("P2"));
    assert!(summary.contains("Maximum genus: 2"));
}

#[test]
fn test_higher_genus_edge_cases() {
    // Genus 0 curve (rational curve)
    let rational_curve = HigherGenusCurve::new(0, 3);
    assert_eq!(rational_curve.canonical_degree, -2); // 2*0 - 2 = -2
    assert_eq!(rational_curve.moduli_stack.dimension, -3); // Needs marked points

    // Very high genus
    let high_genus = HigherGenusCurve::new(10, 5);
    assert_eq!(high_genus.canonical_degree, 18); // 2*10 - 2
    assert_eq!(high_genus.moduli_stack.dimension, 27); // 3*10 - 3

    // Test Riemann-Roch for high genus
    assert!(high_genus.riemann_roch_dimension(25) > 0);
}

#[test]
fn test_jacobian_edge_cases() {
    let curve = HigherGenusCurve::new(1, 2);

    // Genus 1 has 1-dimensional Jacobian (elliptic curve)
    assert_eq!(curve.jacobian.dimension, 1);

    // Test invalid Abel-Jacobi input
    let invalid_aj = curve.jacobian.abel_jacobi_map(-1);
    assert!(invalid_aj.is_err());

    // Test invalid theta characteristic
    let invalid_theta = curve.jacobian.theta_function(&[Rational64::from(0)]);
    assert!(invalid_theta.is_err());
}

#[test]
fn test_virtual_dimension_computations() {
    let curve = HigherGenusCurve::new(2, 4);

    // Test different target spaces
    let p1_result = curve.virtual_gw_invariant("P1", &[]);
    assert!(p1_result.is_ok());

    let p3_result = curve.virtual_gw_invariant("P3", &[]);
    assert!(p3_result.is_ok());

    // Test with many insertions (should give 0 for wrong dimension)
    let many_classes = vec![ChowClass::new(1, Rational64::from(1)); 10];
    let zero_result = curve.virtual_gw_invariant("P2", &many_classes);
    assert_eq!(zero_result.unwrap(), Rational64::from(0));
}

#[test]
fn test_mathematical_consistency() {
    let curve = HigherGenusCurve::new(3, 5);

    // Riemann-Roch should be monotonic for high degrees
    let h0_deg10 = curve.riemann_roch_dimension(10);
    let h0_deg15 = curve.riemann_roch_dimension(15);
    assert!(h0_deg15 >= h0_deg10);

    // Canonical degree should equal 2g - 2
    assert_eq!(curve.canonical_degree, 2 * (curve.genus as i64) - 2);

    // Jacobian dimension should equal genus
    assert_eq!(curve.jacobian.dimension, curve.genus);

    // Brill-Noether number should satisfy expected properties
    let rho1 = curve.brill_noether_number(1, 5);
    let rho2 = curve.brill_noether_number(2, 5);
    assert!(rho2 <= rho1); // Higher r should give smaller ρ
}