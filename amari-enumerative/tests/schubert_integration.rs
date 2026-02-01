//! Integration tests for Schubert calculus extensions
//!
//! These tests verify the correctness of Littlewood-Richardson coefficients,
//! multi-class intersections, and namespace operations.

use amari_enumerative::{
    capability_accessible, lr_coefficient, namespace_intersection, schubert_product, Capability,
    CapabilityId, IntersectionResult, Namespace, NamespaceBuilder, NamespaceIntersection,
    Partition, SchubertCalculus, SchubertClass,
};

// =============================================================================
// Classic Schubert Calculus Tests
// =============================================================================

#[test]
fn test_lines_meeting_four_lines() {
    // Classic problem: how many lines meet 4 general lines in P³?
    // Answer: 2
    //
    // This is computed in Gr(2, 4) with 4 copies of σ_1

    let mut calc = SchubertCalculus::new((2, 4));
    let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

    let classes = vec![
        sigma_1.clone(),
        sigma_1.clone(),
        sigma_1.clone(),
        sigma_1.clone(),
    ];

    let result = calc.multi_intersect(&classes);
    assert_eq!(result, IntersectionResult::Finite(2));
}

#[test]
fn test_planes_meeting_conditions() {
    // Gr(2, 5): 2-planes in 5-space
    // Dimension = 2 × 3 = 6
    //
    // σ_1^6 should give a finite number

    let mut calc = SchubertCalculus::new((2, 5));
    let sigma_1 = SchubertClass::new(vec![1], (2, 5)).unwrap();

    let classes: Vec<_> = (0..6).map(|_| sigma_1.clone()).collect();
    let result = calc.multi_intersect(&classes);

    assert!(matches!(result, IntersectionResult::Finite(_)));
}

#[test]
fn test_schubert_product_commutativity() {
    // Schubert product should be commutative: σ_λ · σ_μ = σ_μ · σ_λ
    let lambda = Partition::new(vec![2, 1]);
    let mu = Partition::new(vec![1, 1]);

    let product1 = schubert_product(&lambda, &mu, (3, 6));
    let product2 = schubert_product(&mu, &lambda, (3, 6));

    assert_eq!(product1, product2);
}

#[test]
fn test_lr_coefficient_symmetry() {
    // LR coefficients are symmetric: c^ν_{λμ} = c^ν_{μλ}
    let lambda = Partition::new(vec![2, 1]);
    let mu = Partition::new(vec![1, 1]);
    let nu = Partition::new(vec![3, 2]);

    let c1 = lr_coefficient(&lambda, &mu, &nu);
    let c2 = lr_coefficient(&mu, &lambda, &nu);

    assert_eq!(c1, c2);
}

#[test]
fn test_pieri_rule() {
    // Pieri's rule: σ_1 · σ_λ is the sum over all λ' obtained by adding
    // a horizontal strip of size 1 to λ

    let lambda = Partition::new(vec![2, 1]);
    let sigma_1 = Partition::new(vec![1]);

    let products = schubert_product(&lambda, &sigma_1, (3, 6));

    // σ_1 · σ_{2,1} = σ_{3,1} + σ_{2,2} + σ_{2,1,1}
    assert!(products.len() >= 2);
}

// =============================================================================
// Littlewood-Richardson Coefficient Tests
// =============================================================================

#[test]
fn test_lr_coefficient_identity() {
    // c^λ_{λ, ∅} = 1 (multiplying by empty partition gives identity)
    let lambda = Partition::new(vec![3, 2, 1]);
    let empty = Partition::empty();

    let coeff = lr_coefficient(&lambda, &empty, &lambda);
    assert_eq!(coeff, 1);
}

#[test]
fn test_lr_coefficient_box_partition() {
    // Classical result: c^{(n-k)^k}_{λ, λ'} where λ' is complement in box
    let lambda = Partition::new(vec![1]);
    let mu = Partition::new(vec![1]);
    let nu = Partition::new(vec![2]);

    // c^{2}_{1,1} = 1
    assert_eq!(lr_coefficient(&lambda, &mu, &nu), 1);
}

#[test]
fn test_lr_coefficient_zero_cases() {
    // Various cases where LR coefficient should be zero

    // Case 1: Nu doesn't contain lambda
    let lambda = Partition::new(vec![3, 2]);
    let mu = Partition::new(vec![1]);
    let nu = Partition::new(vec![2, 1]); // Too small

    assert_eq!(lr_coefficient(&lambda, &mu, &nu), 0);

    // Case 2: Size mismatch
    let lambda2 = Partition::new(vec![1]);
    let mu2 = Partition::new(vec![1]);
    let nu2 = Partition::new(vec![5]); // Size 5 != 1 + 1

    assert_eq!(lr_coefficient(&lambda2, &mu2, &nu2), 0);
}

// =============================================================================
// Namespace and Capability Tests
// =============================================================================

#[test]
fn test_namespace_configuration_count() {
    // Create a namespace in Gr(3, 6)
    let ns = Namespace::full("agent", 3, 6).unwrap();

    // Should have many configurations (full Grassmannian)
    let count = ns.count_configurations();
    assert!(matches!(
        count,
        IntersectionResult::PositiveDimensional { .. }
    ));
}

#[test]
fn test_capability_restricts_configurations() {
    // Adding capabilities reduces the configuration space
    let mut ns = Namespace::full("agent", 2, 4).unwrap();

    // Add capabilities until we get a finite count
    let cap1 = Capability::new("c1", "Cap 1", vec![1], (2, 4)).unwrap();
    let cap2 = Capability::new("c2", "Cap 2", vec![1], (2, 4)).unwrap();
    let cap3 = Capability::new("c3", "Cap 3", vec![1], (2, 4)).unwrap();
    let cap4 = Capability::new("c4", "Cap 4", vec![1], (2, 4)).unwrap();

    ns.grant(cap1).unwrap();
    ns.grant(cap2).unwrap();
    ns.grant(cap3).unwrap();
    ns.grant(cap4).unwrap();

    let count = ns.count_configurations();
    // Four σ_1 conditions in Gr(2,4) gives 2 points
    assert_eq!(count, IntersectionResult::Finite(2));
}

#[test]
fn test_namespace_capability_dependency_chain() {
    // Test a chain of dependencies: admin -> write -> read
    let mut ns = Namespace::full("test", 2, 4).unwrap();

    let read = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();

    let write = Capability::new("write", "Write", vec![1], (2, 4))
        .unwrap()
        .requires(CapabilityId::new("read"));

    let admin = Capability::new("admin", "Admin", vec![1], (2, 4))
        .unwrap()
        .requires(CapabilityId::new("write"));

    // Grant in order should work
    ns.grant(read).unwrap();
    ns.grant(write).unwrap();
    ns.grant(admin).unwrap();

    assert!(ns.has_capability(&CapabilityId::new("read")));
    assert!(ns.has_capability(&CapabilityId::new("write")));
    assert!(ns.has_capability(&CapabilityId::new("admin")));
}

#[test]
fn test_namespace_intersection_same_grassmannian() {
    // Two namespaces in the same Grassmannian
    let ns1 = Namespace::full("ns1", 2, 4).unwrap();
    let ns2 = Namespace::full("ns2", 2, 4).unwrap();

    let result = namespace_intersection(&ns1, &ns2).unwrap();
    // Two full namespaces should have positive-dimensional intersection
    assert!(matches!(result, NamespaceIntersection::Subspace { .. }));
}

#[test]
fn test_namespace_intersection_different_grassmannian() {
    // Two namespaces in different Grassmannians
    let ns1 = Namespace::full("ns1", 2, 4).unwrap();
    let ns2 = Namespace::full("ns2", 3, 6).unwrap();

    let result = namespace_intersection(&ns1, &ns2).unwrap();
    assert_eq!(result, NamespaceIntersection::Incompatible);
}

#[test]
fn test_capability_accessible_check() {
    // Check if a capability is accessible from a namespace
    let ns = Namespace::full("test", 2, 4).unwrap();
    let cap = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();

    // Full namespace should be able to access any capability
    assert!(capability_accessible(&ns, &cap).unwrap());
}

#[test]
fn test_namespace_builder() {
    let read = Capability::new("read", "Read", vec![1], (2, 4)).unwrap();
    let write = Capability::new("write", "Write", vec![1], (2, 4))
        .unwrap()
        .requires(CapabilityId::new("read"));

    let ns = NamespaceBuilder::new("test", 2, 4)
        .position(vec![])
        .with_capability(read)
        .with_capability(write)
        .build()
        .unwrap();

    assert!(ns.has_capability(&CapabilityId::new("read")));
    assert!(ns.has_capability(&CapabilityId::new("write")));
}

// =============================================================================
// Partition Operations Tests
// =============================================================================

#[test]
fn test_partition_conjugate_involution() {
    // Conjugate is an involution: (λ')' = λ
    let lambda = Partition::new(vec![4, 3, 1]);
    let conjugate = lambda.conjugate();
    let double_conjugate = conjugate.conjugate();

    assert_eq!(lambda, double_conjugate);
}

#[test]
fn test_partition_self_conjugate() {
    // Some partitions are self-conjugate
    let staircase = Partition::new(vec![3, 2, 1]);
    assert_eq!(staircase, staircase.conjugate());
}

#[test]
fn test_partition_containment() {
    let large = Partition::new(vec![5, 4, 3, 2, 1]);
    let medium = Partition::new(vec![4, 3, 2]);
    let small = Partition::new(vec![2, 1]);

    assert!(large.contains(&medium));
    assert!(large.contains(&small));
    assert!(medium.contains(&small));
    assert!(!small.contains(&medium));
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

#[test]
fn test_empty_intersection() {
    // Test overdetermined system
    let mut calc = SchubertCalculus::new((2, 4));
    let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

    // 5 conditions exceed the dimension 4
    let classes: Vec<_> = (0..5).map(|_| sigma_1.clone()).collect();
    let result = calc.multi_intersect(&classes);

    assert_eq!(result, IntersectionResult::Empty);
}

#[test]
fn test_single_class_intersection() {
    let mut calc = SchubertCalculus::new((2, 4));
    let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

    let result = calc.multi_intersect(&[sigma_1]);

    // Single class with codimension 1 in dimension 4 space
    assert!(matches!(
        result,
        IntersectionResult::PositiveDimensional { dimension: 3, .. }
    ));
}

#[test]
fn test_fundamental_class() {
    // The fundamental class σ_{(n-k)^k} should be the point class
    let mut calc = SchubertCalculus::new((2, 4));
    let fundamental = SchubertClass::new(vec![2, 2], (2, 4)).unwrap();

    let result = calc.multi_intersect(&[fundamental]);

    // Codimension 4 in dimension 4 = single point
    assert_eq!(result, IntersectionResult::Finite(1));
}
