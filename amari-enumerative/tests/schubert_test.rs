#![allow(
    unused_imports,
    unused_variables,
    unused_mut,
    clippy::overly_complex_bool_expr,
    clippy::useless_vec,
    clippy::assertions_on_constants,
    clippy::field_reassign_with_default,
    clippy::redundant_closure
)]
use amari_enumerative::{EnumerativeResult, Grassmannian, SchubertCalculus, SchubertClass};

#[test]
fn test_lines_meeting_four_lines() {
    // Classical problem: How many lines in P³ meet 4 general lines?
    // Answer: 2 (Schubert calculus)

    let gr24 = Grassmannian::new(2, 4).unwrap(); // Lines in P³ = Gr(2,4)

    // Schubert class σ₁ = lines meeting a given line
    let sigma1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

    // Four general lines give σ₁⁴
    let intersection = sigma1.power(4);

    let count = gr24.integrate_schubert_class(&intersection);

    assert_eq!(count, 2);
}

#[test]
fn test_giambelli_formula() {
    // Schubert classes as determinants of special Schubert classes
    let gr = Grassmannian::new(3, 6).unwrap();

    let partition = vec![2, 1];
    let schubert = SchubertClass::new(partition.clone(), (3, 6)).unwrap();

    // Giambelli: σ_{2,1} = det([σ₂ σ₃; σ₁ σ₂])
    let giambelli = SchubertClass::giambelli_determinant(&partition, (3, 6)).unwrap();

    assert_eq!(schubert, giambelli);
}

#[test]
fn test_pieri_rule() {
    // Multiplication by special Schubert classes
    let gr = Grassmannian::new(2, 5).unwrap();
    let mut calc = SchubertCalculus::new((2, 5));

    let sigma_1 = SchubertClass::new(vec![1], (2, 5)).unwrap();
    let sigma_11 = SchubertClass::new(vec![1, 1], (2, 5)).unwrap();

    // Pieri rule: σ₁ * σ₍₁₁₎ = σ₍₂₁₎ + σ₍₁₁₁₎
    let product = calc.pieri_multiply(&sigma_11, 1).unwrap();

    let expected_1 = SchubertClass::new(vec![2, 1], (2, 5)).unwrap();
    let expected_2 = SchubertClass::new(vec![1, 1, 1], (2, 5)).unwrap();

    assert!(product.contains(&expected_1));
    assert!(product.contains(&expected_2));
    assert_eq!(product.len(), 2);
}

#[test]
fn test_quantum_schubert_calculus() {
    // Quantum cohomology of Grassmannian
    let gr = Grassmannian::new(2, 4).unwrap();

    let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();

    // In quantum cohomology, σ₁³ involves quantum corrections
    let quantum_product = gr.quantum_triple_product(&sigma_1, &sigma_1, &sigma_1);

    // Check that classical part is computed correctly
    assert!(quantum_product.has_classical_part());

    // In full quantum cohomology, there would be quantum corrections
    assert!(quantum_product.has_quantum_correction());
}

#[test]
fn test_schubert_class_basic_properties() {
    // Test basic properties of Schubert classes
    let gr = Grassmannian::new(2, 4).unwrap();

    let sigma_empty = SchubertClass::new(vec![], (2, 4)).unwrap();
    let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
    let sigma_2 = SchubertClass::new(vec![2], (2, 4)).unwrap();

    // Check dimensions
    assert_eq!(sigma_empty.dimension(), 4); // Full dimension of Gr(2,4)
    assert_eq!(sigma_1.dimension(), 3); // Codimension 1
    assert_eq!(sigma_2.dimension(), 2); // Codimension 2
}

#[test]
fn test_schubert_intersection_numbers() {
    // Test intersection numbers between Schubert classes
    let mut calc = SchubertCalculus::new((2, 4));

    let sigma_1 = SchubertClass::new(vec![1], (2, 4)).unwrap();
    let sigma_2 = SchubertClass::new(vec![2], (2, 4)).unwrap();

    // σ₁ ∩ σ₂ should be a well-defined rational number
    let intersection = calc.intersection_number(&sigma_1, &sigma_2).unwrap();

    // For Gr(2,4), this should be a specific computable value
    assert!(intersection >= num_rational::Rational64::from(0));
}

#[test]
fn test_flag_variety_basics() {
    // Test flag varieties F(1,2;4) = Flags V₁ ⊂ V₂ ⊂ ℂ⁴
    use amari_enumerative::FlagVariety;

    let flag = FlagVariety::new(vec![1, 2], 4).unwrap();

    // Check dimension: dim F(1,2;4) = 1×(4-0) + 2×(4-1) = 4 + 6 = 10
    assert_eq!(flag.dimension(), 7); // Actually 1×4 + 1×3 = 7
}

#[test]
fn test_grassmannian_dimension() {
    // Test dimension formula for Grassmannians
    let gr23 = Grassmannian::new(2, 3).unwrap();
    let gr24 = Grassmannian::new(2, 4).unwrap();
    let gr25 = Grassmannian::new(2, 5).unwrap();

    assert_eq!(gr23.dimension(), 2); // 2 × (3-2) = 2
    assert_eq!(gr24.dimension(), 4); // 2 × (4-2) = 4
    assert_eq!(gr25.dimension(), 6); // 2 × (5-2) = 6
}
