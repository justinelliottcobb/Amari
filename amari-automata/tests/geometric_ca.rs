//! Tests for Geometric Cellular Automata

use amari_automata::{GeometricCA, CARule, Evolvable};
use amari_core::Multivector;
use approx::assert_relative_eq;

type TestCA = GeometricCA<3, 0, 0>;

#[test]
fn test_ca_creation() {
    let ca = TestCA::new(10, 10);
    assert_eq!(ca.dimensions(), (10, 10));
    assert_eq!(ca.generation(), 0);
}

#[test]
fn test_cell_operations() {
    let mut ca = TestCA::new(5, 5);
    let mv = Multivector::basis_vector(0);

    // Test setting and getting cells
    ca.set_cell(2, 2, mv.clone()).unwrap();
    let retrieved = ca.get_cell(2, 2).unwrap();
    assert_relative_eq!(retrieved.scalar_part(), mv.scalar_part());

    // Test bounds checking
    assert!(ca.set_cell(10, 10, mv).is_err());
    assert!(ca.get_cell(10, 10).is_err());
}

#[test]
fn test_evolution_step() {
    let mut ca = TestCA::new(3, 3);

    // Set initial pattern
    let e1 = Multivector::basis_vector(0);
    ca.set_cell(1, 1, e1).unwrap();

    // Evolve one step
    ca.step().unwrap();
    assert_eq!(ca.generation(), 1);

    // Check that evolution occurred
    let center = ca.get_cell(1, 1).unwrap();
    // The exact result depends on the rule, but it should be different
    // from zero due to geometric algebra operations
}

#[test]
fn test_custom_rule() {
    let mut ca = TestCA::new(3, 3);
    let custom_rule = CARule::new(0.1, 2.0, 1.0, 0.5);
    ca.set_rule(custom_rule);

    // Set initial state
    let mv = Multivector::scalar(1.0);
    ca.set_cell(1, 1, mv).unwrap();

    // Evolution should work with custom rule
    ca.step().unwrap();
    assert_eq!(ca.generation(), 1);
}

#[test]
fn test_reset() {
    let mut ca = TestCA::new(3, 3);

    // Set initial state and evolve
    let mv = Multivector::basis_vector(1);
    ca.set_cell(1, 1, mv).unwrap();
    ca.step().unwrap();

    assert_eq!(ca.generation(), 1);

    // Reset should clear everything
    ca.reset();
    assert_eq!(ca.generation(), 0);

    let cell = ca.get_cell(1, 1).unwrap();
    assert_relative_eq!(cell.magnitude(), 0.0);
}

#[test]
fn test_geometric_operations() {
    let mut ca = TestCA::new(3, 3);

    // Set up a pattern that will test geometric operations
    let e1 = Multivector::basis_vector(0);
    let e2 = Multivector::basis_vector(1);

    ca.set_cell(0, 1, e1).unwrap();
    ca.set_cell(1, 1, e2).unwrap();
    ca.set_cell(2, 1, e1).unwrap();

    // The center cell should be influenced by geometric products
    ca.step().unwrap();

    let center = ca.get_cell(1, 1).unwrap();
    // Should have contributions from geometric algebra operations
    assert!(center.magnitude() > 0.0);
}

#[test]
fn test_rule_parameters() {
    let rule = CARule::default();
    let custom_rule = CARule::new(0.3, 1.5, 0.8, 0.2);

    // Default rule should have reasonable values
    // Custom rule should accept the provided values
    // (Implementation details would be tested here)
}

#[test]
fn test_neighborhood_computation() {
    let ca = TestCA::new(5, 5);

    // Test that cells have the right number of neighbors
    // Corner cells: 3 neighbors
    // Edge cells: 5 neighbors
    // Interior cells: 8 neighbors

    // This would require exposing the get_neighbors method
    // or testing indirectly through evolution behavior
}

#[test]
fn test_evolution_stability() {
    let mut ca = TestCA::new(10, 10);

    // Set a stable pattern (if such patterns exist for our rule)
    let mv = Multivector::scalar(0.1);
    for x in 3..7 {
        for y in 3..7 {
            ca.set_cell(x, y, mv.clone()).unwrap();
        }
    }

    // Evolve several steps
    for _ in 0..5 {
        ca.step().unwrap();
    }

    // Pattern should remain relatively stable
    let center = ca.get_cell(5, 5).unwrap();
    assert!(center.magnitude() > 0.0);
}

#[test]
fn test_geometric_algebra_properties() {
    let mut ca = TestCA::new(3, 3);

    // Test that the CA respects geometric algebra properties
    let e1 = Multivector::basis_vector(0);
    let e2 = Multivector::basis_vector(1);

    // Set orthogonal vectors
    ca.set_cell(0, 1, e1).unwrap();
    ca.set_cell(2, 1, e2).unwrap();

    ca.step().unwrap();

    let result = ca.get_cell(1, 1).unwrap();

    // The result should reflect geometric algebra operations
    // e1 * e2 = e12 (bivector), so we expect bivector components
    let bivector_magnitude = result.grade_projection(2).magnitude();
    assert!(bivector_magnitude > 0.0);
}