//! Comprehensive Tests for Geometric Cellular Automata

use amari_core::{Multivector, CayleyTable, Rotor, Bivector};
use amari_automata::{GeometricCA, CARule, CellState, Evolvable, RuleType};
use approx::assert_relative_eq;
use std::f64::consts::PI;

#[test]
fn test_multivector_cell_evolution() {
    // Each cell contains a multivector instead of binary state
    let mut ca = GeometricCA::<3, 0, 0>::new(100);

    // Use a custom rule that creates diffusion
    let diffusion_rule = CARule::custom(|center, neighbors| {
        // Simply return the center value (propagation happens to neighbors)
        if center.magnitude() > 0.0 {
            center.clone()
        } else {
            // Take average of non-zero neighbors
            let non_zero_neighbors: Vec<_> = neighbors.iter()
                .filter(|n| n.magnitude() > 0.0)
                .collect();
            if !non_zero_neighbors.is_empty() {
                let sum = non_zero_neighbors.iter()
                    .fold(Multivector::zero(), |acc, &n| acc + n.clone());
                sum * (1.0 / non_zero_neighbors.len() as f64)
            } else {
                Multivector::zero()
            }
        }
    });

    ca.set_rule(diffusion_rule);
    ca.set_cell(50, Multivector::basis_vector(0)).unwrap();
    ca.step();

    // Neighbors affected by diffusion
    let left = ca.get_cell(49);
    let right = ca.get_cell(51);
    assert!(left.magnitude() > 0.0);
    assert!(right.magnitude() > 0.0);
}

#[test]
fn test_ca_rule_as_geometric_operation() {
    // CA rules are geometric products with neighbors
    let rule = CARule::geometric_simple();

    let center = Multivector::<3, 0, 0>::basis_vector(0);
    let neighbors = vec![Multivector::basis_vector(1), Multivector::basis_vector(2)];
    let result = rule.apply(&center, &neighbors);
    assert!(result.magnitude() > 0.0); // Should have some magnitude
}

#[test]
fn test_game_of_life_geometric() {
    // Conway's Game of Life with geometric states
    let mut ca = GeometricCA::<2, 0, 0>::game_of_life(50, 50);

    // Create glider pattern
    ca.set_pattern(10, 10, &[
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ]);

    // Evolve 4 steps (glider period)
    for _ in 0..4 {
        ca.step();
    }

    // Glider should have moved
    assert!(ca.has_pattern_at(11, 11, &[
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ]));
}

#[test]
fn test_reversible_ca_with_group_structure() {
    // Cayley table of group ensures reversibility
    let mut ca = GeometricCA::<2, 0, 0>::reversible(100);
    let initial = ca.clone();

    for _ in 0..10 {
        ca.step();
    }
    for _ in 0..10 {
        ca.step_inverse();
    }

    assert_eq!(ca.state(), initial.state());
}

#[test]
fn test_continuous_ca_with_rotors() {
    // Cells are rotors that compose
    let mut ca = GeometricCA::<3, 0, 0>::rotor_ca(100);
    let rotor = Rotor::from_bivector(&Bivector::from_components(1.0, 0.0, 0.0), PI / 4.0);
    ca.set_cell(50, rotor.as_multivector().clone()).unwrap();
    ca.step();

    let neighbor = ca.get_cell(51);
    assert!(neighbor.bivector_part().magnitude() > 0.0);
}

#[test]
fn test_cayley_table_performance() {
    // O(1) lookups instead of expensive computations
    let mut ca = GeometricCA::<3, 0, 0>::with_cached_cayley(1000);

    // Time should be constant regardless of operation complexity
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        ca.step();
    }
    let duration = start.elapsed();

    // Should complete within reasonable time
    assert!(duration.as_millis() < 1000);
}

#[test]
fn test_multivector_neighborhoods() {
    // Test different neighborhood structures with multivectors
    let mut ca = GeometricCA::<3, 0, 0>::new(10);

    // Set initial multivector state
    ca.set_cell(5, Multivector::basis_vector(0) + Multivector::basis_vector(1)).unwrap();
    ca.step();

    // Check von Neumann neighborhood
    assert!(ca.get_cell(4).magnitude() > 0.0);
    assert!(ca.get_cell(6).magnitude() > 0.0);
}

#[test]
fn test_geometric_grade_preservation() {
    // Test that grade structure is preserved through evolution
    let mut ca = GeometricCA::<3, 0, 0>::grade_preserving(50);

    // Set scalar
    ca.set_cell(10, Multivector::scalar(1.0)).unwrap();
    // Set vector
    ca.set_cell(20, Multivector::basis_vector(0)).unwrap();
    // Set bivector
    ca.set_cell(30, Multivector::from_bivector(&Bivector::from_components(1.0, 0.0, 0.0))).unwrap();

    ca.step();

    // Grade structure should be maintained or predictably transformed
    assert!(ca.get_cell(10).scalar_part() != 0.0 || ca.get_cell(10).magnitude() == 0.0);
}

#[test]
fn test_ca_boundary_conditions() {
    // Test different boundary conditions
    let mut ca_periodic = GeometricCA::<2, 0, 0>::with_boundary_periodic(10);
    let mut ca_fixed = GeometricCA::<2, 0, 0>::with_boundary_fixed(10);

    ca_periodic.set_cell(0, Multivector::basis_vector(0)).unwrap();
    ca_fixed.set_cell(0, Multivector::basis_vector(0)).unwrap();

    ca_periodic.step();
    ca_fixed.step();

    // Periodic should wrap around
    assert!(ca_periodic.get_cell(9).magnitude() > 0.0);
    // Fixed should not
    assert_eq!(ca_fixed.get_cell(9).magnitude(), 0.0);
}

#[test]
fn test_multivector_conservation_laws() {
    // Test conservation of certain multivector quantities
    let mut ca = GeometricCA::<3, 0, 0>::conservative(100);

    // Set initial configuration
    for i in 40..60 {
        ca.set_cell(i, Multivector::scalar(1.0)).unwrap();
    }

    let initial_total = ca.total_magnitude();

    for _ in 0..10 {
        ca.step();
    }

    let final_total = ca.total_magnitude();

    // Should conserve some quantity
    assert_relative_eq!(initial_total, final_total, epsilon = 1e-10);
}