//! Comprehensive Tests for Cayley Graph Navigation

use amari_automata::{CayleyGraphNavigator, GroupElement, Generator, GeometricCA};
use amari_core::Multivector;
use approx::assert_relative_eq;

#[test]
fn test_cayley_graph_as_state_space() {
    // CA states as group Cayley graph
    let navigator = CayleyGraphNavigator::for_group("D4");

    let start = GroupElement::identity();
    let target = GroupElement::from_name("r2f");
    let path = navigator.find_path(start, target);

    assert_eq!(path, vec![
        Generator::Rotation,
        Generator::Rotation,
        Generator::Flip,
    ]);
}

#[test]
fn test_ca_evolution_as_group_action() {
    let mut ca = GeometricCA::with_group_structure("S3");
    ca.set_state(GroupElement::identity());

    ca.apply_generators(&[
        Generator::Transposition(0, 1),
        Generator::Transposition(1, 2),
    ]);

    assert_eq!(ca.state(), GroupElement::from_permutation(&[2, 0, 1]));
}

#[test]
fn test_inverse_problem_as_word_problem() {
    let target = GroupElement::from_name("complex_element");
    let solver = CayleyGraphNavigator::new();
    let word = solver.find_word_for(target);

    let mut element = GroupElement::identity();
    for generator in word {
        element = element.multiply(generator);
    }

    assert_eq!(element, target);
}

#[test]
fn test_geometric_algebra_group_structure() {
    // Test Clifford algebra as a group
    let navigator = CayleyGraphNavigator::for_clifford_algebra(3, 0, 0);

    let e1 = GroupElement::from_multivector(Multivector::e1());
    let e2 = GroupElement::from_multivector(Multivector::e2());
    let e12 = GroupElement::from_multivector(Multivector::e12());

    // e1 * e2 = e12
    let path = navigator.find_path(
        GroupElement::identity(),
        e12.clone()
    );

    let reconstructed = navigator.apply_path(&path);
    assert_eq!(reconstructed, e12);
}

#[test]
fn test_group_orbit_exploration() {
    // Explore all reachable states from a starting point
    let navigator = CayleyGraphNavigator::for_group("A4"); // Alternating group
    let start = GroupElement::identity();

    let orbit = navigator.compute_orbit(start);

    // A4 has 12 elements
    assert_eq!(orbit.len(), 12);

    // All elements should be distinct
    let unique_elements: std::collections::HashSet<_> = orbit.into_iter().collect();
    assert_eq!(unique_elements.len(), 12);
}

#[test]
fn test_shortest_path_algorithms() {
    // Test different path-finding algorithms
    let navigator = CayleyGraphNavigator::for_group("D6");

    let start = GroupElement::identity();
    let target = GroupElement::from_name("r3f2");

    let breadth_first_path = navigator.shortest_path_bfs(start.clone(), target.clone());
    let dijkstra_path = navigator.shortest_path_dijkstra(start.clone(), target.clone());
    let a_star_path = navigator.shortest_path_a_star(start, target);

    // All should find optimal paths (may differ but same length)
    assert_eq!(breadth_first_path.len(), dijkstra_path.len());
    assert_eq!(breadth_first_path.len(), a_star_path.len());
}

#[test]
fn test_group_homomorphisms() {
    // Test mappings between different groups
    let source_nav = CayleyGraphNavigator::for_group("Z4");
    let target_nav = CayleyGraphNavigator::for_group("Z2");

    let homomorphism = source_nav.find_homomorphism(&target_nav);

    // Z4 -> Z2: mod 2 reduction
    let z4_element = GroupElement::from_name("2");
    let z2_image = homomorphism.apply(&z4_element);

    assert_eq!(z2_image, GroupElement::from_name("0"));
}

#[test]
fn test_cayley_table_generation() {
    // Test automatic Cayley table generation
    let generators = vec![
        Generator::Rotation,
        Generator::Reflection,
    ];

    let navigator = CayleyGraphNavigator::from_generators(generators);
    let cayley_table = navigator.generate_cayley_table();

    // Should satisfy group axioms
    assert!(cayley_table.satisfies_associativity());
    assert!(cayley_table.has_identity());
    assert!(cayley_table.all_elements_have_inverse());
}

#[test]
fn test_subgroup_analysis() {
    // Find and analyze subgroups
    let navigator = CayleyGraphNavigator::for_group("S4");

    let subgroups = navigator.find_all_subgroups();

    // S4 has many subgroups of various orders
    assert!(subgroups.iter().any(|sg| sg.order() == 1));  // Trivial
    assert!(subgroups.iter().any(|sg| sg.order() == 2));  // Z2
    assert!(subgroups.iter().any(|sg| sg.order() == 3));  // Z3
    assert!(subgroups.iter().any(|sg| sg.order() == 12)); // A4
}

#[test]
fn test_conjugacy_classes() {
    // Test conjugacy class computation
    let navigator = CayleyGraphNavigator::for_group("D4");

    let conjugacy_classes = navigator.compute_conjugacy_classes();

    // D4 has 5 conjugacy classes
    assert_eq!(conjugacy_classes.len(), 5);

    // Each element should be in exactly one class
    let total_elements: usize = conjugacy_classes.iter().map(|c| c.len()).sum();
    assert_eq!(total_elements, 8); // |D4| = 8
}

#[test]
fn test_group_action_on_sets() {
    // Test group actions on various sets
    let navigator = CayleyGraphNavigator::for_group("S3");

    // Action on a set of 3 elements
    let set = vec!["a", "b", "c"];
    let permutation = GroupElement::from_name("(12)");

    let result = navigator.apply_action(&permutation, &set);

    assert_eq!(result, vec!["b", "a", "c"]);
}

#[test]
fn test_fundamental_domain() {
    // Find fundamental domain for group action
    let navigator = CayleyGraphNavigator::for_group("Z2");

    let space = GeometricSpace::torus(10, 10);
    let fundamental_domain = navigator.find_fundamental_domain(&space);

    // Should cover exactly half the space (Z2 action)
    assert_relative_eq!(fundamental_domain.area(), space.area() / 2.0, epsilon = 0.1);
}

#[test]
fn test_word_metrics() {
    // Test different metrics on the group
    let navigator = CayleyGraphNavigator::for_group("F2"); // Free group on 2 generators

    let element = GroupElement::from_word(&["a", "b", "a^-1", "b^-1"]);

    let word_length = navigator.word_metric(&element);
    let conjugacy_length = navigator.conjugacy_metric(&element);

    assert_eq!(word_length, 4);
    assert!(conjugacy_length <= word_length);
}

#[test]
fn test_random_walks_on_groups() {
    // Test random walk properties
    let navigator = CayleyGraphNavigator::for_group("Z10");

    let mut current = GroupElement::identity();
    let mut positions = vec![current.clone()];

    // Perform 1000 random steps
    for _ in 0..1000 {
        let step = navigator.random_generator();
        current = current.multiply(&step);
        positions.push(current.clone());
    }

    // Should eventually visit all elements (ergodic)
    let unique_positions: std::collections::HashSet<_> = positions.into_iter().collect();
    assert!(unique_positions.len() > 5); // Should visit more than half
}

#[test]
fn test_group_presentation() {
    // Test group presentations and relations
    let presentation = GroupPresentation::new(
        vec!["r", "s"], // generators
        vec![
            "r^4 = 1",    // r has order 4
            "s^2 = 1",    // s has order 2
            "srs = r^-1", // conjugation relation
        ]
    );

    let navigator = CayleyGraphNavigator::from_presentation(presentation);

    // Should recognize this as D4
    assert_eq!(navigator.group_order(), 8);
    assert_eq!(navigator.group_name(), "D4");
}

#[test]
fn test_covering_spaces() {
    // Test universal covers and covering spaces
    let base_navigator = CayleyGraphNavigator::for_group("Z2");
    let covering_navigator = CayleyGraphNavigator::for_group("Z");

    let covering_map = covering_navigator.covering_map_to(&base_navigator);

    // Z -> Z2 covering map
    let lift = covering_map.lift(&GroupElement::from_name("1"));
    assert!(lift.projects_to(&GroupElement::from_name("1")));
}

#[test]
fn test_growth_functions() {
    // Test growth functions for different groups
    let navigator = CayleyGraphNavigator::for_group("Z2");

    let growth_function = navigator.compute_growth_function(10);

    // Z2 has polynomial growth
    assert!(growth_function.is_polynomial());
    assert_eq!(growth_function.degree(), 1);
}

// Helper types for tests
struct GeometricSpace {
    width: usize,
    height: usize,
}

impl GeometricSpace {
    fn torus(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    fn area(&self) -> f64 {
        (self.width * self.height) as f64
    }
}

struct GroupPresentation {
    generators: Vec<String>,
    relations: Vec<String>,
}

impl GroupPresentation {
    fn new(generators: Vec<&str>, relations: Vec<&str>) -> Self {
        Self {
            generators: generators.into_iter().map(|s| s.to_string()).collect(),
            relations: relations.into_iter().map(|s| s.to_string()).collect(),
        }
    }
}