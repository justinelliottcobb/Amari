//! Comprehensive Tests for Self-Assembly System

use amari_automata::{SelfAssembly, Polyomino, TileSet, AssemblyRule, WangTileSet, Shape};
use amari_core::{Rotor, CayleyTable, Multivector};
use std::f64::consts::PI;
use approx::assert_relative_eq;

#[test]
fn test_polyomino_as_multivector() {
    // Polyomino shapes as multivectors
    let l_piece = Polyomino::from_coords(&[
        (0, 0), (0, 1), (0, 2), (1, 0)
    ]);

    let multivector = l_piece.to_multivector();
    let rotor = Rotor::from_angle(PI / 2.0);
    let rotated = rotor.apply(&multivector);
    let rotated_piece = Polyomino::from_multivector(&rotated);

    assert!(rotated_piece.has_cell_at(0, 0));
    assert!(rotated_piece.has_cell_at(1, 0));
    assert!(rotated_piece.has_cell_at(2, 0));
    assert!(rotated_piece.has_cell_at(0, -1));
}

#[test]
fn test_self_assembly_tiling() {
    let tiles = TileSet::new(vec![
        Polyomino::square(),
        Polyomino::l_piece(),
        Polyomino::straight(3),
    ]);

    let target = Shape::rectangle(10, 10);
    let assembler = SelfAssembly::new(tiles);
    let tiling = assembler.find_tiling(&target);

    assert_eq!(tiling.coverage(), 100);
    assert!(!tiling.has_overlaps());
}

#[test]
fn test_assembly_rules_as_group_operations() {
    // Assembly rules use Cayley table
    let rule = AssemblyRule::new(|tile_a, tile_b, cayley| {
        let product = tile_a.geometric_product_via(tile_b, cayley);
        product.scalar_part() > 0.0
    });

    let tile_a = Polyomino::square().to_multivector();
    let tile_b = Polyomino::l_piece().to_multivector();
    let cayley = CayleyTable::for_dimension(2);

    assert!(rule.check(&tile_a, &tile_b, &cayley));
}

#[test]
fn test_wang_tiles_as_clifford_algebra() {
    // Wang tiles with edges as basis vectors
    let tiles = WangTileSet::new_geometric(16);

    let tile_a = tiles.get(0);
    let tile_b = tiles.get(1);

    // Can connect if geometric product is scalar
    let can_connect = tile_a.east_edge()
        .geometric_product(&tile_b.west_edge())
        .is_scalar();

    assert!(can_connect);
}

#[test]
fn test_polyomino_symmetry_group() {
    // Test symmetry operations on polyominoes
    let t_piece = Polyomino::t_piece();

    // Apply all symmetries
    let symmetries = t_piece.symmetry_group();
    assert_eq!(symmetries.len(), 4); // 4-fold rotational symmetry

    // Each symmetry should preserve area
    for symmetric_piece in symmetries {
        assert_eq!(symmetric_piece.area(), t_piece.area());
    }
}

#[test]
fn test_geometric_affinity_calculation() {
    // Test geometric affinity between polyominoes
    let square = Polyomino::square();
    let rectangle = Polyomino::rectangle(2, 1);

    let affinity = square.geometric_affinity(&rectangle);

    // Should have positive affinity (both rectangular shapes)
    assert!(affinity > 0.0);

    // Self-affinity should be maximum
    assert_relative_eq!(square.geometric_affinity(&square), 1.0, epsilon = 1e-10);
}

#[test]
fn test_constrained_assembly() {
    // Assembly with geometric constraints
    let tiles = TileSet::standard_pentominoes();
    let constraints = vec![
        AssemblyConstraint::NoHoles,
        AssemblyConstraint::ConnectedRegion,
        AssemblyConstraint::BoundingBox(8, 8),
    ];

    let assembler = SelfAssembly::with_constraints(constraints);
    let assembly = assembler.assemble_constrained(&tiles);

    assert!(assembly.is_connected());
    assert_eq!(assembly.holes(), 0);
    assert!(assembly.fits_in_box(8, 8));
}

#[test]
fn test_hierarchical_assembly() {
    // Multi-level assembly: molecules -> complexes -> structures
    let atoms = TileSet::atomic_pieces();
    let assembler = SelfAssembly::hierarchical();

    // Level 1: atoms -> molecules
    let molecules = assembler.assemble_level1(&atoms);

    // Level 2: molecules -> complexes
    let complexes = assembler.assemble_level2(&molecules);

    // Level 3: complexes -> final structure
    let final_structure = assembler.assemble_level3(&complexes);

    assert!(final_structure.is_stable());
    assert!(final_structure.satisfies_global_constraints());
}

#[test]
fn test_dynamic_assembly_process() {
    // Assembly as a dynamic process over time
    let tiles = TileSet::random(20);
    let mut assembler = SelfAssembly::dynamic();

    // Initialize with random configuration
    assembler.initialize_random();

    let initial_energy = assembler.total_energy();

    // Run assembly dynamics
    for _ in 0..1000 {
        assembler.dynamics_step();
    }

    let final_energy = assembler.total_energy();

    // Energy should decrease (more stable configuration)
    assert!(final_energy < initial_energy);
    assert!(assembler.is_at_equilibrium());
}

#[test]
fn test_self_replicating_assembly() {
    // Test assembly that can replicate itself
    let template = Polyomino::von_neumann_constructor();
    let raw_materials = TileSet::basic_materials();

    let assembler = SelfAssembly::self_replicating();
    let replica = assembler.replicate(&template, &raw_materials);

    assert_eq!(replica.shape(), template.shape());
    assert!(replica.can_replicate());
}

#[test]
fn test_error_correction_in_assembly() {
    // Test assembly with error correction capabilities
    let perfect_template = Polyomino::complex_pattern();
    let noisy_materials = TileSet::with_defects(0.1); // 10% defect rate

    let assembler = SelfAssembly::with_error_correction();
    let result = assembler.assemble_robust(&perfect_template, &noisy_materials);

    // Should still match template despite defects
    assert!(result.similarity_to(&perfect_template) > 0.95);
}

#[test]
fn test_cooperative_assembly() {
    // Multiple assemblers working together
    let large_target = Shape::complex_3d(50, 50, 50);
    let assemblers = vec![
        SelfAssembly::specialist_a(),
        SelfAssembly::specialist_b(),
        SelfAssembly::coordinator(),
    ];

    let collaborative_result = SelfAssembly::cooperative(&assemblers, &large_target);

    assert!(collaborative_result.is_complete());
    assert!(collaborative_result.assembly_time() < single_assembler_time(&large_target));
}

#[test]
fn test_geometric_algebra_optimization() {
    // Use geometric algebra for assembly optimization
    let tiles = TileSet::irregular_shapes(10);
    let target = Shape::minimize_surface_area();

    let assembler = SelfAssembly::with_geometric_optimization();
    let optimized = assembler.minimize_energy(&tiles, &target);

    // Should find configuration with minimal surface area
    assert!(optimized.surface_area() < naive_assembly_surface_area(&tiles));
}

#[test]
fn test_assembly_with_excluded_volume() {
    // Test assembly respecting physical constraints
    let thick_tiles = TileSet::with_thickness(2.0);
    let assembler = SelfAssembly::with_physics();

    let assembly = assembler.assemble_physical(&thick_tiles);

    // No interpenetration
    assert!(!assembly.has_intersections());
    assert!(assembly.respects_excluded_volume());
}

#[test]
fn test_fractal_assembly_patterns() {
    // Test assembly that creates fractal structures
    let generator = Polyomino::koch_generator();
    let assembler = SelfAssembly::fractal();

    let fractal = assembler.generate_fractal(&generator, 4); // 4 iterations

    assert_relative_eq!(fractal.dimension(), 1.26, epsilon = 0.1); // Koch curve dimension
    assert!(fractal.is_self_similar());
}

// Helper functions for tests
fn single_assembler_time(target: &Shape) -> f64 {
    // Simulated single assembler time
    target.complexity() * 10.0
}

fn naive_assembly_surface_area(tiles: &TileSet) -> f64 {
    // Naive assembly surface area calculation
    tiles.total_perimeter()
}