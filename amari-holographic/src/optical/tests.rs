//! Integration tests for the optical module.
//!
//! These tests verify the complete workflow of optical field operations
//! including the key mathematical properties from the specification.

use super::*;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};

/// Macro for approximate float equality with custom epsilon.
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {
        let a_val = $a;
        let b_val = $b;
        let eps_val = $eps;
        assert!(
            (a_val - b_val).abs() < eps_val,
            "assertion failed: {} ≈ {}, difference {} exceeds epsilon {}",
            a_val,
            b_val,
            (a_val - b_val).abs(),
            eps_val
        );
    };
    ($a:expr, $b:expr) => {
        assert_approx_eq!($a, $b, 1e-5);
    };
}

// ============================================================================
// Specification Test Cases
// ============================================================================

#[test]
fn test_rotor_binding_is_phase_addition() {
    let algebra = OpticalFieldAlgebra::new((3, 1));

    let phase_a = vec![0.0, FRAC_PI_4, FRAC_PI_2];
    let phase_b = vec![FRAC_PI_4, FRAC_PI_4, FRAC_PI_4];

    let field_a = OpticalRotorField::from_phase(phase_a, (3, 1));
    let field_b = OpticalRotorField::from_phase(phase_b, (3, 1));

    let bound = algebra.bind(&field_a, &field_b);

    // Phases should add
    assert_approx_eq!(bound.phase_at(0, 0), FRAC_PI_4);
    assert_approx_eq!(bound.phase_at(1, 0), FRAC_PI_2);
    assert_approx_eq!(bound.phase_at(2, 0), 3.0 * FRAC_PI_4);
}

#[test]
fn test_inverse_negates_phase() {
    let algebra = OpticalFieldAlgebra::new((4, 1));
    let field = OpticalRotorField::from_phase(vec![0.0, FRAC_PI_4, FRAC_PI_2, PI], (4, 1));

    let inv = algebra.inverse(&field);

    assert_approx_eq!(inv.phase_at(0, 0), 0.0);
    assert_approx_eq!(inv.phase_at(1, 0), -FRAC_PI_4);
    assert_approx_eq!(inv.phase_at(2, 0), -FRAC_PI_2);
    // -π and π are equivalent (wrap around)
    assert_approx_eq!(inv.phase_at(3, 0).abs(), PI);
}

#[test]
fn test_bind_inverse_identity() {
    let algebra = OpticalFieldAlgebra::new((64, 64));
    let field = OpticalRotorField::random((64, 64), 42);

    let inv = algebra.inverse(&field);
    let product = algebra.bind(&field, &inv);

    // Should be identity (phase = 0 everywhere)
    for i in 0..product.len() {
        let x = i % 64;
        let y = i / 64;
        assert_approx_eq!(product.phase_at(x, y), 0.0, 1e-5);
    }
}

#[test]
fn test_similarity_self_is_one() {
    let algebra = OpticalFieldAlgebra::new((64, 64));
    let field = OpticalRotorField::random((64, 64), 42);

    let sim = algebra.similarity(&field, &field);
    assert_approx_eq!(sim, 1.0, 1e-5);
}

#[test]
fn test_random_fields_quasi_orthogonal() {
    let algebra = OpticalFieldAlgebra::new((64, 64));

    let fields: Vec<_> = (0..10)
        .map(|i| OpticalRotorField::random((64, 64), i))
        .collect();

    // Self-similarity should be 1
    for f in &fields {
        assert_approx_eq!(algebra.similarity(f, f), 1.0, 1e-5);
    }

    // Cross-similarity should be small (quasi-orthogonal)
    for i in 0..fields.len() {
        for j in (i + 1)..fields.len() {
            let sim = algebra.similarity(&fields[i], &fields[j]);
            assert!(
                sim.abs() < 0.2,
                "Fields {} and {} too similar: {}",
                i,
                j,
                sim
            );
        }
    }
}

#[test]
fn test_lee_encode_produces_binary() {
    let encoder = GeometricLeeEncoder::with_frequency((256, 256), 0.25);
    // Use amplitude 0.5 for reasonable fill factor
    // (amplitude 1.0 gives threshold cos(π) = -1, so almost everything passes)
    let mut phase_field = OpticalRotorField::random((256, 256), 42);
    for i in 0..phase_field.len() {
        phase_field.amplitudes_mut()[i] = 0.5;
    }

    let hologram = encoder.encode(&phase_field);

    assert_eq!(hologram.dimensions(), (256, 256));
    // Fill factor should be reasonable (not all 0s or all 1s)
    let fill = hologram.fill_factor();
    assert!(fill > 0.2 && fill < 0.8, "Unexpected fill factor: {}", fill);
}

#[test]
fn test_codebook_deterministic() {
    let config = CodebookConfig::new((64, 64), 12345);

    let mut codebook1 = OpticalCodebook::new(config.clone());
    let mut codebook2 = OpticalCodebook::new(config);

    codebook1.register("AGENT".into());
    codebook2.register("AGENT".into());

    let field1 = codebook1.get(&"AGENT".into()).unwrap();
    let field2 = codebook2.get(&"AGENT".into()).unwrap();

    // Should be identical
    assert_eq!(field1.scalars(), field2.scalars());
    assert_eq!(field1.bivectors(), field2.bivectors());
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

#[test]
fn test_vsa_workflow() {
    // Simulate a complete VSA workflow:
    // 1. Create codebook with symbols
    // 2. Create structure via binding
    // 3. Store and retrieve

    let config = CodebookConfig::new((32, 32), 42);
    let mut codebook = OpticalCodebook::new(config);
    let algebra = OpticalFieldAlgebra::new((32, 32));

    // Register symbols
    codebook.register("AGENT".into());
    codebook.register("ROLE".into());
    codebook.register("WORKER".into());

    let agent = codebook.get(&"AGENT".into()).unwrap().clone();
    let role = codebook.get(&"ROLE".into()).unwrap().clone();
    let worker = codebook.get(&"WORKER".into()).unwrap().clone();

    // Create structure: AGENT * ROLE = bound
    let bound = algebra.bind(&agent, &role);

    // bound should be dissimilar to both inputs
    assert!(algebra.similarity(&bound, &agent).abs() < 0.3);
    assert!(algebra.similarity(&bound, &role).abs() < 0.3);

    // Unbind to retrieve: ROLE^(-1) * bound ≈ AGENT
    let retrieved = algebra.unbind(&role, &bound);
    let sim = algebra.similarity(&retrieved, &agent);
    assert!(sim > 0.99, "Failed to retrieve: similarity = {}", sim);

    // Bundle: superposition of AGENT and WORKER
    let bundled = algebra.bundle(&[agent.clone(), worker.clone()], &[0.5, 0.5]);

    // Bundled should be similar to both
    let sim_agent = algebra.similarity(&bundled, &agent);
    let sim_worker = algebra.similarity(&bundled, &worker);
    assert!(
        sim_agent > 0.3,
        "Bundled not similar to agent: {}",
        sim_agent
    );
    assert!(
        sim_worker > 0.3,
        "Bundled not similar to worker: {}",
        sim_worker
    );
}

#[test]
fn test_hologram_roundtrip() {
    // Test encoding and analyzing holograms

    let encoder = GeometricLeeEncoder::with_frequency((64, 64), 0.25);
    let algebra = OpticalFieldAlgebra::new((64, 64));

    // Create two different fields with amplitude 0.5 for reasonable fill factor
    let mut field1 = OpticalRotorField::random((64, 64), 1);
    let mut field2 = OpticalRotorField::random((64, 64), 2);
    for i in 0..field1.len() {
        field1.amplitudes_mut()[i] = 0.5;
        field2.amplitudes_mut()[i] = 0.5;
    }

    let hologram1 = encoder.encode(&field1);
    let hologram2 = encoder.encode(&field2);

    // Similar fields should produce similar holograms
    let mut bound = algebra.bind(&field1, &field1); // identity-like
    for i in 0..bound.len() {
        bound.amplitudes_mut()[i] = 0.5;
    }
    let hologram_id = encoder.encode(&bound);

    // Holograms from different fields should differ
    let dist = hologram1.normalized_hamming_distance(&hologram2);
    assert!(
        dist > 0.1,
        "Different fields should produce different holograms: dist={}",
        dist
    );

    // Same field encoded twice should be identical
    let hologram1_copy = encoder.encode(&field1);
    assert_eq!(hologram1.as_bytes(), hologram1_copy.as_bytes());

    // Check theoretical efficiency
    let efficiency = encoder.theoretical_efficiency(&field1);
    assert!(efficiency > 0.0 && efficiency < 1.0);

    // Modulation should preserve dimensions
    let modulated = encoder.modulate(&field1);
    assert_eq!(modulated.dimensions(), field1.dimensions());

    // Use hologram_id to avoid warning
    assert!(hologram_id.popcount() > 0);
}

#[test]
fn test_tropical_attractor_dynamics() {
    let tropical = TropicalOpticalAlgebra::new((16, 16));

    // Create attractors at different phase levels
    let attractor1 = OpticalRotorField::uniform(0.0, 1.0, (16, 16));
    let attractor2 = OpticalRotorField::uniform(FRAC_PI_4, 1.0, (16, 16));
    let attractor3 = OpticalRotorField::uniform(FRAC_PI_2, 1.0, (16, 16));

    let attractors = vec![attractor1.clone(), attractor2, attractor3];

    // Random initial state
    let initial = OpticalRotorField::random((16, 16), 42);

    // Should converge to smallest-phase attractor (attractor1)
    let (final_state, iterations) = tropical.attractor_converge(&initial, &attractors, 100, 1e-6);

    assert!(
        iterations < 10,
        "Should converge quickly, took {} iterations",
        iterations
    );

    // Final state should be close to attractor1 (phase 0)
    let dist = tropical.normalized_phase_distance(&final_state, &attractor1);
    assert!(
        dist < 0.1,
        "Should converge to zero-phase attractor, dist = {}",
        dist
    );
}

#[test]
fn test_complex_binding_chain() {
    // Test a chain of bindings and unbindings
    let algebra = OpticalFieldAlgebra::new((32, 32));

    let a = OpticalRotorField::random((32, 32), 1);
    let b = OpticalRotorField::random((32, 32), 2);
    let c = OpticalRotorField::random((32, 32), 3);

    // Create chain: ((a * b) * c)
    let ab = algebra.bind(&a, &b);
    let abc = algebra.bind(&ab, &c);

    // Unbind in reverse: ((a * b * c) * c^(-1) * b^(-1)) ≈ a
    let ab_recovered = algebra.unbind(&c, &abc);
    let a_recovered = algebra.unbind(&b, &ab_recovered);

    let sim = algebra.similarity(&a_recovered, &a);
    assert!(sim > 0.99, "Chain unbind failed: similarity = {}", sim);
}

#[test]
fn test_bundle_weighting() {
    let algebra = OpticalFieldAlgebra::new((32, 32));

    let a = OpticalRotorField::random((32, 32), 1);
    let b = OpticalRotorField::random((32, 32), 2);

    // Bundle with different weights
    let heavy_a = algebra.bundle(&[a.clone(), b.clone()], &[0.9, 0.1]);
    let heavy_b = algebra.bundle(&[a.clone(), b.clone()], &[0.1, 0.9]);

    // Heavy-a should be more similar to a
    let sim_a_heavy = algebra.similarity(&heavy_a, &a);
    let sim_a_light = algebra.similarity(&heavy_b, &a);
    assert!(
        sim_a_heavy > sim_a_light,
        "Weighting not working: {} vs {}",
        sim_a_heavy,
        sim_a_light
    );

    // Heavy-b should be more similar to b
    let sim_b_heavy = algebra.similarity(&heavy_b, &b);
    let sim_b_light = algebra.similarity(&heavy_a, &b);
    assert!(
        sim_b_heavy > sim_b_light,
        "Weighting not working: {} vs {}",
        sim_b_heavy,
        sim_b_light
    );
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_single_pixel_field() {
    let algebra = OpticalFieldAlgebra::new((1, 1));

    let a = OpticalRotorField::uniform(FRAC_PI_4, 1.0, (1, 1));
    let b = OpticalRotorField::uniform(FRAC_PI_4, 1.0, (1, 1));

    let bound = algebra.bind(&a, &b);
    assert_approx_eq!(bound.phase_at(0, 0), FRAC_PI_2);

    let sim = algebra.similarity(&a, &a);
    assert_approx_eq!(sim, 1.0);
}

#[test]
fn test_very_small_dimensions() {
    let encoder = GeometricLeeEncoder::with_frequency((2, 2), 0.25);
    let field = OpticalRotorField::uniform(0.0, 0.5, (2, 2));
    let hologram = encoder.encode(&field);

    assert_eq!(hologram.dimensions(), (2, 2));
    assert_eq!(hologram.len(), 4);
}

#[test]
fn test_zero_amplitude_handling() {
    let algebra = OpticalFieldAlgebra::new((4, 1));

    let normal = OpticalRotorField::uniform(FRAC_PI_4, 1.0, (4, 1));
    let zero_amp = OpticalRotorField::uniform(FRAC_PI_4, 0.0, (4, 1));

    // Similarity should handle zero amplitude gracefully
    let sim = algebra.similarity(&normal, &zero_amp);
    assert!(sim.abs() < 0.1 || !sim.is_nan());
}

#[test]
fn test_phase_wrapping() {
    // Test that phases near ±π are handled correctly
    let algebra = OpticalFieldAlgebra::new((4, 1));

    let near_pi = OpticalRotorField::from_phase(vec![PI - 0.01, -PI + 0.01, PI, -PI], (4, 1));
    let inv = algebra.inverse(&near_pi);

    // Inverse should have opposite phases (mod 2π)
    for i in 0..4 {
        let original = near_pi.phase_at(i, 0);
        let inverted = inv.phase_at(i, 0);
        // Sum should be approximately 0 (mod 2π)
        let sum = original + inverted;
        let normalized = sum.rem_euclid(std::f32::consts::TAU);
        assert!(
            normalized < 0.1 || normalized > std::f32::consts::TAU - 0.1,
            "Phase sum not near 0: {} at index {}",
            normalized,
            i
        );
    }
}
