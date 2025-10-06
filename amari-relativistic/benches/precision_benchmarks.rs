//! Benchmarks comparing standard and high-precision arithmetic for orbital mechanics

use amari_relativistic::constants;
use amari_relativistic::precision::*;
use amari_relativistic::precision_geodesic::*;
use amari_relativistic::spacetime::SpacetimeVector;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use num_traits::Zero;

/// Benchmark basic precision arithmetic operations
fn bench_precision_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_arithmetic");

    // Standard f64 operations
    group.bench_function("f64_operations", |b| {
        b.iter(|| {
            let x = black_box(2.0_f64);
            let y = black_box(std::f64::consts::PI);

            let sqrt_val = x.sqrt_precise();
            let pow_val = x.powf_precise(y);
            let sin_val = y.sin_precise();
            let cos_val = y.cos_precise();
            let ln_val = y.ln_precise();

            black_box((sqrt_val, pow_val, sin_val, cos_val, ln_val))
        })
    });

    // Physical constants computation
    group.bench_function("physical_constants", |b| {
        b.iter(|| {
            let c = constants::C;
            let g = constants::G;
            let solar_mass = constants::SOLAR_MASS;
            let earth_mass = constants::EARTH_MASS;

            black_box((c, g, solar_mass, earth_mass))
        })
    });

    group.finish();
}

/// Benchmark spacetime vector operations
fn bench_spacetime_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("spacetime_operations");

    // Standard spacetime vector operations
    group.bench_function("standard_spacetime_vector", |b| {
        let v1 = SpacetimeVector::new(1.0, 2.0, 3.0, 4.0);
        let v2 = SpacetimeVector::new(2.0, 1.0, 4.0, 3.0);

        b.iter(|| {
            let dot = black_box(v1.minkowski_dot(&v2));
            let norm_sq = black_box(v1.minkowski_norm_squared());
            let is_timelike = black_box(v1.is_timelike());

            black_box((dot, norm_sq, is_timelike))
        })
    });

    // Precision spacetime vector operations
    group.bench_function("precision_spacetime_vector", |b| {
        let v1 = PrecisionSpacetimeVector::<f64>::new(1.0, 2.0, 3.0, 4.0);
        let v2 = PrecisionSpacetimeVector::<f64>::new(2.0, 1.0, 4.0, 3.0);

        b.iter(|| {
            let dot = black_box(v1.minkowski_dot(&v2));
            let norm_sq = black_box(v1.norm_squared());
            let spatial_mag_sq = black_box(v1.spatial_magnitude_squared());

            black_box((dot, norm_sq, spatial_mag_sq))
        })
    });

    group.finish();
}

/// Benchmark orbital mechanics calculations
fn bench_orbital_mechanics(c: &mut Criterion) {
    let mut group = c.benchmark_group("orbital_mechanics");

    // Compute many orbital parameters for spacecraft trajectory
    group.bench_function("spacecraft_orbital_parameters", |b| {
        b.iter(|| {
            // Typical spacecraft orbital parameters
            let semi_major_axis = black_box(7000e3_f64); // 7000 km
            let eccentricity = black_box(0.01_f64);
            let gravitational_parameter = black_box(3.986004418e14_f64); // Earth μ

            // Orbital period calculation
            let period = 2.0
                * std::f64::consts::PI
                * (semi_major_axis.powi(3) / gravitational_parameter).sqrt_precise();

            // Mean motion
            let mean_motion = 2.0 * std::f64::consts::PI / period;

            // Orbital velocity at periapsis
            let velocity_periapsis = (gravitational_parameter * (1.0 + eccentricity)
                / (semi_major_axis * (1.0 - eccentricity)))
                .sqrt_precise();

            // Specific orbital energy
            let specific_energy = -gravitational_parameter / (2.0 * semi_major_axis);

            black_box((period, mean_motion, velocity_periapsis, specific_energy))
        })
    });

    // High-precision orbital calculations
    group.bench_function("high_precision_orbital_parameters", |b| {
        b.iter(|| {
            let semi_major_axis = StandardFloat::from_f64(black_box(7000e3_f64));
            let eccentricity = StandardFloat::from_f64(black_box(0.01_f64));
            let mu = StandardFloat::from_f64(black_box(3.986004418e14_f64));

            let pi = StandardFloat::from_f64(std::f64::consts::PI);
            let two = StandardFloat::from_f64(2.0);
            let one = StandardFloat::from_f64(1.0);

            // High-precision orbital period
            let period = two
                * pi
                * (semi_major_axis.powf_precise(StandardFloat::from_f64(3.0)) / mu).sqrt_precise();

            // Mean motion with high precision
            let mean_motion = two * pi / period;

            // Periapsis velocity with high precision
            let velocity_periapsis = (mu * (one + eccentricity)
                / (semi_major_axis * (one - eccentricity)))
                .sqrt_precise();

            // Specific energy with high precision
            let specific_energy = StandardFloat::zero() - mu / (two * semi_major_axis);

            black_box((
                period.to_f64(),
                mean_motion.to_f64(),
                velocity_periapsis.to_f64(),
                specific_energy.to_f64(),
            ))
        })
    });

    group.finish();
}

/// Benchmark tolerance comparisons for different precision levels
fn bench_tolerance_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("tolerance_analysis");

    // Test numerical stability in iterative calculations
    group.bench_function("iterative_stability_f64", |b| {
        b.iter(|| {
            let mut value = black_box(1.0_f64);
            let increment = black_box(1e-16_f64);

            // Simulate many small orbital perturbations
            for _ in 0..1000 {
                value += increment;
                value = value.sqrt_precise();
                value = value * value; // Should return to original + increment
            }

            black_box(value)
        })
    });

    // Compare with orbital tolerance
    group.bench_function("orbital_tolerance_check", |b| {
        let tolerance = f64::orbital_tolerance();
        let epsilon = f64::epsilon();

        b.iter(|| {
            let accumulated_error = black_box(1e-10_f64);

            let within_orbital = accumulated_error < tolerance;
            let within_machine = accumulated_error < epsilon;
            let relative_error = accumulated_error / tolerance;

            black_box((within_orbital, within_machine, relative_error))
        })
    });

    group.finish();
}

/// Benchmark memory allocation patterns for different precision types
fn bench_memory_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_patterns");

    // Standard vector allocation
    group.bench_function("standard_vector_allocation", |b| {
        b.iter(|| {
            let vectors: Vec<SpacetimeVector> = (0..1000)
                .map(|i| {
                    let t = i as f64;
                    SpacetimeVector::new(t, t + 1.0, t + 2.0, t + 3.0)
                })
                .collect();

            black_box(vectors.len())
        })
    });

    // Precision vector allocation
    group.bench_function("precision_vector_allocation", |b| {
        b.iter(|| {
            let vectors: Vec<PrecisionSpacetimeVector<f64>> = (0..1000)
                .map(|i| {
                    let t = i as f64;
                    PrecisionSpacetimeVector::new(t, t + 1.0, t + 2.0, t + 3.0)
                })
                .collect();

            black_box(vectors.len())
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_precision_arithmetic,
    bench_spacetime_operations,
    bench_orbital_mechanics,
    bench_tolerance_analysis,
    bench_memory_patterns
);

criterion_main!(benches);
