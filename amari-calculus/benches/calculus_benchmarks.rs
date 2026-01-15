//! Performance benchmarks for amari-calculus geometric calculus operations
//!
//! Measures critical operations for differential operators and vector fields.

use amari_calculus::{curl, divergence, gradient, laplacian, ScalarField, VectorField};
use amari_core::Multivector;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

type Cl3 = Multivector<3, 0, 0>;

/// Quadratic scalar function: f(x) = x[0]^2 + x[1]^2 + x[2]^2
fn quadratic_scalar(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Identity vector function: F(x) = (x[0], x[1], x[2])
fn identity_vector(x: &[f64]) -> Cl3 {
    Cl3::basis_vector(0) * x.first().copied().unwrap_or(0.0)
        + Cl3::basis_vector(1) * x.get(1).copied().unwrap_or(0.0)
        + Cl3::basis_vector(2) * x.get(2).copied().unwrap_or(0.0)
}

/// Benchmark scalar field operations
fn bench_scalar_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_field");

    let field = ScalarField::<3, 0, 0>::new(quadratic_scalar);
    let point = vec![1.0, 0.5, 0.3];

    group.bench_function("evaluate", |b| {
        b.iter(|| black_box(field.evaluate(black_box(&point))))
    });

    group.finish();
}

/// Benchmark vector field operations
fn bench_vector_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_field");

    let field = VectorField::<3, 0, 0>::new(identity_vector);
    let point = vec![1.0, 0.5, 0.3];

    group.bench_function("evaluate", |b| {
        b.iter(|| black_box(field.evaluate(black_box(&point))))
    });

    group.finish();
}

/// Benchmark differential operators
fn bench_differential_operators(c: &mut Criterion) {
    let mut group = c.benchmark_group("differential_operators");

    let scalar_field = ScalarField::<3, 0, 0>::new(quadratic_scalar);
    let vector_field = VectorField::<3, 0, 0>::new(identity_vector);
    let point = vec![1.0, 0.5, 0.3];

    group.bench_function("gradient", |b| {
        b.iter(|| black_box(gradient(black_box(&scalar_field), black_box(&point))))
    });

    group.bench_function("divergence", |b| {
        b.iter(|| black_box(divergence(black_box(&vector_field), black_box(&point))))
    });

    group.bench_function("curl", |b| {
        b.iter(|| black_box(curl(black_box(&vector_field), black_box(&point))))
    });

    group.bench_function("laplacian", |b| {
        b.iter(|| black_box(laplacian(black_box(&scalar_field), black_box(&point))))
    });

    group.finish();
}

/// Benchmark batch field evaluations
fn bench_batch_evaluations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_evaluations");

    let field = ScalarField::<3, 0, 0>::new(quadratic_scalar);

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let points: Vec<Vec<f64>> = (0..*size)
            .map(|i| {
                let t = i as f64 / *size as f64;
                vec![t, 1.0 - t, t * 0.5]
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("scalar_field_batch", size),
            size,
            |b, _| {
                b.iter(|| {
                    let results: Vec<_> = points.iter().map(|p| field.evaluate(p)).collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    calculus_benchmarks,
    bench_scalar_field,
    bench_vector_field,
    bench_differential_operators,
    bench_batch_evaluations,
);

criterion_main!(calculus_benchmarks);
