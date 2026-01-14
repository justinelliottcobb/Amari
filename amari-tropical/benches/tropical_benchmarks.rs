//! Performance benchmarks for amari-tropical (max-plus algebra)
//!
//! Measures critical operations for tropical arithmetic and matrices.

use amari_tropical::{TropicalMatrix, TropicalMultivector, TropicalNumber};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Benchmark basic tropical number arithmetic
fn bench_tropical_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("tropical_arithmetic");

    let a = TropicalNumber::new(2.0);
    let b = TropicalNumber::new(3.0);

    group.bench_function("tropical_add", |b_| {
        b_.iter(|| black_box(black_box(a).tropical_add(black_box(&b))))
    });

    group.bench_function("tropical_mul", |b_| {
        b_.iter(|| black_box(black_box(a).tropical_mul(black_box(&b))))
    });

    group.bench_function("tropical_pow", |b_| {
        b_.iter(|| black_box(black_box(a).tropical_pow(black_box(2.5))))
    });

    group.finish();
}

/// Benchmark tropical number creation
fn bench_tropical_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tropical_creation");

    group.bench_function("new", |b| {
        b.iter(|| black_box(TropicalNumber::new(black_box(5.0))))
    });

    group.bench_function("zero", |b| {
        b.iter(|| black_box(TropicalNumber::<f64>::zero()))
    });

    group.bench_function("one", |b| {
        b.iter(|| black_box(TropicalNumber::<f64>::one()))
    });

    group.bench_function("neg_infinity", |b| {
        b.iter(|| black_box(TropicalNumber::<f64>::neg_infinity()))
    });

    group.finish();
}

/// Benchmark tropical matrix operations
fn bench_tropical_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("tropical_matrix");

    // Matrix creation
    for size in [4, 8, 16, 32].iter() {
        group.bench_with_input(BenchmarkId::new("creation", size), size, |b, &size| {
            b.iter(|| black_box(TropicalMatrix::<f64>::new(black_box(size), black_box(size))))
        });

        group.bench_with_input(BenchmarkId::new("identity", size), size, |b, &size| {
            b.iter(|| black_box(TropicalMatrix::<f64>::identity(black_box(size))))
        });
    }

    group.finish();
}

/// Benchmark tropical matrix multiplication
fn bench_tropical_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("tropical_matmul");

    for size in [4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements((*size * *size) as u64));

        // Create matrices with values
        let data_a: Vec<Vec<f64>> = (0..*size)
            .map(|i| (0..*size).map(|j| (i * size + j) as f64).collect())
            .collect();
        let data_b: Vec<Vec<f64>> = (0..*size)
            .map(|i| (0..*size).map(|j| ((i + j) % size) as f64).collect())
            .collect();

        let mat_a = TropicalMatrix::from_vec(data_a).unwrap();
        let mat_b = TropicalMatrix::from_vec(data_b).unwrap();

        group.bench_with_input(BenchmarkId::new("matmul", size), size, |b, _| {
            b.iter(|| black_box(black_box(&mat_a).tropical_matmul(black_box(&mat_b))))
        });
    }

    group.finish();
}

/// Benchmark tropical multivector operations
fn bench_tropical_multivector(c: &mut Criterion) {
    let mut group = c.benchmark_group("tropical_multivector");

    // Creation
    group.bench_function("new_2d", |b| {
        b.iter(|| black_box(TropicalMultivector::<f64, 2, 0, 0>::new()))
    });

    group.bench_function("new_3d", |b| {
        b.iter(|| black_box(TropicalMultivector::<f64, 3, 0, 0>::new()))
    });

    // From components
    group.bench_function("from_components_4d", |b| {
        let components = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        b.iter(|| {
            black_box(TropicalMultivector::<f64, 3, 0, 0>::from_components(
                black_box(components.clone()),
            ))
        })
    });

    // Operations
    let mv1 = TropicalMultivector::<f64, 3, 0, 0>::from_components(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ])
    .unwrap();
    let mv2 = TropicalMultivector::<f64, 3, 0, 0>::from_components(vec![
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
    ])
    .unwrap();

    group.bench_function("tropical_add", |b| {
        b.iter(|| black_box(black_box(&mv1).tropical_add(black_box(&mv2))))
    });

    group.bench_function("geometric_product", |b| {
        b.iter(|| black_box(black_box(&mv1).geometric_product(black_box(&mv2))))
    });

    group.bench_function("tropical_norm", |b| {
        b.iter(|| black_box(black_box(&mv1).tropical_norm()))
    });

    group.finish();
}

/// Benchmark batch tropical operations
fn bench_batch_tropical(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_tropical");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("batch_add", size), size, |b, &size| {
            let numbers: Vec<TropicalNumber<f64>> =
                (0..size).map(|i| TropicalNumber::new(i as f64)).collect();

            b.iter(|| {
                let mut result = TropicalNumber::neg_infinity();
                for n in &numbers {
                    result = result.tropical_add(n);
                }
                black_box(result)
            })
        });

        group.bench_with_input(BenchmarkId::new("batch_mul", size), size, |b, &size| {
            let numbers: Vec<TropicalNumber<f64>> = (0..size)
                .map(|i| TropicalNumber::new(i as f64 * 0.01))
                .collect();

            b.iter(|| {
                let mut result = TropicalNumber::one();
                for n in &numbers {
                    result = result.tropical_mul(n);
                }
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark tropical shortest path (Viterbi-style computation)
fn bench_shortest_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("shortest_path");

    for size in [4, 8, 16].iter() {
        // Create adjacency matrix with random-ish weights
        let data: Vec<Vec<f64>> = (0..*size)
            .map(|i| {
                (0..*size)
                    .map(|j| {
                        if i == j {
                            0.0
                        } else if (i + j) % 3 == 0 {
                            f64::NEG_INFINITY // No edge
                        } else {
                            ((i * 7 + j * 11) % 20) as f64
                        }
                    })
                    .collect()
            })
            .collect();

        let adj = TropicalMatrix::from_vec(data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("floyd_warshall_iteration", size),
            size,
            |b, _| {
                b.iter(|| {
                    let dist = adj.clone();
                    // One iteration of Floyd-Warshall in tropical algebra
                    let result = dist.tropical_matmul(&dist);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    tropical_benchmarks,
    bench_tropical_arithmetic,
    bench_tropical_creation,
    bench_tropical_matrix,
    bench_tropical_matmul,
    bench_tropical_multivector,
    bench_batch_tropical,
    bench_shortest_path,
);

criterion_main!(tropical_benchmarks);
