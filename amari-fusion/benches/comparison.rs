//! Benchmarks comparing Tropical-Dual-Clifford algebra performance
#![allow(clippy::needless_range_loop)]

use amari_core::Multivector;
use amari_dual::{
    functions::{cross_entropy_loss, softmax},
    DualNumber,
};
use amari_fusion::TropicalDualClifford;
use amari_tropical::{viterbi::TropicalViterbi, TropicalNumber};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Benchmark tropical vs standard softmax computation
fn benchmark_softmax_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_comparison");

    for size in [10, 100, 1000].iter() {
        let logits: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.1).collect();
        let dual_logits: Vec<DualNumber<f64>> =
            logits.iter().map(|&x| DualNumber::variable(x)).collect();

        // Standard softmax using dual numbers
        group.bench_with_input(BenchmarkId::new("standard_softmax", size), size, |b, _| {
            b.iter(|| {
                let result = softmax(black_box(&dual_logits));
                black_box(result)
            });
        });

        // Tropical approximation
        group.bench_with_input(BenchmarkId::new("tropical_softmax", size), size, |b, _| {
            b.iter(|| {
                let tropical_logits: Vec<TropicalNumber<f64>> =
                    logits.iter().map(|&x| TropicalNumber::new(x)).collect();

                // Find max (tropical sum)
                let max_val = tropical_logits
                    .iter()
                    .fold(TropicalNumber::neg_infinity(), |acc, x| acc.tropical_add(x));

                // Normalize (tropical division)
                let result: Vec<TropicalNumber<f64>> = tropical_logits
                    .iter()
                    .map(|x| TropicalNumber::new(x.value() - max_val.value()))
                    .collect();

                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark sequence decoding performance
fn benchmark_sequence_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequence_decoding");

    for seq_len in [10, 50, 100].iter() {
        let vocab_size = 1000;

        // Create transition matrix
        let transitions: Vec<Vec<f64>> = (0..vocab_size)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| fastrand::f64() * 10.0 - 5.0)
                    .collect()
            })
            .collect();

        let emissions: Vec<Vec<f64>> = (0..*seq_len)
            .map(|_| {
                (0..vocab_size)
                    .map(|_| fastrand::f64() * 10.0 - 5.0)
                    .collect()
            })
            .collect();

        let observations: Vec<usize> = (0..*seq_len)
            .map(|_| fastrand::usize(0..vocab_size))
            .collect();

        // Viterbi using tropical algebra
        group.bench_with_input(
            BenchmarkId::new("tropical_viterbi", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    let decoder = TropicalViterbi::new(
                        black_box(transitions.clone()),
                        black_box(emissions.clone()),
                    );
                    let result = decoder.decode(black_box(&observations));
                    black_box(result)
                });
            },
        );

        // Standard Viterbi (log-space)
        group.bench_with_input(
            BenchmarkId::new("standard_viterbi", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    let result = standard_viterbi(
                        black_box(&transitions),
                        black_box(&emissions),
                        black_box(&observations),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Standard Viterbi implementation for comparison
fn standard_viterbi(
    transitions: &[Vec<f64>],
    emissions: &[Vec<f64>],
    observations: &[usize],
) -> Vec<usize> {
    let num_states = transitions.len();
    let seq_len = observations.len();

    if seq_len == 0 || num_states == 0 {
        return Vec::new();
    }

    // Initialize DP table
    let mut dp = vec![vec![f64::NEG_INFINITY; num_states]; seq_len];
    let mut path = vec![vec![0; num_states]; seq_len];

    // Initialize first column
    for state in 0..num_states {
        if observations[0] < emissions[0].len() {
            dp[0][state] = emissions[0][observations[0]];
        }
    }

    // Fill DP table
    for t in 1..seq_len {
        for curr_state in 0..num_states {
            for prev_state in 0..num_states {
                let emission_score = if observations[t] < emissions[t].len() {
                    emissions[t][observations[t]]
                } else {
                    0.0
                };

                let score =
                    dp[t - 1][prev_state] + transitions[prev_state][curr_state] + emission_score;

                if score > dp[t][curr_state] {
                    dp[t][curr_state] = score;
                    path[t][curr_state] = prev_state;
                }
            }
        }
    }

    // Backtrack
    let mut result = vec![0; seq_len];

    // Find best final state
    let mut best_state = 0;
    let mut best_score = f64::NEG_INFINITY;
    for state in 0..num_states {
        if dp[seq_len - 1][state] > best_score {
            best_score = dp[seq_len - 1][state];
            best_state = state;
        }
    }

    result[seq_len - 1] = best_state;

    // Backtrack through path
    for t in (1..seq_len).rev() {
        result[t - 1] = path[t][result[t]];
    }

    result
}

/// Benchmark automatic differentiation performance
fn benchmark_differentiation(c: &mut Criterion) {
    let mut group = c.benchmark_group("differentiation");

    for size in [10, 100, 500].iter() {
        let inputs: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.01).collect();
        let dual_inputs: Vec<DualNumber<f64>> =
            inputs.iter().map(|&x| DualNumber::variable(x)).collect();

        let targets: Vec<f64> = (0..*size).map(|_| fastrand::f64()).collect();

        // Dual number automatic differentiation
        group.bench_with_input(BenchmarkId::new("dual_autodiff", size), size, |b, _| {
            b.iter(|| {
                let loss = cross_entropy_loss(black_box(&dual_inputs), black_box(&targets));
                black_box(loss)
            });
        });

        // Manual gradient computation
        group.bench_with_input(BenchmarkId::new("manual_gradient", size), size, |b, _| {
            b.iter(|| {
                let loss = manual_cross_entropy_gradient(black_box(&inputs), black_box(&targets));
                black_box(loss)
            });
        });
    }

    group.finish();
}

/// Manual gradient computation for comparison
fn manual_cross_entropy_gradient(inputs: &[f64], targets: &[f64]) -> (f64, Vec<f64>) {
    let n = inputs.len();
    if n == 0 {
        return (0.0, Vec::new());
    }

    // Compute softmax
    let max_val = inputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f64> = inputs.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f64 = exp_vals.iter().sum();
    let softmax_vals: Vec<f64> = exp_vals.iter().map(|&x| x / sum_exp).collect();

    // Compute loss
    let mut loss = 0.0;
    for i in 0..n {
        if softmax_vals[i] > 0.0 {
            loss -= targets[i] * softmax_vals[i].ln();
        }
    }

    // Compute gradient
    let mut gradient = vec![0.0; n];
    for i in 0..n {
        gradient[i] = softmax_vals[i] - targets[i];
    }

    (loss, gradient)
}

/// Benchmark geometric operations
fn benchmark_geometric_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_operations");

    for dim in [3, 4, 5].iter() {
        let coeffs1: Vec<f64> = (0..(1 << dim)).map(|i| (i as f64) * 0.1).collect();
        let coeffs2: Vec<f64> = (0..(1 << dim)).map(|i| (i as f64) * 0.05).collect();

        let mv1 = Multivector::<3, 0, 0>::from_coefficients(coeffs1.clone());
        let mv2 = Multivector::<3, 0, 0>::from_coefficients(coeffs2.clone());

        // Clifford geometric product
        group.bench_with_input(BenchmarkId::new("clifford_product", dim), dim, |b, _| {
            b.iter(|| {
                let result = black_box(&mv1).geometric_product(black_box(&mv2));
                black_box(result)
            });
        });

        // Vector dot product (for comparison)
        group.bench_with_input(BenchmarkId::new("vector_dot_product", dim), dim, |b, _| {
            b.iter(|| {
                let mut result = 0.0;
                let len = coeffs1.len().min(coeffs2.len());
                for i in 0..len {
                    result += coeffs1[i] * coeffs2[i];
                }
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark combined TDC operations
fn benchmark_tdc_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdc_operations");

    for size in [8, 16, 32].iter() {
        let logits1: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.1).collect();
        let logits2: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.05).collect();

        let tdc1 = TropicalDualClifford::<f64, 8>::from_logits(&logits1);
        let tdc2 = TropicalDualClifford::<f64, 8>::from_logits(&logits2);

        // TDC evaluation (all three algebras)
        group.bench_with_input(BenchmarkId::new("tdc_evaluation", size), size, |b, _| {
            b.iter(|| {
                let result = black_box(&tdc1).evaluate(black_box(&tdc2));
                black_box(result)
            });
        });

        // TDC distance computation
        group.bench_with_input(BenchmarkId::new("tdc_distance", size), size, |b, _| {
            b.iter(|| {
                let result = black_box(&tdc1).distance(black_box(&tdc2));
                black_box(result)
            });
        });

        // TDC transformation
        group.bench_with_input(BenchmarkId::new("tdc_transform", size), size, |b, _| {
            b.iter(|| {
                let result = black_box(&tdc1).transform(black_box(&tdc2));
                black_box(result)
            });
        });

        // Simple vector operations for comparison
        group.bench_with_input(BenchmarkId::new("vector_operations", size), size, |b, _| {
            b.iter(|| {
                // Euclidean distance
                let mut dist_sq = 0.0;
                let len = logits1.len().min(logits2.len());
                for i in 0..len {
                    let diff = logits1[i] - logits2[i];
                    dist_sq += diff * diff;
                }
                let distance = dist_sq.sqrt();

                // Element-wise product
                let mut product = Vec::with_capacity(len);
                for i in 0..len {
                    product.push(logits1[i] * logits2[i]);
                }

                black_box((distance, product))
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage patterns
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for count in [100, 1000, 5000].iter() {
        // TDC object creation and manipulation
        group.bench_with_input(BenchmarkId::new("tdc_allocation", count), count, |b, _| {
            b.iter(|| {
                let mut objects = Vec::with_capacity(*count);
                for i in 0..*count {
                    let logits = vec![
                        i as f64 * 0.1,
                        (i as f64 + 1.0) * 0.1,
                        (i as f64 + 2.0) * 0.1,
                        (i as f64 + 3.0) * 0.1,
                    ];
                    let tdc = TropicalDualClifford::<f64, 4>::from_logits(&logits);
                    objects.push(tdc);
                }
                black_box(objects)
            });
        });

        // Simple vector allocation for comparison
        group.bench_with_input(
            BenchmarkId::new("vector_allocation", count),
            count,
            |b, _| {
                b.iter(|| {
                    let mut vectors = Vec::with_capacity(*count);
                    for i in 0..*count {
                        let vec = vec![
                            i as f64 * 0.1,
                            (i as f64 + 1.0) * 0.1,
                            (i as f64 + 2.0) * 0.1,
                            (i as f64 + 3.0) * 0.1,
                        ];
                        vectors.push(vec);
                    }
                    black_box(vectors)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark cache efficiency
fn benchmark_cache_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_efficiency");

    // Sequential access pattern
    group.bench_function("sequential_access", |b| {
        let size = 10000;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();

        b.iter(|| {
            let mut sum = 0.0;
            for &value in black_box(&data) {
                sum += value;
            }
            black_box(sum)
        });
    });

    // Random access pattern
    group.bench_function("random_access", |b| {
        let size = 10000;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let indices: Vec<usize> = (0..size).map(|_| fastrand::usize(0..size)).collect();

        b.iter(|| {
            let mut sum = 0.0;
            for &idx in black_box(&indices) {
                sum += data[idx];
            }
            black_box(sum)
        });
    });

    // TDC sequential operations
    group.bench_function("tdc_sequential", |b| {
        let size = 1000;
        let objects: Vec<TropicalDualClifford<f64, 4>> = (0..size)
            .map(|i| {
                let logits = vec![i as f64 * 0.01, (i + 1) as f64 * 0.01, 0.0, 0.0];
                TropicalDualClifford::from_logits(&logits)
            })
            .collect();

        b.iter(|| {
            let mut result = TropicalDualClifford::zero();
            for obj in black_box(&objects) {
                result = result.add(obj);
            }
            black_box(result)
        });
    });

    group.finish();
}

/// Comprehensive performance comparison
fn benchmark_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");

    let problem_sizes = [10, 50, 100, 250];

    for &size in &problem_sizes {
        // Create test data
        let logits: Vec<f64> = (0..size).map(|i| (i as f64) * 0.02 - 1.0).collect();
        let targets: Vec<f64> = (0..size).map(|_| fastrand::f64()).collect();

        // Full TDC pipeline
        group.bench_with_input(
            BenchmarkId::new("full_tdc_pipeline", size),
            &size,
            |b, _| {
                b.iter(|| {
                    // Create TDC object
                    let tdc = TropicalDualClifford::<f64, 8>::from_logits(black_box(&logits));

                    // Extract features
                    let tropical_features = tdc.extract_tropical_features();
                    let dual_features = tdc.extract_dual_features();

                    // Perform operations
                    let sensitivity = tdc.sensitivity_analysis();
                    let most_sensitive = sensitivity.most_sensitive(3);

                    // Distance to self-modified version
                    let modified_logits: Vec<f64> = logits.iter().map(|&x| x * 1.1).collect();
                    let modified_tdc =
                        TropicalDualClifford::<f64, 8>::from_logits(&modified_logits);
                    let distance = tdc.distance(&modified_tdc);

                    black_box((tropical_features, dual_features, most_sensitive, distance))
                });
            },
        );

        // Equivalent operations using standard methods
        group.bench_with_input(
            BenchmarkId::new("standard_pipeline", size),
            &size,
            |b, _| {
                b.iter(|| {
                    // Standard softmax
                    let max_val = logits.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let exp_vals: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
                    let sum_exp: f64 = exp_vals.iter().sum();
                    let softmax_vals: Vec<f64> = exp_vals.iter().map(|&x| x / sum_exp).collect();

                    // Manual gradient computation
                    let (loss, gradient) = manual_cross_entropy_gradient(&logits, &targets);

                    // Find highest gradient components
                    let mut indexed_grad: Vec<(usize, f64)> = gradient
                        .iter()
                        .enumerate()
                        .map(|(i, &g)| (i, g.abs()))
                        .collect();
                    indexed_grad.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let most_sensitive: Vec<usize> =
                        indexed_grad.into_iter().take(3).map(|(i, _)| i).collect();

                    // Euclidean distance
                    let modified_logits: Vec<f64> = logits.iter().map(|&x| x * 1.1).collect();
                    let mut dist_sq = 0.0;
                    for (a, b) in logits.iter().zip(modified_logits.iter()) {
                        let diff = a - b;
                        dist_sq += diff * diff;
                    }
                    let distance = dist_sq.sqrt();

                    black_box((softmax_vals, loss, gradient, most_sensitive, distance))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_softmax_comparison,
    benchmark_sequence_decoding,
    benchmark_differentiation,
    benchmark_geometric_operations,
    benchmark_tdc_operations,
    benchmark_memory_usage,
    benchmark_cache_efficiency,
    benchmark_comprehensive_comparison
);

criterion_main!(benches);
