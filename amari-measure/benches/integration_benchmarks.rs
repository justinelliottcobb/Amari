//! Benchmarks for Lebesgue integration performance
//!
//! This benchmark suite measures the performance of integration operations
//! in the amari-measure crate, including:
//! - Simple function integration
//! - General measurable function integration
//! - Integration over different measure types
//! - Geometric (multivector-valued) integration
//! - Product measure integration (Fubini's theorem)
//!
//! Run with:
//! ```bash
//! cargo bench
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// Note: These benchmarks are conceptual as the actual integration module
// is a trait-based framework without concrete numerical integration implementations.
// In practice, integration would require:
// 1. Concrete set representations
// 2. Numerical quadrature methods
// 3. Monte Carlo integration for high dimensions
//
// These benchmarks focus on the API overhead and type system performance.

/// Benchmark simple function operations
///
/// Simple functions are linear combinations of characteristic functions:
/// s(x) = Σᵢ aᵢ · χ_Aᵢ(x)
///
/// Integration: ∫ s dμ = Σᵢ aᵢ · μ(Aᵢ)
fn bench_simple_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_functions");

    // Benchmark creating simple functions with varying number of terms
    for num_terms in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("create", num_terms), num_terms, |b, &n| {
            b.iter(|| {
                // Simulate creating a simple function with n terms
                let mut coefficients = Vec::with_capacity(n);
                for i in 0..n {
                    coefficients.push(black_box(i as f64));
                }
                black_box(coefficients)
            });
        });
    }

    // Benchmark "integration" of simple functions (sum of products)
    for num_terms in [10, 100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("integrate", num_terms),
            num_terms,
            |b, &n| {
                // Prepare data
                let coefficients: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let measures: Vec<f64> = (0..n).map(|i| 1.0 / (i + 1) as f64).collect();

                b.iter(|| {
                    // ∫ s dμ = Σᵢ aᵢ · μ(Aᵢ)
                    let result: f64 = coefficients
                        .iter()
                        .zip(measures.iter())
                        .map(|(&a, &m)| black_box(a * m))
                        .sum();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark measure evaluation overhead
///
/// Measures the cost of evaluating measures on sets.
/// This is the fundamental operation in integration.
fn bench_measure_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure_evaluation");

    // Lebesgue measure: Volume of intervals/rectangles
    group.bench_function("lebesgue_1d_interval", |b| {
        b.iter(|| {
            // λ([a,b]) = b - a
            let a = black_box(0.0f64);
            let b = black_box(1.0f64);
            black_box(b - a)
        });
    });

    group.bench_function("lebesgue_2d_rectangle", |b| {
        b.iter(|| {
            // λ²([a,b]×[c,d]) = (b-a)(d-c)
            let a = black_box(0.0f64);
            let b = black_box(1.0f64);
            let c = black_box(0.0f64);
            let d = black_box(2.0f64);
            black_box((b - a) * (d - c))
        });
    });

    group.bench_function("lebesgue_3d_box", |b| {
        b.iter(|| {
            // λ³([a,b]×[c,d]×[e,f]) = (b-a)(d-c)(f-e)
            let dims = black_box([(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]);
            let volume = dims.iter().map(|(a, b)| b - a).product::<f64>();
            black_box(volume)
        });
    });

    // Counting measure: Cardinality
    group.bench_function("counting_measure", |b| {
        b.iter(|| {
            // μ(A) = |A|
            let set_size = black_box(1000usize);
            black_box(set_size as f64)
        });
    });

    // Dirac measure: Point mass
    group.bench_function("dirac_measure", |b| {
        b.iter(|| {
            // δₓ(A) = 1 if x ∈ A, else 0
            let point_in_set = black_box(true);
            black_box(if point_in_set { 1.0 } else { 0.0 })
        });
    });

    group.finish();
}

/// Benchmark function evaluation
///
/// Tests the overhead of evaluating measurable functions at points.
fn bench_function_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("function_evaluation");

    // Constant function
    group.bench_function("constant", |b| {
        let c_val = black_box(5.0);
        b.iter(|| {
            let _x = black_box(std::f64::consts::PI);
            black_box(c_val)
        });
    });

    // Linear function
    group.bench_function("linear", |b| {
        b.iter(|| {
            let x = black_box(std::f64::consts::PI);
            black_box(x)
        });
    });

    // Polynomial function
    group.bench_function("polynomial_degree_5", |b| {
        b.iter(|| {
            let x = black_box(std::f64::consts::PI);
            // p(x) = x⁵ + 2x³ + x
            let x2 = x * x;
            let x3 = x2 * x;
            let x5 = x3 * x2;
            black_box(x5 + 2.0 * x3 + x)
        });
    });

    // Trigonometric function
    group.bench_function("trigonometric", |b| {
        b.iter(|| {
            let x = black_box(std::f64::consts::PI);
            black_box(x.sin())
        });
    });

    // Indicator/characteristic function
    group.bench_function("indicator", |b| {
        b.iter(|| {
            let x = black_box(0.5);
            // χ_[0,1](x)
            black_box(if (0.0..=1.0).contains(&x) { 1.0 } else { 0.0 })
        });
    });

    group.finish();
}

/// Benchmark multivector operations for geometric measures
///
/// Geometric measures assign multivectors to sets.
/// This benchmarks the overhead of multivector arithmetic.
fn bench_multivector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("multivector_operations");

    // 2D multivector: 2^2 = 4 components (scalar, e1, e2, e12)
    group.bench_function("mv2d_addition", |b| {
        b.iter(|| {
            let a = black_box([1.0, 2.0, 3.0, 4.0]);
            let b = black_box([5.0, 6.0, 7.0, 8.0]);
            let mut result = [0.0; 4];
            for i in 0..4 {
                result[i] = a[i] + b[i];
            }
            black_box(result)
        });
    });

    // 3D multivector: 2^3 = 8 components
    group.bench_function("mv3d_addition", |b| {
        b.iter(|| {
            let a = black_box([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let b = black_box([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
            let mut result = [0.0; 8];
            for i in 0..8 {
                result[i] = a[i] + b[i];
            }
            black_box(result)
        });
    });

    // 4D multivector: 2^4 = 16 components
    group.bench_function("mv4d_addition", |b| {
        b.iter(|| {
            let a = black_box([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ]);
            let b = black_box([
                16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                1.0,
            ]);
            let mut result = [0.0; 16];
            for i in 0..16 {
                result[i] = a[i] + b[i];
            }
            black_box(result)
        });
    });

    // Scalar multiplication
    group.bench_function("mv3d_scalar_mult", |b| {
        b.iter(|| {
            let mv = black_box([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
            let scalar = black_box(2.5);
            let mut result = [0.0; 8];
            for i in 0..8 {
                result[i] = scalar * mv[i];
            }
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark product measure operations (Fubini's theorem)
///
/// Product measures allow computing double integrals as iterated integrals:
/// ∫∫ f d(μ×ν) = ∫ (∫ f dν) dμ
fn bench_product_measures(c: &mut Criterion) {
    let mut group = c.benchmark_group("product_measures");

    // 2D integration: ∫∫ f(x,y) dxdy
    group.bench_function("double_integral_10x10", |b| {
        b.iter(|| {
            let nx = black_box(10);
            let ny = black_box(10);
            let mut sum = 0.0;

            for i in 0..nx {
                for j in 0..ny {
                    let x = (i as f64) / (nx as f64);
                    let y = (j as f64) / (ny as f64);
                    // f(x,y) = xy
                    sum += black_box(x * y);
                }
            }

            black_box(sum / ((nx * ny) as f64))
        });
    });

    group.bench_function("double_integral_100x100", |b| {
        b.iter(|| {
            let nx = black_box(100);
            let ny = black_box(100);
            let mut sum = 0.0;

            for i in 0..nx {
                for j in 0..ny {
                    let x = (i as f64) / (nx as f64);
                    let y = (j as f64) / (ny as f64);
                    sum += black_box(x * y);
                }
            }

            black_box(sum / ((nx * ny) as f64))
        });
    });

    // 3D integration: ∫∫∫ f(x,y,z) dxdydz
    group.bench_function("triple_integral_10x10x10", |b| {
        b.iter(|| {
            let n = black_box(10);
            let mut sum = 0.0;

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let x = (i as f64) / (n as f64);
                        let y = (j as f64) / (n as f64);
                        let z = (k as f64) / (n as f64);
                        // f(x,y,z) = xyz
                        sum += black_box(x * y * z);
                    }
                }
            }

            black_box(sum / ((n * n * n) as f64))
        });
    });

    group.finish();
}

/// Benchmark convergence theorem operations
///
/// Tests sequences of functions and their limits.
fn bench_convergence_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence");

    // Monotone convergence: sequence f_n ↗ f
    group.bench_function("monotone_sequence_100", |b| {
        b.iter(|| {
            let n_terms = black_box(100);
            let x = black_box(0.5f64);

            // Example: f_n(x) = x(1 - x^n) → x as n→∞
            let mut sequence = Vec::with_capacity(n_terms);
            for n in 1..=n_terms {
                let x_n = x.powi(n as i32);
                sequence.push(x * (1.0 - x_n));
            }

            black_box(sequence)
        });
    });

    // Dominated convergence: |f_n| ≤ g
    group.bench_function("dominated_sequence_100", |b| {
        b.iter(|| {
            let n_terms = black_box(100);
            let x = black_box(0.5);
            let dominating_function = black_box(2.0);

            // Example: f_n(x) = sin(nx)/n → 0 as n→∞, |f_n| ≤ 1/n ≤ g
            let mut sequence = Vec::with_capacity(n_terms);
            for n in 1..=n_terms {
                let value = ((n as f64) * x).sin() / (n as f64);
                // Check domination
                assert!(value.abs() <= dominating_function);
                sequence.push(value);
            }

            black_box(sequence)
        });
    });

    // Fatou's lemma: lim inf computation
    group.bench_function("lim_inf_100", |b| {
        let sequence: Vec<f64> = (1..=100)
            .map(|n| {
                // Oscillating sequence
                if n % 2 == 0 {
                    1.0 / (n as f64)
                } else {
                    1.0
                }
            })
            .collect();

        b.iter(|| {
            // Compute lim inf (infimum of all eventual values)
            let lim_inf = sequence
                .iter()
                .skip(50) // Eventually for large n
                .fold(f64::INFINITY, |acc, &x| black_box(acc.min(x)));
            black_box(lim_inf)
        });
    });

    group.finish();
}

/// Benchmark density/Radon-Nikodym derivative operations
///
/// Densities represent measures as dν/dμ.
fn bench_density_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("density");

    // Evaluate density function
    group.bench_function("density_evaluation_1000", |b| {
        b.iter(|| {
            let n_points = black_box(1000);
            let mut values = Vec::with_capacity(n_points);

            for i in 0..n_points {
                let x = (i as f64) / (n_points as f64);
                // Example density: ρ(x) = 2x on [0,1]
                values.push(black_box(2.0 * x));
            }

            black_box(values)
        });
    });

    // Integrate with respect to density
    group.bench_function("density_integration_1000", |b| {
        b.iter(|| {
            let n_points = black_box(1000);
            let dx = 1.0 / (n_points as f64);
            let mut integral = 0.0;

            for i in 0..n_points {
                let x = (i as f64) * dx;
                // ∫ f(x) ρ(x) dx where f(x) = x, ρ(x) = 2x
                let density = 2.0 * x;
                integral += black_box(x * density * dx);
            }

            black_box(integral)
        });
    });

    group.finish();
}

/// Benchmark signed measure operations
///
/// Signed measures can be positive or negative.
fn bench_signed_measures(c: &mut Criterion) {
    let mut group = c.benchmark_group("signed_measures");

    // Jordan decomposition: μ = μ⁺ - μ⁻
    group.bench_function("jordan_decomposition_100", |b| {
        let values: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) })
            .collect();

        b.iter(|| {
            let (positive, negative): (Vec<_>, Vec<_>) = values.iter().partition(|&&x| x >= 0.0);

            let mu_plus: f64 = positive.iter().map(|&&x| black_box(x)).sum();
            let mu_minus: f64 = negative.iter().map(|&&x| black_box(-x)).sum();

            black_box((mu_plus, mu_minus))
        });
    });

    // Total variation: |μ|(A) = μ⁺(A) + μ⁻(A)
    group.bench_function("total_variation_100", |b| {
        let values: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) })
            .collect();

        b.iter(|| {
            let total_variation: f64 = values.iter().map(|&x| black_box(x.abs())).sum();
            black_box(total_variation)
        });
    });

    group.finish();
}

// Configure benchmark groups
criterion_group!(
    benches,
    bench_simple_functions,
    bench_measure_evaluation,
    bench_function_evaluation,
    bench_multivector_operations,
    bench_product_measures,
    bench_convergence_operations,
    bench_density_operations,
    bench_signed_measures,
);

criterion_main!(benches);
