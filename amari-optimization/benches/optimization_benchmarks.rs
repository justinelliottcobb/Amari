//! Benchmark suite for amari-optimization algorithms
//!
//! This module contains comprehensive benchmarks to measure the performance,
//! scalability, and efficiency of different optimization algorithms.

use amari_optimization::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// High-dimensional test problems for benchmarking
struct HighDimQuadratic {
    dimension: usize,
    condition_number: f64,
}

impl HighDimQuadratic {
    fn new(dimension: usize, condition_number: f64) -> Self {
        Self {
            dimension,
            condition_number,
        }
    }
}

impl ConstrainedObjective<f64> for HighDimQuadratic {
    fn evaluate(&self, x: &[f64]) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let eigenval =
                    1.0 + (self.condition_number - 1.0) * i as f64 / (self.dimension - 1) as f64;
                eigenval * xi * xi
            })
            .sum::<f64>()
            / 2.0
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &xi)| {
                let eigenval =
                    1.0 + (self.condition_number - 1.0) * i as f64 / (self.dimension - 1) as f64;
                eigenval * xi
            })
            .collect()
    }

    fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn inequality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-10.0, 10.0); self.dimension]
    }

    fn num_inequality_constraints(&self) -> usize {
        0
    }
    fn num_equality_constraints(&self) -> usize {
        0
    }
    fn num_variables(&self) -> usize {
        self.dimension
    }
}

/// Rosenbrock function for benchmarking
struct RosenbrockND {
    dimension: usize,
}

impl RosenbrockND {
    fn new(dimension: usize) -> Self {
        assert!(
            dimension >= 2 && dimension % 2 == 0,
            "Dimension must be even and >= 2"
        );
        Self { dimension }
    }
}

impl ConstrainedObjective<f64> for RosenbrockND {
    fn evaluate(&self, x: &[f64]) -> f64 {
        (0..self.dimension / 2)
            .map(|i| {
                let x1 = x[2 * i];
                let x2 = x[2 * i + 1];
                100.0 * (x2 - x1 * x1).powi(2) + (1.0 - x1).powi(2)
            })
            .sum()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let mut grad = vec![0.0; self.dimension];

        for i in 0..self.dimension / 2 {
            let x1 = x[2 * i];
            let x2 = x[2 * i + 1];

            grad[2 * i] = -400.0 * x1 * (x2 - x1 * x1) - 2.0 * (1.0 - x1);
            grad[2 * i + 1] = 200.0 * (x2 - x1 * x1);
        }

        grad
    }

    fn inequality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn inequality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-5.0, 5.0); self.dimension]
    }

    fn num_inequality_constraints(&self) -> usize {
        0
    }
    fn num_equality_constraints(&self) -> usize {
        0
    }
    fn num_variables(&self) -> usize {
        self.dimension
    }
}

/// Constrained test problem with many constraints
struct ConstrainedBenchmark {
    dimension: usize,
    num_constraints: usize,
}

impl ConstrainedBenchmark {
    fn new(dimension: usize, num_constraints: usize) -> Self {
        Self {
            dimension,
            num_constraints,
        }
    }
}

impl ConstrainedObjective<f64> for ConstrainedBenchmark {
    fn evaluate(&self, x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum::<f64>()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| 2.0 * xi).collect()
    }

    fn inequality_constraints(&self, x: &[f64]) -> Vec<f64> {
        (0..self.num_constraints)
            .map(|i| {
                let weight = (i + 1) as f64 / self.num_constraints as f64;
                x.iter().map(|&xi| weight * xi).sum::<f64>() - 1.0
            })
            .collect()
    }

    fn equality_constraints(&self, _x: &[f64]) -> Vec<f64> {
        vec![]
    }

    fn inequality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        (0..self.num_constraints)
            .map(|i| {
                let weight = (i + 1) as f64 / self.num_constraints as f64;
                vec![weight; self.dimension]
            })
            .collect()
    }

    fn equality_jacobian(&self, _x: &[f64]) -> Vec<Vec<f64>> {
        vec![]
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(-2.0, 2.0); self.dimension]
    }

    fn num_inequality_constraints(&self) -> usize {
        self.num_constraints
    }
    fn num_equality_constraints(&self) -> usize {
        0
    }
    fn num_variables(&self) -> usize {
        self.dimension
    }
}

/// Natural gradient benchmark problem
struct HighDimExponentialFamily {
    dimension: usize,
}

impl HighDimExponentialFamily {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl ObjectiveWithFisher<f64> for HighDimExponentialFamily {
    fn evaluate(&self, theta: &[f64]) -> f64 {
        theta
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let weight = 1.0 + 0.1 * i as f64;
                weight * t * t
            })
            .sum::<f64>()
            / 2.0
    }

    fn gradient(&self, theta: &[f64]) -> Vec<f64> {
        theta
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                let weight = 1.0 + 0.1 * i as f64;
                weight * t
            })
            .collect()
    }

    fn fisher_information(&self, _theta: &[f64]) -> Vec<Vec<f64>> {
        let mut matrix = vec![vec![0.0; self.dimension]; self.dimension];
        for (i, row) in matrix.iter_mut().enumerate().take(self.dimension) {
            let weight = 1.0 + 0.1 * i as f64;
            row[i] = weight;
        }
        matrix
    }
}

/// Multi-objective benchmark problem
struct HighDimZDT {
    dimension: usize,
}

impl HighDimZDT {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl MultiObjectiveFunction<f64> for HighDimZDT {
    fn evaluate(&self, x: &[f64]) -> Vec<f64> {
        let f1 = x[0];
        let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (self.dimension - 1) as f64;
        let h = 1.0 - (f1 / g).sqrt();
        let f2 = g * h;

        vec![f1, f2]
    }

    fn num_objectives(&self) -> usize {
        2
    }

    fn num_variables(&self) -> usize {
        self.dimension
    }

    fn variable_bounds(&self) -> Vec<(f64, f64)> {
        vec![(0.0, 1.0); self.dimension]
    }
}

/// Benchmark constrained optimization scaling with problem size
fn bench_constrained_scaling(c: &mut Criterion) {
    use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};

    let mut group = c.benchmark_group("constrained_scaling");
    group.measurement_time(Duration::from_secs(30));

    for dimension in [2, 5, 10, 20, 50].iter() {
        let problem = HighDimQuadratic::new(*dimension, 10.0);
        let optimizer = ConstrainedOptimizer::with_default_config(PenaltyMethod::Exterior);
        let opt_problem: OptimizationProblem<
            1,
            Constrained,
            SingleObjective,
            NonConvex,
            Euclidean,
        > = OptimizationProblem::new();

        group.throughput(Throughput::Elements(*dimension as u64));
        group.bench_with_input(
            BenchmarkId::new("quadratic", dimension),
            dimension,
            |b, &dim| {
                let initial_point = vec![1.0; dim];
                b.iter(|| {
                    let result = optimizer.optimize(
                        black_box(&opt_problem),
                        black_box(&problem),
                        black_box(initial_point.clone()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark constrained optimization with different condition numbers
fn bench_constrained_conditioning(c: &mut Criterion) {
    use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};

    let mut group = c.benchmark_group("constrained_conditioning");
    group.measurement_time(Duration::from_secs(20));

    for condition_number in [1.0, 10.0, 100.0, 1000.0].iter() {
        let problem = HighDimQuadratic::new(10, *condition_number);
        let optimizer =
            ConstrainedOptimizer::with_default_config(PenaltyMethod::AugmentedLagrangian);
        let opt_problem: OptimizationProblem<
            1,
            Constrained,
            SingleObjective,
            NonConvex,
            Euclidean,
        > = OptimizationProblem::new();

        group.bench_with_input(
            BenchmarkId::new("condition", condition_number),
            condition_number,
            |b, &_cond| {
                let initial_point = vec![1.0; 10];
                b.iter(|| {
                    let result = optimizer.optimize(
                        black_box(&opt_problem),
                        black_box(&problem),
                        black_box(initial_point.clone()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark different penalty methods
fn bench_penalty_methods(c: &mut Criterion) {
    use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};

    let mut group = c.benchmark_group("penalty_methods");
    group.measurement_time(Duration::from_secs(15));

    let problem = RosenbrockND::new(10);
    let opt_problem: OptimizationProblem<1, Constrained, SingleObjective, NonConvex, Euclidean> =
        OptimizationProblem::new();

    let methods = [
        ("exterior", PenaltyMethod::Exterior),
        ("interior", PenaltyMethod::Interior),
        ("augmented_lagrangian", PenaltyMethod::AugmentedLagrangian),
    ];

    for (name, method) in methods.iter() {
        let optimizer = ConstrainedOptimizer::with_default_config(*method);

        group.bench_with_input(
            BenchmarkId::new("rosenbrock", name),
            method,
            |b, &_method| {
                let initial_point = vec![0.5; 10];
                b.iter(|| {
                    let result = optimizer.optimize(
                        black_box(&opt_problem),
                        black_box(&problem),
                        black_box(initial_point.clone()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark natural gradient optimization
fn bench_natural_gradient(c: &mut Criterion) {
    use amari_optimization::phantom::{NonConvex, SingleObjective, Statistical, Unconstrained};

    let mut group = c.benchmark_group("natural_gradient");
    group.measurement_time(Duration::from_secs(20));

    for dimension in [5, 10, 20, 50, 100].iter() {
        let objective = HighDimExponentialFamily::new(*dimension);
        let config = NaturalGradientConfig::default();
        let optimizer = NaturalGradientOptimizer::new(config);
        let opt_problem: OptimizationProblem<
            1,
            Unconstrained,
            SingleObjective,
            NonConvex,
            Statistical,
        > = OptimizationProblem::new();

        group.throughput(Throughput::Elements(*dimension as u64));
        group.bench_with_input(
            BenchmarkId::new("exponential_family", dimension),
            dimension,
            |b, &dim| {
                let initial_theta = vec![0.1; dim];
                b.iter(|| {
                    let result = optimizer.optimize_statistical(
                        black_box(&opt_problem),
                        black_box(&objective),
                        black_box(initial_theta.clone()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark multi-objective optimization
fn bench_multi_objective(c: &mut Criterion) {
    use amari_optimization::phantom::{Euclidean, NonConvex, Unconstrained};

    let mut group = c.benchmark_group("multi_objective");
    group.measurement_time(Duration::from_secs(30));

    for dimension in [5, 10, 20, 30].iter() {
        let objective_function = HighDimZDT::new(*dimension);
        let config = MultiObjectiveConfig::default();
        let nsga2 = NsgaII::new(config);
        let opt_problem: OptimizationProblem<
            1,
            Unconstrained,
            MultiObjective,
            NonConvex,
            Euclidean,
        > = OptimizationProblem::new();

        group.throughput(Throughput::Elements(*dimension as u64));
        group.bench_with_input(BenchmarkId::new("zdt", dimension), dimension, |b, &_dim| {
            b.iter(|| {
                let result =
                    nsga2.optimize(black_box(&opt_problem), black_box(&objective_function));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark tropical optimization
fn bench_tropical_optimization(c: &mut Criterion) {
    use amari_tropical::{TropicalMatrix, TropicalNumber};

    let mut group = c.benchmark_group("tropical_optimization");
    group.measurement_time(Duration::from_secs(15));

    for size in [3, 5, 8, 10, 15].iter() {
        let optimizer = TropicalOptimizer::with_default_config();

        // Create tropical linear programming problem
        let objective: Vec<TropicalNumber<f64>> = (0..*size)
            .map(|i| TropicalNumber::new(i as f64 + 1.0))
            .collect();

        let matrix_data: Vec<Vec<f64>> = (0..*size)
            .map(|i| {
                (0..*size)
                    .map(|j| if i == j { 0.0 } else { (i + j) as f64 })
                    .collect()
            })
            .collect();
        let constraint_matrix = TropicalMatrix::from_log_probs(&matrix_data);

        let constraint_rhs: Vec<TropicalNumber<f64>> = (0..*size)
            .map(|i| TropicalNumber::new((*size + i) as f64))
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("linear_program", size), size, |b, &_sz| {
            b.iter(|| {
                let result = optimizer.solve_tropical_linear_program(
                    black_box(&objective),
                    black_box(&constraint_matrix),
                    black_box(&constraint_rhs),
                );
                black_box(result)
            });
        });

        // Benchmark tropical eigenvalue computation
        group.bench_with_input(BenchmarkId::new("eigenvalue", size), size, |b, &_sz| {
            b.iter(|| {
                let result = optimizer.solve_tropical_eigenvalue(black_box(&constraint_matrix));
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark constraint handling overhead
fn bench_constraint_overhead(c: &mut Criterion) {
    use amari_optimization::phantom::{Constrained, Euclidean, NonConvex, SingleObjective};

    let mut group = c.benchmark_group("constraint_overhead");
    group.measurement_time(Duration::from_secs(20));

    for num_constraints in [0, 1, 5, 10, 20, 50].iter() {
        let problem = ConstrainedBenchmark::new(10, *num_constraints);
        let optimizer =
            ConstrainedOptimizer::with_default_config(PenaltyMethod::AugmentedLagrangian);
        let opt_problem: OptimizationProblem<
            1,
            Constrained,
            SingleObjective,
            NonConvex,
            Euclidean,
        > = OptimizationProblem::new();

        group.throughput(Throughput::Elements(*num_constraints as u64));
        group.bench_with_input(
            BenchmarkId::new("constraints", num_constraints),
            num_constraints,
            |b, &_num| {
                let initial_point = vec![0.1; 10];
                b.iter(|| {
                    let result = optimizer.optimize(
                        black_box(&opt_problem),
                        black_box(&problem),
                        black_box(initial_point.clone()),
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Memory allocation benchmark
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");
    group.measurement_time(Duration::from_secs(10));

    for dimension in [10, 50, 100, 200, 500].iter() {
        group.throughput(Throughput::Elements(*dimension as u64));
        group.bench_with_input(
            BenchmarkId::new("vector_creation", dimension),
            dimension,
            |b, &dim| {
                b.iter(|| {
                    let _vectors: Vec<Vec<f64>> = (0..100).map(|_| vec![1.0; dim]).collect();
                    black_box(_vectors)
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("matrix_creation", dimension),
            dimension,
            |b, &dim| {
                b.iter(|| {
                    let _matrix: Vec<Vec<f64>> = (0..dim).map(|_| vec![1.0; dim]).collect();
                    black_box(_matrix)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_constrained_scaling,
    bench_constrained_conditioning,
    bench_penalty_methods,
    bench_natural_gradient,
    bench_multi_objective,
    bench_tropical_optimization,
    bench_constraint_overhead,
    bench_memory_allocation
);

criterion_main!(benches);
