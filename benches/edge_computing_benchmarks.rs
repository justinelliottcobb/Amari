//! Edge Computing Performance Benchmarks - TDD Phase 4
//!
//! Comprehensive benchmarks for WebGPU, WebAssembly, and edge computing
//! performance across different device types and workload sizes.

use amari_core::Multivector;
use amari_info_geom::amari_chentsov_tensor;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Benchmark CPU vs GPU tensor computation scaling
fn bench_tensor_computation_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_computation_scaling");

    let batch_sizes = vec![10, 50, 100, 500, 1000, 5000, 10000];

    for batch_size in batch_sizes {
        let (x_batch, y_batch, z_batch) = create_tensor_batch(batch_size);

        group.throughput(Throughput::Elements(batch_size as u64));

        // CPU baseline
        group.bench_with_input(BenchmarkId::new("cpu", batch_size), &batch_size, |b, _| {
            b.iter(|| {
                let _results: Vec<f64> = x_batch
                    .iter()
                    .zip(y_batch.iter())
                    .zip(z_batch.iter())
                    .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                    .collect();
            });
        });

        // GPU computation (when available)
        if let Ok(rt) = tokio::runtime::Runtime::new() {
            group.bench_with_input(BenchmarkId::new("gpu", batch_size), &batch_size, |b, _| {
                b.to_async(&rt).iter(|| async {
                    // Placeholder for GPU computation
                    compute_tensor_batch_gpu(&x_batch, &y_batch, &z_batch).await
                });
            });
        }

        // WebAssembly computation
        group.bench_with_input(BenchmarkId::new("wasm", batch_size), &batch_size, |b, _| {
            b.iter(|| {
                // Simulated WASM computation with some overhead
                let _results: Vec<f64> = x_batch
                    .iter()
                    .zip(y_batch.iter())
                    .zip(z_batch.iter())
                    .map(|((x, y), z)| {
                        // Simulate WASM call overhead with deterministic computation
                        let overhead_work =
                            (x.norm_squared() + y.norm_squared() + z.norm_squared()) * 0.001;
                        let result = amari_chentsov_tensor(x, y, z);
                        criterion::black_box(overhead_work + result)
                    })
                    .collect();
            });
        });
    }

    group.finish();
}

/// Benchmark memory allocation patterns for different device types
fn bench_memory_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation_patterns");

    let allocation_strategies = vec![
        ("stack", "Stack-allocated arrays"),
        ("heap", "Heap-allocated vectors"),
        ("gpu_buffer", "GPU buffer allocation"),
        ("wasm_linear", "WASM linear memory"),
    ];

    let data_size = 10000;

    for (strategy, description) in allocation_strategies {
        group.bench_function(BenchmarkId::new(strategy, description), |b| {
            match strategy {
                "stack" => {
                    b.iter(|| {
                        // Simulate stack allocation (limited size)
                        let _data: [f64; 1000] = [0.0; 1000];
                    });
                }
                "heap" => {
                    b.iter(|| {
                        let _data: Vec<f64> = vec![0.0; data_size];
                    });
                }
                "gpu_buffer" => {
                    b.iter(|| {
                        // Simulate GPU buffer allocation overhead
                        std::thread::sleep(Duration::from_micros(10));
                        let _data: Vec<f64> = vec![0.0; data_size];
                    });
                }
                "wasm_linear" => {
                    b.iter(|| {
                        // Simulate WASM linear memory allocation
                        std::thread::sleep(Duration::from_nanos(500));
                        let _data: Vec<f64> = vec![0.0; data_size];
                    });
                }
                _ => {}
            }
        });
    }

    group.finish();
}

/// Benchmark different precision levels for edge computing
fn bench_precision_vs_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_vs_performance");

    let precisions = vec![
        ("f32", "Single precision"),
        ("f64", "Double precision"),
        ("reduced", "Reduced precision"),
    ];

    let (x_batch, y_batch, z_batch) = create_tensor_batch(1000);

    for (precision, description) in precisions {
        group.bench_function(BenchmarkId::new(precision, description), |b| {
            match precision {
                "f32" => {
                    b.iter(|| {
                        // Convert to f32 and compute (faster but less precise)
                        let _results: Vec<f32> = x_batch
                            .iter()
                            .zip(y_batch.iter())
                            .zip(z_batch.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z) as f32)
                            .collect();
                    });
                }
                "f64" => {
                    b.iter(|| {
                        let _results: Vec<f64> = x_batch
                            .iter()
                            .zip(y_batch.iter())
                            .zip(z_batch.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();
                    });
                }
                "reduced" => {
                    b.iter(|| {
                        // Simulate reduced precision computation
                        let _results: Vec<f64> = x_batch
                            .iter()
                            .zip(y_batch.iter())
                            .zip(z_batch.iter())
                            .map(|((x, y), z)| {
                                // Reduce precision by rounding
                                let result = amari_chentsov_tensor(x, y, z);
                                (result * 1000.0).round() / 1000.0
                            })
                            .collect();
                    });
                }
                _ => {}
            }
        });
    }

    group.finish();
}

/// Benchmark data transfer patterns for edge computing
fn bench_data_transfer_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_transfer_patterns");

    let transfer_patterns = vec![
        ("zero_copy", "Zero-copy operations"),
        ("copy", "Memory copy operations"),
        ("streaming", "Streaming transfer"),
        ("batched", "Batched transfer"),
    ];

    let data_size = 100000;
    let source_data: Vec<f64> = (0..data_size).map(|i| i as f64).collect();

    for (pattern, description) in transfer_patterns {
        group.throughput(Throughput::Bytes((data_size * 8) as u64)); // 8 bytes per f64

        group.bench_function(BenchmarkId::new(pattern, description), |b| {
            match pattern {
                "zero_copy" => {
                    b.iter(|| {
                        // Simulate zero-copy by just creating a slice
                        let _view = &source_data[..];
                    });
                }
                "copy" => {
                    b.iter(|| {
                        let _copied = source_data.clone();
                    });
                }
                "streaming" => {
                    b.iter(|| {
                        // Simulate streaming by processing in chunks
                        let chunk_size = 1000;
                        for chunk in source_data.chunks(chunk_size) {
                            let _processed: f64 = chunk.iter().sum();
                        }
                    });
                }
                "batched" => {
                    b.iter(|| {
                        // Simulate batched transfer
                        let batch_size = 10000;
                        for batch in source_data.chunks(batch_size) {
                            let _batch_copy = batch.to_vec();
                        }
                    });
                }
                _ => {}
            }
        });
    }

    group.finish();
}

/// Benchmark edge device simulation with different capabilities
fn bench_edge_device_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("edge_device_simulation");

    let device_types = vec![
        ("mobile", "Mobile device simulation", 0.5), // 50% performance
        ("iot", "IoT device simulation", 0.2),       // 20% performance
        ("edge_server", "Edge server simulation", 2.0), // 200% performance
        ("workstation", "Workstation simulation", 4.0), // 400% performance
    ];

    let (x_batch, y_batch, z_batch) = create_tensor_batch(500);

    for (device, description, performance_factor) in device_types {
        let performance_factor: f64 = performance_factor;
        group.bench_function(BenchmarkId::new(device, description), |b| {
            b.iter(|| {
                // Simulate device performance by adding delay
                let delay_micros = (1000.0 / performance_factor) as u64;
                std::thread::sleep(Duration::from_micros(delay_micros));

                // Adjust computation complexity based on device capability
                let batch_limit = (x_batch.len() as f64 * performance_factor.min(1.0)) as usize;
                let limited_batch = &x_batch[..batch_limit];
                let limited_y = &y_batch[..batch_limit];
                let limited_z = &z_batch[..batch_limit];

                let _results: Vec<f64> = limited_batch
                    .iter()
                    .zip(limited_y.iter())
                    .zip(limited_z.iter())
                    .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                    .collect();
            });
        });
    }

    group.finish();
}

/// Benchmark WebGPU vs traditional GPU compute patterns
fn bench_webgpu_vs_traditional_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("webgpu_vs_traditional_gpu");

    let workload_sizes = vec![1000, 5000, 10000, 50000];

    for workload_size in workload_sizes {
        let (x_batch, y_batch, z_batch) = create_tensor_batch(workload_size);

        group.throughput(Throughput::Elements(workload_size as u64));

        // Traditional GPU compute simulation
        group.bench_with_input(
            BenchmarkId::new("traditional_gpu", workload_size),
            &workload_size,
            |b, _| {
                b.iter(|| {
                    // Simulate traditional GPU compute pipeline
                    std::thread::sleep(Duration::from_micros(50)); // GPU setup
                    let _results: Vec<f64> = x_batch
                        .iter()
                        .zip(y_batch.iter())
                        .zip(z_batch.iter())
                        .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                        .collect();
                    std::thread::sleep(Duration::from_micros(20)); // GPU cleanup
                });
            },
        );

        // WebGPU compute simulation
        group.bench_with_input(
            BenchmarkId::new("webgpu", workload_size),
            &workload_size,
            |b, _| {
                b.iter(|| {
                    // Simulate WebGPU compute pipeline with browser overhead
                    std::thread::sleep(Duration::from_micros(100)); // WebGPU setup
                    let _results: Vec<f64> = x_batch
                        .iter()
                        .zip(y_batch.iter())
                        .zip(z_batch.iter())
                        .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                        .collect();
                    std::thread::sleep(Duration::from_micros(30)); // WebGPU cleanup
                });
            },
        );
    }

    group.finish();
}

/// Benchmark adaptive workload distribution
fn bench_adaptive_workload_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("adaptive_workload_distribution");

    let distribution_strategies = vec![
        ("static", "Static distribution"),
        ("dynamic", "Dynamic load balancing"),
        ("predictive", "Predictive scheduling"),
        ("hybrid", "Hybrid CPU+GPU"),
    ];

    let total_workload = 10000;
    let (x_batch, y_batch, z_batch) = create_tensor_batch(total_workload);

    for (strategy, description) in distribution_strategies {
        group.bench_function(BenchmarkId::new(strategy, description), |b| {
            match strategy {
                "static" => {
                    b.iter(|| {
                        // Process everything on one device
                        let _results: Vec<f64> = x_batch
                            .iter()
                            .zip(y_batch.iter())
                            .zip(z_batch.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();
                    });
                }
                "dynamic" => {
                    b.iter(|| {
                        // Simulate dynamic load balancing with overhead
                        std::thread::sleep(Duration::from_micros(50)); // Load balancing overhead

                        let mid = total_workload / 2;
                        let (x1, x2) = x_batch.split_at(mid);
                        let (y1, y2) = y_batch.split_at(mid);
                        let (z1, z2) = z_batch.split_at(mid);

                        // Simulate parallel processing
                        let results1: Vec<f64> = x1
                            .iter()
                            .zip(y1.iter())
                            .zip(z1.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();

                        let results2: Vec<f64> = x2
                            .iter()
                            .zip(y2.iter())
                            .zip(z2.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();

                        let _combined = [results1, results2].concat();
                    });
                }
                "predictive" => {
                    b.iter(|| {
                        // Simulate predictive scheduling
                        std::thread::sleep(Duration::from_micros(100)); // Prediction overhead

                        // Optimal split based on "prediction"
                        let split_point = total_workload * 7 / 10; // 70/30 split
                        let (x1, x2) = x_batch.split_at(split_point);
                        let (y1, y2) = y_batch.split_at(split_point);
                        let (z1, z2) = z_batch.split_at(split_point);

                        let results1: Vec<f64> = x1
                            .iter()
                            .zip(y1.iter())
                            .zip(z1.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();

                        let results2: Vec<f64> = x2
                            .iter()
                            .zip(y2.iter())
                            .zip(z2.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();

                        let _combined = [results1, results2].concat();
                    });
                }
                "hybrid" => {
                    b.iter(|| {
                        // Simulate CPU+GPU hybrid processing
                        std::thread::sleep(Duration::from_micros(75)); // Coordination overhead

                        // 80% on GPU, 20% on CPU
                        let gpu_split = total_workload * 8 / 10;
                        let (gpu_x, cpu_x) = x_batch.split_at(gpu_split);
                        let (gpu_y, cpu_y) = y_batch.split_at(gpu_split);
                        let (gpu_z, cpu_z) = z_batch.split_at(gpu_split);

                        // GPU processing (simulated faster)
                        std::thread::sleep(Duration::from_micros(10));
                        let gpu_results: Vec<f64> = gpu_x
                            .iter()
                            .zip(gpu_y.iter())
                            .zip(gpu_z.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();

                        // CPU processing (simulated slower)
                        let cpu_results: Vec<f64> = cpu_x
                            .iter()
                            .zip(cpu_y.iter())
                            .zip(cpu_z.iter())
                            .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
                            .collect();

                        let _combined = [gpu_results, cpu_results].concat();
                    });
                }
                _ => {}
            }
        });
    }

    group.finish();
}

// Helper functions

fn create_tensor_batch(
    size: usize,
) -> (
    Vec<Multivector<3, 0, 0>>,
    Vec<Multivector<3, 0, 0>>,
    Vec<Multivector<3, 0, 0>>,
) {
    let mut x_batch = Vec::with_capacity(size);
    let mut y_batch = Vec::with_capacity(size);
    let mut z_batch = Vec::with_capacity(size);

    for i in 0..size {
        let mut x = Multivector::zero();
        let mut y = Multivector::zero();
        let mut z = Multivector::zero();

        // Create varied test vectors for realistic benchmarking
        x.set_vector_component(0, 1.0 + (i as f64) * 0.001);
        y.set_vector_component(1, 1.0 + (i as f64) * 0.002);
        z.set_vector_component(2, 1.0 + (i as f64) * 0.003);

        x_batch.push(x);
        y_batch.push(y);
        z_batch.push(z);
    }

    (x_batch, y_batch, z_batch)
}

async fn compute_tensor_batch_gpu(
    x_batch: &[Multivector<3, 0, 0>],
    y_batch: &[Multivector<3, 0, 0>],
    z_batch: &[Multivector<3, 0, 0>],
) -> Vec<f64> {
    // Simulate GPU computation with setup/teardown overhead
    tokio::time::sleep(Duration::from_micros(100)).await; // GPU setup

    let results: Vec<f64> = x_batch
        .iter()
        .zip(y_batch.iter())
        .zip(z_batch.iter())
        .map(|((x, y), z)| amari_chentsov_tensor(x, y, z))
        .collect();

    tokio::time::sleep(Duration::from_micros(50)).await; // GPU cleanup

    results
}

criterion_group!(
    edge_computing_benches,
    bench_tensor_computation_scaling,
    bench_memory_allocation_patterns,
    bench_precision_vs_performance,
    bench_data_transfer_patterns,
    bench_edge_device_simulation,
    bench_webgpu_vs_traditional_gpu,
    bench_adaptive_workload_distribution
);

criterion_main!(edge_computing_benches);
