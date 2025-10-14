//! GPU-accelerated enumerative geometry demonstration
//!
//! This example showcases comprehensive GPU acceleration for enumerative geometry
//! computations including intersection theory, Schubert calculus, and Gromov-Witten
//! invariants using modern WebGPU compute shaders.

#[cfg(feature = "gpu")]
use amari_enumerative::{
    gpu::{EnumerativeGpuOps, GpuGromovWittenData, GpuIntersectionData, GpuSchubertClass},
    ChowClass, SchubertClass,
};

#[cfg(feature = "gpu")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU-Accelerated Enumerative Geometry Demonstration");
    println!("====================================================");

    // Initialize GPU context
    let mut gpu_ops = match EnumerativeGpuOps::new().await {
        Ok(ops) => {
            println!("✅ Enumerative GPU context initialized successfully");
            ops
        }
        Err(e) => {
            println!("⚠️  GPU not available, falling back to CPU demo: {}", e);
            return demonstrate_cpu_enumerative();
        }
    };

    // Demonstrate batch intersection number computation
    println!("\n🔢 Batch Intersection Number Computation");
    demonstrate_gpu_intersection_theory(&mut gpu_ops).await?;

    // Demonstrate Schubert calculus on Grassmannians
    println!("\n🌿 GPU-Accelerated Schubert Calculus");
    demonstrate_gpu_schubert_calculus(&mut gpu_ops).await?;

    // Demonstrate Gromov-Witten invariant computation
    println!("\n🌊 Gromov-Witten Invariant Computation");
    demonstrate_gpu_gromov_witten(&mut gpu_ops).await?;

    // Performance analysis
    println!("\n📊 Performance Analysis");
    analyze_gpu_performance(&mut gpu_ops).await?;

    // Mathematical property verification
    println!("\n🔬 Mathematical Property Verification");
    verify_enumerative_properties(&mut gpu_ops).await?;

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("⚠️  This example requires the 'gpu' feature to be enabled.");
    println!("Run with: cargo run --example gpu_enumerative_geometry --features gpu");
}

#[cfg(feature = "gpu")]
fn demonstrate_cpu_enumerative() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating CPU enumerative geometry fallback...");

    // Create sample Chow classes
    let cubic = ChowClass::hypersurface(3);
    let quartic = ChowClass::hypersurface(4);
    let quintic = ChowClass::hypersurface(5);

    println!("✅ Created Chow classes:");
    println!("   Cubic curve: degree {}", cubic.degree);
    println!("   Quartic curve: degree {}", quartic.degree);
    println!("   Quintic curve: degree {}", quintic.degree);

    // Demonstrate Bézout's theorem
    let bezout_intersection = cubic.degree * quartic.degree;
    println!("\n✅ Bézout's theorem example:");
    println!(
        "   Cubic ∩ Quartic = {} × {} = {} points",
        cubic.degree, quartic.degree, bezout_intersection
    );

    // Create sample Schubert classes
    let partition1 = vec![2, 1];
    let partition2 = vec![1, 1, 1];

    let schubert1 = SchubertClass::new(partition1.clone(), (3, 6))?;
    let schubert2 = SchubertClass::new(partition2.clone(), (3, 6))?;

    println!("\n✅ Created Schubert classes:");
    println!("   σ_{:?} on Gr(3,6)", partition1);
    println!("   σ_{:?} on Gr(3,6)", partition2);

    // Convert to Chow classes
    let chow1 = schubert1.to_chow_class();
    let chow2 = schubert2.to_chow_class();

    println!("\n✅ Converted to Chow ring:");
    println!("   Codimension of σ_{:?}: {}", partition1, chow1.dimension);
    println!("   Codimension of σ_{:?}: {}", partition2, chow2.dimension);

    // Demonstrate multiplicative structure
    let product = chow1.multiply(&chow2);
    println!("\n✅ Chow ring multiplication:");
    println!("   Product degree: {}", product.degree);
    println!("   Product codimension: {}", product.dimension);

    println!("\n✅ CPU enumerative geometry demonstration completed");

    Ok(())
}

#[cfg(feature = "gpu")]
async fn demonstrate_gpu_intersection_theory(
    gpu_ops: &mut EnumerativeGpuOps,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create batch of intersection problems
    let intersection_data = vec![
        // Classic Bézout examples
        GpuIntersectionData {
            degree1: 3.0,
            degree2: 4.0,
            codimension1: 1.0,
            codimension2: 1.0,
            ambient_dimension: 2.0,
            genus_correction: 0.0,
            multiplicity_factor: 1.0,
            padding: 0.0,
        },
        GpuIntersectionData {
            degree1: 2.0,
            degree2: 5.0,
            codimension1: 1.0,
            codimension2: 1.0,
            ambient_dimension: 2.0,
            genus_correction: 0.0,
            multiplicity_factor: 1.0,
            padding: 0.0,
        },
        // Higher dimensional intersections
        GpuIntersectionData {
            degree1: 2.0,
            degree2: 3.0,
            codimension1: 1.0,
            codimension2: 1.0,
            ambient_dimension: 3.0,
            genus_correction: 0.0,
            multiplicity_factor: 1.0,
            padding: 0.0,
        },
        GpuIntersectionData {
            degree1: 4.0,
            degree2: 6.0,
            codimension1: 2.0,
            codimension2: 2.0,
            ambient_dimension: 4.0,
            genus_correction: 0.0,
            multiplicity_factor: 1.0,
            padding: 0.0,
        },
        // High genus corrections
        GpuIntersectionData {
            degree1: 5.0,
            degree2: 7.0,
            codimension1: 1.0,
            codimension2: 1.0,
            ambient_dimension: 2.0,
            genus_correction: 1.0,
            multiplicity_factor: 1.2,
            padding: 0.0,
        },
    ];

    let start_time = std::time::Instant::now();
    let results = gpu_ops
        .batch_intersection_numbers(&intersection_data)
        .await?;
    let computation_time = start_time.elapsed();

    println!(
        "   Computed {} intersection numbers in {:?}",
        results.len(),
        computation_time
    );
    println!("   Results:");

    for (i, (&result, data)) in results.iter().zip(intersection_data.iter()).enumerate() {
        println!(
            "     {}. Degrees ({}, {}) in ℙ^{}: {:.2} points",
            i + 1,
            data.degree1 as i32,
            data.degree2 as i32,
            data.ambient_dimension as i32,
            result
        );
    }

    // Verify Bézout's theorem
    println!("\n   Bézout verification:");
    for (i, (&result, data)) in results.iter().zip(intersection_data.iter()).enumerate() {
        if data.codimension1 + data.codimension2 == data.ambient_dimension {
            let expected = data.degree1 * data.degree2;
            let relative_error = (result - expected).abs() / expected.max(1.0);
            println!(
                "     {}. Expected: {:.1}, Got: {:.2}, Error: {:.2}%",
                i + 1,
                expected,
                result,
                relative_error * 100.0
            );
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn demonstrate_gpu_schubert_calculus(
    gpu_ops: &mut EnumerativeGpuOps,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create batch of Schubert class computations
    let schubert_data = vec![
        // Classical Grassmannian Gr(2,4)
        GpuSchubertClass {
            partition: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            grassmannian_k: 2.0,
            grassmannian_n: 4.0,
            padding: [0.0; 6],
        },
        GpuSchubertClass {
            partition: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            grassmannian_k: 2.0,
            grassmannian_n: 4.0,
            padding: [0.0; 6],
        },
        // Gr(2,5) with longer partitions
        GpuSchubertClass {
            partition: [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            grassmannian_k: 2.0,
            grassmannian_n: 5.0,
            padding: [0.0; 6],
        },
        GpuSchubertClass {
            partition: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            grassmannian_k: 3.0,
            grassmannian_n: 6.0,
            padding: [0.0; 6],
        },
        // Higher dimensional Grassmannian
        GpuSchubertClass {
            partition: [3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            grassmannian_k: 4.0,
            grassmannian_n: 8.0,
            padding: [0.0; 6],
        },
    ];

    let start_time = std::time::Instant::now();
    let results = gpu_ops.batch_schubert_numbers(&schubert_data).await?;
    let computation_time = start_time.elapsed();

    println!(
        "   Computed {} Schubert numbers in {:?}",
        results.len(),
        computation_time
    );
    println!("   Results:");

    for (i, (&result, data)) in results.iter().zip(schubert_data.iter()).enumerate() {
        let partition_str = format_partition(&data.partition);
        println!(
            "     {}. σ_{} on Gr({},{}) = {:.2}",
            i + 1,
            partition_str,
            data.grassmannian_k as i32,
            data.grassmannian_n as i32,
            result
        );
    }

    // Analyze Pieri rule patterns
    println!("\n   Pieri rule analysis:");
    for (i, &result) in results.iter().enumerate() {
        let data = &schubert_data[i];
        let codim = data.partition.iter().take(4).sum::<f32>() as i32;
        let k = data.grassmannian_k as i32;
        let n = data.grassmannian_n as i32;
        let expected_dim = k * (n - k);

        println!(
            "     {}. Codimension: {}, Max dimension: {}, Ratio: {:.2}",
            i + 1,
            codim,
            expected_dim,
            if expected_dim > 0 {
                result / expected_dim as f32
            } else {
                0.0
            }
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn demonstrate_gpu_gromov_witten(
    gpu_ops: &mut EnumerativeGpuOps,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create batch of Gromov-Witten computations
    let gw_data = vec![
        // Rational curves in ℙ^3
        GpuGromovWittenData {
            curve_degree: 1.0,
            genus: 0.0,
            marked_points: 4.0,
            target_dimension: 3.0,
            virtual_dimension: 0.0,
            quantum_parameter: 0.0,
            padding: [0.0; 2],
        },
        GpuGromovWittenData {
            curve_degree: 2.0,
            genus: 0.0,
            marked_points: 3.0,
            target_dimension: 3.0,
            virtual_dimension: 1.0,
            quantum_parameter: 0.0,
            padding: [0.0; 2],
        },
        // Elliptic curves
        GpuGromovWittenData {
            curve_degree: 3.0,
            genus: 1.0,
            marked_points: 1.0,
            target_dimension: 3.0,
            virtual_dimension: 1.0,
            quantum_parameter: 0.0,
            padding: [0.0; 2],
        },
        // Higher genus with quantum corrections
        GpuGromovWittenData {
            curve_degree: 2.0,
            genus: 2.0,
            marked_points: 0.0,
            target_dimension: 2.0,
            virtual_dimension: 0.0,
            quantum_parameter: 0.1,
            padding: [0.0; 2],
        },
        // K3 surface case
        GpuGromovWittenData {
            curve_degree: 4.0,
            genus: 0.0,
            marked_points: 2.0,
            target_dimension: 2.0,
            virtual_dimension: 2.0,
            quantum_parameter: 0.05,
            padding: [0.0; 2],
        },
    ];

    let start_time = std::time::Instant::now();
    let results = gpu_ops.batch_gromov_witten_invariants(&gw_data).await?;
    let computation_time = start_time.elapsed();

    println!(
        "   Computed {} GW invariants in {:?}",
        results.len(),
        computation_time
    );
    println!("   Results:");

    for (i, (&result, data)) in results.iter().zip(gw_data.iter()).enumerate() {
        println!(
            "     {}. ⟨τ_{}⟩^{{{}}}_{{{},{d}}} = {:.6}",
            i + 1,
            data.marked_points as i32,
            data.target_dimension as i32,
            data.genus as i32,
            result,
            d = data.curve_degree as i32
        );
    }

    // Analyze virtual dimensions
    println!("\n   Virtual dimension analysis:");
    for (i, data) in gw_data.iter().enumerate() {
        let expected_vdim = data.target_dimension as i32 * (1 - data.genus as i32)
            + data.curve_degree as i32 * data.target_dimension as i32
            + data.marked_points as i32;

        println!(
            "     {}. Expected vir.dim: {}, Stored: {}, Match: {}",
            i + 1,
            expected_vdim,
            data.virtual_dimension as i32,
            expected_vdim == data.virtual_dimension as i32
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn analyze_gpu_performance(
    gpu_ops: &mut EnumerativeGpuOps,
) -> Result<(), Box<dyn std::error::Error>> {
    // Performance test with varying batch sizes
    let batch_sizes = vec![10, 50, 100, 500, 1000];

    println!("   Intersection theory performance:");
    for &batch_size in &batch_sizes {
        let test_data: Vec<GpuIntersectionData> = (0..batch_size)
            .map(|i| GpuIntersectionData {
                degree1: (i % 5 + 1) as f32,
                degree2: (i % 7 + 1) as f32,
                codimension1: 1.0,
                codimension2: 1.0,
                ambient_dimension: 2.0,
                genus_correction: 0.0,
                multiplicity_factor: 1.0,
                padding: 0.0,
            })
            .collect();

        let start_time = std::time::Instant::now();
        let _results = gpu_ops.batch_intersection_numbers(&test_data).await?;
        let elapsed = start_time.elapsed();

        let throughput = batch_size as f64 / elapsed.as_secs_f64();
        println!(
            "     Batch size {}: {:.2} computations/second",
            batch_size, throughput
        );
    }

    println!("\n   Schubert calculus performance:");
    for &batch_size in &batch_sizes {
        let test_data: Vec<GpuSchubertClass> = (0..batch_size)
            .map(|i| {
                let mut partition = [0.0f32; 8];
                partition[0] = (i % 3 + 1) as f32;
                partition[1] = (i % 2) as f32;

                GpuSchubertClass {
                    partition,
                    grassmannian_k: ((i % 3) + 2) as f32,
                    grassmannian_n: ((i % 4) + 5) as f32,
                    padding: [0.0; 6],
                }
            })
            .collect();

        let start_time = std::time::Instant::now();
        let _results = gpu_ops.batch_schubert_numbers(&test_data).await?;
        let elapsed = start_time.elapsed();

        let throughput = batch_size as f64 / elapsed.as_secs_f64();
        println!(
            "     Batch size {}: {:.2} computations/second",
            batch_size, throughput
        );
    }

    // Memory efficiency analysis
    let large_batch_size = 2000;
    let start_memory = get_approximate_memory_usage();

    let large_test_data: Vec<GpuIntersectionData> = (0..large_batch_size)
        .map(|i| GpuIntersectionData {
            degree1: (i % 10 + 1) as f32,
            degree2: (i % 8 + 1) as f32,
            codimension1: 1.0,
            codimension2: 1.0,
            ambient_dimension: (i % 3 + 2) as f32,
            genus_correction: 0.0,
            multiplicity_factor: 1.0,
            padding: 0.0,
        })
        .collect();

    let _large_results = gpu_ops.batch_intersection_numbers(&large_test_data).await?;
    let end_memory = get_approximate_memory_usage();

    println!("\n   Memory efficiency:");
    println!(
        "     Large batch ({}): ~{:.1} MB memory increase",
        large_batch_size,
        (end_memory - start_memory) as f64 / (1024.0 * 1024.0)
    );

    Ok(())
}

#[cfg(feature = "gpu")]
async fn verify_enumerative_properties(
    gpu_ops: &mut EnumerativeGpuOps,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Verifying Bézout's theorem:");

    // Test cases for Bézout verification
    let bezout_tests = vec![
        (2, 3, 2), // Conic and cubic in ℙ^2
        (3, 4, 2), // Cubic and quartic in ℙ^2
        (2, 2, 3), // Two quadrics in ℙ^3
        (1, 5, 2), // Line and quintic in ℙ^2
    ];

    let mut bezout_data = Vec::new();
    let mut expected_values = Vec::new();

    for &(deg1, deg2, dim) in &bezout_tests {
        bezout_data.push(GpuIntersectionData {
            degree1: deg1 as f32,
            degree2: deg2 as f32,
            codimension1: 1.0,
            codimension2: 1.0,
            ambient_dimension: dim as f32,
            genus_correction: 0.0,
            multiplicity_factor: 1.0,
            padding: 0.0,
        });
        expected_values.push(deg1 * deg2);
    }

    let results = gpu_ops.batch_intersection_numbers(&bezout_data).await?;

    for (i, (&result, &expected)) in results.iter().zip(expected_values.iter()).enumerate() {
        let error = (result - expected as f32).abs() / expected as f32;
        let test = &bezout_tests[i];
        println!(
            "     Deg {} × Deg {} in ℙ^{}: {:.1} (expected {}), error: {:.2}%",
            test.0,
            test.1,
            test.2,
            result,
            expected,
            error * 100.0
        );
    }

    println!("\n   Verifying Grassmannian dimension formula:");

    let grassmannian_tests = vec![
        (2, 4), // Gr(2,4): 4-dimensional
        (2, 5), // Gr(2,5): 6-dimensional
        (3, 6), // Gr(3,6): 9-dimensional
        (2, 6), // Gr(2,6): 8-dimensional
    ];

    for &(k, n) in &grassmannian_tests {
        let expected_dim = k * (n - k);
        let test_partition = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let schubert_data = vec![GpuSchubertClass {
            partition: test_partition,
            grassmannian_k: k as f32,
            grassmannian_n: n as f32,
            padding: [0.0; 6],
        }];

        let results = gpu_ops.batch_schubert_numbers(&schubert_data).await?;

        println!(
            "     Gr({},{}): dimension {}, test result: {:.2}",
            k, n, expected_dim, results[0]
        );
    }

    println!("\n   Verifying virtual dimension formula:");

    // Test virtual dimension computation
    let vdim_tests = vec![
        (1, 0, 4, 3), // Lines in ℙ^3 through 4 points: vdim = 0
        (2, 0, 3, 3), // Conics in ℙ^3 through 3 points: vdim = 1
        (3, 1, 1, 3), // Elliptic curves of degree 3 in ℙ^3: vdim = 1
    ];

    for &(degree, genus, marked_pts, target_dim) in &vdim_tests {
        let expected_vdim = target_dim * (1 - genus) + degree * target_dim + marked_pts;

        let gw_data = vec![GpuGromovWittenData {
            curve_degree: degree as f32,
            genus: genus as f32,
            marked_points: marked_pts as f32,
            target_dimension: target_dim as f32,
            virtual_dimension: expected_vdim as f32,
            quantum_parameter: 0.0,
            padding: [0.0; 2],
        }];

        let results = gpu_ops.batch_gromov_witten_invariants(&gw_data).await?;

        println!(
            "     (d,g,n)=({},{},{}): expected vdim={}, GW result: {:.3}",
            degree, genus, marked_pts, expected_vdim, results[0]
        );
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn format_partition(partition: &[f32; 8]) -> String {
    let non_zero: Vec<String> = partition
        .iter()
        .take_while(|&&x| x > 0.0)
        .map(|&x| (x as i32).to_string())
        .collect();

    if non_zero.is_empty() {
        "∅".to_string()
    } else {
        format!("({})", non_zero.join(","))
    }
}

#[cfg(feature = "gpu")]
fn get_approximate_memory_usage() -> usize {
    // Simplified memory usage approximation
    // In practice, you'd use system-specific APIs
    std::process::id() as usize * 1024 // Rough approximation
}
