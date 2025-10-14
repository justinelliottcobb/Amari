//! GPU-Accelerated Information Geometry Demo
//!
//! This example demonstrates GPU-accelerated information geometry computations using WebGPU
//! for high-performance Fisher information matrix calculations, Bregman divergences,
//! and statistical manifold operations.

use amari_info_geom::{DuallyFlatManifold, InfoGeomError};

#[cfg(feature = "gpu")]
use amari_info_geom::{GpuBregmanData, GpuFisherData, GpuStatisticalManifold, InfoGeomGpuOps};

use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), InfoGeomError> {
    println!("🦀 Amari GPU Information Geometry Demo");
    println!("======================================\n");

    #[cfg(feature = "gpu")]
    {
        // Run GPU demonstrations
        gpu_fisher_information_demo().await?;
        gpu_bregman_divergence_demo().await?;
        gpu_statistical_manifold_demo().await?;
        gpu_performance_comparison().await?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!(
            "⚠️  GPU features not enabled. Compile with --features gpu to see GPU acceleration."
        );
        cpu_fallback_demo()?;
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn gpu_fisher_information_demo() -> Result<(), InfoGeomError> {
    println!("🚀 GPU Fisher Information Matrix Computation");
    println!("--------------------------------------------");

    // Initialize GPU operations
    match InfoGeomGpuOps::new().await {
        Ok(mut gpu_ops) => {
            println!("✅ GPU context initialized successfully");

            // Create test statistical distributions
            let statistical_points = vec![
                // Gaussian distribution parameters
                GpuFisherData {
                    param0: 0.5,  // μ (mean)
                    param1: -0.5, // -1/(2σ²) (precision parameter)
                    dimension: 2.0,
                    manifold_type: 0.0, // Exponential family
                    regularization: 1e-8,
                    connection_alpha: 0.0, // e-connection
                    ..Default::default()
                },
                // Another Gaussian with different parameters
                GpuFisherData {
                    param0: 1.0,
                    param1: -0.25,
                    dimension: 2.0,
                    manifold_type: 0.0,
                    regularization: 1e-8,
                    connection_alpha: 0.0,
                    ..Default::default()
                },
                // Probability simplex point
                GpuFisherData {
                    param0: 0.3, // p₁
                    param1: 0.4, // p₂
                    param2: 0.2, // p₃
                    param3: 0.1, // p₄
                    dimension: 4.0,
                    manifold_type: 1.0, // Probability simplex
                    regularization: 1e-8,
                    ..Default::default()
                },
            ];

            println!(
                "🔢 Computing Fisher information matrices for {} points...",
                statistical_points.len()
            );
            let start_time = Instant::now();

            let fisher_matrices = gpu_ops
                .batch_fisher_information(&statistical_points)
                .await?;

            let computation_time = start_time.elapsed();

            println!("✅ Fisher computation completed in {:?}", computation_time);
            println!(
                "   📊 Computed {} 4×4 Fisher matrices",
                fisher_matrices.len()
            );

            // Analyze Fisher matrices
            for (i, matrix) in fisher_matrices.iter().enumerate() {
                println!(
                    "   🔍 Matrix {}: First few elements: [{:.6}, {:.6}, {:.6}, {:.6}]",
                    i, matrix[0], matrix[1], matrix[4], matrix[5]
                );

                // Check positive definiteness via diagonal elements
                let is_positive_definite =
                    matrix[0] > 0.0 && matrix[5] > 0.0 && matrix[10] > 0.0 && matrix[15] > 0.0;
                println!("      Positive definite: {}", is_positive_definite);
            }

            println!();
        }
        Err(_) => {
            println!("⚠️  GPU not available, using CPU fallback");
            cpu_fallback_demo()?;
        }
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn gpu_bregman_divergence_demo() -> Result<(), InfoGeomError> {
    println!("📏 GPU Bregman Divergence Computation");
    println!("-------------------------------------");

    let mut gpu_ops = match InfoGeomGpuOps::new().await {
        Ok(ops) => ops,
        Err(_) => {
            println!("⚠️  GPU not available");
            return Ok(());
        }
    };

    // Create test cases for different Bregman divergences
    let bregman_data = vec![
        // KL divergence (entropy potential)
        GpuBregmanData {
            p_param0: 0.4,
            p_param1: 0.3,
            p_param2: 0.2,
            p_param3: 0.1,
            q_param0: 0.25,
            q_param1: 0.25,
            q_param2: 0.25,
            q_param3: 0.25,
            potential_type: 1.0, // Entropy
            potential_scale: 1.0,
            regularization: 1e-12,
            ..Default::default()
        },
        // Squared Euclidean distance (quadratic potential)
        GpuBregmanData {
            p_param0: 2.0,
            p_param1: 1.5,
            p_param2: 0.0,
            p_param3: 0.0,
            q_param0: 1.0,
            q_param1: 2.0,
            q_param2: 0.0,
            q_param3: 0.0,
            potential_type: 0.0, // Quadratic
            potential_scale: 1.0,
            regularization: 1e-12,
            ..Default::default()
        },
        // Exponential family divergence
        GpuBregmanData {
            p_param0: 0.8,
            p_param1: -0.4,
            p_param2: 0.0,
            p_param3: 0.0,
            q_param0: 0.6,
            q_param1: -0.3,
            q_param2: 0.0,
            q_param3: 0.0,
            potential_type: 2.0, // Exponential
            potential_scale: 0.1,
            regularization: 1e-12,
            ..Default::default()
        },
    ];

    println!(
        "🧮 Computing Bregman divergences for {} pairs...",
        bregman_data.len()
    );
    let start_time = Instant::now();

    let divergences = gpu_ops.batch_bregman_divergence(&bregman_data).await?;

    let computation_time = start_time.elapsed();

    println!(
        "✅ Bregman divergence computation completed in {:?}",
        computation_time
    );

    let divergence_names = ["KL Divergence", "Squared Euclidean", "Exponential Family"];
    for (&divergence, name) in divergences.iter().zip(divergence_names.iter()) {
        println!("   📊 {}: {:.6}", name, divergence);

        // Verify non-negativity (fundamental property of Bregman divergences)
        if divergence >= 0.0 {
            println!("      ✓ Non-negative (as expected)");
        } else {
            println!("      ⚠️  Negative value detected (numerical issue)");
        }
    }

    println!();
    Ok(())
}

#[cfg(feature = "gpu")]
async fn gpu_statistical_manifold_demo() -> Result<(), InfoGeomError> {
    println!("🌐 GPU Statistical Manifold Operations");
    println!("-------------------------------------");

    let mut gpu_ops = match InfoGeomGpuOps::new().await {
        Ok(ops) => ops,
        Err(_) => {
            println!("⚠️  GPU not available");
            return Ok(());
        }
    };

    // Create statistical manifold points with natural and expectation parameters
    let manifold_data = vec![
        GpuStatisticalManifold {
            eta0: 1.0,
            eta1: -0.5,
            eta2: 0.0,
            eta3: 0.0, // Natural parameters
            mu0: 0.8,
            mu1: 0.6,
            mu2: 0.0,
            mu3: 0.0,              // Expectation parameters
            alpha_connection: 0.0, // e-connection
            fisher_metric_det: 1.0,
            entropy: 1.2,
            temperature: 1.0,
            convergence_threshold: 1e-6,
            ..Default::default()
        },
        GpuStatisticalManifold {
            eta0: 0.5,
            eta1: -0.3,
            eta2: 0.0,
            eta3: 0.0,
            mu0: 0.7,
            mu1: 0.5,
            mu2: 0.0,
            mu3: 0.0,
            alpha_connection: 1.0, // m-connection
            fisher_metric_det: 0.8,
            entropy: 1.1,
            temperature: 1.0,
            convergence_threshold: 1e-6,
            ..Default::default()
        },
        GpuStatisticalManifold {
            eta0: 0.0,
            eta1: -0.2,
            eta2: 0.0,
            eta3: 0.0,
            mu0: 0.5,
            mu1: 0.4,
            mu2: 0.0,
            mu3: 0.0,
            alpha_connection: -0.5, // α-connection with α = -0.5
            fisher_metric_det: 0.9,
            entropy: 1.0,
            temperature: 1.0,
            convergence_threshold: 1e-6,
            ..Default::default()
        },
    ];

    println!(
        "🧮 Computing KL divergences for {} manifold points...",
        manifold_data.len()
    );
    let start_time = Instant::now();

    let kl_divergences = gpu_ops.batch_kl_divergence(&manifold_data).await?;

    let computation_time = start_time.elapsed();

    println!(
        "✅ KL divergence computation completed in {:?}",
        computation_time
    );

    for (i, &kl) in kl_divergences.iter().enumerate() {
        let connection_type = match manifold_data[i].alpha_connection {
            x if x.abs() < 1e-8 => "e-connection (α=0)",
            x if (x - 1.0).abs() < 1e-8 => "m-connection (α=1)",
            x => &format!("α-connection (α={:.1})", x),
        };

        println!("   📊 Point {} ({}): KL = {:.6}", i, connection_type, kl);
    }

    // Demonstrate manifold operations
    println!("\n🔧 Computing manifold operations...");
    let updated_manifolds = gpu_ops.batch_manifold_operations(&manifold_data).await?;

    println!("✅ Updated {} manifold points", updated_manifolds.len());
    for (i, manifold) in updated_manifolds.iter().enumerate() {
        println!(
            "   📍 Point {}: Fisher det = {:.6}, Entropy = {:.6}",
            i, manifold.fisher_metric_det, manifold.entropy
        );
    }

    println!();
    Ok(())
}

#[cfg(feature = "gpu")]
async fn gpu_performance_comparison() -> Result<(), InfoGeomError> {
    println!("⚡ GPU vs CPU Performance Comparison");
    println!("------------------------------------");

    let batch_sizes = [10, 100, 500];

    for &batch_size in &batch_sizes {
        println!("📏 Testing batch size: {}", batch_size);

        // GPU timing
        let gpu_time = match InfoGeomGpuOps::new().await {
            Ok(mut gpu_ops) => {
                let test_data: Vec<GpuFisherData> = (0..batch_size)
                    .map(|i| GpuFisherData {
                        param0: (i as f32) / (batch_size as f32),
                        param1: -(i as f32) / (batch_size as f32 * 2.0),
                        dimension: 2.0,
                        manifold_type: 0.0,
                        regularization: 1e-8,
                        ..Default::default()
                    })
                    .collect();

                let start = Instant::now();
                let _ = gpu_ops.batch_fisher_information(&test_data).await?;
                start.elapsed()
            }
            Err(_) => {
                println!("   ⚠️  GPU not available");
                continue;
            }
        };

        // CPU timing (simplified)
        let cpu_time = {
            let start = Instant::now();

            // Simulate CPU computation
            for i in 0..batch_size {
                let manifold = DuallyFlatManifold::new(2, 0.0);
                let point = vec![
                    (i as f64) / (batch_size as f64),
                    1.0 - (i as f64) / (batch_size as f64),
                ];
                let _ = manifold.fisher_metric_at(&point);
            }

            start.elapsed()
        };

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!("   ⏱️  GPU time: {:?}", gpu_time);
        println!("   ⏱️  CPU time: {:?}", cpu_time);
        println!("   🚀 GPU speedup: {:.2}x", speedup);
        println!();
    }

    Ok(())
}

fn cpu_fallback_demo() -> Result<(), InfoGeomError> {
    println!("🖥️  CPU Information Geometry Demo");
    println!("--------------------------------");

    // Create a dually flat manifold
    let manifold = DuallyFlatManifold::new(3, 0.0);

    println!("🔢 Computing Fisher information matrices...");
    let start_time = Instant::now();

    // Test points on probability simplex
    let test_points = [
        vec![0.5, 0.3, 0.2],
        vec![0.4, 0.4, 0.2],
        vec![0.6, 0.2, 0.2],
    ];

    for (i, point) in test_points.iter().enumerate() {
        let fisher_matrix = manifold.fisher_metric_at(point);
        let eigenvalues = fisher_matrix.eigenvalues();

        println!("   📊 Point {}: Fisher eigenvalues: {:?}", i, eigenvalues);

        // Check positive definiteness
        let is_positive_definite = eigenvalues.iter().all(|&x| x > 0.0);
        println!("      Positive definite: {}", is_positive_definite);
    }

    // Compute Bregman divergences
    println!("\n📏 Computing Bregman divergences...");
    let p = vec![0.4, 0.3, 0.3];
    let q = vec![0.33, 0.33, 0.34];

    let kl_div = manifold.bregman_divergence(&p, &q);
    println!("   📊 KL divergence: {:.6}", kl_div);

    let computation_time = start_time.elapsed();
    println!("✅ CPU computation completed in {:?}", computation_time);

    Ok(())
}
