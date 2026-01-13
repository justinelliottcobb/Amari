//! # Topological Data Analysis (TDA) Example
//!
//! Demonstrates practical applications of computational topology for data analysis.
//!
//! ## Mathematical Background
//!
//! TDA uses persistent homology to extract topological features from data:
//! - Shape detection (loops, voids, clusters)
//! - Noise filtering via persistence thresholds
//! - Feature extraction for machine learning
//! - Stability under perturbations (bottleneck distance)
//!
//! Run with: `cargo run --bin topological_data_analysis`

use amari_topology::{
    persistence::{Filtration, PersistentHomology, PersistenceDiagram, bottleneck_distance},
    tda::{PointCloud, VietorisRips, AlphaComplex, WassersteinDistance},
    statistics::{PersistenceLandscape, PersistenceImage, BottleneckStability},
};
use rand::Rng;
use nalgebra::DVector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("            TOPOLOGICAL DATA ANALYSIS DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Point Cloud from Different Shapes
    // =========================================================================
    println!("Part 1: Shape Detection via Persistent Homology");
    println!("────────────────────────────────────────────────\n");

    // Generate point cloud from a circle
    let n_circle = 50;
    let circle_points: Vec<DVector<f64>> = (0..n_circle)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_circle as f64;
            // Add small noise
            let mut rng = rand::thread_rng();
            let noise_r = 1.0 + 0.05 * rng.gen::<f64>();
            let noise_theta = theta + 0.05 * rng.gen::<f64>();
            DVector::from_vec(vec![noise_r * noise_theta.cos(), noise_r * noise_theta.sin()])
        })
        .collect();

    let circle_cloud = PointCloud::new(circle_points.clone())?;
    println!("Circle point cloud: {} points", n_circle);

    // Compute persistent homology via Vietoris-Rips
    let rips = VietorisRips::new(&circle_cloud, 2.0)?;
    let ph_circle = PersistentHomology::compute(&rips.filtration())?;

    println!("\nH₀ (connected components):");
    let h0 = ph_circle.diagram(0);
    let h0_persistent: Vec<_> = h0.pairs()
        .iter()
        .filter(|(b, d)| d - b > 0.1)
        .collect();
    println!("  Significant features (persistence > 0.1): {}", h0_persistent.len());

    println!("\nH₁ (loops):");
    let h1 = ph_circle.diagram(1);
    let h1_persistent: Vec<_> = h1.pairs()
        .iter()
        .filter(|(b, d)| d - b > 0.3)
        .collect();
    println!("  Significant features (persistence > 0.3): {}", h1_persistent.len());
    for (birth, death) in &h1_persistent {
        println!("    ({:.3}, {:.3}) - persistence = {:.3}", birth, death, death - birth);
    }
    println!("\n  → Circle detected! (1 significant H₁ feature)");

    // Generate point cloud from a figure-8
    println!("\n\nFigure-8 point cloud:");
    let n_figure8 = 60;
    let figure8_points: Vec<DVector<f64>> = (0..n_figure8)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / n_figure8 as f64;
            // Lemniscate of Bernoulli: (x,y) = (cos(t), sin(t)cos(t))
            let mut rng = rand::thread_rng();
            let x = t.cos() + 0.05 * rng.gen::<f64>();
            let y = t.sin() * t.cos() + 0.05 * rng.gen::<f64>();
            DVector::from_vec(vec![x, y])
        })
        .collect();

    let figure8_cloud = PointCloud::new(figure8_points)?;
    let rips_8 = VietorisRips::new(&figure8_cloud, 2.0)?;
    let ph_8 = PersistentHomology::compute(&rips_8.filtration())?;

    let h1_8 = ph_8.diagram(1);
    let h1_8_persistent: Vec<_> = h1_8.pairs()
        .iter()
        .filter(|(b, d)| d - b > 0.2)
        .collect();
    println!("  Significant H₁ features: {}", h1_8_persistent.len());
    println!("  → Figure-8 detected! (2 significant H₁ features = 2 loops)");

    // =========================================================================
    // Part 2: Clustering via H₀ Persistence
    // =========================================================================
    println!("\n\nPart 2: Clustering via H₀ Persistence");
    println!("──────────────────────────────────────\n");

    // Generate 3 clusters
    let mut cluster_points = Vec::new();
    let centers = [(0.0, 0.0), (3.0, 0.0), (1.5, 2.5)];
    let mut rng = rand::thread_rng();

    for (cx, cy) in centers.iter() {
        for _ in 0..20 {
            let x = cx + 0.3 * rng.gen::<f64>() - 0.15;
            let y = cy + 0.3 * rng.gen::<f64>() - 0.15;
            cluster_points.push(DVector::from_vec(vec![x, y]));
        }
    }

    let cluster_cloud = PointCloud::new(cluster_points)?;
    println!("Point cloud: 60 points in 3 clusters");

    let rips_cluster = VietorisRips::new(&cluster_cloud, 5.0)?;
    let ph_cluster = PersistentHomology::compute(&rips_cluster.filtration())?;

    let h0_cluster = ph_cluster.diagram(0);
    println!("\nH₀ persistence diagram:");

    // Sort by persistence (death - birth)
    let mut pairs: Vec<_> = h0_cluster.pairs().iter()
        .map(|(b, d)| (*b, *d, d - b))
        .collect();
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

    for (birth, death, pers) in pairs.iter().take(5) {
        let death_str = if death.is_infinite() { "∞".to_string() } else { format!("{:.3}", death) };
        println!("    ({:.3}, {}) - persistence = {:.3}",
                 birth, death_str,
                 if death.is_infinite() { f64::INFINITY } else { *pers });
    }

    // Find optimal number of clusters
    let significant_gaps: Vec<_> = pairs.iter()
        .filter(|(_, d, _)| !d.is_infinite())
        .filter(|(_, _, p)| *p > 0.5)
        .collect();

    println!("\nSignificant H₀ gaps (persistence > 0.5): {}", significant_gaps.len());
    println!("Estimated number of clusters: {} (essential feature + {} merges)",
             significant_gaps.len() + 1, significant_gaps.len());

    // =========================================================================
    // Part 3: Alpha Complex (More Efficient for Low Dimensions)
    // =========================================================================
    println!("\n\nPart 3: Alpha Complex Construction");
    println!("────────────────────────────────────\n");

    println!("Alpha complex vs Vietoris-Rips:");
    println!("  - Alpha: Based on Delaunay triangulation");
    println!("  - Alpha: Smaller complex, same homology");
    println!("  - Alpha: Better for low dimensions (2D, 3D)");
    println!("  - Rips: Simpler, works in any dimension");

    // Use circle points for comparison
    let alpha = AlphaComplex::new(&circle_cloud)?;
    let rips_full = VietorisRips::new(&circle_cloud, 2.0)?;

    println!("\nCircle point cloud comparison:");
    println!("  Rips complex size: {} simplices", rips_full.num_simplices());
    println!("  Alpha complex size: {} simplices", alpha.num_simplices());

    let ph_alpha = PersistentHomology::compute(&alpha.filtration())?;
    let h1_alpha = ph_alpha.diagram(1);

    println!("\nH₁ from Alpha complex:");
    for (birth, death) in h1_alpha.pairs().iter().take(3) {
        println!("    ({:.3}, {:.3})", birth, death);
    }

    // =========================================================================
    // Part 4: Persistence Landscape (Vectorization)
    // =========================================================================
    println!("\n\nPart 4: Persistence Landscape");
    println!("──────────────────────────────\n");

    println!("Persistence landscapes convert diagrams to functions:");
    println!("  λₖ(t) = kth largest value of min(t-b, d-t) over all (b,d)");
    println!("  - Stable: small changes in data → small changes in landscape");
    println!("  - Vectorizable: can use as features for ML");

    let landscape = PersistenceLandscape::from_diagram(&h1, 100)?;

    println!("\nCircle H₁ landscape (first 3 layers):");
    let sample_t = landscape.domain_max() * 0.5;
    for k in 0..3 {
        let value = landscape.evaluate(k, sample_t);
        println!("  λ_{} at t={:.2}: {:.4}", k, sample_t, value);
    }

    // Landscape norm (L²)
    let l2_norm = landscape.l2_norm();
    println!("\nL² norm of landscape: {:.4}", l2_norm);
    println!("  (Larger norm → more persistent features)");

    // =========================================================================
    // Part 5: Persistence Images (Alternative Vectorization)
    // =========================================================================
    println!("\n\nPart 5: Persistence Images");
    println!("───────────────────────────\n");

    println!("Persistence images are 2D histograms of persistence diagrams:");
    println!("  - Weight points by persistence (long-lived features matter more)");
    println!("  - Convolve with Gaussian kernel");
    println!("  - Result: Fixed-size feature vector for ML");

    let pi = PersistenceImage::from_diagram(&h1, 20, 20)?;
    let pi_vector = pi.to_vector();

    println!("\nCircle H₁ persistence image: {}x{} = {} features",
             pi.resolution().0, pi.resolution().1, pi_vector.len());

    let nonzero = pi_vector.iter().filter(|&&x| x > 1e-6).count();
    println!("  Non-zero entries: {}", nonzero);
    println!("  Max value: {:.6}", pi_vector.iter().cloned().fold(0.0f64, f64::max));

    // =========================================================================
    // Part 6: Stability - Bottleneck Distance
    // =========================================================================
    println!("\n\nPart 6: Stability via Bottleneck Distance");
    println!("──────────────────────────────────────────\n");

    println!("Bottleneck distance measures diagram similarity:");
    println!("  d_B(D₁, D₂) = inf_γ sup_p |p - γ(p)|_∞");
    println!("  where γ matches points between diagrams");

    // Create slightly perturbed circle
    let perturbed_circle: Vec<DVector<f64>> = circle_points.iter()
        .map(|p| {
            let mut rng = rand::thread_rng();
            let noise = DVector::from_vec(vec![
                0.1 * rng.gen::<f64>() - 0.05,
                0.1 * rng.gen::<f64>() - 0.05,
            ]);
            p + noise
        })
        .collect();

    let perturbed_cloud = PointCloud::new(perturbed_circle)?;
    let rips_perturbed = VietorisRips::new(&perturbed_cloud, 2.0)?;
    let ph_perturbed = PersistentHomology::compute(&rips_perturbed.filtration())?;
    let h1_perturbed = ph_perturbed.diagram(1);

    let bottleneck = bottleneck_distance(&h1, &h1_perturbed)?;
    println!("\nOriginal circle vs perturbed circle:");
    println!("  Perturbation magnitude: ~0.1");
    println!("  Bottleneck distance: {:.4}", bottleneck);
    println!("  → Small perturbation → small distance (stability!)");

    // Compare circle to figure-8
    let bottleneck_different = bottleneck_distance(&h1, &h1_8)?;
    println!("\nCircle vs Figure-8:");
    println!("  Bottleneck distance: {:.4}", bottleneck_different);
    println!("  → Different shapes → large distance");

    // =========================================================================
    // Part 7: Wasserstein Distance (Earth Mover's Distance)
    // =========================================================================
    println!("\n\nPart 7: Wasserstein Distance");
    println!("─────────────────────────────\n");

    println!("Wasserstein distance (p-Wasserstein):");
    println!("  W_p(D₁, D₂) = (inf_γ Σ |p - γ(p)|^p)^(1/p)");
    println!("  - More sensitive to all differences (not just worst)");
    println!("  - Commonly use p=1 or p=2");

    let wasserstein = WassersteinDistance::compute(&h1, &h1_perturbed, 2)?;
    println!("\nOriginal circle vs perturbed (2-Wasserstein):");
    println!("  W₂ distance: {:.4}", wasserstein);

    let wasserstein_diff = WassersteinDistance::compute(&h1, &h1_8, 2)?;
    println!("\nCircle vs Figure-8 (2-Wasserstein):");
    println!("  W₂ distance: {:.4}", wasserstein_diff);

    // =========================================================================
    // Part 8: Feature Extraction for Machine Learning
    // =========================================================================
    println!("\n\nPart 8: TDA Features for Machine Learning");
    println!("──────────────────────────────────────────\n");

    println!("Common TDA feature extraction strategies:");
    println!();
    println!("1. Persistence Statistics:");
    let stats = compute_persistence_stats(&h1);
    println!("   - Number of features: {}", stats.count);
    println!("   - Total persistence: {:.4}", stats.total);
    println!("   - Max persistence: {:.4}", stats.max);
    println!("   - Mean persistence: {:.4}", stats.mean);
    println!("   - Std persistence: {:.4}", stats.std);
    println!();
    println!("2. Persistence Landscape: {} values", landscape.to_vector(50)?.len());
    println!("3. Persistence Image: {} values", pi_vector.len());
    println!();
    println!("4. Betti Curve:");
    let betti_curve = compute_betti_curve(&h1, 50);
    println!("   - Sample β₁ values at 50 thresholds");
    println!("   - Max β₁: {}", betti_curve.iter().max().unwrap_or(&0));
    println!();
    println!("5. Silhouette:");
    println!("   - Weighted sum of landscapes: φ(t) = Σ wₖ λₖ(t)");
    println!("   - Common weights: wₖ = k^(-p) for some p > 0");

    // =========================================================================
    // Part 9: Noisy Data and Persistence Threshold
    // =========================================================================
    println!("\n\nPart 9: Denoising via Persistence Threshold");
    println!("────────────────────────────────────────────\n");

    // Generate circle with significant noise
    let n_noisy = 40;
    let noisy_circle: Vec<DVector<f64>> = (0..n_noisy)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_noisy as f64;
            let mut rng = rand::thread_rng();
            let r = 1.0 + 0.3 * (rng.gen::<f64>() - 0.5);  // Larger noise
            DVector::from_vec(vec![r * theta.cos(), r * theta.sin()])
        })
        .collect();

    let noisy_cloud = PointCloud::new(noisy_circle)?;
    let rips_noisy = VietorisRips::new(&noisy_cloud, 2.0)?;
    let ph_noisy = PersistentHomology::compute(&rips_noisy.filtration())?;
    let h1_noisy = ph_noisy.diagram(1);

    println!("Noisy circle (30% noise):");
    println!("  Total H₁ features: {}", h1_noisy.pairs().len());

    let thresholds = [0.05, 0.1, 0.2, 0.3];
    println!("\n  Threshold | Features | Interpretation");
    println!("  ──────────┼──────────┼─────────────────────────────");
    for thresh in thresholds {
        let count = h1_noisy.pairs().iter()
            .filter(|(b, d)| d - b > thresh)
            .count();
        let interpretation = match count {
            0 => "Too strict - signal lost",
            1 => "Correct - circle detected",
            2..=3 => "Slightly noisy",
            _ => "Too lenient - noise included",
        };
        println!("  {:9.2} | {:8} | {}", thresh, count, interpretation);
    }

    println!("\n  → Choose threshold where significant features stabilize");

    // =========================================================================
    // Part 10: Summary
    // =========================================================================
    println!("\n\nSummary: TDA Pipeline");
    println!("─────────────────────\n");

    println!("1. Data → Point Cloud");
    println!("   - Embed data as points in metric space");
    println!();
    println!("2. Point Cloud → Filtration");
    println!("   - Vietoris-Rips: General purpose, high-dimensional");
    println!("   - Alpha Complex: Efficient for low dimensions");
    println!();
    println!("3. Filtration → Persistence Diagrams");
    println!("   - Track birth/death of topological features");
    println!("   - H₀: clusters, H₁: loops, H₂: voids");
    println!();
    println!("4. Diagrams → Features");
    println!("   - Statistics, landscapes, images");
    println!("   - Use for ML classification/regression");
    println!();
    println!("5. Compare Diagrams");
    println!("   - Bottleneck distance: worst-case difference");
    println!("   - Wasserstein distance: total difference");
    println!();
    println!("Key insight: Persistent features (high death-birth) are signal,");
    println!("             short-lived features are noise.");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}

/// Persistence statistics for a diagram
struct PersistenceStats {
    count: usize,
    total: f64,
    max: f64,
    mean: f64,
    std: f64,
}

fn compute_persistence_stats(diagram: &PersistenceDiagram) -> PersistenceStats {
    let persistences: Vec<f64> = diagram.pairs()
        .iter()
        .filter(|(_, d)| !d.is_infinite())
        .map(|(b, d)| d - b)
        .collect();

    let count = persistences.len();
    if count == 0 {
        return PersistenceStats { count: 0, total: 0.0, max: 0.0, mean: 0.0, std: 0.0 };
    }

    let total: f64 = persistences.iter().sum();
    let max = persistences.iter().cloned().fold(0.0f64, f64::max);
    let mean = total / count as f64;
    let variance: f64 = persistences.iter()
        .map(|p| (p - mean).powi(2))
        .sum::<f64>() / count as f64;
    let std = variance.sqrt();

    PersistenceStats { count, total, max, mean, std }
}

fn compute_betti_curve(diagram: &PersistenceDiagram, n_samples: usize) -> Vec<usize> {
    let pairs = diagram.pairs();
    if pairs.is_empty() {
        return vec![0; n_samples];
    }

    let max_death = pairs.iter()
        .filter(|(_, d)| !d.is_infinite())
        .map(|(_, d)| *d)
        .fold(0.0f64, f64::max);

    (0..n_samples)
        .map(|i| {
            let t = max_death * i as f64 / (n_samples - 1) as f64;
            pairs.iter()
                .filter(|(b, d)| *b <= t && t < *d)
                .count()
        })
        .collect()
}
