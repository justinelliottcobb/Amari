//! # Persistent Homology Example
//!
//! Demonstrates persistent homology computation for topological data analysis (TDA).
//!
//! ## Mathematical Background
//!
//! Persistent homology tracks how topological features (connected components,
//! loops, voids) appear and disappear as we grow a filtration parameter.
//!
//! Key concepts:
//! - Filtration: Nested sequence of simplicial complexes K₀ ⊆ K₁ ⊆ ... ⊆ Kₙ
//! - Birth time: When a feature first appears
//! - Death time: When a feature merges/fills in
//! - Persistence: Death - Birth (lifetime of feature)
//! - Persistence diagram: Plot of (birth, death) pairs
//!
//! Run with: `cargo run --bin persistent_homology`

use amari_topology::{
    persistence::{Filtration, PersistentHomology, PersistenceDiagram},
    simplex::Simplex,
    complex::SimplicialComplex,
};
use rand::Rng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                PERSISTENT HOMOLOGY DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Simple Filtration Example
    // =========================================================================
    println!("Part 1: Simple Filtration");
    println!("─────────────────────────\n");

    // Build a simple filtration: 3 vertices → edges → triangle
    let mut filtration = Filtration::new();

    // Time 0: Add vertices
    filtration.add(Simplex::vertex(0), 0.0)?;
    filtration.add(Simplex::vertex(1), 0.0)?;
    filtration.add(Simplex::vertex(2), 0.0)?;

    // Time 1: Add edge [0,1]
    filtration.add(Simplex::edge(0, 1), 1.0)?;

    // Time 2: Add edge [1,2] - now 0,1,2 connected
    filtration.add(Simplex::edge(1, 2), 2.0)?;

    // Time 3: Add edge [0,2] - creates a cycle (loop)
    filtration.add(Simplex::edge(0, 2), 3.0)?;

    // Time 4: Fill in triangle - kills the cycle
    filtration.add(Simplex::triangle(0, 1, 2), 4.0)?;

    println!("Filtration steps:");
    println!("  t=0: Add vertices 0, 1, 2  (3 components born)");
    println!("  t=1: Add edge [0,1]        (components 0,1 merge → 1 dies)");
    println!("  t=2: Add edge [1,2]        (component 2 merges → dies)");
    println!("  t=3: Add edge [0,2]        (cycle born!)");
    println!("  t=4: Add triangle [0,1,2]  (cycle filled → dies)");

    // Compute persistent homology
    let ph = PersistentHomology::compute(&filtration)?;

    println!("\nPersistence diagrams:");

    // H₀ (connected components)
    let diagram_h0 = ph.diagram(0);
    println!("\n  H₀ (connected components):");
    for (birth, death) in diagram_h0.pairs() {
        let persistence = death - birth;
        let death_str = if death.is_infinite() { "∞".to_string() } else { format!("{:.1}", death) };
        println!("    ({:.1}, {}) - persistence = {}",
                 birth, death_str,
                 if death.is_infinite() { "∞ (essential)".to_string() } else { format!("{:.1}", persistence) });
    }

    // H₁ (loops)
    let diagram_h1 = ph.diagram(1);
    println!("\n  H₁ (loops/cycles):");
    if diagram_h1.pairs().is_empty() {
        println!("    (no persistent features)");
    }
    for (birth, death) in diagram_h1.pairs() {
        let persistence = death - birth;
        println!("    ({:.1}, {:.1}) - persistence = {:.1}", birth, death, persistence);
    }

    // =========================================================================
    // Part 2: Vietoris-Rips Complex
    // =========================================================================
    println!("\n\nPart 2: Vietoris-Rips Complex from Point Cloud");
    println!("────────────────────────────────────────────────\n");

    // Generate point cloud: 3 clusters
    let points = vec![
        // Cluster 1
        (0.0, 0.0), (0.1, 0.1), (0.0, 0.2),
        // Cluster 2
        (1.0, 0.0), (1.1, 0.1), (1.0, 0.2),
        // Cluster 3
        (0.5, 1.0), (0.6, 1.1), (0.5, 1.2),
    ];

    println!("Point cloud: 9 points in 3 clusters");
    println!("  Cluster 1: around (0, 0)");
    println!("  Cluster 2: around (1, 0)");
    println!("  Cluster 3: around (0.5, 1)");

    // Compute distance matrix
    let n = points.len();
    let mut distances = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            distances[i][j] = (dx*dx + dy*dy).sqrt();
        }
    }

    // Build Rips filtration
    let rips = Filtration::rips_from_distances(&distances, 2)?;

    println!("\nVietoris-Rips filtration:");
    println!("  Add edge [i,j] at distance d(i,j)");
    println!("  Add triangle [i,j,k] at max(d(i,j), d(j,k), d(i,k))");

    let ph_rips = PersistentHomology::compute(&rips)?;

    let diagram_h0_rips = ph_rips.diagram(0);
    println!("\n  H₀ persistence diagram:");

    let mut pairs: Vec<_> = diagram_h0_rips.pairs().iter().collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    for (birth, death) in pairs.iter().take(5) {
        let death_str = if death.is_infinite() { "∞".to_string() } else { format!("{:.3}", death) };
        println!("    ({:.3}, {})", birth, death_str);
    }
    if pairs.len() > 5 {
        println!("    ... ({} more pairs)", pairs.len() - 5);
    }

    // Find significant features (high persistence)
    println!("\n  Interpretation:");
    println!("    - Short-lived features (low persistence): noise");
    println!("    - Long-lived features (high persistence): true structure");
    println!("    - 3 clusters → expect 2 significant deaths in H₀");

    // =========================================================================
    // Part 3: Detecting a Loop
    // =========================================================================
    println!("\n\nPart 3: Detecting Topological Loops");
    println!("────────────────────────────────────\n");

    // Points arranged in a circle
    let n_circle = 12;
    let circle_points: Vec<(f64, f64)> = (0..n_circle)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / n_circle as f64;
            (theta.cos(), theta.sin())
        })
        .collect();

    println!("Circle point cloud: {} points on unit circle", n_circle);

    // Compute distances
    let mut circle_distances = vec![vec![0.0; n_circle]; n_circle];
    for i in 0..n_circle {
        for j in 0..n_circle {
            let dx = circle_points[i].0 - circle_points[j].0;
            let dy = circle_points[i].1 - circle_points[j].1;
            circle_distances[i][j] = (dx*dx + dy*dy).sqrt();
        }
    }

    let rips_circle = Filtration::rips_from_distances(&circle_distances, 2)?;
    let ph_circle = PersistentHomology::compute(&rips_circle)?;

    println!("\nH₁ (loops) persistence diagram:");
    let diagram_h1_circle = ph_circle.diagram(1);

    for (birth, death) in diagram_h1_circle.pairs() {
        let persistence = death - birth;
        let significance = if persistence > 0.5 { "*** SIGNIFICANT ***" } else { "" };
        println!("    ({:.3}, {:.3}) - persistence = {:.3} {}",
                 birth, death, persistence, significance);
    }

    println!("\n  The circle should have one significant H₁ feature!");
    println!("  (The loop appears when neighbors connect, dies when center fills)");

    // =========================================================================
    // Part 4: Betti Numbers
    // =========================================================================
    println!("\n\nPart 4: Betti Numbers");
    println!("──────────────────────\n");

    println!("Betti numbers βₖ count independent k-dimensional holes:");
    println!("  β₀ = # connected components");
    println!("  β₁ = # independent loops");
    println!("  β₂ = # enclosed voids");

    // Torus would have β₀=1, β₁=2, β₂=1
    // But we'll show the triangle example

    let mut triangle_filt = Filtration::new();
    for v in 0..3 {
        triangle_filt.add(Simplex::vertex(v), 0.0)?;
    }
    triangle_filt.add(Simplex::edge(0, 1), 1.0)?;
    triangle_filt.add(Simplex::edge(1, 2), 1.0)?;
    triangle_filt.add(Simplex::edge(0, 2), 1.0)?;
    // Don't fill in triangle

    let ph_triangle = PersistentHomology::compute(&triangle_filt)?;
    let betti = ph_triangle.betti_numbers_at(1.5);

    println!("\nTriangle boundary (without interior) at t=1.5:");
    println!("  β₀ = {} (one connected component)", betti.get(&0).unwrap_or(&0));
    println!("  β₁ = {} (one loop)", betti.get(&1).unwrap_or(&0));

    // =========================================================================
    // Part 5: Persistence Landscape / Statistics
    // =========================================================================
    println!("\n\nPart 5: Persistence Statistics");
    println!("───────────────────────────────\n");

    // Use the circle example
    let diagram = ph_circle.diagram(1);

    let total_persistence: f64 = diagram.pairs()
        .iter()
        .map(|(b, d)| d - b)
        .sum();

    let max_persistence = diagram.pairs()
        .iter()
        .map(|(b, d)| d - b)
        .fold(0.0f64, |a, b| a.max(b));

    let num_features = diagram.pairs().len();

    println!("H₁ statistics for circle point cloud:");
    println!("  Number of features: {}", num_features);
    println!("  Total persistence: {:.3}", total_persistence);
    println!("  Maximum persistence: {:.3}", max_persistence);
    println!("  Average persistence: {:.3}",
             if num_features > 0 { total_persistence / num_features as f64 } else { 0.0 });

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
