//! # Morse Theory Example
//!
//! Demonstrates Morse theory for analyzing the topology of level sets
//! and critical points of functions.
//!
//! ## Mathematical Background
//!
//! Morse theory studies how topology changes as we sweep through level sets
//! f⁻¹((-∞, t]) of a smooth function f.
//!
//! Critical points are where ∇f = 0:
//! - Index 0 (minimum): All eigenvalues of Hessian positive
//! - Index 1 (saddle): One negative eigenvalue
//! - Index 2 (maximum): All eigenvalues negative
//!
//! Morse inequalities: #(index-k critical points) ≥ βₖ
//!
//! Run with: `cargo run --bin morse_theory`

use amari_topology::morse::{
    MorseFunction, CriticalPoint, CriticalPointType,
    find_critical_points_2d, MorseComplex,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    MORSE THEORY DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Height Function on a Surface
    // =========================================================================
    println!("Part 1: Height Function Example");
    println!("────────────────────────────────\n");

    // Consider height function on a torus standing upright
    // This is simulated with a function that has 4 critical points:
    // 2 minima, 2 saddles, 2 maxima... wait, torus has 1 min, 2 saddles, 1 max

    println!("Height function on a torus (standing upright):");
    println!("  - Bottom: 1 minimum (index 0)");
    println!("  - Lower rim: 1 saddle (index 1) - hole appears");
    println!("  - Upper rim: 1 saddle (index 1) - hole disappears");
    println!("  - Top: 1 maximum (index 2)");
    println!("\nMorse inequalities satisfied:");
    println!("  #minima (1) ≥ β₀ (1)  ✓");
    println!("  #saddles (2) ≥ β₁ (2)  ✓  (torus has 2 loops)");
    println!("  #maxima (1) ≥ β₂ (1)  ✓");

    // =========================================================================
    // Part 2: Critical Points of 2D Functions
    // =========================================================================
    println!("\n\nPart 2: Critical Points of 2D Functions");
    println!("────────────────────────────────────────\n");

    // Simple quadratic: f(x,y) = x² + y² (paraboloid)
    println!("Function f(x,y) = x² + y²  (paraboloid)");

    let paraboloid = |x: f64, y: f64| x*x + y*y;

    let critical_points = find_critical_points_2d(
        paraboloid,
        (-2.0, 2.0),  // x range
        (-2.0, 2.0),  // y range
        50,           // resolution
    )?;

    println!("  Critical points found: {}", critical_points.len());
    for cp in &critical_points {
        println!("    ({:.3}, {:.3}): {:?}, f = {:.3}",
                 cp.position.0, cp.position.1, cp.point_type, cp.value);
    }

    // Saddle function: f(x,y) = x² - y² (hyperbolic paraboloid)
    println!("\nFunction f(x,y) = x² - y²  (saddle surface)");

    let saddle = |x: f64, y: f64| x*x - y*y;

    let saddle_critical = find_critical_points_2d(
        saddle,
        (-2.0, 2.0),
        (-2.0, 2.0),
        50,
    )?;

    println!("  Critical points found: {}", saddle_critical.len());
    for cp in &saddle_critical {
        println!("    ({:.3}, {:.3}): {:?}, f = {:.3}",
                 cp.position.0, cp.position.1, cp.point_type, cp.value);
    }

    // Monkey saddle: f(x,y) = x³ - 3xy² (degenerate)
    println!("\nFunction f(x,y) = x³ - 3xy²  (monkey saddle - degenerate)");

    let monkey_saddle = |x: f64, y: f64| x.powi(3) - 3.0*x*y*y;

    let monkey_critical = find_critical_points_2d(
        monkey_saddle,
        (-2.0, 2.0),
        (-2.0, 2.0),
        50,
    )?;

    println!("  Critical points found: {}", monkey_critical.len());
    for cp in &monkey_critical {
        println!("    ({:.3}, {:.3}): {:?}, f = {:.3}",
                 cp.position.0, cp.position.1, cp.point_type, cp.value);
    }

    // =========================================================================
    // Part 3: Multiple Critical Points
    // =========================================================================
    println!("\n\nPart 3: Multiple Critical Points");
    println!("─────────────────────────────────\n");

    // Double-well: f(x,y) = (x² - 1)² + y²
    println!("Double-well: f(x,y) = (x² - 1)² + y²");
    println!("  Expected: 2 minima at (±1, 0), 1 saddle at (0, 0)");

    let double_well = |x: f64, y: f64| (x*x - 1.0).powi(2) + y*y;

    let dw_critical = find_critical_points_2d(
        double_well,
        (-2.0, 2.0),
        (-2.0, 2.0),
        100,
    )?;

    println!("\n  Critical points found:");
    for cp in &dw_critical {
        let type_str = match cp.point_type {
            CriticalPointType::Minimum => "minimum",
            CriticalPointType::Maximum => "maximum",
            CriticalPointType::Saddle => "saddle",
            CriticalPointType::Degenerate => "degenerate",
        };
        println!("    ({:6.3}, {:6.3}): {} (f = {:.3})",
                 cp.position.0, cp.position.1, type_str, cp.value);
    }

    // =========================================================================
    // Part 4: Morse Inequalities Verification
    // =========================================================================
    println!("\n\nPart 4: Morse Inequalities");
    println!("──────────────────────────\n");

    println!("The Morse inequalities relate critical points to Betti numbers:");
    println!("  mₖ ≥ βₖ  (weak Morse inequality)");
    println!("  Σ(-1)ᵏ mₖ = χ  (Euler characteristic)");
    println!();

    // Count critical points by index for the double-well
    let mut count_by_type = std::collections::HashMap::new();
    for cp in &dw_critical {
        *count_by_type.entry(&cp.point_type).or_insert(0) += 1;
    }

    let m0 = *count_by_type.get(&CriticalPointType::Minimum).unwrap_or(&0);
    let m1 = *count_by_type.get(&CriticalPointType::Saddle).unwrap_or(&0);
    let m2 = *count_by_type.get(&CriticalPointType::Maximum).unwrap_or(&0);

    println!("For the double-well function:");
    println!("  m₀ (minima) = {}", m0);
    println!("  m₁ (saddles) = {}", m1);
    println!("  m₂ (maxima) = {}", m2);
    println!();
    println!("  Euler characteristic: m₀ - m₁ + m₂ = {} - {} + {} = {}",
             m0, m1, m2, m0 as i32 - m1 as i32 + m2 as i32);
    println!("  (For the plane restricted to this region)");

    // =========================================================================
    // Part 5: Level Set Evolution
    // =========================================================================
    println!("\n\nPart 5: Level Set Evolution");
    println!("────────────────────────────\n");

    println!("How topology changes as we sweep through level sets of f(x,y) = (x²-1)² + y²:");
    println!();
    println!("  Level t | Description");
    println!("  ────────┼────────────────────────────────────────────");
    println!("  t < 0   | Empty set");
    println!("  t = 0   | Two points appear (minima at (±1,0))");
    println!("  0 < t < 1| Two separate disks growing");
    println!("  t = 1   | Disks touch at saddle point (0,0)");
    println!("  t > 1   | Single connected region (figure-8 opens)");
    println!();
    println!("  β₀ changes: 0 → 2 → 1 (components merge at saddle)");

    // =========================================================================
    // Part 6: Gradient Flow
    // =========================================================================
    println!("\n\nPart 6: Gradient Flow");
    println!("─────────────────────\n");

    println!("Gradient flow: dx/dt = -∇f(x)");
    println!("  - Flows downhill toward minima");
    println!("  - Stable manifolds of saddles separate basins of attraction");
    println!();

    // Simulate gradient descent from various starting points
    println!("Gradient descent from various starting points (double-well):");

    let start_points = vec![
        (0.5, 0.5),
        (-0.5, 0.5),
        (0.1, 0.0),
        (-0.1, 0.0),
        (1.5, 0.3),
    ];

    for (x0, y0) in start_points {
        let mut x = x0;
        let mut y = y0;
        let dt = 0.01;

        for _ in 0..1000 {
            // Gradient of (x² - 1)² + y²
            let gx = 4.0 * x * (x*x - 1.0);
            let gy = 2.0 * y;

            x -= dt * gx;
            y -= dt * gy;

            // Check convergence
            if gx*gx + gy*gy < 1e-10 {
                break;
            }
        }

        let basin = if x > 0.0 { "right minimum (+1, 0)" }
                   else { "left minimum (-1, 0)" };
        println!("  ({:5.2}, {:5.2}) → ({:6.3}, {:6.3}) : {}",
                 x0, y0, x, y, basin);
    }

    println!("\nNote: Points with x₀ ≈ 0 can go either way (on separatrix)");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
