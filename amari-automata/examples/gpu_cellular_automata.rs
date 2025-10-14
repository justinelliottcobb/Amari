//! GPU-Accelerated Cellular Automata Demo
//!
//! This example demonstrates GPU-accelerated cellular automata using WebGPU for
//! high-performance evolution of geometric cellular automata systems.

use amari_automata::{
    AutomataResult, Evolvable, GeometricCA, RuleType,
};

#[cfg(feature = "gpu")]
use amari_automata::{
    AutomataGpuOps, GpuCellData, GpuEvolutionParams, GpuRuleConfig,
};

use amari_core::Multivector;
use std::time::Instant;

#[tokio::main]
async fn main() -> AutomataResult<()> {
    println!("🦀 Amari GPU Cellular Automata Demo");
    println!("===================================\n");

    #[cfg(feature = "gpu")]
    {
        // Run GPU demonstrations
        gpu_cellular_automata_demo().await?;
        gpu_performance_comparison().await?;
        gpu_rule_variants_demo().await?;
        gpu_large_scale_simulation().await?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        println!("⚠️  GPU features not enabled. Compile with --features gpu to see GPU acceleration.");
        cpu_fallback_demo()?;
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn gpu_cellular_automata_demo() -> AutomataResult<()> {
    println!("🚀 GPU Cellular Automata Evolution");
    println!("----------------------------------");

    // Initialize GPU operations
    match AutomataGpuOps::new().await {
        Ok(mut gpu_ops) => {
            println!("✅ GPU context initialized successfully");

            // Create initial configuration with geometric patterns
            let grid_size = 64;
            let total_cells = grid_size * grid_size;

            let mut initial_cells = vec![GpuCellData::default(); total_cells];

            // Set up a glider pattern (adapted for geometric CA)
            let glider_pattern = [
                (32, 32), (33, 32), (34, 32), // Line
                (34, 33), (33, 34),           // L-shape continuation
            ];

            for &(x, y) in &glider_pattern {
                let index = y * grid_size + x;
                if index < total_cells {
                    initial_cells[index] = GpuCellData {
                        scalar: 1.0,
                        e1: 0.5,
                        e2: 0.5,
                        e3: 0.0,
                        e12: 0.25,
                        e13: 0.0,
                        e23: 0.0,
                        e123: 0.0,
                        generation: 0.0,
                        neighborhood_size: 8.0,
                        rule_type: 0.0,
                        boundary_condition: 0.0,
                        padding: [0.0; 4],
                    };
                }
            }

            // Configure geometric CA rule
            let rule_configs = vec![GpuRuleConfig {
                rule_type: 0.0, // Geometric rule
                threshold: 0.3,
                damping_factor: 0.8,
                energy_conservation: 0.95,
                time_step: 1.0,
                spatial_scale: 1.0,
                geometric_weight: 1.0,
                nonlinear_factor: 0.1,
                boundary_type: 0.0, // Periodic
                neighborhood_radius: 1.0,
                evolution_speed: 1.0,
                stability_factor: 0.9,
                padding: [0.0; 4],
            }];

            // Evolution parameters
            let evolution_params = GpuEvolutionParams {
                grid_width: grid_size as f32,
                grid_height: grid_size as f32,
                total_cells: total_cells as f32,
                steps_per_batch: 10.0,
                current_generation: 0.0,
                max_generations: 100.0,
                convergence_threshold: 0.001,
                energy_scale: 1.0,
                workgroup_size_x: 16.0,
                workgroup_size_y: 16.0,
                parallel_factor: 1.0,
                memory_optimization: 1.0,
                padding: [0.0; 4],
            };

            // Evolve the cellular automaton
            println!("🔄 Evolving geometric cellular automaton...");
            let start_time = Instant::now();

            let evolved_cells = gpu_ops
                .batch_evolve_ca(&initial_cells, &rule_configs, &evolution_params)
                .await?;

            let evolution_time = start_time.elapsed();

            // Calculate energy metrics
            let initial_energy = gpu_ops.calculate_total_energy(&initial_cells).await?;
            let final_energy = gpu_ops.calculate_total_energy(&evolved_cells).await?;

            println!("✅ Evolution completed in {:?}", evolution_time);
            println!("   📊 Initial energy: {:.6}", initial_energy);
            println!("   📊 Final energy: {:.6}", final_energy);
            println!("   📊 Energy change: {:.6}", final_energy - initial_energy);
            println!("   📊 Active cells: {}/{}",
                evolved_cells.iter().filter(|c| cell_magnitude(c) > 0.1).count(),
                total_cells);

            // Analyze pattern evolution
            analyze_pattern_evolution(&initial_cells, &evolved_cells, grid_size);

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
async fn gpu_performance_comparison() -> AutomataResult<()> {
    println!("⚡ GPU vs CPU Performance Comparison");
    println!("------------------------------------");

    let grid_sizes = [32, 64, 128];
    let generations = 50;

    for &grid_size in &grid_sizes {
        println!("📏 Testing {}×{} grid with {} generations", grid_size, grid_size, generations);

        // GPU timing
        let gpu_time = match AutomataGpuOps::new().await {
            Ok(mut gpu_ops) => {
                let total_cells = grid_size * grid_size;
                let initial_cells = vec![random_cell(); total_cells];
                let rule_configs = vec![GpuRuleConfig::default()];
                let evolution_params = GpuEvolutionParams {
                    grid_width: grid_size as f32,
                    grid_height: grid_size as f32,
                    total_cells: total_cells as f32,
                    steps_per_batch: generations as f32,
                    ..Default::default()
                };

                let start = Instant::now();
                let _ = gpu_ops.batch_evolve_ca(&initial_cells, &rule_configs, &evolution_params).await?;
                start.elapsed()
            }
            Err(_) => {
                println!("   ⚠️  GPU not available");
                continue;
            }
        };

        // CPU timing
        let cpu_time = {
            let mut ca = GeometricCA::<3, 0, 0>::new_2d(grid_size, grid_size);

            // Set random initial state
            for x in 0..grid_size {
                for y in 0..grid_size {
                    if (x + y) % 3 == 0 {  // Simple deterministic pattern instead of rand
                        let _ = ca.set_cell_2d(x, y, Multivector::scalar(0.8));
                    }
                }
            }

            let start = Instant::now();
            for _ in 0..generations {
                let _ = ca.step();
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

#[cfg(feature = "gpu")]
async fn gpu_rule_variants_demo() -> AutomataResult<()> {
    println!("🎯 GPU Cellular Automata Rule Variants");
    println!("--------------------------------------");

    let mut gpu_ops = match AutomataGpuOps::new().await {
        Ok(ops) => ops,
        Err(_) => {
            println!("⚠️  GPU not available");
            return Ok(());
        }
    };

    let grid_size = 32;
    let total_cells = grid_size * grid_size;
    let initial_cells = vec![random_cell(); total_cells];

    let rule_types = [
        ("Geometric", 0.0),
        ("Game of Life", 1.0),
        ("Conservative", 5.0),
        ("Rotor CA", 3.0),
    ];

    for (rule_name, rule_type) in &rule_types {
        println!("🔬 Testing {} rule:", rule_name);

        let rule_configs = vec![GpuRuleConfig {
            rule_type: *rule_type,
            threshold: 0.5,
            damping_factor: 0.7,
            energy_conservation: if *rule_type == 5.0 { 1.0 } else { 0.9 },
            ..Default::default()
        }];

        let evolution_params = GpuEvolutionParams {
            grid_width: grid_size as f32,
            grid_height: grid_size as f32,
            total_cells: total_cells as f32,
            steps_per_batch: 20.0,
            ..Default::default()
        };

        let start_time = Instant::now();
        let evolved_cells = gpu_ops
            .batch_evolve_ca(&initial_cells, &rule_configs, &evolution_params)
            .await?;
        let evolution_time = start_time.elapsed();

        let initial_energy = gpu_ops.calculate_total_energy(&initial_cells).await?;
        let final_energy = gpu_ops.calculate_total_energy(&evolved_cells).await?;
        let energy_conservation = (final_energy / initial_energy) * 100.0;

        println!("   ⏱️  Evolution time: {:?}", evolution_time);
        println!("   📊 Energy conservation: {:.1}%", energy_conservation);
        println!("   📊 Active cells: {}/{}",
            evolved_cells.iter().filter(|c| cell_magnitude(c) > 0.1).count(),
            total_cells);
        println!();
    }

    Ok(())
}

#[cfg(feature = "gpu")]
async fn gpu_large_scale_simulation() -> AutomataResult<()> {
    println!("🌐 Large-Scale GPU Cellular Automata Simulation");
    println!("-----------------------------------------------");

    let mut gpu_ops = match AutomataGpuOps::new().await {
        Ok(ops) => ops,
        Err(_) => {
            println!("⚠️  GPU not available for large-scale simulation");
            return Ok(());
        }
    };

    let grid_size = 256; // 256×256 = 65,536 cells
    let total_cells = grid_size * grid_size;
    let generations = 100;

    println!("📏 Grid size: {}×{} ({} cells)", grid_size, grid_size, total_cells);
    println!("🔄 Generations: {}", generations);

    // Create complex initial pattern
    let mut initial_cells = vec![GpuCellData::default(); total_cells];

    // Add multiple patterns
    add_glider_pattern(&mut initial_cells, grid_size, 64, 64);
    add_glider_pattern(&mut initial_cells, grid_size, 192, 64);
    add_glider_pattern(&mut initial_cells, grid_size, 64, 192);
    add_glider_pattern(&mut initial_cells, grid_size, 192, 192);

    // Add some random noise
    for i in 0..total_cells {
        if (i % 97) == 0 {  // Deterministic sparse pattern
            initial_cells[i] = random_cell();
        }
    }

    let rule_configs = vec![GpuRuleConfig {
        rule_type: 0.0, // Geometric
        threshold: 0.4,
        damping_factor: 0.75,
        energy_conservation: 0.98,
        stability_factor: 0.95,
        ..Default::default()
    }];

    let evolution_params = GpuEvolutionParams {
        grid_width: grid_size as f32,
        grid_height: grid_size as f32,
        total_cells: total_cells as f32,
        steps_per_batch: generations as f32,
        workgroup_size_x: 16.0,
        workgroup_size_y: 16.0,
        parallel_factor: 4.0,
        memory_optimization: 1.0,
        ..Default::default()
    };

    println!("🚀 Starting large-scale evolution...");
    let start_time = Instant::now();

    let evolved_cells = gpu_ops
        .batch_evolve_ca(&initial_cells, &rule_configs, &evolution_params)
        .await?;

    let total_time = start_time.elapsed();

    // Performance metrics
    let cells_per_second = (total_cells as f64 * generations as f64) / total_time.as_secs_f64();
    let initial_energy = gpu_ops.calculate_total_energy(&initial_cells).await?;
    let final_energy = gpu_ops.calculate_total_energy(&evolved_cells).await?;

    println!("✅ Large-scale simulation completed!");
    println!("   ⏱️  Total time: {:?}", total_time);
    println!("   🚀 Performance: {:.0} cells/second", cells_per_second);
    println!("   📊 Initial energy: {:.6}", initial_energy);
    println!("   📊 Final energy: {:.6}", final_energy);
    println!("   📊 Energy ratio: {:.3}", final_energy / initial_energy);

    let active_initial = initial_cells.iter().filter(|c| cell_magnitude(c) > 0.1).count();
    let active_final = evolved_cells.iter().filter(|c| cell_magnitude(c) > 0.1).count();
    println!("   📊 Active cells: {} → {} ({:+})",
        active_initial, active_final, active_final as i32 - active_initial as i32);

    Ok(())
}

fn cpu_fallback_demo() -> AutomataResult<()> {
    println!("🖥️  CPU Cellular Automata Demo");
    println!("-----------------------------");

    // Create a smaller CA for CPU demo
    let mut ca = GeometricCA::<3, 0, 0>::new_2d(32, 32);

    // Set up a simple pattern
    let _ = ca.set_cell_2d(16, 16, Multivector::scalar(1.0));
    let _ = ca.set_cell_2d(17, 16, Multivector::basis_vector(0));
    let _ = ca.set_cell_2d(16, 17, Multivector::basis_vector(1));

    println!("🔄 Evolving 32×32 CA for 50 generations...");
    let start_time = Instant::now();

    for _ in 0..50 {
        ca.step()?;
    }

    let cpu_time = start_time.elapsed();

    println!("✅ CPU evolution completed in {:?}", cpu_time);
    println!("   📊 Generation: {}", ca.generation());
    println!("   📊 Total energy: {:.6}", ca.total_energy());

    Ok(())
}

#[cfg(feature = "gpu")]
fn random_cell() -> GpuCellData {
    // Deterministic "random" cell generation
    GpuCellData {
        scalar: 0.7,
        e1: 0.2,
        e2: -0.1,
        e3: 0.3,
        e12: 0.15,
        e13: -0.1,
        e23: 0.05,
        e123: 0.0,
        generation: 0.0,
        neighborhood_size: 8.0,
        rule_type: 0.0,
        boundary_condition: 0.0,
        padding: [0.0; 4],
    }
}

#[cfg(feature = "gpu")]
fn cell_magnitude(cell: &GpuCellData) -> f32 {
    (cell.scalar * cell.scalar + cell.e1 * cell.e1 + cell.e2 * cell.e2 + cell.e3 * cell.e3 +
     cell.e12 * cell.e12 + cell.e13 * cell.e13 + cell.e23 * cell.e23 + cell.e123 * cell.e123).sqrt()
}

#[cfg(feature = "gpu")]
fn add_glider_pattern(cells: &mut [GpuCellData], grid_size: usize, start_x: usize, start_y: usize) {
    let pattern = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 2)];

    for &(dx, dy) in &pattern {
        let x = start_x + dx;
        let y = start_y + dy;
        if x < grid_size && y < grid_size {
            let index = y * grid_size + x;
            cells[index] = GpuCellData {
                scalar: 1.0,
                e1: 0.7,
                e2: 0.3,
                e12: 0.5,
                ..Default::default()
            };
        }
    }
}

#[cfg(feature = "gpu")]
fn analyze_pattern_evolution(
    initial: &[GpuCellData],
    evolved: &[GpuCellData],
    grid_size: usize,
) {
    println!("🔍 Pattern Evolution Analysis:");

    // Calculate pattern metrics
    let initial_active = initial.iter().filter(|c| cell_magnitude(c) > 0.1).count();
    let evolved_active = evolved.iter().filter(|c| cell_magnitude(c) > 0.1).count();

    println!("   📊 Active cells: {} → {}", initial_active, evolved_active);

    // Calculate center of mass
    let initial_com = calculate_center_of_mass(initial, grid_size);
    let evolved_com = calculate_center_of_mass(evolved, grid_size);

    println!("   📍 Center of mass: ({:.1}, {:.1}) → ({:.1}, {:.1})",
        initial_com.0, initial_com.1, evolved_com.0, evolved_com.1);

    let displacement = ((evolved_com.0 - initial_com.0).powi(2) +
                        (evolved_com.1 - initial_com.1).powi(2)).sqrt();
    println!("   📏 Pattern displacement: {:.2} cells", displacement);
}

#[cfg(feature = "gpu")]
fn calculate_center_of_mass(cells: &[GpuCellData], grid_size: usize) -> (f32, f32) {
    let mut total_mass = 0.0;
    let mut weighted_x = 0.0;
    let mut weighted_y = 0.0;

    for (i, cell) in cells.iter().enumerate() {
        let mass = cell_magnitude(cell);
        if mass > 0.1 {
            let x = (i % grid_size) as f32;
            let y = (i / grid_size) as f32;

            total_mass += mass;
            weighted_x += x * mass;
            weighted_y += y * mass;
        }
    }

    if total_mass > 0.0 {
        (weighted_x / total_mass, weighted_y / total_mass)
    } else {
        (0.0, 0.0)
    }
}