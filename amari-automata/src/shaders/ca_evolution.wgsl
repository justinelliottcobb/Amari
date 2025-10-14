// CA Evolution Compute Shader
// Performs one evolutionary step for geometric cellular automata

struct GpuCellData {
    scalar: f32,
    e1: f32,
    e2: f32,
    e3: f32,
    e12: f32,
    e13: f32,
    e23: f32,
    e123: f32,
    generation: f32,
    neighborhood_size: f32,
    rule_type: f32,
    boundary_condition: f32,
    padding: array<f32, 4>,
}

struct GpuRuleConfig {
    rule_type: f32,
    threshold: f32,
    damping_factor: f32,
    energy_conservation: f32,
    time_step: f32,
    spatial_scale: f32,
    geometric_weight: f32,
    nonlinear_factor: f32,
    boundary_type: f32,
    neighborhood_radius: f32,
    evolution_speed: f32,
    stability_factor: f32,
    padding: array<f32, 4>,
}

@group(0) @binding(0) var<storage, read> input_cells: array<GpuCellData>;
@group(0) @binding(1) var<storage, read> rule_configs: array<GpuRuleConfig>;
@group(0) @binding(2) var<storage, read_write> output_cells: array<GpuCellData>;

// Geometric algebra operations
fn geometric_product(a: GpuCellData, b: GpuCellData) -> GpuCellData {
    var result: GpuCellData;

    // Simplified geometric product for 3D space (3,0,0)
    result.scalar = a.scalar * b.scalar + a.e1 * b.e1 + a.e2 * b.e2 + a.e3 * b.e3
                   - a.e12 * b.e12 - a.e13 * b.e13 - a.e23 * b.e23 - a.e123 * b.e123;

    result.e1 = a.scalar * b.e1 + a.e1 * b.scalar - a.e2 * b.e12 - a.e3 * b.e13
               + a.e12 * b.e2 + a.e13 * b.e3 - a.e23 * b.e123 + a.e123 * b.e23;

    result.e2 = a.scalar * b.e2 + a.e1 * b.e12 + a.e2 * b.scalar - a.e3 * b.e23
               - a.e12 * b.e1 + a.e13 * b.e123 + a.e23 * b.e3 + a.e123 * b.e13;

    result.e3 = a.scalar * b.e3 + a.e1 * b.e13 + a.e2 * b.e23 + a.e3 * b.scalar
               - a.e12 * b.e123 - a.e13 * b.e1 - a.e23 * b.e2 + a.e123 * b.e12;

    result.e12 = a.scalar * b.e12 + a.e1 * b.e2 - a.e2 * b.e1 + a.e3 * b.e123
                + a.e12 * b.scalar + a.e13 * b.e23 - a.e23 * b.e13 + a.e123 * b.e3;

    result.e13 = a.scalar * b.e13 + a.e1 * b.e3 - a.e2 * b.e123 - a.e3 * b.e1
                - a.e12 * b.e23 + a.e13 * b.scalar + a.e23 * b.e12 + a.e123 * b.e2;

    result.e23 = a.scalar * b.e23 + a.e1 * b.e123 + a.e2 * b.e3 - a.e3 * b.e2
                + a.e12 * b.e13 - a.e13 * b.e12 + a.e23 * b.scalar + a.e123 * b.e1;

    result.e123 = a.scalar * b.e123 + a.e1 * b.e23 + a.e2 * b.e13 + a.e3 * b.e12
                 + a.e12 * b.e3 + a.e13 * b.e2 + a.e23 * b.e1 + a.e123 * b.scalar;

    return result;
}

fn cell_magnitude(cell: GpuCellData) -> f32 {
    return sqrt(cell.scalar * cell.scalar + cell.e1 * cell.e1 + cell.e2 * cell.e2 + cell.e3 * cell.e3 +
                cell.e12 * cell.e12 + cell.e13 * cell.e13 + cell.e23 * cell.e23 + cell.e123 * cell.e123);
}

fn normalize_cell(cell: GpuCellData) -> GpuCellData {
    let mag = cell_magnitude(cell);
    var result = cell;
    if (mag > 0.0) {
        let inv_mag = 1.0 / mag;
        result.scalar *= inv_mag;
        result.e1 *= inv_mag;
        result.e2 *= inv_mag;
        result.e3 *= inv_mag;
        result.e12 *= inv_mag;
        result.e13 *= inv_mag;
        result.e23 *= inv_mag;
        result.e123 *= inv_mag;
    }
    return result;
}

fn zero_cell() -> GpuCellData {
    var result: GpuCellData;
    result.scalar = 0.0;
    result.e1 = 0.0;
    result.e2 = 0.0;
    result.e3 = 0.0;
    result.e12 = 0.0;
    result.e13 = 0.0;
    result.e23 = 0.0;
    result.e123 = 0.0;
    return result;
}

fn scalar_cell(value: f32) -> GpuCellData {
    var result: GpuCellData;
    result.scalar = value;
    result.e1 = 0.0;
    result.e2 = 0.0;
    result.e3 = 0.0;
    result.e12 = 0.0;
    result.e13 = 0.0;
    result.e23 = 0.0;
    result.e123 = 0.0;
    return result;
}

// CA rules implementation
fn apply_geometric_rule(center: GpuCellData, neighbors: array<GpuCellData, 8>, rule: GpuRuleConfig) -> GpuCellData {
    var result = center;

    // Apply geometric product with neighbors (unrolled for WGSL compatibility)
    var neighbor_magnitude: f32;

    neighbor_magnitude = cell_magnitude(neighbors[0]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[0]);
    }

    neighbor_magnitude = cell_magnitude(neighbors[1]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[1]);
    }

    neighbor_magnitude = cell_magnitude(neighbors[2]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[2]);
    }

    neighbor_magnitude = cell_magnitude(neighbors[3]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[3]);
    }

    neighbor_magnitude = cell_magnitude(neighbors[4]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[4]);
    }

    neighbor_magnitude = cell_magnitude(neighbors[5]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[5]);
    }

    neighbor_magnitude = cell_magnitude(neighbors[6]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[6]);
    }

    neighbor_magnitude = cell_magnitude(neighbors[7]);
    if (neighbor_magnitude > rule.threshold) {
        result = geometric_product(result, neighbors[7]);
    }

    // Apply damping and normalization
    let magnitude = cell_magnitude(result);
    if (magnitude > rule.threshold) {
        result = normalize_cell(result);
        let damped_magnitude = magnitude * rule.damping_factor;
        result.scalar *= damped_magnitude;
        result.e1 *= damped_magnitude;
        result.e2 *= damped_magnitude;
        result.e3 *= damped_magnitude;
        result.e12 *= damped_magnitude;
        result.e13 *= damped_magnitude;
        result.e23 *= damped_magnitude;
        result.e123 *= damped_magnitude;
    } else {
        result = zero_cell();
    }

    return result;
}

fn apply_game_of_life_rule(center: GpuCellData, neighbors: array<GpuCellData, 8>, rule: GpuRuleConfig) -> GpuCellData {
    var alive_neighbors = 0u;

    // Count alive neighbors (unrolled for WGSL compatibility)
    if (cell_magnitude(neighbors[0]) > rule.threshold) {
        alive_neighbors++;
    }
    if (cell_magnitude(neighbors[1]) > rule.threshold) {
        alive_neighbors++;
    }
    if (cell_magnitude(neighbors[2]) > rule.threshold) {
        alive_neighbors++;
    }
    if (cell_magnitude(neighbors[3]) > rule.threshold) {
        alive_neighbors++;
    }
    if (cell_magnitude(neighbors[4]) > rule.threshold) {
        alive_neighbors++;
    }
    if (cell_magnitude(neighbors[5]) > rule.threshold) {
        alive_neighbors++;
    }
    if (cell_magnitude(neighbors[6]) > rule.threshold) {
        alive_neighbors++;
    }
    if (cell_magnitude(neighbors[7]) > rule.threshold) {
        alive_neighbors++;
    }

    let center_alive = cell_magnitude(center) > rule.threshold;

    if (center_alive) {
        // Alive cell survives with 2 or 3 neighbors
        if (alive_neighbors == 2u || alive_neighbors == 3u) {
            return center;
        } else {
            return zero_cell();
        }
    } else {
        // Dead cell becomes alive with exactly 3 neighbors
        if (alive_neighbors == 3u) {
            return scalar_cell(1.0);
        } else {
            return zero_cell();
        }
    }
}

fn apply_conservative_rule(center: GpuCellData, neighbors: array<GpuCellData, 8>, rule: GpuRuleConfig) -> GpuCellData {
    var total_energy = cell_magnitude(center) * cell_magnitude(center);

    // Calculate total energy in neighborhood (unrolled for WGSL compatibility)
    let neighbor_mag_0 = cell_magnitude(neighbors[0]);
    total_energy += neighbor_mag_0 * neighbor_mag_0;

    let neighbor_mag_1 = cell_magnitude(neighbors[1]);
    total_energy += neighbor_mag_1 * neighbor_mag_1;

    let neighbor_mag_2 = cell_magnitude(neighbors[2]);
    total_energy += neighbor_mag_2 * neighbor_mag_2;

    let neighbor_mag_3 = cell_magnitude(neighbors[3]);
    total_energy += neighbor_mag_3 * neighbor_mag_3;

    let neighbor_mag_4 = cell_magnitude(neighbors[4]);
    total_energy += neighbor_mag_4 * neighbor_mag_4;

    let neighbor_mag_5 = cell_magnitude(neighbors[5]);
    total_energy += neighbor_mag_5 * neighbor_mag_5;

    let neighbor_mag_6 = cell_magnitude(neighbors[6]);
    total_energy += neighbor_mag_6 * neighbor_mag_6;

    let neighbor_mag_7 = cell_magnitude(neighbors[7]);
    total_energy += neighbor_mag_7 * neighbor_mag_7;

    let avg_energy = total_energy / 9.0; // Center + 8 neighbors
    let target_magnitude = sqrt(avg_energy);

    var result = normalize_cell(center);
    let current_magnitude = cell_magnitude(center);

    if (current_magnitude > 0.0) {
        let scale_factor = target_magnitude * rule.energy_conservation;
        result.scalar *= scale_factor;
        result.e1 *= scale_factor;
        result.e2 *= scale_factor;
        result.e3 *= scale_factor;
        result.e12 *= scale_factor;
        result.e13 *= scale_factor;
        result.e23 *= scale_factor;
        result.e123 *= scale_factor;
    }

    return result;
}

fn apply_rotor_rule(center: GpuCellData, neighbors: array<GpuCellData, 8>, rule: GpuRuleConfig) -> GpuCellData {
    var result = center;

    // Accumulate bivector parts from neighbors (rotors) - unrolled for WGSL compatibility
    let bivector_magnitude_0 = sqrt(neighbors[0].e12 * neighbors[0].e12 +
                                   neighbors[0].e13 * neighbors[0].e13 +
                                   neighbors[0].e23 * neighbors[0].e23);
    if (bivector_magnitude_0 > rule.threshold) {
        result.e12 += neighbors[0].e12 * rule.geometric_weight;
        result.e13 += neighbors[0].e13 * rule.geometric_weight;
        result.e23 += neighbors[0].e23 * rule.geometric_weight;
    }

    let bivector_magnitude_1 = sqrt(neighbors[1].e12 * neighbors[1].e12 +
                                   neighbors[1].e13 * neighbors[1].e13 +
                                   neighbors[1].e23 * neighbors[1].e23);
    if (bivector_magnitude_1 > rule.threshold) {
        result.e12 += neighbors[1].e12 * rule.geometric_weight;
        result.e13 += neighbors[1].e13 * rule.geometric_weight;
        result.e23 += neighbors[1].e23 * rule.geometric_weight;
    }

    let bivector_magnitude_2 = sqrt(neighbors[2].e12 * neighbors[2].e12 +
                                   neighbors[2].e13 * neighbors[2].e13 +
                                   neighbors[2].e23 * neighbors[2].e23);
    if (bivector_magnitude_2 > rule.threshold) {
        result.e12 += neighbors[2].e12 * rule.geometric_weight;
        result.e13 += neighbors[2].e13 * rule.geometric_weight;
        result.e23 += neighbors[2].e23 * rule.geometric_weight;
    }

    let bivector_magnitude_3 = sqrt(neighbors[3].e12 * neighbors[3].e12 +
                                   neighbors[3].e13 * neighbors[3].e13 +
                                   neighbors[3].e23 * neighbors[3].e23);
    if (bivector_magnitude_3 > rule.threshold) {
        result.e12 += neighbors[3].e12 * rule.geometric_weight;
        result.e13 += neighbors[3].e13 * rule.geometric_weight;
        result.e23 += neighbors[3].e23 * rule.geometric_weight;
    }

    let bivector_magnitude_4 = sqrt(neighbors[4].e12 * neighbors[4].e12 +
                                   neighbors[4].e13 * neighbors[4].e13 +
                                   neighbors[4].e23 * neighbors[4].e23);
    if (bivector_magnitude_4 > rule.threshold) {
        result.e12 += neighbors[4].e12 * rule.geometric_weight;
        result.e13 += neighbors[4].e13 * rule.geometric_weight;
        result.e23 += neighbors[4].e23 * rule.geometric_weight;
    }

    let bivector_magnitude_5 = sqrt(neighbors[5].e12 * neighbors[5].e12 +
                                   neighbors[5].e13 * neighbors[5].e13 +
                                   neighbors[5].e23 * neighbors[5].e23);
    if (bivector_magnitude_5 > rule.threshold) {
        result.e12 += neighbors[5].e12 * rule.geometric_weight;
        result.e13 += neighbors[5].e13 * rule.geometric_weight;
        result.e23 += neighbors[5].e23 * rule.geometric_weight;
    }

    let bivector_magnitude_6 = sqrt(neighbors[6].e12 * neighbors[6].e12 +
                                   neighbors[6].e13 * neighbors[6].e13 +
                                   neighbors[6].e23 * neighbors[6].e23);
    if (bivector_magnitude_6 > rule.threshold) {
        result.e12 += neighbors[6].e12 * rule.geometric_weight;
        result.e13 += neighbors[6].e13 * rule.geometric_weight;
        result.e23 += neighbors[6].e23 * rule.geometric_weight;
    }

    let bivector_magnitude_7 = sqrt(neighbors[7].e12 * neighbors[7].e12 +
                                   neighbors[7].e13 * neighbors[7].e13 +
                                   neighbors[7].e23 * neighbors[7].e23);
    if (bivector_magnitude_7 > rule.threshold) {
        result.e12 += neighbors[7].e12 * rule.geometric_weight;
        result.e13 += neighbors[7].e13 * rule.geometric_weight;
        result.e23 += neighbors[7].e23 * rule.geometric_weight;
    }

    return result;
}

// Get Moore neighborhood (8 neighbors) for a cell
fn get_neighbors(cell_index: u32, grid_width: u32, grid_height: u32) -> array<GpuCellData, 8> {
    var neighbors: array<GpuCellData, 8>;
    let x = cell_index % grid_width;
    let y = cell_index / grid_width;

    var neighbor_idx = 0u;

    // Check all 8 directions around the cell
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) {
                continue; // Skip center cell
            }

            let nx = i32(x) + dx;
            let ny = i32(y) + dy;

            // Handle periodic boundary conditions
            var wrapped_x = nx;
            var wrapped_y = ny;

            if (nx < 0) {
                wrapped_x = i32(grid_width) - 1;
            } else if (nx >= i32(grid_width)) {
                wrapped_x = 0;
            }

            if (ny < 0) {
                wrapped_y = i32(grid_height) - 1;
            } else if (ny >= i32(grid_height)) {
                wrapped_y = 0;
            }

            let neighbor_cell_index = u32(wrapped_y) * grid_width + u32(wrapped_x);

            if (neighbor_cell_index < arrayLength(&input_cells)) {
                neighbors[neighbor_idx] = input_cells[neighbor_cell_index];
            } else {
                neighbors[neighbor_idx] = zero_cell();
            }

            neighbor_idx++;
        }
    }

    return neighbors;
}

@compute @workgroup_size(256)
fn ca_evolution_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;

    if (cell_index >= arrayLength(&input_cells)) {
        return;
    }

    let center_cell = input_cells[cell_index];
    let rule_config = rule_configs[0]; // Use first rule for now

    // Estimate grid dimensions (assuming square grid)
    let total_cells = arrayLength(&input_cells);
    let grid_width = u32(sqrt(f32(total_cells)));
    let grid_height = grid_width;

    let neighbors = get_neighbors(cell_index, grid_width, grid_height);

    var new_cell: GpuCellData;

    // Apply appropriate rule based on rule type
    if (rule_config.rule_type == 0.0) {
        // Geometric rule
        new_cell = apply_geometric_rule(center_cell, neighbors, rule_config);
    } else if (rule_config.rule_type == 1.0) {
        // Game of Life rule
        new_cell = apply_game_of_life_rule(center_cell, neighbors, rule_config);
    } else if (rule_config.rule_type == 5.0) {
        // Conservative rule
        new_cell = apply_conservative_rule(center_cell, neighbors, rule_config);
    } else if (rule_config.rule_type == 3.0) {
        // Rotor rule
        new_cell = apply_rotor_rule(center_cell, neighbors, rule_config);
    } else {
        // Default to geometric rule
        new_cell = apply_geometric_rule(center_cell, neighbors, rule_config);
    }

    // Update generation and metadata
    new_cell.generation = center_cell.generation + 1.0;
    new_cell.neighborhood_size = 8.0;
    new_cell.rule_type = rule_config.rule_type;
    new_cell.boundary_condition = rule_config.boundary_type;

    output_cells[cell_index] = new_cell;
}