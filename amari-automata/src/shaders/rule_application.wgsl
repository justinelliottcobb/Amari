// Rule Application Compute Shader
// Applies CA rules to individual cells with their neighborhoods

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

// Cell addition operation
fn add_cells(a: GpuCellData, b: GpuCellData) -> GpuCellData {
    var result: GpuCellData;
    result.scalar = a.scalar + b.scalar;
    result.e1 = a.e1 + b.e1;
    result.e2 = a.e2 + b.e2;
    result.e3 = a.e3 + b.e3;
    result.e12 = a.e12 + b.e12;
    result.e13 = a.e13 + b.e13;
    result.e23 = a.e23 + b.e23;
    result.e123 = a.e123 + b.e123;
    result.generation = max(a.generation, b.generation);
    result.neighborhood_size = a.neighborhood_size;
    result.rule_type = a.rule_type;
    result.boundary_condition = a.boundary_condition;
    return result;
}

// Scalar multiplication
fn scale_cell(cell: GpuCellData, scale: f32) -> GpuCellData {
    var result = cell;
    result.scalar *= scale;
    result.e1 *= scale;
    result.e2 *= scale;
    result.e3 *= scale;
    result.e12 *= scale;
    result.e13 *= scale;
    result.e23 *= scale;
    result.e123 *= scale;
    return result;
}

// Inner product (scalar result)
fn inner_product(a: GpuCellData, b: GpuCellData) -> f32 {
    return a.scalar * b.scalar + a.e1 * b.e1 + a.e2 * b.e2 + a.e3 * b.e3 +
           a.e12 * b.e12 + a.e13 * b.e13 + a.e23 * b.e23 + a.e123 * b.e123;
}

// Outer product (for grade-2 elements)
fn outer_product(a: GpuCellData, b: GpuCellData) -> GpuCellData {
    var result: GpuCellData;

    // Simplified outer product focusing on grade preservation
    result.scalar = 0.0;
    result.e1 = a.scalar * b.e1 - a.e1 * b.scalar;
    result.e2 = a.scalar * b.e2 - a.e2 * b.scalar;
    result.e3 = a.scalar * b.e3 - a.e3 * b.scalar;
    result.e12 = a.scalar * b.e12 + a.e1 * b.e2 - a.e2 * b.e1 - a.e12 * b.scalar;
    result.e13 = a.scalar * b.e13 + a.e1 * b.e3 - a.e3 * b.e1 - a.e13 * b.scalar;
    result.e23 = a.scalar * b.e23 + a.e2 * b.e3 - a.e3 * b.e2 - a.e23 * b.scalar;
    result.e123 = a.scalar * b.e123 + a.e1 * b.e23 + a.e2 * b.e13 + a.e3 * b.e12
                 + a.e12 * b.e3 + a.e13 * b.e2 + a.e23 * b.e1 + a.e123 * b.scalar;

    return result;
}

// Grade projection
fn grade_projection(cell: GpuCellData, grade: u32) -> GpuCellData {
    var result: GpuCellData;

    if (grade == 0u) {
        result.scalar = cell.scalar;
        result.e1 = 0.0;
        result.e2 = 0.0;
        result.e3 = 0.0;
        result.e12 = 0.0;
        result.e13 = 0.0;
        result.e23 = 0.0;
        result.e123 = 0.0;
    } else if (grade == 1u) {
        result.scalar = 0.0;
        result.e1 = cell.e1;
        result.e2 = cell.e2;
        result.e3 = cell.e3;
        result.e12 = 0.0;
        result.e13 = 0.0;
        result.e23 = 0.0;
        result.e123 = 0.0;
    } else if (grade == 2u) {
        result.scalar = 0.0;
        result.e1 = 0.0;
        result.e2 = 0.0;
        result.e3 = 0.0;
        result.e12 = cell.e12;
        result.e13 = cell.e13;
        result.e23 = cell.e23;
        result.e123 = 0.0;
    } else if (grade == 3u) {
        result.scalar = 0.0;
        result.e1 = 0.0;
        result.e2 = 0.0;
        result.e3 = 0.0;
        result.e12 = 0.0;
        result.e13 = 0.0;
        result.e23 = 0.0;
        result.e123 = cell.e123;
    } else {
        result = cell; // Return original if invalid grade
    }

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

// Apply rule based on configuration
fn apply_rule(center: GpuCellData, rule: GpuRuleConfig) -> GpuCellData {
    var result = center;

    if (rule.rule_type == 0.0) {
        // Geometric rule - normalize if above threshold
        let mag = cell_magnitude(center);
        if (mag > rule.threshold) {
            result = normalize_cell(center);
            result = scale_cell(result, mag * rule.damping_factor);
        } else {
            result.scalar = 0.0;
            result.e1 = 0.0;
            result.e2 = 0.0;
            result.e3 = 0.0;
            result.e12 = 0.0;
            result.e13 = 0.0;
            result.e23 = 0.0;
            result.e123 = 0.0;
        }
    } else if (rule.rule_type == 2.0) {
        // Reversible rule - apply time-reversed operation
        result = scale_cell(center, -rule.time_step);
    } else if (rule.rule_type == 4.0) {
        // Grade-preserving rule - project to original grade
        let original_grade = 0u; // Simplified - would need to determine actual grade
        if (cell_magnitude(grade_projection(center, 0u)) > 0.1) {
            result = grade_projection(result, 0u);
        } else if (cell_magnitude(grade_projection(center, 1u)) > 0.1) {
            result = grade_projection(result, 1u);
        } else if (cell_magnitude(grade_projection(center, 2u)) > 0.1) {
            result = grade_projection(result, 2u);
        } else {
            result = grade_projection(result, 3u);
        }
    } else {
        // Default geometric transformation
        result = normalize_cell(center);
    }

    // Apply stability factor
    result = scale_cell(result, rule.stability_factor);

    return result;
}

@compute @workgroup_size(256)
fn rule_application_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;

    if (cell_index >= arrayLength(&input_cells)) {
        return;
    }

    let input_cell = input_cells[cell_index];
    let rule_index = min(cell_index, arrayLength(&rule_configs) - 1u);
    let rule_config = rule_configs[rule_index];

    var output_cell = apply_rule(input_cell, rule_config);

    // Update generation and metadata
    output_cell.generation = input_cell.generation + rule_config.time_step;
    output_cell.neighborhood_size = input_cell.neighborhood_size;
    output_cell.rule_type = rule_config.rule_type;
    output_cell.boundary_condition = rule_config.boundary_type;

    output_cells[cell_index] = output_cell;
}