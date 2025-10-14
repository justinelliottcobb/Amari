// Neighbor Extraction Compute Shader
// Extracts neighborhoods for all cells in parallel

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

struct GridParams {
    width: f32,
    height: f32,
    total_cells: f32,
    boundary_type: f32,
}

@group(0) @binding(0) var<storage, read> input_cells: array<GpuCellData>;
@group(0) @binding(1) var<uniform> grid_params: GridParams;
@group(0) @binding(2) var<storage, read_write> neighborhood_output: array<GpuCellData>;

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
    result.generation = 0.0;
    result.neighborhood_size = 0.0;
    result.rule_type = 0.0;
    result.boundary_condition = 0.0;
    return result;
}

// Get neighbor at relative position (dx, dy) from cell (x, y)
fn get_neighbor_at(x: u32, y: u32, dx: i32, dy: i32, grid_width: u32, grid_height: u32, boundary_type: f32) -> GpuCellData {
    let nx = i32(x) + dx;
    let ny = i32(y) + dy;

    var final_x: i32;
    var final_y: i32;
    var is_valid = true;

    if (boundary_type == 0.0) {
        // Periodic boundary conditions
        if (nx < 0) {
            final_x = i32(grid_width) - 1;
        } else if (nx >= i32(grid_width)) {
            final_x = 0;
        } else {
            final_x = nx;
        }

        if (ny < 0) {
            final_y = i32(grid_height) - 1;
        } else if (ny >= i32(grid_height)) {
            final_y = 0;
        } else {
            final_y = ny;
        }
    } else if (boundary_type == 1.0) {
        // Fixed boundary conditions - out of bounds returns zero
        if (nx < 0 || nx >= i32(grid_width) || ny < 0 || ny >= i32(grid_height)) {
            is_valid = false;
            final_x = 0;
            final_y = 0;
        } else {
            final_x = nx;
            final_y = ny;
        }
    } else {
        // Reflecting boundary conditions
        if (nx < 0) {
            final_x = -nx;
        } else if (nx >= i32(grid_width)) {
            final_x = i32(grid_width) - 1 - (nx - i32(grid_width));
        } else {
            final_x = nx;
        }

        if (ny < 0) {
            final_y = -ny;
        } else if (ny >= i32(grid_height)) {
            final_y = i32(grid_height) - 1 - (ny - i32(grid_height));
        } else {
            final_y = ny;
        }

        // Clamp to valid range
        final_x = max(0, min(final_x, i32(grid_width) - 1));
        final_y = max(0, min(final_y, i32(grid_height) - 1));
    }

    if (!is_valid) {
        return zero_cell();
    }

    let neighbor_index = u32(final_y) * grid_width + u32(final_x);

    if (neighbor_index < arrayLength(&input_cells)) {
        return input_cells[neighbor_index];
    } else {
        return zero_cell();
    }
}

// Extract Moore neighborhood (8 neighbors) for a cell
fn extract_moore_neighborhood(cell_index: u32, grid_width: u32, grid_height: u32, boundary_type: f32) -> array<GpuCellData, 8> {
    var neighbors: array<GpuCellData, 8>;

    let x = cell_index % grid_width;
    let y = cell_index / grid_width;

    var neighbor_idx = 0u;

    // Moore neighborhood: 8 surrounding cells
    let offsets: array<vec2<i32>, 8> = array<vec2<i32>, 8>(
        vec2<i32>(-1, -1), // Top-left
        vec2<i32>( 0, -1), // Top
        vec2<i32>( 1, -1), // Top-right
        vec2<i32>(-1,  0), // Left
        vec2<i32>( 1,  0), // Right
        vec2<i32>(-1,  1), // Bottom-left
        vec2<i32>( 0,  1), // Bottom
        vec2<i32>( 1,  1)  // Bottom-right
    );

    // Unrolled for WGSL compatibility (variable array indexing not allowed)
    neighbors[0] = get_neighbor_at(x, y, offsets[0].x, offsets[0].y, grid_width, grid_height, boundary_type);
    neighbors[1] = get_neighbor_at(x, y, offsets[1].x, offsets[1].y, grid_width, grid_height, boundary_type);
    neighbors[2] = get_neighbor_at(x, y, offsets[2].x, offsets[2].y, grid_width, grid_height, boundary_type);
    neighbors[3] = get_neighbor_at(x, y, offsets[3].x, offsets[3].y, grid_width, grid_height, boundary_type);
    neighbors[4] = get_neighbor_at(x, y, offsets[4].x, offsets[4].y, grid_width, grid_height, boundary_type);
    neighbors[5] = get_neighbor_at(x, y, offsets[5].x, offsets[5].y, grid_width, grid_height, boundary_type);
    neighbors[6] = get_neighbor_at(x, y, offsets[6].x, offsets[6].y, grid_width, grid_height, boundary_type);
    neighbors[7] = get_neighbor_at(x, y, offsets[7].x, offsets[7].y, grid_width, grid_height, boundary_type);

    return neighbors;
}

// Extract Von Neumann neighborhood (4 neighbors) for a cell
fn extract_von_neumann_neighborhood(cell_index: u32, grid_width: u32, grid_height: u32, boundary_type: f32) -> array<GpuCellData, 4> {
    var neighbors: array<GpuCellData, 4>;

    let x = cell_index % grid_width;
    let y = cell_index / grid_width;

    // Von Neumann neighborhood: 4 orthogonal neighbors
    let offsets: array<vec2<i32>, 4> = array<vec2<i32>, 4>(
        vec2<i32>( 0, -1), // Top
        vec2<i32>(-1,  0), // Left
        vec2<i32>( 1,  0), // Right
        vec2<i32>( 0,  1)  // Bottom
    );

    // Unrolled for WGSL compatibility (variable array indexing not allowed)
    neighbors[0] = get_neighbor_at(x, y, offsets[0].x, offsets[0].y, grid_width, grid_height, boundary_type);
    neighbors[1] = get_neighbor_at(x, y, offsets[1].x, offsets[1].y, grid_width, grid_height, boundary_type);
    neighbors[2] = get_neighbor_at(x, y, offsets[2].x, offsets[2].y, grid_width, grid_height, boundary_type);
    neighbors[3] = get_neighbor_at(x, y, offsets[3].x, offsets[3].y, grid_width, grid_height, boundary_type);

    return neighbors;
}

@compute @workgroup_size(256)
fn neighbor_extraction_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_index = global_id.x;

    if (cell_index >= arrayLength(&input_cells)) {
        return;
    }

    let grid_width = u32(grid_params.width);
    let grid_height = u32(grid_params.height);
    let boundary_type = grid_params.boundary_type;

    // Extract Moore neighborhood (8 neighbors)
    let neighbors = extract_moore_neighborhood(cell_index, grid_width, grid_height, boundary_type);

    // Store neighbors in output buffer
    // Each cell gets 8 neighbors stored sequentially
    let base_output_index = cell_index * 8u;

    // Store neighbors in output buffer (unrolled for WGSL compatibility)
    if (base_output_index + 0u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 0u] = neighbors[0];
    }
    if (base_output_index + 1u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 1u] = neighbors[1];
    }
    if (base_output_index + 2u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 2u] = neighbors[2];
    }
    if (base_output_index + 3u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 3u] = neighbors[3];
    }
    if (base_output_index + 4u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 4u] = neighbors[4];
    }
    if (base_output_index + 5u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 5u] = neighbors[5];
    }
    if (base_output_index + 6u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 6u] = neighbors[6];
    }
    if (base_output_index + 7u < arrayLength(&neighborhood_output)) {
        neighborhood_output[base_output_index + 7u] = neighbors[7];
    }
}