// Energy Calculation Compute Shader
// Calculates total energy of the cellular automata system

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

@group(0) @binding(0) var<storage, read> cells: array<GpuCellData>;
@group(0) @binding(1) var<storage, read_write> total_energy: array<f32>;

// Local workgroup memory for reduction
var<workgroup> local_energy: array<f32, 256>;

fn cell_energy(cell: GpuCellData) -> f32 {
    // Calculate energy as magnitude squared (L2 norm squared)
    return cell.scalar * cell.scalar + cell.e1 * cell.e1 + cell.e2 * cell.e2 + cell.e3 * cell.e3 +
           cell.e12 * cell.e12 + cell.e13 * cell.e13 + cell.e23 * cell.e23 + cell.e123 * cell.e123;
}

@compute @workgroup_size(256)
fn energy_calculation_main(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(local_invocation_id) local_id: vec3<u32>,
                          @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let thread_id = local_id.x;
    let global_thread_id = global_id.x;

    // Initialize local energy to 0
    local_energy[thread_id] = 0.0;

    // Each thread processes multiple cells if needed
    let cells_per_thread = (arrayLength(&cells) + 255u) / 256u;

    for (var i = 0u; i < cells_per_thread; i++) {
        let cell_index = global_thread_id * cells_per_thread + i;

        if (cell_index < arrayLength(&cells)) {
            local_energy[thread_id] += cell_energy(cells[cell_index]);
        }
    }

    // Synchronize threads in workgroup
    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_id < stride) {
            local_energy[thread_id] += local_energy[thread_id + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the result for this workgroup
    if (thread_id == 0u) {
        // For simplicity, we'll assume single workgroup
        // In a full implementation, would need another reduction pass
        total_energy[0] = local_energy[0];
    }
}