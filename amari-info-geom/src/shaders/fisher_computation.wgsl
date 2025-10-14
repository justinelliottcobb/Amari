// Fisher Information Matrix Compute Shader
// Computes Fisher information matrices for statistical manifolds

struct GpuFisherData {
    // Statistical parameters (up to 8 components)
    param0: f32,
    param1: f32,
    param2: f32,
    param3: f32,
    param4: f32,
    param5: f32,
    param6: f32,
    param7: f32,
    // Fisher metric properties
    dimension: f32,
    log_partition_value: f32,
    regularization: f32,
    numerical_stability: f32,
    // Manifold geometry
    curvature_scalar: f32,
    connection_alpha: f32,
    manifold_type: f32,
    entropy_value: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<GpuFisherData>;
@group(0) @binding(1) var<storage, read_write> output_matrices: array<f32>;

// Compute log partition function for exponential families
fn log_partition_function(params: array<f32, 8>, dim: u32) -> f32 {
    var result = 0.0;

    // For exponential families: ψ(η) = log ∫ exp(η·x) μ(dx)
    // Simplified computation for common distributions

    if (dim >= 2u) {
        // Gaussian case: ψ(η₁,η₂) = -η₁²/(4η₂) - 0.5*log(-2η₂)
        let eta1 = params[0];
        let eta2 = params[1];

        if (eta2 < -1e-8) {
            result = -(eta1 * eta1) / (4.0 * eta2) - 0.5 * log(-2.0 * eta2);
        } else {
            // Regularize for numerical stability
            result = eta1 * eta1 * 0.1 + abs(eta2) * 0.5;
        }
    } else {
        // Exponential case: ψ(η) = -log(-η) for η < 0
        let eta = params[0];
        if (eta < -1e-8) {
            result = -log(-eta);
        } else {
            result = abs(eta) * 0.5;
        }
    }

    return result;
}

// Compute second derivatives (Hessian) for Fisher metric
fn compute_hessian_element(params: array<f32, 8>, dim: u32, i: u32, j: u32, eps: f32) -> f32 {
    // Finite difference approximation for ∂²ψ/∂ηᵢ∂ηⱼ

    var params_pp = params;
    var params_pm = params;
    var params_mp = params;
    var params_mm = params;

    // Adjust parameters for finite differences
    if (i < 8u && j < 8u) {
        params_pp[i] += eps;
        params_pp[j] += eps;

        params_pm[i] += eps;
        params_pm[j] -= eps;

        params_mp[i] -= eps;
        params_mp[j] += eps;

        params_mm[i] -= eps;
        params_mm[j] -= eps;
    }

    // Second-order finite difference: (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h))/(4h²)
    let psi_pp = log_partition_function(params_pp, dim);
    let psi_pm = log_partition_function(params_pm, dim);
    let psi_mp = log_partition_function(params_mp, dim);
    let psi_mm = log_partition_function(params_mm, dim);

    let hessian_element = (psi_pp - psi_pm - psi_mp + psi_mm) / (4.0 * eps * eps);

    return hessian_element;
}

// Compute Fisher information matrix for a single data point
fn compute_fisher_matrix(data: GpuFisherData) -> array<f32, 16> {
    var fisher_matrix: array<f32, 16>;

    // Extract parameters
    let params = array<f32, 8>(
        data.param0, data.param1, data.param2, data.param3,
        data.param4, data.param5, data.param6, data.param7
    );

    let dim = u32(data.dimension);
    let eps = 1e-6; // Finite difference step size

    // Fisher information matrix: gᵢⱼ = ∂²ψ/∂ηᵢ∂ηⱼ
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let matrix_idx = i * 4u + j;

            if (i < dim && j < dim) {
                var element = compute_hessian_element(params, dim, i, j, eps);

                // Add regularization for numerical stability
                if (i == j) {
                    element += data.regularization;
                }

                // Ensure positive definiteness
                if (i == j && element < data.numerical_stability) {
                    element = data.numerical_stability;
                }

                fisher_matrix[matrix_idx] = element;
            } else {
                fisher_matrix[matrix_idx] = 0.0;
            }
        }
    }

    return fisher_matrix;
}

// Compute Fisher information for probability distributions
fn compute_distribution_fisher(data: GpuFisherData) -> array<f32, 16> {
    var fisher_matrix: array<f32, 16>;

    let params = array<f32, 8>(
        data.param0, data.param1, data.param2, data.param3,
        data.param4, data.param5, data.param6, data.param7
    );

    let dim = u32(data.dimension);

    // For discrete probability distributions: gᵢⱼ = δᵢⱼ/pᵢ (diagonal Fisher metric)
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let matrix_idx = i * 4u + j;

            if (i < dim && j < dim) {
                if (i == j && i < 8u) {
                    let prob = params[i];
                    if (prob > data.numerical_stability) {
                        fisher_matrix[matrix_idx] = 1.0 / prob;
                    } else {
                        fisher_matrix[matrix_idx] = 1.0 / data.numerical_stability;
                    }
                } else {
                    fisher_matrix[matrix_idx] = 0.0;
                }
            } else {
                fisher_matrix[matrix_idx] = 0.0;
            }
        }
    }

    return fisher_matrix;
}

// Compute Riemannian curvature scalar
fn compute_curvature_scalar(fisher_matrix: array<f32, 16>, dim: u32) -> f32 {
    // Simplified curvature computation
    // For statistical manifolds, the curvature is related to the Fisher metric

    var trace = 0.0;
    var determinant = 1.0;

    // Compute trace and determinant
    for (var i = 0u; i < dim; i++) {
        let diagonal_idx = i * 4u + i;
        if (diagonal_idx < 16u) {
            trace += fisher_matrix[diagonal_idx];
            determinant *= fisher_matrix[diagonal_idx];
        }
    }

    // Scalar curvature approximation: R ∼ trace/det for diagonal metrics
    if (determinant > 1e-12) {
        return trace / sqrt(determinant);
    } else {
        return 0.0;
    }
}

@compute @workgroup_size(256)
fn fisher_computation_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let data_index = global_id.x;

    if (data_index >= arrayLength(&input_data)) {
        return;
    }

    let data = input_data[data_index];

    // Choose computation method based on manifold type
    var fisher_matrix: array<f32, 16>;

    if (data.manifold_type == 0.0) {
        // Exponential family manifold
        fisher_matrix = compute_fisher_matrix(data);
    } else {
        // Probability simplex manifold
        fisher_matrix = compute_distribution_fisher(data);
    }

    // Compute additional geometric properties
    let curvature = compute_curvature_scalar(fisher_matrix, u32(data.dimension));

    // Store results in output buffer
    let output_offset = data_index * 16u;

    for (var i = 0u; i < 16u; i++) {
        let output_idx = output_offset + i;
        if (output_idx < arrayLength(&output_matrices)) {
            output_matrices[output_idx] = fisher_matrix[i];
        }
    }
}