// Statistical Manifold Operations Compute Shader
// Performs various operations on statistical manifolds including geodesics, connections, and curvature

struct GpuStatisticalManifold {
    // Natural parameters (η)
    eta0: f32,
    eta1: f32,
    eta2: f32,
    eta3: f32,
    // Expectation parameters (μ)
    mu0: f32,
    mu1: f32,
    mu2: f32,
    mu3: f32,
    // Manifold structure
    alpha_connection: f32,
    fisher_metric_det: f32,
    entropy: f32,
    temperature: f32,
    // Computational parameters
    batch_id: f32,
    convergence_threshold: f32,
    max_iterations: f32,
    stability_factor: f32,
}

@group(0) @binding(0) var<storage, read> input_manifolds: array<GpuStatisticalManifold>;
@group(0) @binding(1) var<storage, read_write> output_manifolds: array<GpuStatisticalManifold>;

// Compute Fisher information matrix determinant
fn compute_fisher_determinant(eta: array<f32, 4>) -> f32 {
    // For exponential families, Fisher metric: g_ij = ∂²ψ/∂η_i∂η_j
    // Compute 2x2 Hessian for simplicity

    let eps = 1e-6;

    // Compute second derivatives using finite differences
    var hessian: array<array<f32, 2>, 2>;

    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            var eta_pp = eta;
            var eta_pm = eta;
            var eta_mp = eta;
            var eta_mm = eta;

            eta_pp[i] += eps; eta_pp[j] += eps;
            eta_pm[i] += eps; eta_pm[j] -= eps;
            eta_mp[i] -= eps; eta_mp[j] += eps;
            eta_mm[i] -= eps; eta_mm[j] -= eps;

            let psi_pp = log_partition_function(eta_pp);
            let psi_pm = log_partition_function(eta_pm);
            let psi_mp = log_partition_function(eta_mp);
            let psi_mm = log_partition_function(eta_mm);

            hessian[i][j] = (psi_pp - psi_pm - psi_mp + psi_mm) / (4.0 * eps * eps);
        }
    }

    // Compute 2x2 determinant
    return hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];
}

// Log partition function (same as in other shaders)
fn log_partition_function(eta: array<f32, 4>) -> f32 {
    let eta1 = eta[0];
    let eta2 = eta[1];

    if (eta2 < -1e-8) {
        return -(eta1 * eta1) / (4.0 * eta2) - 0.5 * log(-2.0 * eta2);
    } else {
        return eta1 * eta1 * 0.1 + abs(eta2) * 0.5;
    }
}

// Compute expectation parameters from natural parameters
fn compute_expectation_parameters(eta: array<f32, 4>) -> array<f32, 4> {
    var mu: array<f32, 4>;
    let eps = 1e-6;

    // μ = ∇ψ(η)
    for (var i = 0u; i < 4u; i++) {
        var eta_plus = eta;
        var eta_minus = eta;

        eta_plus[i] += eps;
        eta_minus[i] -= eps;

        let psi_plus = log_partition_function(eta_plus);
        let psi_minus = log_partition_function(eta_minus);

        mu[i] = (psi_plus - psi_minus) / (2.0 * eps);
    }

    return mu;
}

// Compute entropy
fn compute_entropy(eta: array<f32, 4>, mu: array<f32, 4>) -> f32 {
    // For exponential families: H = ⟨η,μ⟩ - ψ(η)
    var inner_product = 0.0;
    for (var i = 0; i < 4; i++) {
        inner_product += eta[i] * mu[i];
    }

    let psi = log_partition_function(eta);
    return inner_product - psi;
}

// Compute α-connection Christoffel symbols (simplified)
fn compute_christoffel_symbols(eta: array<f32, 4>, alpha: f32) -> array<f32, 8> {
    // Christoffel symbols for α-connection: Γᵢⱼᵏ⁽ᵅ⁾
    // Simplified computation for 2D case

    var symbols: array<f32, 8>; // Store flattened Γᵢⱼᵏ for i,j,k ∈ {0,1}

    let eps = 1e-6;

    // Compute Fisher metric components
    var fisher: array<array<f32, 2>, 2>;
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            var eta_pp = eta;
            eta_pp[i] += eps;
            eta_pp[j] += eps;

            var eta_mm = eta;
            eta_mm[i] -= eps;
            eta_mm[j] -= eps;

            let psi_pp = log_partition_function(eta_pp);
            let psi_mm = log_partition_function(eta_mm);

            fisher[i][j] = (psi_pp - 2.0 * log_partition_function(eta) + psi_mm) / (eps * eps);
        }
    }

    // Compute third derivatives for Amari-Chentsov tensor
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            for (var k = 0u; k < 2u; k++) {
                let symbol_idx = i * 4u + j * 2u + k;

                // Simplified: Γᵢⱼᵏ⁽ᵅ⁾ = Γᵢⱼᵏ⁽⁰⁾ + (α/2) * Tᵢⱼᵏ
                // where Γ⁽⁰⁾ is the 0-connection (exponential) and T is Amari-Chentsov tensor

                var christoffel = 0.0;

                // Simplified computation based on the structure of exponential families
                if (i == j && j == k) {
                    christoffel = alpha * 0.1; // Simplified tensor component
                } else if (i == j || j == k || i == k) {
                    christoffel = alpha * 0.05;
                }

                if (symbol_idx < 8u) {
                    symbols[symbol_idx] = christoffel;
                }
            }
        }
    }

    return symbols;
}

// Geodesic equation integration (simplified Euler method)
fn geodesic_step(
    eta: array<f32, 4>,
    eta_dot: array<f32, 4>,
    christoffel: array<f32, 8>,
    dt: f32
) -> array<f32, 4> {
    var new_eta_dot: array<f32, 4>;

    // Geodesic equation: d²ηᵏ/dt² + Σᵢⱼ Γᵢⱼᵏ (dηᵢ/dt)(dηⱼ/dt) = 0
    for (var k = 0u; k < 2u; k++) {
        var acceleration = 0.0;

        for (var i = 0u; i < 2u; i++) {
            for (var j = 0u; j < 2u; j++) {
                let symbol_idx = i * 4u + j * 2u + k;
                if (symbol_idx < 8u) {
                    acceleration -= christoffel[symbol_idx] * eta_dot[i] * eta_dot[j];
                }
            }
        }

        new_eta_dot[k] = eta_dot[k] + acceleration * dt;
    }

    // Update remaining components (simplified)
    new_eta_dot[2] = eta_dot[2];
    new_eta_dot[3] = eta_dot[3];

    return new_eta_dot;
}

// Parallel transport along geodesic
fn parallel_transport(
    vector: array<f32, 4>,
    eta: array<f32, 4>,
    eta_dot: array<f32, 4>,
    christoffel: array<f32, 8>,
    dt: f32
) -> array<f32, 4> {
    var transported_vector: array<f32, 4>;

    // Parallel transport equation: DV/dt + Σᵢⱼ Γᵢⱼᵏ (dγᵢ/dt) Vⱼ = 0
    for (var k = 0u; k < 2u; k++) {
        var connection_term = 0.0;

        for (var i = 0u; i < 2u; i++) {
            for (var j = 0u; j < 2u; j++) {
                let symbol_idx = i * 4u + j * 2u + k;
                if (symbol_idx < 8u) {
                    connection_term += christoffel[symbol_idx] * eta_dot[i] * vector[j];
                }
            }
        }

        transported_vector[k] = vector[k] - connection_term * dt;
    }

    transported_vector[2] = vector[2];
    transported_vector[3] = vector[3];

    return transported_vector;
}

// Update manifold properties
fn update_manifold(manifold: GpuStatisticalManifold) -> GpuStatisticalManifold {
    let eta = array<f32, 4>(manifold.eta0, manifold.eta1, manifold.eta2, manifold.eta3);
    var updated = manifold;

    // Recompute expectation parameters
    let mu = compute_expectation_parameters(eta);
    updated.mu0 = mu[0];
    updated.mu1 = mu[1];
    updated.mu2 = mu[2];
    updated.mu3 = mu[3];

    // Update Fisher metric determinant
    updated.fisher_metric_det = compute_fisher_determinant(eta);

    // Update entropy
    updated.entropy = compute_entropy(eta, mu);

    // Ensure numerical stability
    if (abs(updated.fisher_metric_det) < updated.convergence_threshold) {
        updated.fisher_metric_det = updated.convergence_threshold;
    }

    return updated;
}

// Compute geodesic between two points
fn compute_geodesic_midpoint(
    manifold1: GpuStatisticalManifold,
    manifold2: GpuStatisticalManifold
) -> GpuStatisticalManifold {
    let eta1 = array<f32, 4>(manifold1.eta0, manifold1.eta1, manifold1.eta2, manifold1.eta3);
    let eta2 = array<f32, 4>(manifold2.eta0, manifold2.eta1, manifold2.eta2, manifold2.eta3);

    var midpoint = manifold1;

    // Simple geodesic midpoint approximation
    // For α-connections, this involves solving ODEs, but we use linear interpolation as approximation
    let t = 0.5; // Midpoint parameter

    if (abs(manifold1.alpha_connection) < 1e-8) {
        // e-connection (α = 0): midpoint in natural parameter space
        midpoint.eta0 = (1.0 - t) * eta1[0] + t * eta2[0];
        midpoint.eta1 = (1.0 - t) * eta1[1] + t * eta2[1];
        midpoint.eta2 = (1.0 - t) * eta1[2] + t * eta2[2];
        midpoint.eta3 = (1.0 - t) * eta1[3] + t * eta2[3];
    } else {
        // General α-connection: more complex interpolation
        // Simplified: use weighted combination based on Fisher metric
        let w1 = 1.0 / (1.0 + manifold1.fisher_metric_det);
        let w2 = 1.0 / (1.0 + manifold2.fisher_metric_det);
        let total_weight = w1 + w2;

        midpoint.eta0 = (w1 * eta1[0] + w2 * eta2[0]) / total_weight;
        midpoint.eta1 = (w1 * eta1[1] + w2 * eta2[1]) / total_weight;
        midpoint.eta2 = (w1 * eta1[2] + w2 * eta2[2]) / total_weight;
        midpoint.eta3 = (w1 * eta1[3] + w2 * eta2[3]) / total_weight;
    }

    return update_manifold(midpoint);
}

@compute @workgroup_size(256)
fn statistical_manifold_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let manifold_index = global_id.x;

    if (manifold_index >= arrayLength(&input_manifolds)) {
        return;
    }

    let manifold = input_manifolds[manifold_index];

    // Update manifold properties
    var updated_manifold = update_manifold(manifold);

    // If working with pairs, compute geodesic properties
    if (manifold_index > 0u && manifold_index < arrayLength(&input_manifolds)) {
        let reference_index = manifold_index - 1u;
        if (reference_index < arrayLength(&input_manifolds)) {
            let reference_manifold = input_manifolds[reference_index];
            let geodesic_midpoint = compute_geodesic_midpoint(updated_manifold, reference_manifold);

            // Store some geodesic information in unused fields
            updated_manifold.batch_id = geodesic_midpoint.entropy;
        }
    }

    // Store result
    if (manifold_index < arrayLength(&output_manifolds)) {
        output_manifolds[manifold_index] = updated_manifold;
    }
}