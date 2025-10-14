// Bregman Divergence Compute Shader
// Computes Bregman divergences between probability distributions and statistical models

struct GpuBregmanData {
    // Point p parameters
    p_param0: f32,
    p_param1: f32,
    p_param2: f32,
    p_param3: f32,
    // Point q parameters
    q_param0: f32,
    q_param1: f32,
    q_param2: f32,
    q_param3: f32,
    // Potential function parameters
    potential_type: f32,
    potential_scale: f32,
    potential_offset: f32,
    regularization: f32,
    // Gradient info
    gradient_p0: f32,
    gradient_p1: f32,
    gradient_q0: f32,
    gradient_q1: f32,
}

@group(0) @binding(0) var<storage, read> input_data: array<GpuBregmanData>;
@group(0) @binding(1) var<storage, read_write> output_divergences: array<f32>;

// Potential functions for different Bregman divergences
fn potential_function(params: array<f32, 4>, potential_type: f32, scale: f32, offset: f32) -> f32 {
    var result = 0.0;

    if (potential_type == 0.0) {
        // Quadratic potential: φ(x) = (1/2)||x||²
        for (var i = 0; i < 4; i++) {
            result += params[i] * params[i];
        }
        result = result * 0.5 * scale + offset;
    } else if (potential_type == 1.0) {
        // Entropy potential: φ(x) = Σ xᵢ log(xᵢ) (for probability vectors)
        for (var i = 0; i < 4; i++) {
            let p = params[i];
            if (p > 1e-12) {
                result += p * log(p);
            }
        }
        result = result * scale + offset;
    } else if (potential_type == 2.0) {
        // Exponential potential: φ(x) = exp(||x||²)
        var norm_sq = 0.0;
        for (var i = 0; i < 4; i++) {
            norm_sq += params[i] * params[i];
        }
        result = exp(norm_sq * scale) + offset;
    } else if (potential_type == 3.0) {
        // Log-sum-exp potential: φ(x) = log(Σ exp(xᵢ))
        var max_val = params[0];
        for (var i = 1; i < 4; i++) {
            if (params[i] > max_val) {
                max_val = params[i];
            }
        }

        var sum_exp = 0.0;
        for (var i = 0; i < 4; i++) {
            sum_exp += exp((params[i] - max_val) * scale);
        }

        result = max_val + log(sum_exp) / scale + offset;
    } else {
        // Linear potential: φ(x) = scale * Σ xᵢ + offset
        for (var i = 0; i < 4; i++) {
            result += params[i];
        }
        result = result * scale + offset;
    }

    return result;
}

// Compute gradient using finite differences
fn compute_gradient(params: array<f32, 4>, potential_type: f32, scale: f32, offset: f32, eps: f32) -> array<f32, 4> {
    var gradient: array<f32, 4>;

    for (var i = 0; i < 4; i++) {
        var params_plus = params;
        var params_minus = params;

        params_plus[i] += eps;
        params_minus[i] -= eps;

        let phi_plus = potential_function(params_plus, potential_type, scale, offset);
        let phi_minus = potential_function(params_minus, potential_type, scale, offset);

        gradient[i] = (phi_plus - phi_minus) / (2.0 * eps);
    }

    return gradient;
}

// Compute analytical gradient for known potentials
fn analytical_gradient(params: array<f32, 4>, potential_type: f32, scale: f32) -> array<f32, 4> {
    var gradient: array<f32, 4>;

    if (potential_type == 0.0) {
        // Quadratic: ∇φ(x) = scale * x
        for (var i = 0; i < 4; i++) {
            gradient[i] = scale * params[i];
        }
    } else if (potential_type == 1.0) {
        // Entropy: ∇φ(x) = scale * (log(x) + 1)
        for (var i = 0; i < 4; i++) {
            let p = params[i];
            if (p > 1e-12) {
                gradient[i] = scale * (log(p) + 1.0);
            } else {
                gradient[i] = scale * (log(1e-12) + 1.0);
            }
        }
    } else if (potential_type == 2.0) {
        // Exponential: ∇φ(x) = 2 * scale * exp(scale * ||x||²) * x
        var norm_sq = 0.0;
        for (var i = 0; i < 4; i++) {
            norm_sq += params[i] * params[i];
        }
        let exp_factor = exp(scale * norm_sq);
        for (var i = 0; i < 4; i++) {
            gradient[i] = 2.0 * scale * exp_factor * params[i];
        }
    } else {
        // Default to finite differences
        return compute_gradient(params, potential_type, scale, 0.0, 1e-6);
    }

    return gradient;
}

// Compute KL divergence for probability distributions
fn kl_divergence(p_params: array<f32, 4>, q_params: array<f32, 4>) -> f32 {
    var kl = 0.0;

    // D_KL(P||Q) = Σ p_i log(p_i / q_i)
    for (var i = 0; i < 4; i++) {
        let p_i = p_params[i];
        let q_i = q_params[i];

        if (p_i > 1e-12 && q_i > 1e-12) {
            kl += p_i * log(p_i / q_i);
        } else if (p_i > 1e-12) {
            // q_i is zero or very small - KL divergence is infinite, use large value
            kl += p_i * 20.0;
        }
        // If p_i is zero, contribution is zero (0 * log(0/q) = 0)
    }

    return kl;
}

// Compute Jensen-Shannon divergence
fn jensen_shannon_divergence(p_params: array<f32, 4>, q_params: array<f32, 4>) -> f32 {
    // JS(P,Q) = (1/2)[D_KL(P||M) + D_KL(Q||M)] where M = (P+Q)/2
    var m_params: array<f32, 4>;
    for (var i = 0; i < 4; i++) {
        m_params[i] = (p_params[i] + q_params[i]) * 0.5;
    }

    let kl_pm = kl_divergence(p_params, m_params);
    let kl_qm = kl_divergence(q_params, m_params);

    return 0.5 * (kl_pm + kl_qm);
}

// Main Bregman divergence computation
fn compute_bregman_divergence(data: GpuBregmanData) -> f32 {
    let p_params = array<f32, 4>(data.p_param0, data.p_param1, data.p_param2, data.p_param3);
    let q_params = array<f32, 4>(data.q_param0, data.q_param1, data.q_param2, data.q_param3);

    // Bregman divergence: D_φ(p,q) = φ(p) - φ(q) - ⟨∇φ(q), p-q⟩

    let phi_p = potential_function(p_params, data.potential_type, data.potential_scale, data.potential_offset);
    let phi_q = potential_function(q_params, data.potential_type, data.potential_scale, data.potential_offset);

    // Compute or use precomputed gradient
    var grad_q: array<f32, 4>;

    // Use precomputed gradients if available, otherwise compute analytically
    if (abs(data.gradient_q0) > 1e-12 || abs(data.gradient_q1) > 1e-12) {
        grad_q[0] = data.gradient_q0;
        grad_q[1] = data.gradient_q1;
        grad_q[2] = 0.0; // Extended if needed
        grad_q[3] = 0.0;
    } else {
        grad_q = analytical_gradient(q_params, data.potential_type, data.potential_scale);
    }

    // Compute inner product ⟨∇φ(q), p-q⟩
    var inner_product = 0.0;
    for (var i = 0; i < 4; i++) {
        let diff = p_params[i] - q_params[i];
        inner_product += grad_q[i] * diff;
    }

    var divergence = phi_p - phi_q - inner_product;

    // Add regularization to ensure non-negativity
    if (divergence < 0.0) {
        divergence = divergence + data.regularization;
    }

    return max(divergence, 0.0);
}

// Specialized divergences for common cases
fn compute_specialized_divergence(data: GpuBregmanData) -> f32 {
    let p_params = array<f32, 4>(data.p_param0, data.p_param1, data.p_param2, data.p_param3);
    let q_params = array<f32, 4>(data.q_param0, data.q_param1, data.q_param2, data.q_param3);

    if (data.potential_type == 1.0) {
        // For entropy potential, this gives KL divergence
        return kl_divergence(p_params, q_params);
    } else if (data.potential_type == 0.0) {
        // For quadratic potential, this gives squared Euclidean distance
        var squared_distance = 0.0;
        for (var i = 0; i < 4; i++) {
            let diff = p_params[i] - q_params[i];
            squared_distance += diff * diff;
        }
        return squared_distance * 0.5 * data.potential_scale;
    } else {
        // Use general Bregman divergence computation
        return compute_bregman_divergence(data);
    }
}

@compute @workgroup_size(256)
fn bregman_divergence_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let data_index = global_id.x;

    if (data_index >= arrayLength(&input_data)) {
        return;
    }

    let data = input_data[data_index];

    // Compute the appropriate divergence
    let divergence = compute_specialized_divergence(data);

    // Store result
    if (data_index < arrayLength(&output_divergences)) {
        output_divergences[data_index] = divergence;
    }
}