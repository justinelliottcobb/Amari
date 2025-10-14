// KL Divergence Compute Shader
// Computes KL divergences and related information-theoretic quantities

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
@group(0) @binding(1) var<storage, read_write> output_divergences: array<f32>;

// Compute log partition function from natural parameters
fn log_partition_function(eta: array<f32, 4>) -> f32 {
    // For exponential families: ψ(η) = log Z(η)
    // Simplified implementations for common cases

    // Gaussian: ψ(η₁,η₂) = -η₁²/(4η₂) - 0.5*log(-2η₂)
    let eta1 = eta[0];
    let eta2 = eta[1];

    if (eta2 < -1e-8) {
        return -(eta1 * eta1) / (4.0 * eta2) - 0.5 * log(-2.0 * eta2);
    } else {
        // Regularized version for numerical stability
        return eta1 * eta1 * 0.1 + abs(eta2) * 0.5;
    }
}

// Compute expectation parameters from natural parameters
fn natural_to_expectation(eta: array<f32, 4>) -> array<f32, 4> {
    var mu: array<f32, 4>;

    // μ = ∇ψ(η) - gradient of log partition function
    let eps = 1e-6;

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

// Compute natural parameters from expectation parameters (inverse transform)
fn expectation_to_natural(mu: array<f32, 4>, max_iter: f32) -> array<f32, 4> {
    var eta = mu; // Initial guess
    let iterations = u32(max_iter);
    let tolerance = 1e-6;

    // Newton-Raphson iteration: η^(k+1) = η^k - F^(-1)(η^k) * (∇ψ(η^k) - μ)
    for (var iter = 0u; iter < iterations; iter++) {
        let computed_mu = natural_to_expectation(eta);

        // Compute residual
        var residual_norm = 0.0;
        for (var i = 0; i < 4; i++) {
            let diff = computed_mu[i] - mu[i];
            residual_norm += diff * diff;
        }

        if (sqrt(residual_norm) < tolerance) {
            break;
        }

        // Simplified update (assuming Fisher metric is approximately identity)
        for (var i = 0; i < 4; i++) {
            eta[i] -= 0.1 * (computed_mu[i] - mu[i]);
        }
    }

    return eta;
}

// KL divergence in terms of natural and expectation parameters
fn kl_divergence_exponential_family(
    eta_p: array<f32, 4>,
    eta_q: array<f32, 4>,
    mu_p: array<f32, 4>
) -> f32 {
    // KL(P||Q) = ⟨η_p - η_q, μ_p⟩ - ψ(η_p) + ψ(η_q)

    var inner_product = 0.0;
    for (var i = 0; i < 4; i++) {
        inner_product += (eta_p[i] - eta_q[i]) * mu_p[i];
    }

    let psi_p = log_partition_function(eta_p);
    let psi_q = log_partition_function(eta_q);

    return inner_product - psi_p + psi_q;
}

// Compute mutual information
fn mutual_information(joint_eta: array<f32, 4>, marginal1_eta: array<f32, 4>, marginal2_eta: array<f32, 4>) -> f32 {
    // I(X;Y) = H(X) + H(Y) - H(X,Y)
    // For exponential families: H = ⟨η,μ⟩ - ψ(η)

    let joint_mu = natural_to_expectation(joint_eta);
    let marginal1_mu = natural_to_expectation(marginal1_eta);
    let marginal2_mu = natural_to_expectation(marginal2_eta);

    // Entropy = ⟨η,μ⟩ - ψ(η)
    var h_joint = 0.0;
    var h_marg1 = 0.0;
    var h_marg2 = 0.0;

    for (var i = 0; i < 4; i++) {
        h_joint += joint_eta[i] * joint_mu[i];
        if (i < 2) {
            h_marg1 += marginal1_eta[i] * marginal1_mu[i];
            h_marg2 += marginal2_eta[i] * marginal2_mu[i];
        }
    }

    h_joint -= log_partition_function(joint_eta);
    h_marg1 -= log_partition_function(marginal1_eta);
    h_marg2 -= log_partition_function(marginal2_eta);

    return h_marg1 + h_marg2 - h_joint;
}

// α-divergence (generalization of KL divergence)
fn alpha_divergence(
    eta_p: array<f32, 4>,
    eta_q: array<f32, 4>,
    mu_p: array<f32, 4>,
    mu_q: array<f32, 4>,
    alpha: f32
) -> f32 {
    if (abs(alpha) < 1e-8) {
        // α = 0: KL divergence
        return kl_divergence_exponential_family(eta_p, eta_q, mu_p);
    } else if (abs(alpha - 1.0) < 1e-8) {
        // α = 1: Reverse KL divergence
        return kl_divergence_exponential_family(eta_q, eta_p, mu_q);
    } else {
        // General α-divergence
        // D_α(P||Q) = (1/(α(1-α))) * [α*⟨η_p,μ_p⟩ + (1-α)*⟨η_q,μ_q⟩ - ⟨α*η_p + (1-α)*η_q, α*μ_p + (1-α)*μ_q⟩]

        var term1 = 0.0;
        var term2 = 0.0;
        var term3 = 0.0;

        for (var i = 0; i < 4; i++) {
            term1 += eta_p[i] * mu_p[i];
            term2 += eta_q[i] * mu_q[i];

            let combined_eta = alpha * eta_p[i] + (1.0 - alpha) * eta_q[i];
            let combined_mu = alpha * mu_p[i] + (1.0 - alpha) * mu_q[i];
            term3 += combined_eta * combined_mu;
        }

        let coeff = 1.0 / (alpha * (1.0 - alpha));
        return coeff * (alpha * term1 + (1.0 - alpha) * term2 - term3);
    }
}

// Compute geodesic distance on statistical manifold
fn geodesic_distance(eta_p: array<f32, 4>, eta_q: array<f32, 4>, alpha: f32) -> f32 {
    // For α-connections, geodesic distance involves integration
    // Simplified approximation using α-divergence

    let mu_p = natural_to_expectation(eta_p);
    let mu_q = natural_to_expectation(eta_q);

    // Symmetric α-divergence
    let d_alpha_pq = alpha_divergence(eta_p, eta_q, mu_p, mu_q, alpha);
    let d_alpha_qp = alpha_divergence(eta_q, eta_p, mu_q, mu_p, alpha);

    return sqrt(0.5 * (d_alpha_pq + d_alpha_qp));
}

// Main computation function
fn compute_kl_and_related(manifold: GpuStatisticalManifold, reference_manifold: GpuStatisticalManifold) -> f32 {
    let eta_p = array<f32, 4>(manifold.eta0, manifold.eta1, manifold.eta2, manifold.eta3);
    let mu_p = array<f32, 4>(manifold.mu0, manifold.mu1, manifold.mu2, manifold.mu3);

    let eta_q = array<f32, 4>(reference_manifold.eta0, reference_manifold.eta1, reference_manifold.eta2, reference_manifold.eta3);
    let mu_q = array<f32, 4>(reference_manifold.mu0, reference_manifold.mu1, reference_manifold.mu2, reference_manifold.mu3);

    // Choose computation based on connection type
    if (abs(manifold.alpha_connection) < 1e-8) {
        // Standard KL divergence (α = 0)
        return kl_divergence_exponential_family(eta_p, eta_q, mu_p);
    } else {
        // α-divergence
        return alpha_divergence(eta_p, eta_q, mu_p, mu_q, manifold.alpha_connection);
    }
}

@compute @workgroup_size(256)
fn kl_divergence_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let manifold_index = global_id.x;

    if (manifold_index >= arrayLength(&input_manifolds)) {
        return;
    }

    // For simplicity, compute KL divergence between each manifold and the first one as reference
    let manifold = input_manifolds[manifold_index];
    let reference_manifold = input_manifolds[0];

    let divergence = compute_kl_and_related(manifold, reference_manifold);

    // Store result
    if (manifold_index < arrayLength(&output_divergences)) {
        output_divergences[manifold_index] = divergence;
    }
}