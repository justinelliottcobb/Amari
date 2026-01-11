//! WGSL compute shaders for GPU-accelerated dynamics
//!
//! This module contains WebGPU Shading Language (WGSL) compute shaders
//! for parallel dynamical systems computation.

/// Batch trajectory computation shader
///
/// Computes multiple trajectories in parallel using RK4 integration.
/// Each workgroup handles one trajectory.
pub const BATCH_TRAJECTORY_SHADER: &str = r#"
// Batch trajectory computation using RK4 integration
// Each invocation computes one full trajectory

struct Config {
    dt: f32,
    steps: u32,
    dim: u32,
    system_type: u32,
}

@group(0) @binding(0) var<storage, read> initial_conditions: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

// Lorenz system: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z
fn lorenz(x: f32, y: f32, z: f32, sigma: f32, rho: f32, beta: f32) -> vec3<f32> {
    return vec3<f32>(
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    );
}

// Van der Pol oscillator: dx/dt = y, dy/dt = mu*(1-x^2)*y - x
fn van_der_pol(x: f32, y: f32, mu: f32) -> vec2<f32> {
    return vec2<f32>(
        y,
        mu * (1.0 - x * x) * y - x
    );
}

// Duffing oscillator: dx/dt = y, dy/dt = -delta*y - alpha*x - beta*x^3
fn duffing(x: f32, y: f32, delta: f32, alpha: f32, beta: f32) -> vec2<f32> {
    return vec2<f32>(
        y,
        -delta * y - alpha * x - beta * x * x * x
    );
}

// Rössler system: dx/dt = -y-z, dy/dt = x+a*y, dz/dt = b + z*(x-c)
fn rossler(x: f32, y: f32, z: f32, a: f32, b: f32, c: f32) -> vec3<f32> {
    return vec3<f32>(
        -y - z,
        x + a * y,
        b + z * (x - c)
    );
}

// Simple pendulum: dtheta/dt = omega, domega/dt = -g/L * sin(theta)
fn pendulum(theta: f32, omega: f32, g_over_L: f32) -> vec2<f32> {
    return vec2<f32>(
        omega,
        -g_over_L * sin(theta)
    );
}

// RK4 step for 3D system
fn rk4_step_3d(
    state: vec3<f32>,
    dt: f32,
    system_type: u32,
    p0: f32, p1: f32, p2: f32
) -> vec3<f32> {
    var k1: vec3<f32>;
    var k2: vec3<f32>;
    var k3: vec3<f32>;
    var k4: vec3<f32>;

    if (system_type == 0u) {
        // Lorenz
        k1 = lorenz(state.x, state.y, state.z, p0, p1, p2);
        let s2 = state + 0.5 * dt * k1;
        k2 = lorenz(s2.x, s2.y, s2.z, p0, p1, p2);
        let s3 = state + 0.5 * dt * k2;
        k3 = lorenz(s3.x, s3.y, s3.z, p0, p1, p2);
        let s4 = state + dt * k3;
        k4 = lorenz(s4.x, s4.y, s4.z, p0, p1, p2);
    } else if (system_type == 3u) {
        // Rössler
        k1 = rossler(state.x, state.y, state.z, p0, p1, p2);
        let s2 = state + 0.5 * dt * k1;
        k2 = rossler(s2.x, s2.y, s2.z, p0, p1, p2);
        let s3 = state + 0.5 * dt * k2;
        k3 = rossler(s3.x, s3.y, s3.z, p0, p1, p2);
        let s4 = state + dt * k3;
        k4 = rossler(s4.x, s4.y, s4.z, p0, p1, p2);
    } else {
        // Default to Lorenz
        k1 = lorenz(state.x, state.y, state.z, p0, p1, p2);
        let s2 = state + 0.5 * dt * k1;
        k2 = lorenz(s2.x, s2.y, s2.z, p0, p1, p2);
        let s3 = state + 0.5 * dt * k2;
        k3 = lorenz(s3.x, s3.y, s3.z, p0, p1, p2);
        let s4 = state + dt * k3;
        k4 = lorenz(s4.x, s4.y, s4.z, p0, p1, p2);
    }

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// RK4 step for 2D system
fn rk4_step_2d(
    state: vec2<f32>,
    dt: f32,
    system_type: u32,
    p0: f32, p1: f32, p2: f32
) -> vec2<f32> {
    var k1: vec2<f32>;
    var k2: vec2<f32>;
    var k3: vec2<f32>;
    var k4: vec2<f32>;

    if (system_type == 1u) {
        // Van der Pol
        k1 = van_der_pol(state.x, state.y, p0);
        let s2 = state + 0.5 * dt * k1;
        k2 = van_der_pol(s2.x, s2.y, p0);
        let s3 = state + 0.5 * dt * k2;
        k3 = van_der_pol(s3.x, s3.y, p0);
        let s4 = state + dt * k3;
        k4 = van_der_pol(s4.x, s4.y, p0);
    } else if (system_type == 2u) {
        // Duffing
        k1 = duffing(state.x, state.y, p0, p1, p2);
        let s2 = state + 0.5 * dt * k1;
        k2 = duffing(s2.x, s2.y, p0, p1, p2);
        let s3 = state + 0.5 * dt * k2;
        k3 = duffing(s3.x, s3.y, p0, p1, p2);
        let s4 = state + dt * k3;
        k4 = duffing(s4.x, s4.y, p0, p1, p2);
    } else if (system_type == 4u) {
        // Pendulum
        k1 = pendulum(state.x, state.y, p0);
        let s2 = state + 0.5 * dt * k1;
        k2 = pendulum(s2.x, s2.y, p0);
        let s3 = state + 0.5 * dt * k2;
        k3 = pendulum(s3.x, s3.y, p0);
        let s4 = state + dt * k3;
        k4 = pendulum(s4.x, s4.y, p0);
    } else {
        // Default to Van der Pol
        k1 = van_der_pol(state.x, state.y, p0);
        let s2 = state + 0.5 * dt * k1;
        k2 = van_der_pol(s2.x, s2.y, p0);
        let s3 = state + 0.5 * dt * k2;
        k3 = van_der_pol(s3.x, s3.y, p0);
        let s4 = state + dt * k3;
        k4 = van_der_pol(s4.x, s4.y, p0);
    }

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let dim = config.dim;

    // Read initial condition
    let ic_offset = idx * dim;

    // Read parameters (shared across all trajectories or per-trajectory)
    let p0 = params[0];
    let p1 = params[1];
    let p2 = params[2];

    if (dim == 3u) {
        var state = vec3<f32>(
            initial_conditions[ic_offset],
            initial_conditions[ic_offset + 1u],
            initial_conditions[ic_offset + 2u]
        );

        // Integrate for the specified number of steps
        for (var i = 0u; i < config.steps; i = i + 1u) {
            state = rk4_step_3d(state, config.dt, config.system_type, p0, p1, p2);
        }

        // Write final state
        let out_offset = idx * dim;
        output[out_offset] = state.x;
        output[out_offset + 1u] = state.y;
        output[out_offset + 2u] = state.z;
    } else {
        var state = vec2<f32>(
            initial_conditions[ic_offset],
            initial_conditions[ic_offset + 1u]
        );

        // Integrate for the specified number of steps
        for (var i = 0u; i < config.steps; i = i + 1u) {
            state = rk4_step_2d(state, config.dt, config.system_type, p0, p1, p2);
        }

        // Write final state
        let out_offset = idx * dim;
        output[out_offset] = state.x;
        output[out_offset + 1u] = state.y;
    }
}
"#;

/// Flow field computation shader
///
/// Computes the vector field on a 2D grid for phase portrait visualization.
pub const FLOW_FIELD_SHADER: &str = r#"
// Flow field computation for phase portrait visualization
// Each invocation computes the vector field at one grid point

struct Config {
    width: u32,
    height: u32,
    x_min: f32,
    x_max: f32,
    y_min: f32,
    y_max: f32,
    system_type: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> grid_points: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

// Van der Pol oscillator
fn van_der_pol(x: f32, y: f32, mu: f32) -> vec2<f32> {
    return vec2<f32>(y, mu * (1.0 - x * x) * y - x);
}

// Duffing oscillator (unforced)
fn duffing(x: f32, y: f32, delta: f32, alpha: f32, beta: f32) -> vec2<f32> {
    return vec2<f32>(y, -delta * y - alpha * x - beta * x * x * x);
}

// Simple pendulum
fn pendulum(theta: f32, omega: f32, g_over_L: f32) -> vec2<f32> {
    return vec2<f32>(omega, -g_over_L * sin(theta));
}

// Simple harmonic oscillator
fn harmonic(x: f32, y: f32, omega: f32) -> vec2<f32> {
    return vec2<f32>(y, -omega * omega * x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= config.width || j >= config.height) {
        return;
    }

    // Compute grid position
    let dx = (config.x_max - config.x_min) / f32(config.width - 1u);
    let dy = (config.y_max - config.y_min) / f32(config.height - 1u);
    let x = config.x_min + f32(i) * dx;
    let y = config.y_min + f32(j) * dy;

    // Read parameters
    let p0 = params[0];
    let p1 = params[1];
    let p2 = params[2];

    // Compute vector field
    var v: vec2<f32>;

    if (config.system_type == 1u) {
        v = van_der_pol(x, y, p0);
    } else if (config.system_type == 2u) {
        v = duffing(x, y, p0, p1, p2);
    } else if (config.system_type == 4u) {
        v = pendulum(x, y, p0);
    } else {
        v = harmonic(x, y, p0);
    }

    // Write output
    let out_idx = (j * config.width + i) * 2u;
    output[out_idx] = v.x;
    output[out_idx + 1u] = v.y;
}
"#;

/// Bifurcation diagram computation shader
///
/// Computes attractors for many parameter values in parallel.
pub const BIFURCATION_SHADER: &str = r#"
// Bifurcation diagram computation
// Each invocation computes the attractor for one parameter value

struct Config {
    dt: f32,
    transient_steps: u32,
    sample_steps: u32,
    samples_per_param: u32,
    param_min: f32,
    param_max: f32,
    num_params: u32,
    system_type: u32,
}

@group(0) @binding(0) var<storage, read> initial_condition: array<f32>;
@group(0) @binding(1) var<storage, read> fixed_params: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

// Logistic map: x_{n+1} = r * x_n * (1 - x_n)
fn logistic_map(x: f32, r: f32) -> f32 {
    return r * x * (1.0 - x);
}

// Hénon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n
fn henon_map(x: f32, y: f32, a: f32, b: f32) -> vec2<f32> {
    return vec2<f32>(1.0 - a * x * x + y, b * x);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let param_idx = global_id.x;

    if (param_idx >= config.num_params) {
        return;
    }

    // Compute parameter value
    let param_range = config.param_max - config.param_min;
    let param = config.param_min + f32(param_idx) * param_range / f32(config.num_params - 1u);

    // Initialize state
    var x = initial_condition[0];
    var y = initial_condition[1];

    // Fixed parameters
    let p1 = fixed_params[0];

    // Transient phase
    for (var i = 0u; i < config.transient_steps; i = i + 1u) {
        if (config.system_type == 0u) {
            // Logistic map
            x = logistic_map(x, param);
        } else {
            // Hénon map
            let new_state = henon_map(x, y, param, p1);
            x = new_state.x;
            y = new_state.y;
        }
    }

    // Sampling phase
    let out_offset = param_idx * config.samples_per_param;
    for (var i = 0u; i < config.samples_per_param; i = i + 1u) {
        // Take sample_steps between each sample
        for (var j = 0u; j < config.sample_steps; j = j + 1u) {
            if (config.system_type == 0u) {
                x = logistic_map(x, param);
            } else {
                let new_state = henon_map(x, y, param, p1);
                x = new_state.x;
                y = new_state.y;
            }
        }
        output[out_offset + i] = x;
    }
}
"#;

/// Lyapunov exponent computation shader
///
/// Computes the largest Lyapunov exponent via trajectory separation.
pub const LYAPUNOV_SHADER: &str = r#"
// Largest Lyapunov exponent computation
// Uses trajectory separation method

struct Config {
    dt: f32,
    steps: u32,
    renorm_steps: u32,
    epsilon: f32,
    system_type: u32,
    dim: u32,
    _padding0: u32,
    _padding1: u32,
}

@group(0) @binding(0) var<storage, read> initial_conditions: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> config: Config;

fn lorenz(x: f32, y: f32, z: f32, sigma: f32, rho: f32, beta: f32) -> vec3<f32> {
    return vec3<f32>(sigma * (y - x), x * (rho - z) - y, x * y - beta * z);
}

fn rk4_lorenz(state: vec3<f32>, dt: f32, sigma: f32, rho: f32, beta: f32) -> vec3<f32> {
    let k1 = lorenz(state.x, state.y, state.z, sigma, rho, beta);
    let s2 = state + 0.5 * dt * k1;
    let k2 = lorenz(s2.x, s2.y, s2.z, sigma, rho, beta);
    let s3 = state + 0.5 * dt * k2;
    let k3 = lorenz(s3.x, s3.y, s3.z, sigma, rho, beta);
    let s4 = state + dt * k3;
    let k4 = lorenz(s4.x, s4.y, s4.z, sigma, rho, beta);
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Read initial condition
    let ic_offset = idx * config.dim;
    var state = vec3<f32>(
        initial_conditions[ic_offset],
        initial_conditions[ic_offset + 1u],
        initial_conditions[ic_offset + 2u]
    );

    // Perturbed state
    var perturbed = state + vec3<f32>(config.epsilon, 0.0, 0.0);

    let sigma = params[0];
    let rho = params[1];
    let beta = params[2];

    var lyapunov_sum: f32 = 0.0;
    var renorm_count: u32 = 0u;

    for (var step = 0u; step < config.steps; step = step + 1u) {
        // Evolve both trajectories
        state = rk4_lorenz(state, config.dt, sigma, rho, beta);
        perturbed = rk4_lorenz(perturbed, config.dt, sigma, rho, beta);

        // Renormalization
        if ((step + 1u) % config.renorm_steps == 0u) {
            let diff = perturbed - state;
            let dist = length(diff);

            if (dist > 0.0) {
                lyapunov_sum = lyapunov_sum + log(dist / config.epsilon);
                renorm_count = renorm_count + 1u;

                // Renormalize perturbed trajectory
                perturbed = state + diff * (config.epsilon / dist);
            }
        }
    }

    // Compute average Lyapunov exponent
    let total_time = f32(config.steps) * config.dt;
    let lyapunov = lyapunov_sum / total_time;

    output[idx] = lyapunov;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_sources_valid() {
        // Just check that the shader strings are non-empty
        assert!(!BATCH_TRAJECTORY_SHADER.is_empty());
        assert!(!FLOW_FIELD_SHADER.is_empty());
        assert!(!BIFURCATION_SHADER.is_empty());
        assert!(!LYAPUNOV_SHADER.is_empty());
    }

    #[test]
    fn test_shader_contains_main() {
        assert!(BATCH_TRAJECTORY_SHADER.contains("fn main"));
        assert!(FLOW_FIELD_SHADER.contains("fn main"));
        assert!(BIFURCATION_SHADER.contains("fn main"));
        assert!(LYAPUNOV_SHADER.contains("fn main"));
    }
}
