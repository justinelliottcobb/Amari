//! WebAssembly bindings for dynamical systems
//!
//! This module provides WASM-compatible interfaces for running dynamical
//! systems simulations in web browsers.
//!
//! # Usage from JavaScript
//!
//! ```javascript
//! import init, { WasmLorenz, WasmTrajectory } from 'amari-dynamics';
//!
//! await init();
//!
//! // Create a Lorenz system
//! const lorenz = WasmLorenz.classic();
//!
//! // Simulate a trajectory
//! const trajectory = lorenz.simulate([1.0, 1.0, 1.0], 0.01, 10000);
//!
//! // Get the trajectory data
//! const points = trajectory.getPoints();
//! ```

use wasm_bindgen::prelude::*;

/// WASM-compatible trajectory result
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmTrajectory {
    /// Flattened point data (x, y, z, x, y, z, ...)
    points: Vec<f64>,
    /// Time values
    times: Vec<f64>,
    /// Dimension of state space
    dim: usize,
}

#[wasm_bindgen]
impl WasmTrajectory {
    /// Get the number of points in the trajectory
    #[wasm_bindgen(js_name = getLength)]
    pub fn length(&self) -> usize {
        self.times.len()
    }

    /// Get the dimension of the state space
    #[wasm_bindgen(js_name = getDimension)]
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Get all points as a flat array
    #[wasm_bindgen(js_name = getPoints)]
    pub fn get_points(&self) -> Vec<f64> {
        self.points.clone()
    }

    /// Get all time values
    #[wasm_bindgen(js_name = getTimes)]
    pub fn get_times(&self) -> Vec<f64> {
        self.times.clone()
    }

    /// Get a specific point by index
    #[wasm_bindgen(js_name = getPoint)]
    pub fn get_point(&self, idx: usize) -> Option<Vec<f64>> {
        if idx >= self.times.len() {
            return None;
        }
        let start = idx * self.dim;
        let end = start + self.dim;
        Some(self.points[start..end].to_vec())
    }

    /// Get the final state
    #[wasm_bindgen(js_name = getFinalState)]
    pub fn get_final_state(&self) -> Option<Vec<f64>> {
        if self.times.is_empty() {
            return None;
        }
        self.get_point(self.times.len() - 1)
    }
}

/// WASM-compatible Lorenz system
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct WasmLorenz {
    sigma: f64,
    rho: f64,
    beta: f64,
}

#[wasm_bindgen]
impl WasmLorenz {
    /// Create a new Lorenz system with custom parameters
    #[wasm_bindgen(constructor)]
    pub fn new(sigma: f64, rho: f64, beta: f64) -> Self {
        Self { sigma, rho, beta }
    }

    /// Create the classic Lorenz system (sigma=10, rho=28, beta=8/3)
    #[wasm_bindgen]
    pub fn classic() -> Self {
        Self::new(10.0, 28.0, 8.0 / 3.0)
    }

    /// Get sigma parameter
    #[wasm_bindgen(getter)]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Get rho parameter
    #[wasm_bindgen(getter)]
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Get beta parameter
    #[wasm_bindgen(getter)]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Compute the vector field at a point
    #[wasm_bindgen(js_name = vectorField)]
    pub fn vector_field(&self, x: f64, y: f64, z: f64) -> Vec<f64> {
        vec![
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        ]
    }

    /// Simulate a trajectory using RK4
    #[wasm_bindgen]
    pub fn simulate(&self, initial: Vec<f64>, dt: f64, steps: usize) -> WasmTrajectory {
        let mut points = Vec::with_capacity((steps + 1) * 3);
        let mut times = Vec::with_capacity(steps + 1);

        let mut x = initial.first().copied().unwrap_or(1.0);
        let mut y = initial.get(1).copied().unwrap_or(1.0);
        let mut z = initial.get(2).copied().unwrap_or(1.0);

        // Store initial point
        points.push(x);
        points.push(y);
        points.push(z);
        times.push(0.0);

        for step in 0..steps {
            // RK4 integration
            let (k1x, k1y, k1z) = self.derivatives(x, y, z);
            let (k2x, k2y, k2z) =
                self.derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z);
            let (k3x, k3y, k3z) =
                self.derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z);
            let (k4x, k4y, k4z) = self.derivatives(x + dt * k3x, y + dt * k3y, z + dt * k3z);

            x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
            z += dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);

            points.push(x);
            points.push(y);
            points.push(z);
            times.push((step + 1) as f64 * dt);
        }

        WasmTrajectory {
            points,
            times,
            dim: 3,
        }
    }

    /// Internal: compute derivatives
    fn derivatives(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        )
    }
}

/// WASM-compatible Van der Pol oscillator
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct WasmVanDerPol {
    mu: f64,
}

#[wasm_bindgen]
impl WasmVanDerPol {
    /// Create a new Van der Pol oscillator
    #[wasm_bindgen(constructor)]
    pub fn new(mu: f64) -> Self {
        Self { mu }
    }

    /// Get mu parameter
    #[wasm_bindgen(getter)]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Compute the vector field at a point
    #[wasm_bindgen(js_name = vectorField)]
    pub fn vector_field(&self, x: f64, y: f64) -> Vec<f64> {
        vec![y, self.mu * (1.0 - x * x) * y - x]
    }

    /// Simulate a trajectory using RK4
    #[wasm_bindgen]
    pub fn simulate(&self, initial: Vec<f64>, dt: f64, steps: usize) -> WasmTrajectory {
        let mut points = Vec::with_capacity((steps + 1) * 2);
        let mut times = Vec::with_capacity(steps + 1);

        let mut x = initial.first().copied().unwrap_or(2.0);
        let mut y = initial.get(1).copied().unwrap_or(0.0);

        points.push(x);
        points.push(y);
        times.push(0.0);

        for step in 0..steps {
            let (k1x, k1y) = self.derivatives(x, y);
            let (k2x, k2y) = self.derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
            let (k3x, k3y) = self.derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
            let (k4x, k4y) = self.derivatives(x + dt * k3x, y + dt * k3y);

            x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);

            points.push(x);
            points.push(y);
            times.push((step + 1) as f64 * dt);
        }

        WasmTrajectory {
            points,
            times,
            dim: 2,
        }
    }

    fn derivatives(&self, x: f64, y: f64) -> (f64, f64) {
        (y, self.mu * (1.0 - x * x) * y - x)
    }
}

/// WASM-compatible Duffing oscillator
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct WasmDuffing {
    delta: f64,
    alpha: f64,
    beta: f64,
}

#[wasm_bindgen]
impl WasmDuffing {
    /// Create a new Duffing oscillator
    #[wasm_bindgen(constructor)]
    pub fn new(delta: f64, alpha: f64, beta: f64) -> Self {
        Self { delta, alpha, beta }
    }

    /// Create a double-well Duffing oscillator
    #[wasm_bindgen(js_name = doubleWell)]
    pub fn double_well() -> Self {
        Self::new(0.1, -1.0, 1.0)
    }

    /// Simulate a trajectory
    #[wasm_bindgen]
    pub fn simulate(&self, initial: Vec<f64>, dt: f64, steps: usize) -> WasmTrajectory {
        let mut points = Vec::with_capacity((steps + 1) * 2);
        let mut times = Vec::with_capacity(steps + 1);

        let mut x = initial.first().copied().unwrap_or(1.0);
        let mut y = initial.get(1).copied().unwrap_or(0.0);

        points.push(x);
        points.push(y);
        times.push(0.0);

        for step in 0..steps {
            let (k1x, k1y) = self.derivatives(x, y);
            let (k2x, k2y) = self.derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y);
            let (k3x, k3y) = self.derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y);
            let (k4x, k4y) = self.derivatives(x + dt * k3x, y + dt * k3y);

            x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);

            points.push(x);
            points.push(y);
            times.push((step + 1) as f64 * dt);
        }

        WasmTrajectory {
            points,
            times,
            dim: 2,
        }
    }

    fn derivatives(&self, x: f64, y: f64) -> (f64, f64) {
        (y, -self.delta * y - self.alpha * x - self.beta * x * x * x)
    }
}

/// WASM-compatible Rössler system
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct WasmRossler {
    a: f64,
    b: f64,
    c: f64,
}

#[wasm_bindgen]
impl WasmRossler {
    /// Create a new Rössler system
    #[wasm_bindgen(constructor)]
    pub fn new(a: f64, b: f64, c: f64) -> Self {
        Self { a, b, c }
    }

    /// Create the classic Rössler system
    #[wasm_bindgen]
    pub fn classic() -> Self {
        Self::new(0.2, 0.2, 5.7)
    }

    /// Simulate a trajectory
    #[wasm_bindgen]
    pub fn simulate(&self, initial: Vec<f64>, dt: f64, steps: usize) -> WasmTrajectory {
        let mut points = Vec::with_capacity((steps + 1) * 3);
        let mut times = Vec::with_capacity(steps + 1);

        let mut x = initial.first().copied().unwrap_or(1.0);
        let mut y = initial.get(1).copied().unwrap_or(1.0);
        let mut z = initial.get(2).copied().unwrap_or(1.0);

        points.push(x);
        points.push(y);
        points.push(z);
        times.push(0.0);

        for step in 0..steps {
            let (k1x, k1y, k1z) = self.derivatives(x, y, z);
            let (k2x, k2y, k2z) =
                self.derivatives(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z);
            let (k3x, k3y, k3z) =
                self.derivatives(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z);
            let (k4x, k4y, k4z) = self.derivatives(x + dt * k3x, y + dt * k3y, z + dt * k3z);

            x += dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
            y += dt / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
            z += dt / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z);

            points.push(x);
            points.push(y);
            points.push(z);
            times.push((step + 1) as f64 * dt);
        }

        WasmTrajectory {
            points,
            times,
            dim: 3,
        }
    }

    fn derivatives(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        (-y - z, x + self.a * y, self.b + z * (x - self.c))
    }
}

/// WASM-compatible simple pendulum
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct WasmPendulum {
    g_over_l: f64,
}

#[wasm_bindgen]
impl WasmPendulum {
    /// Create a new simple pendulum
    #[wasm_bindgen(constructor)]
    pub fn new(g_over_l: f64) -> Self {
        Self { g_over_l }
    }

    /// Create a standard pendulum (g/L = 1)
    #[wasm_bindgen]
    pub fn standard() -> Self {
        Self::new(1.0)
    }

    /// Simulate a trajectory
    #[wasm_bindgen]
    pub fn simulate(&self, initial: Vec<f64>, dt: f64, steps: usize) -> WasmTrajectory {
        let mut points = Vec::with_capacity((steps + 1) * 2);
        let mut times = Vec::with_capacity(steps + 1);

        let mut theta = initial
            .first()
            .copied()
            .unwrap_or(std::f64::consts::PI / 4.0);
        let mut omega = initial.get(1).copied().unwrap_or(0.0);

        points.push(theta);
        points.push(omega);
        times.push(0.0);

        for step in 0..steps {
            let (k1t, k1o) = self.derivatives(theta, omega);
            let (k2t, k2o) = self.derivatives(theta + 0.5 * dt * k1t, omega + 0.5 * dt * k1o);
            let (k3t, k3o) = self.derivatives(theta + 0.5 * dt * k2t, omega + 0.5 * dt * k2o);
            let (k4t, k4o) = self.derivatives(theta + dt * k3t, omega + dt * k3o);

            theta += dt / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t);
            omega += dt / 6.0 * (k1o + 2.0 * k2o + 2.0 * k3o + k4o);

            points.push(theta);
            points.push(omega);
            times.push((step + 1) as f64 * dt);
        }

        WasmTrajectory {
            points,
            times,
            dim: 2,
        }
    }

    fn derivatives(&self, theta: f64, omega: f64) -> (f64, f64) {
        (omega, -self.g_over_l * theta.sin())
    }
}

/// Compute a flow field for phase portrait visualization
#[wasm_bindgen(js_name = computeFlowField)]
pub fn compute_flow_field(
    system_type: &str,
    params: Vec<f64>,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    resolution: usize,
) -> Vec<f64> {
    let mut result = Vec::with_capacity(resolution * resolution * 2);

    let dx = (x_max - x_min) / (resolution - 1) as f64;
    let dy = (y_max - y_min) / (resolution - 1) as f64;

    for j in 0..resolution {
        for i in 0..resolution {
            let x = x_min + i as f64 * dx;
            let y = y_min + j as f64 * dy;

            let (vx, vy) = match system_type {
                "vanderpol" => {
                    let mu = params.first().copied().unwrap_or(1.0);
                    (y, mu * (1.0 - x * x) * y - x)
                }
                "duffing" => {
                    let delta = params.first().copied().unwrap_or(0.1);
                    let alpha = params.get(1).copied().unwrap_or(-1.0);
                    let beta = params.get(2).copied().unwrap_or(1.0);
                    (y, -delta * y - alpha * x - beta * x * x * x)
                }
                "pendulum" => {
                    let g_over_l = params.first().copied().unwrap_or(1.0);
                    (y, -g_over_l * x.sin())
                }
                "harmonic" => {
                    let omega = params.first().copied().unwrap_or(1.0);
                    (y, -omega * omega * x)
                }
                _ => (y, -x), // Default: simple harmonic
            };

            result.push(vx);
            result.push(vy);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_lorenz() {
        let lorenz = WasmLorenz::classic();
        assert!((lorenz.sigma() - 10.0).abs() < 1e-10);
        assert!((lorenz.rho() - 28.0).abs() < 1e-10);

        let trajectory = lorenz.simulate(vec![1.0, 1.0, 1.0], 0.01, 100);
        assert_eq!(trajectory.length(), 101);
        assert_eq!(trajectory.dimension(), 3);
    }

    #[test]
    fn test_wasm_vanderpol() {
        let vdp = WasmVanDerPol::new(1.0);
        let trajectory = vdp.simulate(vec![2.0, 0.0], 0.01, 100);
        assert_eq!(trajectory.length(), 101);
        assert_eq!(trajectory.dimension(), 2);
    }

    #[test]
    fn test_wasm_duffing() {
        let duffing = WasmDuffing::double_well();
        let trajectory = duffing.simulate(vec![1.0, 0.0], 0.01, 100);
        assert_eq!(trajectory.length(), 101);
    }

    #[test]
    fn test_wasm_rossler() {
        let rossler = WasmRossler::classic();
        let trajectory = rossler.simulate(vec![1.0, 1.0, 1.0], 0.01, 100);
        assert_eq!(trajectory.length(), 101);
        assert_eq!(trajectory.dimension(), 3);
    }

    #[test]
    fn test_wasm_pendulum() {
        let pendulum = WasmPendulum::standard();
        let trajectory = pendulum.simulate(vec![0.5, 0.0], 0.01, 100);
        assert_eq!(trajectory.length(), 101);
    }

    #[test]
    fn test_flow_field() {
        let field = compute_flow_field("vanderpol", vec![1.0], -3.0, 3.0, -3.0, 3.0, 10);
        assert_eq!(field.len(), 10 * 10 * 2);
    }

    #[test]
    fn test_trajectory_get_point() {
        let lorenz = WasmLorenz::classic();
        let trajectory = lorenz.simulate(vec![1.0, 1.0, 1.0], 0.01, 10);

        let point = trajectory.get_point(0).unwrap();
        assert_eq!(point.len(), 3);
        assert!((point[0] - 1.0).abs() < 1e-10);

        let final_state = trajectory.get_final_state().unwrap();
        assert_eq!(final_state.len(), 3);
    }
}
