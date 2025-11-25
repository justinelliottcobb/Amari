//! Geodesics and parallel transport on Riemannian manifolds
//!
//! This module implements geodesic equations and parallel transport,
//! which generalize the concept of "straight lines" to curved spaces.

use super::Connection;

/// Geodesic solver for Riemannian manifolds
///
/// Geodesics are curves that parallel transport their own tangent vector,
/// generalizing straight lines to curved spaces. They satisfy the geodesic equation:
///
/// ```text
/// d²x^i/dt² + Γ^i_jk (dx^j/dt)(dx^k/dt) = 0
/// ```
///
/// ## Properties
///
/// - Locally distance-minimizing curves
/// - Zero acceleration in the manifold's connection
/// - Free-fall trajectories in general relativity
///
/// ## Examples
///
/// ```
/// use amari_calculus::manifold::{MetricTensor, Connection, GeodesicSolver};
///
/// let metric = MetricTensor::<2>::sphere(1.0);
/// let connection = Connection::from_metric(&metric);
/// let solver = GeodesicSolver::new(&connection);
///
/// // Initial position and velocity
/// let pos = vec![std::f64::consts::PI / 4.0, 0.0];
/// let vel = vec![0.1, 0.0];
///
/// // Evolve geodesic for small time step
/// let (new_pos, new_vel) = solver.step(&pos, &vel, 0.01);
/// ```
pub struct GeodesicSolver<'a, const DIM: usize> {
    connection: &'a Connection<DIM>,
}

impl<'a, const DIM: usize> GeodesicSolver<'a, DIM> {
    /// Create a new geodesic solver
    ///
    /// # Arguments
    ///
    /// * `connection` - The connection (Christoffel symbols)
    pub fn new(connection: &'a Connection<DIM>) -> Self {
        Self { connection }
    }

    /// Compute geodesic acceleration at a point with given velocity
    ///
    /// Returns d²x^i/dt² = -Γ^i_jk v^j v^k
    ///
    /// # Arguments
    ///
    /// * `pos` - Current position x^i
    /// * `vel` - Current velocity dx^i/dt
    ///
    /// # Returns
    ///
    /// Acceleration vector d²x^i/dt²
    pub fn acceleration(&self, pos: &[f64], vel: &[f64]) -> Vec<f64> {
        let mut accel = vec![0.0; DIM];

        #[allow(clippy::needless_range_loop)]
        for i in 0..DIM {
            let mut acc_i = 0.0;

            // a^i = -Γ^i_jk v^j v^k
            #[allow(clippy::needless_range_loop)]
            for j in 0..DIM {
                #[allow(clippy::needless_range_loop)]
                for k in 0..DIM {
                    let gamma = self.connection.christoffel(i, j, k, pos);
                    acc_i -= gamma * vel[j] * vel[k];
                }
            }

            accel[i] = acc_i;
        }

        accel
    }

    /// Evolve geodesic by one time step using Runge-Kutta 4
    ///
    /// Integrates the geodesic equation using RK4 for numerical stability.
    ///
    /// # Arguments
    ///
    /// * `pos` - Current position
    /// * `vel` - Current velocity
    /// * `dt` - Time step
    ///
    /// # Returns
    ///
    /// (new_pos, new_vel) after time dt
    pub fn step(&self, pos: &[f64], vel: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
        // RK4 for position and velocity
        let k1_vel = vel.to_vec();
        let k1_acc = self.acceleration(pos, vel);

        let mid_pos1: Vec<f64> = pos
            .iter()
            .zip(&k1_vel)
            .map(|(p, v)| p + 0.5 * dt * v)
            .collect();
        let mid_vel1: Vec<f64> = vel
            .iter()
            .zip(&k1_acc)
            .map(|(v, a)| v + 0.5 * dt * a)
            .collect();

        let k2_vel = mid_vel1.clone();
        let k2_acc = self.acceleration(&mid_pos1, &mid_vel1);

        let mid_pos2: Vec<f64> = pos
            .iter()
            .zip(&k2_vel)
            .map(|(p, v)| p + 0.5 * dt * v)
            .collect();
        let mid_vel2: Vec<f64> = vel
            .iter()
            .zip(&k2_acc)
            .map(|(v, a)| v + 0.5 * dt * a)
            .collect();

        let k3_vel = mid_vel2.clone();
        let k3_acc = self.acceleration(&mid_pos2, &mid_vel2);

        let end_pos: Vec<f64> = pos.iter().zip(&k3_vel).map(|(p, v)| p + dt * v).collect();
        let end_vel: Vec<f64> = vel.iter().zip(&k3_acc).map(|(v, a)| v + dt * a).collect();

        let k4_vel = end_vel.clone();
        let k4_acc = self.acceleration(&end_pos, &end_vel);

        // Combine RK4 contributions
        let new_pos: Vec<f64> = pos
            .iter()
            .enumerate()
            .map(|(i, p)| {
                p + (dt / 6.0) * (k1_vel[i] + 2.0 * k2_vel[i] + 2.0 * k3_vel[i] + k4_vel[i])
            })
            .collect();

        let new_vel: Vec<f64> = vel
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v + (dt / 6.0) * (k1_acc[i] + 2.0 * k2_acc[i] + 2.0 * k3_acc[i] + k4_acc[i])
            })
            .collect();

        (new_pos, new_vel)
    }

    /// Compute full geodesic trajectory
    ///
    /// # Arguments
    ///
    /// * `initial_pos` - Starting position
    /// * `initial_vel` - Starting velocity
    /// * `t_max` - Total integration time
    /// * `dt` - Time step
    ///
    /// # Returns
    ///
    /// Vector of (position, velocity) pairs at each time step
    pub fn trajectory(
        &self,
        initial_pos: &[f64],
        initial_vel: &[f64],
        t_max: f64,
        dt: f64,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let num_steps = (t_max / dt) as usize;
        let mut trajectory = Vec::with_capacity(num_steps + 1);

        let mut pos = initial_pos.to_vec();
        let mut vel = initial_vel.to_vec();

        trajectory.push((pos.clone(), vel.clone()));

        for _ in 0..num_steps {
            let (new_pos, new_vel) = self.step(&pos, &vel, dt);
            pos = new_pos;
            vel = new_vel;
            trajectory.push((pos.clone(), vel.clone()));
        }

        trajectory
    }
}

/// Parallel transport along a curve
///
/// Parallel transport moves a vector along a curve while keeping it
/// "as constant as possible" relative to the connection. The equation is:
///
/// ```text
/// ∇_γ'(t) V = dV^i/dt + Γ^i_jk V^j γ'^k = 0
/// ```
///
/// ## Properties
///
/// - Preserves vector length (for metric connections)
/// - Preserves angles between vectors
/// - Path-dependent in curved spaces (holonomy)
///
/// ## Applications
///
/// - Foucault pendulum (parallel transport on Earth)
/// - Berry phase in quantum mechanics
/// - Frame dragging in general relativity
pub struct ParallelTransport<'a, const DIM: usize> {
    connection: &'a Connection<DIM>,
}

impl<'a, const DIM: usize> ParallelTransport<'a, DIM> {
    /// Create a new parallel transport solver
    pub fn new(connection: &'a Connection<DIM>) -> Self {
        Self { connection }
    }

    /// Compute time derivative of vector under parallel transport
    ///
    /// Returns dV^i/dt = -Γ^i_jk V^j γ'^k
    ///
    /// # Arguments
    ///
    /// * `pos` - Current position on curve
    /// * `tangent` - Tangent vector to curve γ'^k
    /// * `vector` - Vector being transported V^j
    pub fn rate(&self, pos: &[f64], tangent: &[f64], vector: &[f64]) -> Vec<f64> {
        let mut rate = vec![0.0; DIM];

        #[allow(clippy::needless_range_loop)]
        for i in 0..DIM {
            let mut r_i = 0.0;

            // dV^i/dt = -Γ^i_jk V^j γ'^k
            #[allow(clippy::needless_range_loop)]
            for j in 0..DIM {
                #[allow(clippy::needless_range_loop)]
                for k in 0..DIM {
                    let gamma = self.connection.christoffel(i, j, k, pos);
                    r_i -= gamma * vector[j] * tangent[k];
                }
            }

            rate[i] = r_i;
        }

        rate
    }

    /// Parallel transport vector along geodesic for one step
    ///
    /// # Arguments
    ///
    /// * `pos` - Current position
    /// * `tangent` - Tangent to curve (geodesic velocity)
    /// * `vector` - Vector to transport
    /// * `dt` - Time step
    ///
    /// # Returns
    ///
    /// Transported vector after time dt
    pub fn step(&self, pos: &[f64], tangent: &[f64], vector: &[f64], dt: f64) -> Vec<f64> {
        // Simple Euler integration for parallel transport
        let rate = self.rate(pos, tangent, vector);

        vector.iter().zip(&rate).map(|(v, r)| v + dt * r).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::MetricTensor;

    #[test]
    fn test_euclidean_geodesic() {
        // In Euclidean space, geodesics are straight lines
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);
        let solver = GeodesicSolver::new(&connection);

        let pos = vec![0.0, 0.0];
        let vel = vec![1.0, 1.0];

        // Acceleration should be zero in flat space
        let accel = solver.acceleration(&pos, &vel);
        for a in &accel {
            assert!(
                a.abs() < 1e-10,
                "Euclidean geodesic should have zero acceleration, got {}",
                a
            );
        }

        // Step should give linear motion
        let (new_pos, new_vel) = solver.step(&pos, &vel, 0.1);

        // Position should increase linearly: x(t) = x0 + v*t
        assert!((new_pos[0] - 0.1).abs() < 1e-6);
        assert!((new_pos[1] - 0.1).abs() < 1e-6);

        // Velocity should remain constant
        assert!((new_vel[0] - 1.0).abs() < 1e-6);
        assert!((new_vel[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sphere_geodesic_acceleration() {
        // On a sphere, geodesics curve
        let metric = MetricTensor::<2>::sphere(1.0);
        let connection = Connection::from_metric(&metric);
        let solver = GeodesicSolver::new(&connection);

        let pos = vec![std::f64::consts::PI / 4.0, 0.0];
        let vel = vec![0.0, 1.0]; // Moving in φ direction

        let accel = solver.acceleration(&pos, &vel);

        // Should have non-zero acceleration due to curvature
        // (acceleration pulls toward equator)
        assert!(
            accel.iter().any(|a| a.abs() > 0.01),
            "Sphere geodesic should have non-zero acceleration"
        );
    }

    #[test]
    fn test_parallel_transport_euclidean() {
        // In Euclidean space, parallel transport doesn't change vectors
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);
        let transport = ParallelTransport::new(&connection);

        let pos = vec![1.0, 2.0];
        let tangent = vec![0.5, 0.5];
        let vector = vec![1.0, 0.0];

        // Rate of change should be zero
        let rate = transport.rate(&pos, &tangent, &vector);
        for r in &rate {
            assert!(
                r.abs() < 1e-10,
                "Euclidean parallel transport rate should be zero, got {}",
                r
            );
        }

        // Step should not change vector
        let new_vector = transport.step(&pos, &tangent, &vector, 0.1);
        for (v, nv) in vector.iter().zip(&new_vector) {
            assert!(
                (v - nv).abs() < 1e-6,
                "Vector should not change under Euclidean parallel transport"
            );
        }
    }

    #[test]
    fn test_parallel_transport_sphere() {
        // On sphere, parallel transport along non-geodesic changes vectors
        let metric = MetricTensor::<2>::sphere(1.0);
        let connection = Connection::from_metric(&metric);
        let transport = ParallelTransport::new(&connection);

        let pos = vec![std::f64::consts::PI / 4.0, 0.0];
        let tangent = vec![0.0, 1.0]; // Moving in φ direction
        let vector = vec![1.0, 0.0]; // Vector in θ direction

        let rate = transport.rate(&pos, &tangent, &vector);

        // Should have non-zero rate due to curvature
        assert!(
            rate.iter().any(|r| r.abs() > 0.01),
            "Sphere parallel transport should have non-zero rate"
        );
    }

    #[test]
    fn test_geodesic_trajectory() {
        // Test that we can compute a trajectory without panicking
        let metric = MetricTensor::<2>::euclidean();
        let connection = Connection::from_metric(&metric);
        let solver = GeodesicSolver::new(&connection);

        let pos = vec![0.0, 0.0];
        let vel = vec![1.0, 0.5];

        let traj = solver.trajectory(&pos, &vel, 1.0, 0.1);

        assert!(traj.len() > 1, "Trajectory should have multiple points");

        // For Euclidean space, final position should be approximately pos + vel * t
        let (final_pos, _) = &traj[traj.len() - 1];
        assert!((final_pos[0] - 1.0).abs() < 0.05);
        assert!((final_pos[1] - 0.5).abs() < 0.05);
    }
}
