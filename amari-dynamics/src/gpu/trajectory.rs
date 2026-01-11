//! GPU-accelerated trajectory computation
//!
//! This module provides methods for computing trajectories on the GPU.

use super::{
    BatchTrajectoryConfig, BatchTrajectoryResult, FlowFieldConfig, FlowFieldResult, GpuDynamics,
};
use crate::error::{DynamicsError, Result};
use wgpu::util::DeviceExt;

impl GpuDynamics {
    /// Compute multiple trajectories in parallel
    ///
    /// # Arguments
    ///
    /// * `initial_conditions` - Flattened initial conditions (dim values per trajectory)
    /// * `params` - System parameters (e.g., [sigma, rho, beta] for Lorenz)
    /// * `config` - Trajectory computation configuration
    ///
    /// # Returns
    ///
    /// Final states of all trajectories
    pub async fn batch_trajectories(
        &self,
        initial_conditions: &[f32],
        params: &[f32],
        config: &BatchTrajectoryConfig,
    ) -> Result<BatchTrajectoryResult> {
        let dim = config.dim as usize;
        let num_trajectories = initial_conditions.len() / dim;

        if initial_conditions.len() % dim != 0 {
            return Err(DynamicsError::invalid_parameter(
                "Initial conditions length must be divisible by dimension",
            ));
        }

        // Create buffers
        let ic_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Initial Conditions Buffer"),
                contents: bytemuck::cast_slice(initial_conditions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Ensure params has at least 3 elements
        let mut params_padded = vec![0.0f32; 4];
        for (i, &p) in params.iter().take(4).enumerate() {
            params_padded[i] = p;
        }

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parameters Buffer"),
                contents: bytemuck::cast_slice(&params_padded),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (num_trajectories * dim * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: bytemuck::bytes_of(config),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = self.trajectory_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Trajectory Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ic_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Trajectory Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Trajectory Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.trajectory_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (64 threads per workgroup)
            let workgroups = (num_trajectories as u32).div_ceil(64);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await
            .map_err(|_| DynamicsError::numerical_instability("GPU", "Buffer map cancelled"))?
            .map_err(|e| {
                DynamicsError::numerical_instability("GPU", format!("Buffer map error: {:?}", e))
            })?;

        let data = buffer_slice.get_mapped_range();
        let final_states: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        Ok(BatchTrajectoryResult {
            final_states,
            count: num_trajectories,
            dim,
        })
    }

    /// Compute flow field on a 2D grid
    ///
    /// # Arguments
    ///
    /// * `params` - System parameters
    /// * `config` - Flow field configuration
    ///
    /// # Returns
    ///
    /// Vector field values on the grid
    pub async fn compute_flow_field(
        &self,
        params: &[f32],
        config: &FlowFieldConfig,
    ) -> Result<FlowFieldResult> {
        let width = config.width as usize;
        let height = config.height as usize;
        let num_points = width * height;

        // Create grid points (not used directly in shader, but could be for custom systems)
        let grid_size = num_points * 2 * std::mem::size_of::<f32>();
        let grid_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Buffer"),
            size: grid_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Ensure params has at least 3 elements
        let mut params_padded = vec![0.0f32; 4];
        for (i, &p) in params.iter().take(4).enumerate() {
            params_padded[i] = p;
        }

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parameters Buffer"),
                contents: bytemuck::cast_slice(&params_padded),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (num_points * 2 * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: bytemuck::bytes_of(config),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group
        let bind_group_layout = self.flow_field_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Flow Field Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Flow Field Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Flow Field Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.flow_field_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (8x8 threads per workgroup)
            let workgroups_x = (width as u32).div_ceil(8);
            let workgroups_y = (height as u32).div_ceil(8);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        // Copy output to staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await
            .map_err(|_| DynamicsError::numerical_instability("GPU", "Buffer map cancelled"))?
            .map_err(|e| {
                DynamicsError::numerical_instability("GPU", format!("Buffer map error: {:?}", e))
            })?;

        let data = buffer_slice.get_mapped_range();
        let vectors: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        Ok(FlowFieldResult {
            vectors,
            width,
            height,
        })
    }
}

/// Adaptive GPU/CPU dispatcher for trajectory computation
pub struct AdaptiveDynamics {
    gpu: Option<GpuDynamics>,
    /// Threshold for GPU dispatch (number of trajectories)
    pub gpu_threshold: usize,
}

impl AdaptiveDynamics {
    /// Create a new adaptive dispatcher
    ///
    /// Attempts to create GPU context; falls back to CPU-only if unavailable.
    pub async fn new() -> Self {
        let gpu = GpuDynamics::new().await.ok();
        Self {
            gpu,
            gpu_threshold: 100, // Default: use GPU for 100+ trajectories
        }
    }

    /// Create a CPU-only dispatcher
    pub fn cpu_only() -> Self {
        Self {
            gpu: None,
            gpu_threshold: usize::MAX,
        }
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Compute trajectories adaptively
    ///
    /// Uses GPU if available and batch size exceeds threshold.
    pub async fn batch_trajectories(
        &self,
        initial_conditions: &[f32],
        params: &[f32],
        config: &BatchTrajectoryConfig,
    ) -> Result<BatchTrajectoryResult> {
        let dim = config.dim as usize;
        let num_trajectories = initial_conditions.len() / dim;

        // Use GPU if available and batch size is large enough
        if let Some(ref gpu) = self.gpu {
            if num_trajectories >= self.gpu_threshold {
                return gpu
                    .batch_trajectories(initial_conditions, params, config)
                    .await;
            }
        }

        // Fall back to CPU computation
        Self::cpu_batch_trajectories(initial_conditions, params, config)
    }

    /// CPU fallback for batch trajectory computation
    fn cpu_batch_trajectories(
        initial_conditions: &[f32],
        params: &[f32],
        config: &BatchTrajectoryConfig,
    ) -> Result<BatchTrajectoryResult> {
        let dim = config.dim as usize;
        let num_trajectories = initial_conditions.len() / dim;
        let dt = config.dt as f64;
        let steps = config.steps as usize;

        let mut final_states = vec![0.0f32; num_trajectories * dim];

        // Process each trajectory
        for t in 0..num_trajectories {
            let offset = t * dim;

            if dim == 3 {
                // 3D system (Lorenz, Rössler)
                let mut x = initial_conditions[offset] as f64;
                let mut y = initial_conditions[offset + 1] as f64;
                let mut z = initial_conditions[offset + 2] as f64;

                let sigma = params.first().copied().unwrap_or(10.0) as f64;
                let rho = params.get(1).copied().unwrap_or(28.0) as f64;
                let beta = params.get(2).copied().unwrap_or(8.0 / 3.0) as f64;

                for _ in 0..steps {
                    // RK4 step for Lorenz
                    let (x1, y1, z1) = Self::lorenz_derivative(x, y, z, sigma, rho, beta);
                    let (x2, y2, z2) = Self::lorenz_derivative(
                        x + 0.5 * dt * x1,
                        y + 0.5 * dt * y1,
                        z + 0.5 * dt * z1,
                        sigma,
                        rho,
                        beta,
                    );
                    let (x3, y3, z3) = Self::lorenz_derivative(
                        x + 0.5 * dt * x2,
                        y + 0.5 * dt * y2,
                        z + 0.5 * dt * z2,
                        sigma,
                        rho,
                        beta,
                    );
                    let (x4, y4, z4) = Self::lorenz_derivative(
                        x + dt * x3,
                        y + dt * y3,
                        z + dt * z3,
                        sigma,
                        rho,
                        beta,
                    );

                    x += dt / 6.0 * (x1 + 2.0 * x2 + 2.0 * x3 + x4);
                    y += dt / 6.0 * (y1 + 2.0 * y2 + 2.0 * y3 + y4);
                    z += dt / 6.0 * (z1 + 2.0 * z2 + 2.0 * z3 + z4);
                }

                final_states[offset] = x as f32;
                final_states[offset + 1] = y as f32;
                final_states[offset + 2] = z as f32;
            } else {
                // 2D system (Van der Pol, etc.)
                let mut x = initial_conditions[offset] as f64;
                let mut y = initial_conditions[offset + 1] as f64;

                let mu = params.first().copied().unwrap_or(1.0) as f64;

                for _ in 0..steps {
                    // RK4 step for Van der Pol
                    let (x1, y1) = Self::vanderpol_derivative(x, y, mu);
                    let (x2, y2) =
                        Self::vanderpol_derivative(x + 0.5 * dt * x1, y + 0.5 * dt * y1, mu);
                    let (x3, y3) =
                        Self::vanderpol_derivative(x + 0.5 * dt * x2, y + 0.5 * dt * y2, mu);
                    let (x4, y4) = Self::vanderpol_derivative(x + dt * x3, y + dt * y3, mu);

                    x += dt / 6.0 * (x1 + 2.0 * x2 + 2.0 * x3 + x4);
                    y += dt / 6.0 * (y1 + 2.0 * y2 + 2.0 * y3 + y4);
                }

                final_states[offset] = x as f32;
                final_states[offset + 1] = y as f32;
            }
        }

        Ok(BatchTrajectoryResult {
            final_states,
            count: num_trajectories,
            dim,
        })
    }

    /// Lorenz system derivatives
    fn lorenz_derivative(
        x: f64,
        y: f64,
        z: f64,
        sigma: f64,
        rho: f64,
        beta: f64,
    ) -> (f64, f64, f64) {
        (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
    }

    /// Van der Pol oscillator derivatives
    fn vanderpol_derivative(x: f64, y: f64, mu: f64) -> (f64, f64) {
        (y, mu * (1.0 - x * x) * y - x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_batch_trajectories_lorenz() {
        // Test CPU fallback for Lorenz system
        let initial_conditions = vec![1.0f32, 1.0, 1.0, 0.1, 0.1, 0.1];
        let params = vec![10.0f32, 28.0, 8.0 / 3.0];
        let config = BatchTrajectoryConfig {
            dt: 0.01,
            steps: 100,
            dim: 3,
            system_type: 0,
        };

        let result =
            AdaptiveDynamics::cpu_batch_trajectories(&initial_conditions, &params, &config)
                .unwrap();

        assert_eq!(result.count, 2);
        assert_eq!(result.dim, 3);
        assert_eq!(result.final_states.len(), 6);

        // States should have evolved from initial conditions
        let state0 = result.get_state(0).unwrap();
        assert!(state0[0] != 1.0 || state0[1] != 1.0 || state0[2] != 1.0);
    }

    #[test]
    fn test_cpu_batch_trajectories_vanderpol() {
        // Test CPU fallback for Van der Pol
        let initial_conditions = vec![2.0f32, 0.0, 0.5, 0.5];
        let params = vec![1.0f32];
        let config = BatchTrajectoryConfig {
            dt: 0.01,
            steps: 100,
            dim: 2,
            system_type: 1,
        };

        let result =
            AdaptiveDynamics::cpu_batch_trajectories(&initial_conditions, &params, &config)
                .unwrap();

        assert_eq!(result.count, 2);
        assert_eq!(result.dim, 2);
        assert_eq!(result.final_states.len(), 4);
    }

    #[test]
    fn test_adaptive_cpu_only() {
        let adaptive = AdaptiveDynamics::cpu_only();
        assert!(!adaptive.has_gpu());
    }
}
