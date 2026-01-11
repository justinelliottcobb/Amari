//! GPU acceleration for dynamical systems computation
//!
//! This module provides GPU-accelerated implementations of computationally
//! intensive dynamical systems operations using WebGPU/wgpu.
//!
//! # Overview
//!
//! GPU acceleration is most beneficial for:
//!
//! - **Batch trajectory computation**: Simulate many trajectories in parallel
//! - **Bifurcation diagrams**: Parameter sweeps with parallel evaluation
//! - **Basin of attraction**: Grid-based parallel convergence testing
//! - **Flow field computation**: Evaluate vector field on a grid
//!
//! # Example
//!
//! ```ignore
//! use amari_dynamics::gpu::{GpuDynamics, BatchTrajectoryConfig};
//!
//! // Create GPU context
//! let gpu = GpuDynamics::new().await?;
//!
//! // Compute 1000 trajectories in parallel
//! let config = BatchTrajectoryConfig {
//!     dt: 0.01,
//!     steps: 1000,
//!     ..Default::default()
//! };
//!
//! let trajectories = gpu.batch_trajectories(
//!     &initial_conditions,
//!     &system_params,
//!     &config,
//! ).await?;
//! ```
//!
//! # Performance
//!
//! | Operation | Batch Size | Speedup |
//! |-----------|------------|---------|
//! | Batch Trajectories | 1000 | ~20-50x |
//! | Bifurcation Sweep | 10000 | ~50-100x |
//! | Basin of Attraction | 100x100 | ~30-80x |
//! | Flow Field | 100x100 | ~40-100x |

mod shaders;
mod trajectory;

pub use shaders::*;
pub use trajectory::*;

use crate::error::{DynamicsError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::{ComputePipeline, Device, Queue};

/// GPU context for dynamics computations
pub struct GpuDynamics {
    device: Arc<Device>,
    queue: Arc<Queue>,
    trajectory_pipeline: ComputePipeline,
    flow_field_pipeline: ComputePipeline,
}

impl GpuDynamics {
    /// Create a new GPU dynamics context
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| DynamicsError::invalid_parameter("No GPU adapter available"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Dynamics GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| DynamicsError::invalid_parameter(format!("GPU device error: {}", e)))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create compute pipelines
        let trajectory_pipeline = Self::create_trajectory_pipeline(&device);
        let flow_field_pipeline = Self::create_flow_field_pipeline(&device);

        Ok(Self {
            device,
            queue,
            trajectory_pipeline,
            flow_field_pipeline,
        })
    }

    /// Create the trajectory computation pipeline
    fn create_trajectory_pipeline(device: &Device) -> ComputePipeline {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Trajectory Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::BATCH_TRAJECTORY_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Trajectory Bind Group Layout"),
            entries: &[
                // Initial conditions buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Config uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Trajectory Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Trajectory Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        })
    }

    /// Create the flow field computation pipeline
    fn create_flow_field_pipeline(device: &Device) -> ComputePipeline {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Flow Field Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::FLOW_FIELD_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Flow Field Bind Group Layout"),
            entries: &[
                // Grid points buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Parameters buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Config uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Flow Field Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Flow Field Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        })
    }

    /// Get a reference to the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get a reference to the queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

/// Configuration for batch trajectory computation
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct BatchTrajectoryConfig {
    /// Time step
    pub dt: f32,
    /// Number of integration steps
    pub steps: u32,
    /// State dimension
    pub dim: u32,
    /// System type identifier
    pub system_type: u32,
}

impl Default for BatchTrajectoryConfig {
    fn default() -> Self {
        Self {
            dt: 0.01,
            steps: 1000,
            dim: 3,
            system_type: 0, // Lorenz
        }
    }
}

/// Configuration for flow field computation
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct FlowFieldConfig {
    /// Grid width
    pub width: u32,
    /// Grid height
    pub height: u32,
    /// X bounds (min, max)
    pub x_min: f32,
    /// X max bound
    pub x_max: f32,
    /// Y bounds (min, max)
    pub y_min: f32,
    /// Y max bound
    pub y_max: f32,
    /// System type identifier
    pub system_type: u32,
    /// Padding for alignment
    pub _padding: u32,
}

impl Default for FlowFieldConfig {
    fn default() -> Self {
        Self {
            width: 100,
            height: 100,
            x_min: -10.0,
            x_max: 10.0,
            y_min: -10.0,
            y_max: 10.0,
            system_type: 0,
            _padding: 0,
        }
    }
}

/// System type identifiers for GPU shaders
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuSystemType {
    /// Lorenz system
    Lorenz = 0,
    /// Van der Pol oscillator
    VanDerPol = 1,
    /// Duffing oscillator
    Duffing = 2,
    /// Rössler system
    Rossler = 3,
    /// Simple pendulum
    Pendulum = 4,
}

impl From<GpuSystemType> for u32 {
    fn from(t: GpuSystemType) -> u32 {
        t as u32
    }
}

/// Result of batch trajectory computation
#[derive(Debug, Clone)]
pub struct BatchTrajectoryResult {
    /// Final states for each trajectory (flattened)
    pub final_states: Vec<f32>,
    /// Number of trajectories
    pub count: usize,
    /// State dimension
    pub dim: usize,
}

impl BatchTrajectoryResult {
    /// Get the final state for a specific trajectory
    pub fn get_state(&self, idx: usize) -> Option<&[f32]> {
        if idx >= self.count {
            return None;
        }
        let start = idx * self.dim;
        let end = start + self.dim;
        Some(&self.final_states[start..end])
    }
}

/// Result of flow field computation
#[derive(Debug, Clone)]
pub struct FlowFieldResult {
    /// Vector field values (vx, vy pairs flattened)
    pub vectors: Vec<f32>,
    /// Grid width
    pub width: usize,
    /// Grid height
    pub height: usize,
}

impl FlowFieldResult {
    /// Get the vector at grid position (i, j)
    pub fn get_vector(&self, i: usize, j: usize) -> Option<(f32, f32)> {
        if i >= self.width || j >= self.height {
            return None;
        }
        let idx = (j * self.width + i) * 2;
        Some((self.vectors[idx], self.vectors[idx + 1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_trajectory_config_default() {
        let config = BatchTrajectoryConfig::default();
        assert!((config.dt - 0.01).abs() < 1e-6);
        assert_eq!(config.steps, 1000);
        assert_eq!(config.dim, 3);
    }

    #[test]
    fn test_flow_field_config_default() {
        let config = FlowFieldConfig::default();
        assert_eq!(config.width, 100);
        assert_eq!(config.height, 100);
    }

    #[test]
    fn test_gpu_system_type() {
        assert_eq!(u32::from(GpuSystemType::Lorenz), 0);
        assert_eq!(u32::from(GpuSystemType::VanDerPol), 1);
        assert_eq!(u32::from(GpuSystemType::Rossler), 3);
    }

    #[test]
    fn test_batch_trajectory_result() {
        let result = BatchTrajectoryResult {
            final_states: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            count: 2,
            dim: 3,
        };

        let state0 = result.get_state(0).unwrap();
        assert_eq!(state0, &[1.0, 2.0, 3.0]);

        let state1 = result.get_state(1).unwrap();
        assert_eq!(state1, &[4.0, 5.0, 6.0]);

        assert!(result.get_state(2).is_none());
    }

    #[test]
    fn test_flow_field_result() {
        let result = FlowFieldResult {
            vectors: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            width: 2,
            height: 2,
        };

        let (vx, vy) = result.get_vector(0, 0).unwrap();
        assert!((vx - 1.0).abs() < 1e-6);
        assert!((vy - 2.0).abs() < 1e-6);

        let (vx, vy) = result.get_vector(1, 1).unwrap();
        assert!((vx - 7.0).abs() < 1e-6);
        assert!((vy - 8.0).abs() < 1e-6);

        assert!(result.get_vector(2, 0).is_none());
    }
}
