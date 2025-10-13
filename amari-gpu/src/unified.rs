//! Unified GPU acceleration infrastructure for all mathematical domains
//!
//! This module provides a common interface and infrastructure for GPU acceleration
//! across tropical algebra, automatic differentiation, fusion systems, and other
//! mathematical domains in the Amari library.

use crate::GpuError;
use std::collections::HashMap;
use thiserror::Error;
use wgpu::util::DeviceExt;

#[derive(Error, Debug)]
pub enum UnifiedGpuError {
    #[error("GPU error: {0}")]
    Gpu(#[from] GpuError),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),
}

pub type UnifiedGpuResult<T> = Result<T, UnifiedGpuError>;

/// Universal trait for GPU-accelerated mathematical operations
pub trait GpuAccelerated<T> {
    /// Convert data to GPU buffer format
    fn to_gpu_buffer(&self, context: &GpuContext) -> UnifiedGpuResult<wgpu::Buffer>;

    /// Reconstruct data from GPU buffer
    fn from_gpu_buffer(buffer: &wgpu::Buffer, context: &GpuContext) -> UnifiedGpuResult<T>;

    /// Execute GPU operation
    fn gpu_operation(
        &self,
        operation: &str,
        context: &GpuContext,
        params: &GpuOperationParams,
    ) -> UnifiedGpuResult<T>;
}

/// GPU operation parameters for flexible operation dispatch
#[derive(Debug, Clone)]
pub struct GpuOperationParams {
    /// Operation-specific parameters
    pub params: HashMap<String, GpuParam>,
    /// Batch size for operations
    pub batch_size: usize,
    /// Workgroup size for compute shaders
    pub workgroup_size: (u32, u32, u32),
}

/// Parameter types for GPU operations
#[derive(Debug, Clone)]
pub enum GpuParam {
    Float(f32),
    Double(f64),
    Integer(i32),
    UnsignedInteger(u32),
    Buffer(String), // Buffer identifier
    Array(Vec<f32>),
}

impl Default for GpuOperationParams {
    fn default() -> Self {
        Self {
            params: HashMap::new(),
            batch_size: 1,
            workgroup_size: (1, 1, 1),
        }
    }
}

/// Unified GPU context managing device, queue, and shader cache
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    shader_cache: HashMap<String, wgpu::ComputePipeline>,
    #[allow(dead_code)]
    buffer_pool: GpuBufferPool,
}

impl GpuContext {
    /// Initialize GPU context with WebGPU
    pub async fn new() -> UnifiedGpuResult<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                UnifiedGpuError::Gpu(GpuError::InitializationError(
                    "No GPU adapter found".to_string(),
                ))
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Amari Unified GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| UnifiedGpuError::Gpu(GpuError::InitializationError(e.to_string())))?;

        Ok(Self {
            device,
            queue,
            shader_cache: HashMap::new(),
            buffer_pool: GpuBufferPool::new(),
        })
    }

    /// Get or compile compute shader
    pub fn get_compute_pipeline(
        &mut self,
        shader_key: &str,
        shader_source: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> UnifiedGpuResult<&wgpu::ComputePipeline> {
        if !self.shader_cache.contains_key(shader_key) {
            let shader_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("{} Shader", shader_key)),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{} Pipeline Layout", shader_key)),
                        bind_group_layouts: &[bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&format!("{} Pipeline", shader_key)),
                        layout: Some(&pipeline_layout),
                        module: &shader_module,
                        entry_point: "main",
                    });

            self.shader_cache
                .insert(shader_key.to_string(), compute_pipeline);
        }

        Ok(self
            .shader_cache
            .get(shader_key)
            .expect("Pipeline should exist"))
    }

    /// Create buffer with data
    pub fn create_buffer_with_data<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    /// Create empty buffer
    pub fn create_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Execute compute shader
    pub fn execute_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroup_count: (u32, u32, u32),
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(
                workgroup_count.0,
                workgroup_count.1,
                workgroup_count.2,
            );
        }

        self.queue.submit([encoder.finish()]);
    }

    /// Read buffer data back to CPU
    pub async fn read_buffer<T: bytemuck::Pod + Clone>(
        &self,
        buffer: &wgpu::Buffer,
        size: u64,
    ) -> UnifiedGpuResult<Vec<T>> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.queue.submit([encoder.finish()]);

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.await
            .map_err(|_| UnifiedGpuError::InvalidOperation("Buffer read timeout".to_string()))?
            .map_err(|e| UnifiedGpuError::InvalidOperation(format!("Buffer map failed: {}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

/// GPU buffer pool for efficient memory management
pub struct GpuBufferPool {
    _pools: HashMap<String, Vec<wgpu::Buffer>>, // Future: implement buffer pooling
}

impl GpuBufferPool {
    pub fn new() -> Self {
        Self {
            _pools: HashMap::new(),
        }
    }

    // Future: Add buffer pooling methods
    // pub fn get_buffer(&mut self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer
    // pub fn return_buffer(&mut self, buffer: wgpu::Buffer)
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Smart GPU/CPU dispatch based on workload characteristics
pub struct GpuDispatcher {
    gpu_context: Option<GpuContext>,
    cpu_threshold: usize,
    gpu_threshold: usize,
}

impl GpuDispatcher {
    /// Create new dispatcher with GPU context
    pub async fn new() -> UnifiedGpuResult<Self> {
        let gpu_context = (GpuContext::new().await).ok(); // Graceful fallback to CPU-only

        Ok(Self {
            gpu_context,
            cpu_threshold: 100,  // Use CPU for small workloads
            gpu_threshold: 1000, // Use GPU for large workloads
        })
    }

    /// Determine optimal compute strategy
    pub fn should_use_gpu(&self, workload_size: usize) -> bool {
        self.gpu_context.is_some()
            && workload_size >= self.cpu_threshold
            && workload_size >= self.gpu_threshold
    }

    /// Execute operation with optimal strategy
    pub async fn execute<T, F, G>(&mut self, workload_size: usize, gpu_op: G, cpu_op: F) -> T
    where
        F: FnOnce() -> T,
        G: FnOnce(&mut GpuContext) -> UnifiedGpuResult<T>,
    {
        if self.should_use_gpu(workload_size) {
            if let Some(ref mut ctx) = self.gpu_context {
                if let Ok(result) = gpu_op(ctx) {
                    return result;
                }
            }
        }

        // Fallback to CPU
        cpu_op()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_context_creation() {
        // Test should pass even without GPU (graceful fallback)
        let _result = GpuContext::new().await;
        // Don't assert success since GPU might not be available in CI
    }

    #[tokio::test]
    async fn test_gpu_dispatcher() {
        let dispatcher = GpuDispatcher::new().await;
        assert!(dispatcher.is_ok());
    }

    #[test]
    fn test_gpu_operation_params() {
        let mut params = GpuOperationParams::default();
        params
            .params
            .insert("scale".to_string(), GpuParam::Float(2.0));
        params.batch_size = 100;

        assert_eq!(params.batch_size, 100);
        match params.params.get("scale") {
            Some(GpuParam::Float(val)) => assert_eq!(*val, 2.0),
            _ => panic!("Expected float parameter"),
        }
    }
}
