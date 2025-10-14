//! Unified GPU acceleration infrastructure for all mathematical domains
//!
//! This module provides a common interface and infrastructure for GPU acceleration
//! across tropical algebra, automatic differentiation, fusion systems, and other
//! mathematical domains in the Amari library.

use crate::GpuError;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
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

/// Shared GPU context for efficient resource management across all crates
#[derive(Clone)]
pub struct SharedGpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_info: wgpu::AdapterInfo,
    buffer_pool: Arc<std::sync::Mutex<EnhancedGpuBufferPool>>,
    shader_cache: Arc<std::sync::Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>>,
    creation_time: Instant,
}

impl SharedGpuContext {
    /// Get the global shared GPU context (singleton pattern)
    /// Note: This creates a new context each time for now. In production,
    /// this would be a proper singleton with atomic initialization.
    pub async fn global() -> UnifiedGpuResult<&'static Self> {
        let context = Self::new().await?;
        // Leak the context to make it 'static - in production, this would be managed properly
        Ok(Box::leak(Box::new(context)))
    }

    /// Create a new shared GPU context
    async fn new() -> UnifiedGpuResult<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                UnifiedGpuError::InvalidOperation("No suitable GPU adapter found".into())
            })?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Shared Amari GPU Device"),
                    required_features: wgpu::Features::TIMESTAMP_QUERY,
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                UnifiedGpuError::InvalidOperation(format!("Device request failed: {:?}", e))
            })?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            buffer_pool: Arc::new(std::sync::Mutex::new(EnhancedGpuBufferPool::new())),
            shader_cache: Arc::new(std::sync::Mutex::new(HashMap::new())),
            creation_time: Instant::now(),
        })
    }

    /// Get the device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get the queue
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Get adapter info
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Get or create a buffer from the pool
    pub fn get_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> wgpu::Buffer {
        if let Ok(mut pool) = self.buffer_pool.lock() {
            pool.get_or_create(&self.device, size, usage, label)
        } else {
            // Fallback if mutex is poisoned
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            })
        }
    }

    /// Return a buffer to the pool for reuse
    pub fn return_buffer(&self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        if let Ok(mut pool) = self.buffer_pool.lock() {
            pool.return_buffer(buffer, size, usage);
        }
        // If mutex is poisoned, just drop the buffer
    }

    /// Get or create a compute pipeline from cache
    pub fn get_compute_pipeline(
        &self,
        shader_key: &str,
        shader_source: &str,
        entry_point: &str,
    ) -> UnifiedGpuResult<Arc<wgpu::ComputePipeline>> {
        let cache_key = format!("{}:{}", shader_key, entry_point);

        if let Ok(mut cache) = self.shader_cache.lock() {
            if let Some(pipeline) = cache.get(&cache_key) {
                return Ok(Arc::clone(pipeline));
            }

            // Create new pipeline
            let shader_module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&format!("{} Shader", shader_key)),
                    source: wgpu::ShaderSource::Wgsl(shader_source.into()),
                });

            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("{} Bind Group Layout", shader_key)),
                        entries: &[
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
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&format!("{} Pipeline Layout", shader_key)),
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{} Pipeline", shader_key)),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point,
                });

            let pipeline_arc = Arc::new(pipeline);
            cache.insert(cache_key, Arc::clone(&pipeline_arc));
            Ok(pipeline_arc)
        } else {
            Err(UnifiedGpuError::InvalidOperation(
                "Failed to access shader cache".into(),
            ))
        }
    }

    /// Get buffer pool statistics
    pub fn buffer_pool_stats(&self) -> BufferPoolStats {
        if let Ok(pool) = self.buffer_pool.lock() {
            pool.get_stats()
        } else {
            BufferPoolStats::default()
        }
    }

    /// Get uptime of this context
    pub fn uptime(&self) -> std::time::Duration {
        self.creation_time.elapsed()
    }

    /// Get optimal workgroup configuration for given operation type and data size
    pub fn get_optimal_workgroup(&self, operation: &str, data_size: usize) -> (u32, u32, u32) {
        match operation {
            "matrix_multiply" | "matrix_operation" => {
                // 2D workgroups optimized for matrix operations
                // Use larger workgroups for better occupancy
                (16, 16, 1)
            }
            "vector_operation" | "reduce" | "scan" => {
                // 1D operations - prefer large workgroups for coalesced memory access
                let workgroup_size = if data_size > 10000 {
                    256 // Large batches benefit from maximum occupancy
                } else if data_size > 1000 {
                    128 // Medium batches
                } else {
                    64 // Small batches
                };
                (workgroup_size, 1, 1)
            }
            "geometric_algebra" | "clifford_algebra" => {
                // GA operations with moderate computational complexity
                (128, 1, 1)
            }
            "cellular_automata" | "ca_evolution" => {
                // 2D grid operations, optimized for spatial locality
                (16, 16, 1)
            }
            "neural_network" | "batch_processing" => {
                // Large 1D workgroups for high-throughput batch processing
                (256, 1, 1)
            }
            "information_geometry" | "fisher_information" | "bregman_divergence" => {
                // Statistical manifold computations - large workgroups
                (256, 1, 1)
            }
            "tropical_algebra" | "tropical_matrix" => {
                // Tropical operations, moderate workgroup size
                (128, 1, 1)
            }
            "dual_number" | "automatic_differentiation" => {
                // AD operations, balanced workgroup size
                (128, 1, 1)
            }
            "fusion_system" | "llm_evaluation" => {
                // Complex fusion operations, large workgroups
                (256, 1, 1)
            }
            "enumerative_geometry" | "intersection_theory" => {
                // Geometric computations, moderate workgroups
                (64, 1, 1)
            }
            _ => (64, 1, 1), // Conservative default for unknown operations
        }
    }

    /// Generate optimized WGSL workgroup declaration for operation
    pub fn get_workgroup_declaration(&self, operation: &str, data_size: usize) -> String {
        let (x, y, z) = self.get_optimal_workgroup(operation, data_size);

        if y == 1 && z == 1 {
            format!("@compute @workgroup_size({})", x)
        } else if z == 1 {
            format!("@compute @workgroup_size({}, {})", x, y)
        } else {
            format!("@compute @workgroup_size({}, {}, {})", x, y, z)
        }
    }
}

/// Enhanced buffer pool with statistics and eviction policies
pub struct EnhancedGpuBufferPool {
    pools: HashMap<(u64, wgpu::BufferUsages), Vec<wgpu::Buffer>>,
    stats: HashMap<(u64, wgpu::BufferUsages), PoolEntryStats>,
    total_created: u64,
    total_reused: u64,
    last_cleanup: Instant,
}

#[derive(Debug, Clone, Default)]
pub struct PoolEntryStats {
    pub created_count: u64,
    pub reused_count: u64,
    pub last_used: Option<Instant>,
    pub total_size_bytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct BufferPoolStats {
    pub total_buffers_created: u64,
    pub total_buffers_reused: u64,
    pub current_pooled_count: usize,
    pub total_pooled_memory_mb: f32,
    pub hit_rate_percent: f32,
}

impl EnhancedGpuBufferPool {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
            stats: HashMap::new(),
            total_created: 0,
            total_reused: 0,
            last_cleanup: Instant::now(),
        }
    }
}

impl Default for EnhancedGpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

impl EnhancedGpuBufferPool {
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> wgpu::Buffer {
        let key = (size, usage);

        // Try to reuse from pool
        if let Some(buffers) = self.pools.get_mut(&key) {
            if let Some(buffer) = buffers.pop() {
                self.total_reused += 1;
                self.stats.entry(key).or_default().reused_count += 1;
                self.stats.get_mut(&key).unwrap().last_used = Some(Instant::now());
                return buffer;
            }
        }

        // Create new buffer
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: false,
        });

        self.total_created += 1;
        let stats = self.stats.entry(key).or_default();
        stats.created_count += 1;
        stats.total_size_bytes += size;
        stats.last_used = Some(Instant::now());

        // Periodic cleanup
        if self.last_cleanup.elapsed().as_secs() > 30 {
            self.cleanup_old_buffers();
        }

        buffer
    }

    pub fn return_buffer(&mut self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        let key = (size, usage);
        self.pools.entry(key).or_default().push(buffer);
    }

    pub fn get_stats(&self) -> BufferPoolStats {
        let total_ops = self.total_created + self.total_reused;
        let hit_rate = if total_ops > 0 {
            (self.total_reused as f32 / total_ops as f32) * 100.0
        } else {
            0.0
        };

        let current_pooled_count = self.pools.values().map(|v| v.len()).sum();
        let total_pooled_memory_mb: f32 = self
            .pools
            .iter()
            .map(|((size, _usage), buffers)| {
                (*size as f32 * buffers.len() as f32) / 1024.0 / 1024.0
            })
            .sum();

        BufferPoolStats {
            total_buffers_created: self.total_created,
            total_buffers_reused: self.total_reused,
            current_pooled_count,
            total_pooled_memory_mb,
            hit_rate_percent: hit_rate,
        }
    }

    fn cleanup_old_buffers(&mut self) {
        let now = Instant::now();
        let cleanup_threshold = std::time::Duration::from_secs(300); // 5 minutes

        self.pools.retain(|&key, buffers| {
            if let Some(stats) = self.stats.get(&key) {
                if let Some(last_used) = stats.last_used {
                    if now.duration_since(last_used) > cleanup_threshold {
                        // Remove old unused buffers
                        buffers.clear();
                        return false;
                    }
                }
            }
            true
        });

        self.last_cleanup = now;
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
    #[ignore = "GPU hardware required, may fail in CI/CD environments"]
    async fn test_gpu_context_creation() {
        // Test should pass even without GPU (graceful fallback)
        let _result = GpuContext::new().await;
        // Don't assert success since GPU might not be available in CI
    }

    #[tokio::test]
    #[ignore = "GPU hardware required, may fail in CI/CD environments"]
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
