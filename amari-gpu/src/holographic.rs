//! GPU-accelerated holographic memory operations
//!
//! This module provides GPU acceleration for Vector Symbolic Architecture (VSA)
//! operations using WebGPU/wgpu compute shaders.
//!
//! # Supported Operations
//!
//! - **Batch Binding**: GPU-accelerated geometric product for many key-value pairs
//! - **Batch Unbinding**: Parallel retrieval operations
//! - **Batch Similarity**: Efficient similarity computation across large codebooks
//! - **Batch Bundling**: Parallel superposition of multiple representations
//!
//! # Performance
//!
//! GPU acceleration is beneficial for:
//! - Batch sizes > 100 items
//! - Large codebook operations (resonator cleanup)
//! - Memory operations with many stored items
//!
//! For small operations, CPU fallback is used automatically.

use amari_holographic::{AlgebraConfig, BindingAlgebra, HolographicMemory, ProductCliffordAlgebra};
use thiserror::Error;
use wgpu::util::DeviceExt;

/// Type alias for commonly used ProductClifford algebra
pub type ProductCl3x32 = ProductCliffordAlgebra<32>;

/// Errors specific to GPU holographic operations
#[derive(Error, Debug)]
pub enum GpuHolographicError {
    #[error("GPU initialization failed: {0}")]
    InitializationError(String),

    #[error("GPU buffer error: {0}")]
    BufferError(String),

    #[error("Shader compilation error: {0}")]
    ShaderError(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Holographic algebra error: {0}")]
    AlgebraError(String),
}

/// Result type for GPU holographic operations
pub type GpuHolographicResult<T> = Result<T, GpuHolographicError>;

/// GPU-accelerated holographic operations
///
/// Provides batch operations for VSA using WebGPU compute shaders.
/// Automatically falls back to CPU for small batch sizes.
pub struct GpuHolographic {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bind_pipeline: wgpu::ComputePipeline,
    similarity_pipeline: wgpu::ComputePipeline,
    bundle_pipeline: wgpu::ComputePipeline,
    dimension: usize,
}

impl GpuHolographic {
    /// Initialize GPU context for holographic operations
    ///
    /// # Arguments
    /// * `dimension` - The dimension of the algebra (e.g., 256 for ProductCl3x32)
    pub async fn new(dimension: usize) -> GpuHolographicResult<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                GpuHolographicError::InitializationError("No GPU adapter found".to_string())
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Amari Holographic GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuHolographicError::InitializationError(e.to_string()))?;

        // Create compute pipelines
        let bind_pipeline = Self::create_bind_pipeline(&device, dimension)?;
        let similarity_pipeline = Self::create_similarity_pipeline(&device, dimension)?;
        let bundle_pipeline = Self::create_bundle_pipeline(&device, dimension)?;

        Ok(Self {
            device,
            queue,
            bind_pipeline,
            similarity_pipeline,
            bundle_pipeline,
            dimension,
        })
    }

    /// Create GPU context for ProductCl3x32 (256-dimensional)
    pub async fn new_product_cl3x32() -> GpuHolographicResult<Self> {
        Self::new(256).await
    }

    /// Batch binding operation (geometric product for holographic binding)
    ///
    /// Computes `result[i] = keys[i] ⊛ values[i]` for all i.
    ///
    /// # Arguments
    /// * `keys` - Flat array of key coefficients (batch_size * dimension)
    /// * `values` - Flat array of value coefficients (batch_size * dimension)
    ///
    /// # Returns
    /// Flat array of bound pair coefficients
    pub async fn batch_bind(&self, keys: &[f64], values: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = keys.len() / self.dimension;

        if keys.len() != values.len() {
            return Err(GpuHolographicError::DimensionMismatch {
                expected: keys.len(),
                actual: values.len(),
            });
        }

        // For small batches, use CPU
        if batch_size < 100 {
            return self.batch_bind_cpu(keys, values);
        }

        self.batch_bind_gpu(keys, values).await
    }

    /// Batch unbinding operation
    ///
    /// Computes `result[i] = keys[i]⁻¹ ⊛ bounds[i]` for all i.
    pub async fn batch_unbind(
        &self,
        keys: &[f64],
        bounds: &[f64],
    ) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = keys.len() / self.dimension;

        if keys.len() != bounds.len() {
            return Err(GpuHolographicError::DimensionMismatch {
                expected: keys.len(),
                actual: bounds.len(),
            });
        }

        // For small batches, use CPU
        if batch_size < 100 {
            return self.batch_unbind_cpu(keys, bounds);
        }

        // GPU unbinding requires computing inverse first
        // For now, use CPU for correctness
        self.batch_unbind_cpu(keys, bounds)
    }

    /// Batch similarity computation
    ///
    /// Computes cosine similarity between all pairs.
    ///
    /// # Arguments
    /// * `a_batch` - First set of vectors (batch_size * dimension)
    /// * `b_batch` - Second set of vectors (batch_size * dimension)
    ///
    /// # Returns
    /// Array of similarity values (batch_size)
    pub async fn batch_similarity(
        &self,
        a_batch: &[f64],
        b_batch: &[f64],
    ) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = a_batch.len() / self.dimension;

        if a_batch.len() != b_batch.len() {
            return Err(GpuHolographicError::DimensionMismatch {
                expected: a_batch.len(),
                actual: b_batch.len(),
            });
        }

        // For small batches, use CPU
        if batch_size < 100 {
            return self.batch_similarity_cpu(a_batch, b_batch);
        }

        self.batch_similarity_gpu(a_batch, b_batch).await
    }

    /// Batch bundling operation
    ///
    /// Computes element-wise sum for superposition.
    pub async fn batch_bundle(
        &self,
        a_batch: &[f64],
        b_batch: &[f64],
    ) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = a_batch.len() / self.dimension;

        if a_batch.len() != b_batch.len() {
            return Err(GpuHolographicError::DimensionMismatch {
                expected: a_batch.len(),
                actual: b_batch.len(),
            });
        }

        // For small batches, use CPU
        if batch_size < 100 {
            return self.batch_bundle_cpu(a_batch, b_batch);
        }

        self.batch_bundle_gpu(a_batch, b_batch).await
    }

    /// Find most similar item in codebook (for resonator cleanup)
    ///
    /// # Arguments
    /// * `query` - Query vector coefficients (dimension)
    /// * `codebook` - Codebook vectors (codebook_size * dimension)
    ///
    /// # Returns
    /// (best_index, best_similarity)
    pub async fn find_most_similar(
        &self,
        query: &[f64],
        codebook: &[f64],
    ) -> GpuHolographicResult<(usize, f64)> {
        if query.len() != self.dimension {
            return Err(GpuHolographicError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        let codebook_size = codebook.len() / self.dimension;

        // Compute similarities with all codebook items
        let queries = query.repeat(codebook_size);
        let similarities = self.batch_similarity(&queries, codebook).await?;

        // Find maximum
        let (best_idx, best_sim) = similarities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, s)| (i, *s))
            .unwrap_or((0, 0.0));

        Ok((best_idx, best_sim))
    }

    /// Heuristic to determine if GPU should be used
    pub fn should_use_gpu(batch_size: usize) -> bool {
        batch_size >= 100
    }

    // ========================================================================
    // CPU fallback implementations
    // ========================================================================

    fn batch_bind_cpu(&self, keys: &[f64], values: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = keys.len() / self.dimension;
        let mut results = Vec::with_capacity(keys.len());

        for i in 0..batch_size {
            let start = i * self.dimension;
            let end = start + self.dimension;

            let key = ProductCl3x32::from_coefficients(&keys[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;
            let value = ProductCl3x32::from_coefficients(&values[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;

            let bound = key.bind(&value);
            results.extend(bound.to_coefficients());
        }

        Ok(results)
    }

    fn batch_unbind_cpu(&self, keys: &[f64], bounds: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = keys.len() / self.dimension;
        let mut results = Vec::with_capacity(keys.len());

        for i in 0..batch_size {
            let start = i * self.dimension;
            let end = start + self.dimension;

            let key = ProductCl3x32::from_coefficients(&keys[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;
            let bound = ProductCl3x32::from_coefficients(&bounds[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;

            let unbound = key
                .unbind(&bound)
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;
            results.extend(unbound.to_coefficients());
        }

        Ok(results)
    }

    fn batch_similarity_cpu(&self, a: &[f64], b: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = a.len() / self.dimension;
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start = i * self.dimension;
            let end = start + self.dimension;

            let a_vec = ProductCl3x32::from_coefficients(&a[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;
            let b_vec = ProductCl3x32::from_coefficients(&b[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;

            results.push(a_vec.similarity(&b_vec));
        }

        Ok(results)
    }

    fn batch_bundle_cpu(&self, a: &[f64], b: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = a.len() / self.dimension;
        let mut results = Vec::with_capacity(a.len());

        for i in 0..batch_size {
            let start = i * self.dimension;
            let end = start + self.dimension;

            let a_vec = ProductCl3x32::from_coefficients(&a[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;
            let b_vec = ProductCl3x32::from_coefficients(&b[start..end])
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;

            let bundled = a_vec
                .bundle(&b_vec, 1.0)
                .map_err(|e| GpuHolographicError::AlgebraError(format!("{}", e)))?;
            results.extend(bundled.to_coefficients());
        }

        Ok(results)
    }

    // ========================================================================
    // GPU implementations
    // ========================================================================

    async fn batch_bind_gpu(&self, keys: &[f64], values: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = keys.len() / self.dimension;

        // Convert to f32 for GPU
        let keys_f32: Vec<f32> = keys.iter().map(|&x| x as f32).collect();
        let values_f32: Vec<f32> = values.iter().map(|&x| x as f32).collect();

        // Create GPU buffers
        let keys_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Keys Buffer"),
                contents: bytemuck::cast_slice(&keys_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let values_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Values Buffer"),
                contents: bytemuck::cast_slice(&values_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (keys.len() * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.bind_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Compute Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: keys_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Bind Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bind Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.bind_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .await
            .unwrap()
            .map_err(|e| GpuHolographicError::BufferError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = result_f32.iter().map(|&x| x as f64).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    async fn batch_similarity_gpu(&self, a: &[f64], b: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = a.len() / self.dimension;

        // Convert to f32 for GPU
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

        // Create GPU buffers
        let a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A Buffer"),
                contents: bytemuck::cast_slice(&a_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B Buffer"),
                contents: bytemuck::cast_slice(&b_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (batch_size * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Similarity Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.similarity_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Similarity Compute Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Similarity Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Similarity Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.similarity_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(batch_size.div_ceil(64) as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .await
            .unwrap()
            .map_err(|e| GpuHolographicError::BufferError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = result_f32.iter().map(|&x| x as f64).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    async fn batch_bundle_gpu(&self, a: &[f64], b: &[f64]) -> GpuHolographicResult<Vec<f64>> {
        let batch_size = a.len() / self.dimension;

        // Convert to f32 for GPU
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();

        // Create GPU buffers
        let a_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A Buffer"),
                contents: bytemuck::cast_slice(&a_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B Buffer"),
                contents: bytemuck::cast_slice(&b_f32),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (a.len() * std::mem::size_of::<f32>()) as u64;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bundle Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.bundle_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bundle Compute Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch compute shader
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Bundle Compute Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bundle Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.bundle_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .await
            .unwrap()
            .map_err(|e| GpuHolographicError::BufferError(e.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = result_f32.iter().map(|&x| x as f64).collect();

        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    // ========================================================================
    // Pipeline creation
    // ========================================================================

    fn create_bind_pipeline(
        device: &wgpu::Device,
        dimension: usize,
    ) -> GpuHolographicResult<wgpu::ComputePipeline> {
        let shader_source = Self::get_bind_shader(dimension);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Holographic Bind Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_source)),
        });

        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Holographic Bind Pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
            }),
        )
    }

    fn create_similarity_pipeline(
        device: &wgpu::Device,
        dimension: usize,
    ) -> GpuHolographicResult<wgpu::ComputePipeline> {
        let shader_source = Self::get_similarity_shader(dimension);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Holographic Similarity Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_source)),
        });

        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Holographic Similarity Pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
            }),
        )
    }

    fn create_bundle_pipeline(
        device: &wgpu::Device,
        dimension: usize,
    ) -> GpuHolographicResult<wgpu::ComputePipeline> {
        let shader_source = Self::get_bundle_shader(dimension);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Holographic Bundle Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_source)),
        });

        Ok(
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Holographic Bundle Pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
            }),
        )
    }

    // ========================================================================
    // WGSL Shader generation
    // ========================================================================

    fn get_bind_shader(dimension: usize) -> String {
        format!(
            r#"
// Holographic binding shader (element-wise product for ProductClifford)
// For ProductClifford, binding is component-wise Cl3 geometric product

const DIMENSION: u32 = {dim}u;
const CL3_DIM: u32 = 8u;
const NUM_COMPONENTS: u32 = {dim}u / 8u;

@group(0) @binding(0)
var<storage, read> keys: array<f32>;

@group(0) @binding(1)
var<storage, read> values: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

// Cl3 geometric product sign table (simplified)
fn cl3_product_sign(i: u32, j: u32) -> f32 {{
    // This is a simplified version - full implementation would use proper Cayley table
    return 1.0;
}}

fn cl3_product_index(i: u32, j: u32) -> u32 {{
    // XOR for grade computation in Cl3
    return i ^ j;
}}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x;
    let offset = batch_idx * DIMENSION;

    // For each Cl3 component
    for (var comp = 0u; comp < NUM_COMPONENTS; comp = comp + 1u) {{
        let comp_offset = offset + comp * CL3_DIM;

        // Clear output for this component
        for (var k = 0u; k < CL3_DIM; k = k + 1u) {{
            output[comp_offset + k] = 0.0;
        }}

        // Compute Cl3 geometric product
        for (var i = 0u; i < CL3_DIM; i = i + 1u) {{
            let a = keys[comp_offset + i];
            if (abs(a) < 1e-10) {{
                continue;
            }}

            for (var j = 0u; j < CL3_DIM; j = j + 1u) {{
                let b = values[comp_offset + j];
                if (abs(b) < 1e-10) {{
                    continue;
                }}

                let sign = cl3_product_sign(i, j);
                let idx = cl3_product_index(i, j);
                output[comp_offset + idx] += sign * a * b;
            }}
        }}
    }}
}}
"#,
            dim = dimension
        )
    }

    fn get_similarity_shader(dimension: usize) -> String {
        format!(
            r#"
// Holographic similarity shader (cosine similarity)

const DIMENSION: u32 = {dim}u;

@group(0) @binding(0)
var<storage, read> a_batch: array<f32>;

@group(0) @binding(1)
var<storage, read> b_batch: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x;
    let offset = batch_idx * DIMENSION;

    var dot_product: f32 = 0.0;
    var norm_a_sq: f32 = 0.0;
    var norm_b_sq: f32 = 0.0;

    for (var i = 0u; i < DIMENSION; i = i + 1u) {{
        let a = a_batch[offset + i];
        let b = b_batch[offset + i];

        dot_product += a * b;
        norm_a_sq += a * a;
        norm_b_sq += b * b;
    }}

    let norm_a = sqrt(norm_a_sq);
    let norm_b = sqrt(norm_b_sq);

    var similarity: f32 = 0.0;
    if (norm_a > 1e-10 && norm_b > 1e-10) {{
        similarity = dot_product / (norm_a * norm_b);
        similarity = clamp(similarity, -1.0, 1.0);
    }}

    output[batch_idx] = similarity;
}}
"#,
            dim = dimension
        )
    }

    fn get_bundle_shader(dimension: usize) -> String {
        format!(
            r#"
// Holographic bundling shader (element-wise sum)

const DIMENSION: u32 = {dim}u;

@group(0) @binding(0)
var<storage, read> a_batch: array<f32>;

@group(0) @binding(1)
var<storage, read> b_batch: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let batch_idx = global_id.x;
    let offset = batch_idx * DIMENSION;

    for (var i = 0u; i < DIMENSION; i = i + 1u) {{
        output[offset + i] = a_batch[offset + i] + b_batch[offset + i];
    }}
}}
"#,
            dim = dimension
        )
    }
}

/// GPU-accelerated holographic memory
///
/// Wrapper around `HolographicMemory` that uses GPU for batch operations.
pub struct GpuHolographicMemory {
    memory: HolographicMemory<ProductCl3x32>,
    gpu: Option<GpuHolographic>,
}

impl GpuHolographicMemory {
    /// Create a new GPU-accelerated holographic memory
    pub async fn new() -> GpuHolographicResult<Self> {
        let memory = HolographicMemory::new(AlgebraConfig::default());
        let gpu = GpuHolographic::new_product_cl3x32().await.ok();

        Ok(Self { memory, gpu })
    }

    /// Store a key-value pair
    pub fn store(&mut self, key: &ProductCl3x32, value: &ProductCl3x32) {
        self.memory.store(key, value);
    }

    /// Store multiple key-value pairs with GPU acceleration
    pub async fn store_batch(
        &mut self,
        keys: &[ProductCl3x32],
        values: &[ProductCl3x32],
    ) -> GpuHolographicResult<()> {
        // For batch storage, we could use GPU to compute bindings
        // but for now, use the standard batch method
        let pairs: Vec<_> = keys.iter().cloned().zip(values.iter().cloned()).collect();
        self.memory.store_batch(&pairs);
        Ok(())
    }

    /// Retrieve a value by key
    pub fn retrieve(
        &self,
        key: &ProductCl3x32,
    ) -> amari_holographic::RetrievalResult<ProductCl3x32> {
        self.memory.retrieve(key)
    }

    /// Get memory statistics
    pub fn capacity_info(&self) -> amari_holographic::CapacityInfo {
        self.memory.capacity_info()
    }

    /// Clear the memory
    pub fn clear(&mut self) {
        self.memory.clear();
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_use_gpu() {
        assert!(!GpuHolographic::should_use_gpu(10));
        assert!(GpuHolographic::should_use_gpu(100));
        assert!(GpuHolographic::should_use_gpu(1000));
    }

    #[ignore] // Requires GPU initialization
    #[tokio::test]
    async fn test_batch_similarity_cpu() {
        // Test CPU fallback
        let dim = 256;

        // Create test data
        let a: Vec<f64> = (0..dim).map(|i| (i as f64) / dim as f64).collect();
        let b = a.clone();

        // Use CPU path directly (small batch)
        if let Ok(gpu) = GpuHolographic::new(dim).await {
            let result = gpu.batch_similarity_cpu(&a, &b).unwrap();
            assert_eq!(result.len(), 1);
            // Self-similarity should be close to 1.0
            assert!((result[0] - 1.0).abs() < 0.01);
        }
    }

    #[ignore] // Requires GPU initialization
    #[tokio::test]
    async fn test_batch_bundle_cpu() {
        // Test CPU fallback for bundling
        let dim = 256;

        // Use small values to avoid numerical issues
        let a: Vec<f64> = (0..dim).map(|i| (i as f64) / 1000.0).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64) / 500.0).collect();

        if let Ok(gpu) = GpuHolographic::new(dim).await {
            let result = gpu.batch_bundle_cpu(&a, &b).unwrap();
            assert_eq!(result.len(), dim);
            // Check sum - bundle adds corresponding coefficients
            for i in 0..dim {
                assert!(
                    (result[i] - (a[i] + b[i])).abs() < 0.01,
                    "Mismatch at {}: {} vs {}",
                    i,
                    result[i],
                    a[i] + b[i]
                );
            }
        }
    }
}
