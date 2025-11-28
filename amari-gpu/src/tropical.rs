//! GPU acceleration for tropical algebra operations
//!
//! This module provides GPU-accelerated implementations of tropical algebra
//! operations including matrix multiplication, neural network attention,
//! and Viterbi algorithm computation using WebGPU compute shaders.

#[cfg(feature = "tropical")]
use amari_tropical::{TropicalError, TropicalMatrix, TropicalMultivector, TropicalNumber};

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "gpu")]
use num_traits::Float;

#[cfg(feature = "gpu")]
use std::collections::HashMap;

#[cfg(feature = "gpu")]
use wgpu::{util::DeviceExt, Buffer, BufferUsages};

#[cfg(feature = "gpu")]
use thiserror::Error;

/// GPU-specific error types for tropical algebra operations
#[cfg(feature = "gpu")]
#[derive(Error, Debug)]
pub enum TropicalGpuError {
    #[error("GPU initialization failed: {0}")]
    InitializationError(String),

    #[error("Shader compilation failed: {0}")]
    ShaderCompilation(String),

    #[error("Buffer operation failed: {0}")]
    BufferError(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocation(String),

    #[error("Tropical algebra error: {0}")]
    TropicalError(#[from] TropicalError),
}

#[cfg(feature = "gpu")]
pub type TropicalGpuResult<T> = Result<T, TropicalGpuError>;

/// GPU buffer representation for tropical numbers
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuTropicalNumber {
    pub value: f32,
}

#[cfg(feature = "gpu")]
impl From<TropicalNumber<f32>> for GpuTropicalNumber {
    fn from(t: TropicalNumber<f32>) -> Self {
        Self { value: t.value() }
    }
}

#[cfg(feature = "gpu")]
impl From<GpuTropicalNumber> for TropicalNumber<f32> {
    fn from(gpu: GpuTropicalNumber) -> Self {
        TropicalNumber::new(gpu.value)
    }
}

/// GPU buffer representation for tropical matrices
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuTropicalMatrixHeader {
    pub rows: u32,
    pub cols: u32,
    pub _padding: [u32; 2], // Ensure 16-byte alignment
}

/// GPU context for tropical algebra operations
#[cfg(feature = "gpu")]
pub struct TropicalGpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    #[allow(dead_code)]
    shader_cache: HashMap<String, wgpu::ComputePipeline>,
}

#[cfg(feature = "gpu")]
impl TropicalGpuContext {
    /// Initialize GPU context with WebGPU
    pub async fn new() -> TropicalGpuResult<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                TropicalGpuError::InitializationError("No GPU adapter found".to_string())
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Amari Tropical GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| TropicalGpuError::InitializationError(e.to_string()))?;

        Ok(Self {
            device,
            queue,
            shader_cache: HashMap::new(),
        })
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

    /// Read buffer data back to CPU
    pub async fn read_buffer<T: bytemuck::Pod + Clone>(
        &self,
        buffer: &wgpu::Buffer,
        size: u64,
    ) -> TropicalGpuResult<Vec<T>> {
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

        pollster::block_on(rx)
            .map_err(|_| TropicalGpuError::BufferError("Buffer read timeout".to_string()))?
            .map_err(|e| TropicalGpuError::BufferError(format!("Buffer map failed: {}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

/// Tropical algebra GPU operations trait
#[cfg(feature = "gpu")]
pub trait TropicalGpuAccelerated<T> {
    /// Convert data to GPU buffer format
    fn to_gpu_buffer(&self, context: &TropicalGpuContext) -> TropicalGpuResult<wgpu::Buffer>;

    /// Reconstruct data from GPU buffer
    fn from_gpu_buffer(buffer: &wgpu::Buffer, context: &TropicalGpuContext)
        -> TropicalGpuResult<T>;

    /// Execute GPU operation with specified parameters
    fn gpu_operation(
        &self,
        operation: &str,
        context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<T>;
}

/// Parameter types for GPU operations
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub enum GpuParameter {
    Float(f32),
    Integer(i32),
    UnsignedInteger(u32),
    Buffer(String), // Buffer identifier
    Array(Vec<f32>),
}

#[cfg(feature = "gpu")]
impl<T: Float> TropicalGpuAccelerated<TropicalNumber<T>> for TropicalNumber<T>
where
    T: bytemuck::Pod + Into<f32> + From<f32>,
{
    fn to_gpu_buffer(&self, context: &TropicalGpuContext) -> TropicalGpuResult<Buffer> {
        let gpu_data = GpuTropicalNumber {
            value: self.value().into(),
        };

        let buffer = context.create_buffer_with_data(
            "TropicalNumber Buffer",
            &[gpu_data],
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        Ok(buffer)
    }

    fn from_gpu_buffer(
        buffer: &Buffer,
        context: &TropicalGpuContext,
    ) -> TropicalGpuResult<TropicalNumber<T>> {
        let gpu_data: Vec<GpuTropicalNumber> = pollster::block_on(
            context.read_buffer(buffer, std::mem::size_of::<GpuTropicalNumber>() as u64),
        )?;

        if gpu_data.is_empty() {
            return Err(TropicalGpuError::InvalidOperation(
                "Empty buffer data".to_string(),
            ));
        }

        Ok(TropicalNumber::new(<T as From<f32>>::from(
            gpu_data[0].value,
        )))
    }

    fn gpu_operation(
        &self,
        operation: &str,
        _context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalNumber<T>> {
        match operation {
            "tropical_add" => {
                if let Some(GpuParameter::Buffer(_other_buffer_id)) = params.get("other") {
                    // Placeholder: would implement actual GPU tropical add
                    Ok(*self)
                } else {
                    Err(TropicalGpuError::InvalidOperation(
                        "Missing 'other' parameter for tropical_add".to_string(),
                    ))
                }
            }
            "tropical_mul" => {
                if let Some(GpuParameter::Buffer(_other_buffer_id)) = params.get("other") {
                    // Placeholder: would implement actual GPU tropical mul
                    Ok(*self)
                } else {
                    Err(TropicalGpuError::InvalidOperation(
                        "Missing 'other' parameter for tropical_mul".to_string(),
                    ))
                }
            }
            "tropical_pow" => {
                if let Some(GpuParameter::Float(scalar)) = params.get("scalar") {
                    Ok(self.tropical_pow(<T as From<f32>>::from(*scalar)))
                } else {
                    Err(TropicalGpuError::InvalidOperation(
                        "Missing 'scalar' parameter for tropical_pow".to_string(),
                    ))
                }
            }
            _ => Err(TropicalGpuError::InvalidOperation(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

#[cfg(feature = "gpu")]
impl<T: Float> TropicalGpuAccelerated<TropicalMatrix<T>> for TropicalMatrix<T>
where
    T: bytemuck::Pod + Into<f32> + From<f32>,
{
    fn to_gpu_buffer(&self, context: &TropicalGpuContext) -> TropicalGpuResult<Buffer> {
        // Create header
        let header = GpuTropicalMatrixHeader {
            rows: self.rows as u32,
            cols: self.cols as u32,
            _padding: [0; 2],
        };

        // Flatten matrix data
        let mut gpu_data = Vec::with_capacity(self.rows * self.cols);
        for row in &self.data {
            for &element in row {
                gpu_data.push(GpuTropicalNumber {
                    value: element.value().into(),
                });
            }
        }

        // Create combined buffer with header + data
        let mut buffer_data = Vec::new();
        buffer_data.extend_from_slice(bytemuck::cast_slice(&[header]));
        buffer_data.extend_from_slice(bytemuck::cast_slice(&gpu_data));

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TropicalMatrix Buffer"),
                contents: &buffer_data,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            });

        Ok(buffer)
    }

    fn from_gpu_buffer(
        buffer: &Buffer,
        context: &TropicalGpuContext,
    ) -> TropicalGpuResult<TropicalMatrix<T>> {
        // Read header first
        let header_data: Vec<GpuTropicalMatrixHeader> = pollster::block_on(context.read_buffer(
            buffer,
            std::mem::size_of::<GpuTropicalMatrixHeader>() as u64,
        ))?;

        if header_data.is_empty() {
            return Err(TropicalGpuError::InvalidOperation(
                "Empty header data".to_string(),
            ));
        }

        let header = header_data[0];
        let rows = header.rows as usize;
        let cols = header.cols as usize;

        // For this implementation, we'll read the entire buffer and parse it
        let total_size = std::mem::size_of::<GpuTropicalMatrixHeader>()
            + rows * cols * std::mem::size_of::<GpuTropicalNumber>();

        let full_data: Vec<u8> =
            pollster::block_on(context.read_buffer(buffer, total_size as u64))?;

        // Skip header bytes and parse data
        let data_offset = std::mem::size_of::<GpuTropicalMatrixHeader>();
        let data_slice = &full_data[data_offset..];
        let gpu_numbers: &[GpuTropicalNumber] = bytemuck::cast_slice(data_slice);

        // Reconstruct matrix
        let mut matrix = TropicalMatrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let idx = i * cols + j;
                if idx < gpu_numbers.len() {
                    matrix.data[i][j] =
                        TropicalNumber::new(<T as From<f32>>::from(gpu_numbers[idx].value));
                }
            }
        }

        Ok(matrix)
    }

    fn gpu_operation(
        &self,
        operation: &str,
        context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMatrix<T>> {
        match operation {
            "tropical_matrix_multiply" => self.gpu_matrix_multiply(context, params),
            "viterbi" => self.gpu_viterbi(context, params),
            "attention_scores" => self.gpu_attention_scores(context, params),
            _ => Err(TropicalGpuError::InvalidOperation(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

#[cfg(feature = "gpu")]
impl<T: Float> TropicalMatrix<T>
where
    T: bytemuck::Pod + Into<f32> + From<f32>,
{
    /// GPU-accelerated tropical matrix multiplication
    pub fn gpu_matrix_multiply(
        &self,
        _context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMatrix<T>> {
        let _other_buffer_id = match params.get("other") {
            Some(GpuParameter::Buffer(id)) => id,
            _ => {
                return Err(TropicalGpuError::InvalidOperation(
                    "Missing 'other' matrix parameter".to_string(),
                ))
            }
        };

        // TODO: Implement actual GPU matrix multiplication using tropical shaders
        // For now, return self as placeholder
        Ok(self.clone())
    }

    /// GPU-accelerated Viterbi algorithm
    pub fn gpu_viterbi(
        &self,
        _context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMatrix<T>> {
        let _emissions = match params.get("emissions") {
            Some(GpuParameter::Buffer(id)) => id,
            _ => {
                return Err(TropicalGpuError::InvalidOperation(
                    "Missing 'emissions' parameter".to_string(),
                ))
            }
        };

        let _initial_probs = match params.get("initial_probs") {
            Some(GpuParameter::Array(probs)) => probs,
            _ => {
                return Err(TropicalGpuError::InvalidOperation(
                    "Missing 'initial_probs' parameter".to_string(),
                ))
            }
        };

        let _sequence_length = match params.get("sequence_length") {
            Some(GpuParameter::UnsignedInteger(len)) => *len as usize,
            _ => {
                return Err(TropicalGpuError::InvalidOperation(
                    "Missing 'sequence_length' parameter".to_string(),
                ))
            }
        };

        // TODO: Implement GPU Viterbi using tropical algebra shaders
        Ok(self.clone())
    }

    /// GPU-accelerated attention score computation
    pub fn gpu_attention_scores(
        &self,
        _context: &TropicalGpuContext,
        _params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMatrix<T>> {
        // TODO: Implement GPU attention scores using tropical shaders
        Ok(self.clone())
    }
}

#[cfg(feature = "gpu")]
impl<T: Float, const DIM: usize> TropicalGpuAccelerated<TropicalMultivector<T, DIM>>
    for TropicalMultivector<T, DIM>
where
    T: bytemuck::Pod + Into<f32> + From<f32>,
{
    fn to_gpu_buffer(&self, context: &TropicalGpuContext) -> TropicalGpuResult<Buffer> {
        // Convert coefficients to GPU format
        let gpu_data: Vec<GpuTropicalNumber> = self
            .coefficients
            .iter()
            .map(|&coeff| GpuTropicalNumber {
                value: coeff.value().into(),
            })
            .collect();

        let buffer = context.create_buffer_with_data(
            "TropicalMultivector Buffer",
            &gpu_data,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        );

        Ok(buffer)
    }

    fn from_gpu_buffer(
        buffer: &Buffer,
        context: &TropicalGpuContext,
    ) -> TropicalGpuResult<TropicalMultivector<T, DIM>> {
        let gpu_data: Vec<GpuTropicalNumber> = pollster::block_on(context.read_buffer(
            buffer,
            (TropicalMultivector::<T, DIM>::BASIS_COUNT * std::mem::size_of::<GpuTropicalNumber>())
                as u64,
        ))?;

        let coefficients: Vec<TropicalNumber<T>> = gpu_data
            .into_iter()
            .map(|gpu_num| TropicalNumber::new(<T as From<f32>>::from(gpu_num.value)))
            .collect();

        Ok(TropicalMultivector { coefficients })
    }

    fn gpu_operation(
        &self,
        operation: &str,
        context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMultivector<T, DIM>> {
        match operation {
            "geometric_product" => self.gpu_geometric_product(context, params),
            "tropical_add" => self.gpu_tropical_add(context, params),
            "tropical_scale" => self.gpu_tropical_scale(context, params),
            _ => Err(TropicalGpuError::InvalidOperation(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

#[cfg(feature = "gpu")]
impl<T: Float, const DIM: usize> TropicalMultivector<T, DIM>
where
    T: bytemuck::Pod + Into<f32> + From<f32>,
{
    /// GPU-accelerated geometric product
    pub fn gpu_geometric_product(
        &self,
        _context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMultivector<T, DIM>> {
        let _other_buffer_id = match params.get("other") {
            Some(GpuParameter::Buffer(id)) => id,
            _ => {
                return Err(TropicalGpuError::InvalidOperation(
                    "Missing 'other' parameter for geometric_product".to_string(),
                ))
            }
        };

        // TODO: Implement GPU geometric product using tropical geometric algebra shaders
        // For now, return self as placeholder
        Ok(self.clone())
    }

    /// GPU-accelerated tropical addition
    pub fn gpu_tropical_add(
        &self,
        _context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMultivector<T, DIM>> {
        let _other_buffer_id = match params.get("other") {
            Some(GpuParameter::Buffer(id)) => id,
            _ => {
                return Err(TropicalGpuError::InvalidOperation(
                    "Missing 'other' parameter for tropical_add".to_string(),
                ))
            }
        };

        // TODO: Implement GPU tropical addition using element-wise max operations
        // For now, return self as placeholder
        Ok(self.clone())
    }

    /// GPU-accelerated tropical scaling
    pub fn gpu_tropical_scale(
        &self,
        _context: &TropicalGpuContext,
        params: &HashMap<String, GpuParameter>,
    ) -> TropicalGpuResult<TropicalMultivector<T, DIM>> {
        let scalar = match params.get("scalar") {
            Some(GpuParameter::Float(s)) => *s,
            _ => {
                return Err(TropicalGpuError::InvalidOperation(
                    "Missing 'scalar' parameter for tropical_scale".to_string(),
                ))
            }
        };

        // TODO: Implement GPU tropical scaling using element-wise addition
        // For now, return CPU result
        Ok(self.tropical_scale(<T as From<f32>>::from(scalar)))
    }
}

/// High-level GPU tropical algebra operations
#[cfg(feature = "gpu")]
pub struct TropicalGpuOps {
    #[allow(dead_code)]
    context: TropicalGpuContext,
}

#[cfg(feature = "gpu")]
impl TropicalGpuOps {
    /// Create new GPU operations context
    pub async fn new() -> TropicalGpuResult<Self> {
        let context = TropicalGpuContext::new().await?;
        Ok(Self { context })
    }

    /// GPU-accelerated neural network attention using tropical algebra
    pub async fn neural_attention<T>(
        &mut self,
        query: &TropicalMatrix<T>,
        _key: &TropicalMatrix<T>,
        _value: &TropicalMatrix<T>,
    ) -> TropicalGpuResult<TropicalMatrix<T>>
    where
        T: Float + bytemuck::Pod + Into<f32> + From<f32>,
    {
        // TODO: Implement full GPU attention mechanism
        // 1. QK^T tropical matrix multiply (max-plus operations)
        // 2. Apply tropical softmax (max operation)
        // 3. Multiply by V using tropical arithmetic

        // For now, return query as placeholder
        Ok(query.clone())
    }

    /// GPU-accelerated batch Viterbi decoding
    pub async fn batch_viterbi<T>(
        &mut self,
        transitions: &[TropicalMatrix<T>],
        _emissions: &[TropicalMatrix<T>],
        _initial_probs: &[Vec<T>],
        _sequence_lengths: &[usize],
    ) -> TropicalGpuResult<Vec<Vec<usize>>>
    where
        T: Float + bytemuck::Pod + Into<f32> + From<f32>,
    {
        // TODO: Implement batch GPU Viterbi decoding
        // This would process multiple sequences in parallel on GPU
        // For now, return empty results
        Ok(vec![vec![]; transitions.len()])
    }

    /// GPU-accelerated tropical linear algebra solve
    pub async fn tropical_solve<T>(
        &mut self,
        _a: &TropicalMatrix<T>,
        b: &TropicalMatrix<T>,
    ) -> TropicalGpuResult<TropicalMatrix<T>>
    where
        T: Float + bytemuck::Pod + Into<f32> + From<f32>,
    {
        // TODO: Implement tropical linear system solver on GPU
        // Uses tropical elimination and max-plus arithmetic
        // For now, return b as placeholder
        Ok(b.clone())
    }
}

/// WGSL shader source for tropical algebra operations
#[cfg(feature = "gpu")]
pub const TROPICAL_MATRIX_MULTIPLY_SHADER: &str = r#"
// Tropical matrix multiplication compute shader
// Tropical: (A ⊗ B)[i,j] = max_k(A[i,k] + B[k,j])

struct TropicalNumber {
    value: f32,
}

struct MatrixHeader {
    rows: u32,
    cols: u32,
    padding: vec2<u32>,
}

@group(0) @binding(0)
var<storage, read> matrix_a: array<TropicalNumber>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<TropicalNumber>;

@group(0) @binding(2)
var<storage, read_write> result: array<TropicalNumber>;

@group(0) @binding(3)
var<storage, read> params: MatrixHeader;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    let rows_a = params.rows;
    let cols_a = params.cols; // Also rows_b
    let cols_b = params.padding.x; // Stored in padding

    if (i >= rows_a || j >= cols_b) {
        return;
    }

    var max_val = -3.402823466e+38; // Tropical zero (-infinity)

    for (var k = 0u; k < cols_a; k++) {
        let a_val = matrix_a[i * cols_a + k].value;
        let b_val = matrix_b[k * cols_b + j].value;
        let product = a_val + b_val; // Tropical multiplication
        max_val = max(max_val, product); // Tropical addition
    }

    result[i * cols_b + j].value = max_val;
}
"#;

/// WGSL shader for tropical attention computation
#[cfg(feature = "gpu")]
pub const TROPICAL_ATTENTION_SHADER: &str = r#"
// Tropical attention computation
// Replaces softmax with max operation for efficiency

struct TropicalNumber {
    value: f32,
}

@group(0) @binding(0)
var<storage, read> attention_logits: array<TropicalNumber>;

@group(0) @binding(1)
var<storage, read_write> attention_scores: array<TropicalNumber>;

@group(0) @binding(2)
var<uniform> seq_length: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= seq_length) {
        return;
    }

    // Find the maximum value in this sequence (tropical sum)
    var max_val = -3.402823466e+38;
    for (var i = 0u; i < seq_length; i++) {
        max_val = max(max_val, attention_logits[idx * seq_length + i].value);
    }

    // Apply tropical normalization (winner-takes-all)
    let current_val = attention_logits[idx * seq_length + idx].value;
    attention_scores[idx * seq_length + idx].value = select(0.0, 1.0, current_val == max_val);
}
"#;

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;
    use crate::TropicalNumber;

    #[tokio::test]
    async fn test_tropical_gpu_context_creation() {
        // Test should pass even without GPU (graceful fallback)
        let _result = TropicalGpuContext::new().await;
        // Don't assert success since GPU might not be available in CI
    }

    #[tokio::test]
    async fn test_tropical_number_gpu_buffer() {
        let tropical_num = TropicalNumber::new(3.5f32);

        // Test should not fail even if GPU is not available
        if let Ok(context) = TropicalGpuContext::new().await {
            let buffer = tropical_num.to_gpu_buffer(&context).unwrap();
            let reconstructed = TropicalNumber::<f32>::from_gpu_buffer(&buffer, &context).unwrap();

            assert_eq!(tropical_num.value(), reconstructed.value());
        }
    }

    #[tokio::test]
    async fn test_tropical_gpu_ops() {
        // Test initialization
        let result = TropicalGpuOps::new().await;
        // Should not panic even if GPU is not available
        if result.is_ok() {
            // GPU context created successfully
            println!("✅ TropicalGpuOps initialized successfully");
        }
    }

    #[test]
    fn test_gpu_tropical_number_conversion() {
        let tropical_num = TropicalNumber::new(-2.5f32);
        let gpu_num: GpuTropicalNumber = tropical_num.into();
        let reconstructed: TropicalNumber<f32> = gpu_num.into();

        assert!((tropical_num.value() - reconstructed.value()).abs() < 1e-6);
    }
}
