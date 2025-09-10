//! GPU acceleration for geometric algebra operations using WebGPU/wgpu

use amari_core::Multivector;
use wgpu::util::DeviceExt;
use thiserror::Error;
use bytemuck::{Pod, Zeroable};

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("Failed to initialize GPU: {0}")]
    InitializationError(String),
    
    #[error("GPU buffer error: {0}")]
    BufferError(String),
    
    #[error("Shader compilation error: {0}")]
    ShaderError(String),
}

/// GPU-accelerated Clifford algebra operations
pub struct GpuCliffordAlgebra {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    cayley_buffer: wgpu::Buffer,
    dim: usize,
    basis_count: usize,
}

impl GpuCliffordAlgebra {
    /// Initialize GPU context and compile shaders
    pub async fn new<const P: usize, const Q: usize, const R: usize>() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| GpuError::InitializationError("No GPU adapter found".to_string()))?;
        
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Amari GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::InitializationError(e.to_string()))?;
        
        let dim = P + Q + R;
        let basis_count = 1 << dim;
        
        // Generate and upload Cayley table
        let cayley_table = Self::generate_cayley_table::<P, Q, R>();
        let cayley_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cayley Table"),
            contents: bytemuck::cast_slice(&cayley_table),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Geometric Product Shader"),
            source: wgpu::ShaderSource::Wgsl(GEOMETRIC_PRODUCT_SHADER.into()),
        });
        
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
        
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Geometric Product Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        
        Ok(Self {
            device,
            queue,
            compute_pipeline,
            cayley_buffer,
            dim,
            basis_count,
        })
    }
    
    /// Generate Cayley table as flat array for GPU
    fn generate_cayley_table<const P: usize, const Q: usize, const R: usize>() -> Vec<CayleyEntry> {
        use amari_core::cayley::CayleyTable;
        
        let table = CayleyTable::<P, Q, R>::get();
        let basis_count = 1 << (P + Q + R);
        let mut flat_table = Vec::with_capacity(basis_count * basis_count);
        
        for i in 0..basis_count {
            for j in 0..basis_count {
                let (sign, index) = table.get_product(i, j);
                flat_table.push(CayleyEntry {
                    sign: sign as f32,
                    index: index as u32,
                });
            }
        }
        
        flat_table
    }
    
    /// Perform batch geometric product on GPU
    pub async fn batch_geometric_product(
        &self,
        a_batch: &[f64],
        b_batch: &[f64],
    ) -> Result<Vec<f64>, GpuError> {
        let batch_size = a_batch.len() / self.basis_count;
        
        if a_batch.len() != b_batch.len() {
            return Err(GpuError::BufferError("Input batches must have same size".to_string()));
        }
        
        // Convert to f32 for GPU
        let a_f32: Vec<f32> = a_batch.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b_batch.iter().map(|&x| x as f32).collect();
        
        // Create GPU buffers
        let a_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("A Buffer"),
            contents: bytemuck::cast_slice(&a_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let b_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("B Buffer"),
            contents: bytemuck::cast_slice(&b_f32),
            usage: wgpu::BufferUsages::STORAGE,
        });
        
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (a_batch.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (a_batch.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group
        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.cayley_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }
        
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (a_batch.len() * std::mem::size_of::<f32>()) as u64,
        );
        
        self.queue.submit(Some(encoder.finish()));
        
        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        receiver.await.unwrap().map_err(|e| GpuError::BufferError(e.to_string()))?;
        
        let data = buffer_slice.get_mapped_range();
        let result_f32: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<f64> = result_f32.iter().map(|&x| x as f64).collect();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
    
    /// Heuristic to determine if GPU should be used
    pub fn should_use_gpu(operation_count: usize) -> bool {
        // GPU is beneficial for batch operations with many multivectors
        operation_count >= 100
    }
}

/// Cayley table entry for GPU
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CayleyEntry {
    sign: f32,
    index: u32,
}

/// WGSL compute shader for geometric product
const GEOMETRIC_PRODUCT_SHADER: &str = r#"
struct CayleyEntry {
    sign: f32,
    index: u32,
}

@group(0) @binding(0)
var<storage, read> cayley_table: array<CayleyEntry>;

@group(0) @binding(1)
var<storage, read> a_batch: array<f32>;

@group(0) @binding(2)
var<storage, read> b_batch: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

const BASIS_COUNT: u32 = 8u; // For 3D Clifford algebra

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let offset = batch_idx * BASIS_COUNT;
    
    // Clear output
    for (var k = 0u; k < BASIS_COUNT; k = k + 1u) {
        output[offset + k] = 0.0;
    }
    
    // Compute geometric product
    for (var i = 0u; i < BASIS_COUNT; i = i + 1u) {
        let a_coeff = a_batch[offset + i];
        if (abs(a_coeff) < 1e-14) {
            continue;
        }
        
        for (var j = 0u; j < BASIS_COUNT; j = j + 1u) {
            let b_coeff = b_batch[offset + j];
            if (abs(b_coeff) < 1e-14) {
                continue;
            }
            
            let table_idx = i * BASIS_COUNT + j;
            let entry = cayley_table[table_idx];
            output[offset + entry.index] += entry.sign * a_coeff * b_coeff;
        }
    }
}
"#;

/// Adaptive GPU/CPU dispatcher
pub struct AdaptiveCompute {
    gpu: Option<GpuCliffordAlgebra>,
}

impl AdaptiveCompute {
    /// Create with optional GPU acceleration
    pub async fn new<const P: usize, const Q: usize, const R: usize>() -> Self {
        let gpu = GpuCliffordAlgebra::new::<P, Q, R>().await.ok();
        Self { gpu }
    }
    
    /// Perform geometric product, automatically choosing CPU or GPU
    pub async fn geometric_product<const P: usize, const Q: usize, const R: usize>(
        &self,
        a: &Multivector<P, Q, R>,
        b: &Multivector<P, Q, R>,
    ) -> Multivector<P, Q, R> {
        // For single operations, always use CPU
        a.geometric_product(b)
    }
    
    /// Batch geometric product with adaptive dispatch
    pub async fn batch_geometric_product(
        &self,
        a_batch: &[f64],
        b_batch: &[f64],
    ) -> Result<Vec<f64>, GpuError> {
        let batch_size = a_batch.len() / 8; // Assuming 3D
        
        if let Some(gpu) = &self.gpu {
            if GpuCliffordAlgebra::should_use_gpu(batch_size) {
                return gpu.batch_geometric_product(a_batch, b_batch).await;
            }
        }
        
        // Fallback to CPU
        let mut result = Vec::with_capacity(a_batch.len());
        for i in 0..batch_size {
            let start = i * 8;
            let end = start + 8;
            
            let a = Multivector::<3, 0, 0>::from_coefficients(a_batch[start..end].to_vec());
            let b = Multivector::<3, 0, 0>::from_coefficients(b_batch[start..end].to_vec());
            let product = a.geometric_product(&b);
            
            for j in 0..8 {
                result.push(product.get(j));
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_should_use_gpu() {
        assert!(!GpuCliffordAlgebra::should_use_gpu(10));
        assert!(GpuCliffordAlgebra::should_use_gpu(1000));
    }
}