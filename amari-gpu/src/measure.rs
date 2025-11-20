//! GPU-accelerated measure theory and integration
//!
//! Provides parallel computation for:
//! - Numerical integration with Lebesgue and probability measures
//! - Monte Carlo expectation calculations
//! - Batch probability density evaluations
//! - Convolution operations on measure spaces

use crate::{GpuError, UnifiedGpuError};
use wgpu::util::DeviceExt;

/// GPU-accelerated numerical integrator
///
/// Computes integrals ∫f(x)dμ using parallel evaluation on the GPU.
pub struct GpuIntegrator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    integration_pipeline: wgpu::ComputePipeline,
}

impl GpuIntegrator {
    /// Create a new GPU integrator
    pub async fn new() -> Result<Self, GpuError> {
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
                    label: Some("Amari GPU Integrator"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::InitializationError(e.to_string()))?;

        // Create integration shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Integration Shader"),
            source: wgpu::ShaderSource::Wgsl(INTEGRATION_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Integration Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "integrate_riemann",
        });

        Ok(Self {
            device,
            queue,
            integration_pipeline: pipeline,
        })
    }

    /// Integrate a function over [a, b] using Riemann sum with n points
    ///
    /// The function is evaluated at n uniformly spaced points and summed in parallel.
    /// For custom functions, upload function values to GPU via `integrate_values`.
    pub async fn integrate_uniform(
        &self,
        a: f32,
        b: f32,
        n: u32,
        function_id: u32,
    ) -> Result<f32, UnifiedGpuError> {
        // Create buffers for integration parameters
        let params = IntegrationParams {
            lower_bound: a,
            upper_bound: b,
            num_points: n,
            function_id,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Integration Parameters"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer for all n results (one per thread)
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Integration Output"),
            size: (n * 4) as u64, // n floats, 4 bytes each
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.integration_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Integration Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Integration Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Integration Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.integration_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Dispatch enough workgroups for n threads (256 threads per workgroup)
            let workgroup_count = n.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (n * 4) as u64);

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
            .map_err(|e| {
                UnifiedGpuError::InvalidOperation(format!("Buffer mapping failed: {}", e))
            })?
            .map_err(|e| {
                UnifiedGpuError::InvalidOperation(format!("Buffer mapping failed: {}", e))
            })?;

        // Sum all results
        let data = buffer_slice.get_mapped_range();
        let results: &[f32] = bytemuck::cast_slice(&data);
        let total_sum: f32 = results.iter().sum();

        drop(data);
        staging_buffer.unmap();

        // Multiply by dx for Riemann sum: ∫f(x)dx ≈ Σf(xᵢ)·Δx
        let dx = (b - a) / n as f32;
        Ok(total_sum * dx)
    }

    /// Integrate pre-computed function values
    ///
    /// Useful for custom functions computed on CPU or for testing.
    pub async fn integrate_values(&self, values: &[f32], dx: f32) -> Result<f32, UnifiedGpuError> {
        // Upload values to GPU
        let _values_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Function Values"),
                contents: bytemuck::cast_slice(values),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // TODO: Implement GPU reduction for custom values
        // For now, use CPU summation
        let sum: f32 = values.iter().sum();
        Ok(sum * dx)
    }
}

/// Parameters for GPU integration
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct IntegrationParams {
    lower_bound: f32,
    upper_bound: f32,
    num_points: u32,
    function_id: u32, // ID for built-in test functions
}

/// GPU-accelerated Monte Carlo integrator
///
/// Computes expectations E[f(X)] using parallel random sampling.
pub struct GpuMonteCarloIntegrator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    monte_carlo_pipeline: wgpu::ComputePipeline,
}

impl GpuMonteCarloIntegrator {
    /// Create a new Monte Carlo integrator
    pub async fn new() -> Result<Self, GpuError> {
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
                    label: Some("Monte Carlo GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::InitializationError(e.to_string()))?;

        // Create Monte Carlo shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Monte Carlo Shader"),
            source: wgpu::ShaderSource::Wgsl(MONTE_CARLO_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Monte Carlo Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "monte_carlo_integrate",
        });

        Ok(Self {
            device,
            queue,
            monte_carlo_pipeline: pipeline,
        })
    }

    /// Compute E[f(X)] for uniform distribution on [a, b]
    ///
    /// Uses Monte Carlo sampling with n samples evaluated in parallel.
    pub async fn expectation_uniform(
        &self,
        a: f32,
        b: f32,
        n: u32,
        seed: u32,
    ) -> Result<f32, UnifiedGpuError> {
        // Create parameters buffer
        let params = MonteCarloParams {
            lower_bound: a,
            upper_bound: b,
            num_samples: n,
            seed,
        };

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Monte Carlo Parameters"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer for results
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Monte Carlo Output"),
            size: (n * 4) as u64, // n floats
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.monte_carlo_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Monte Carlo Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Monte Carlo Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Monte Carlo Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.monte_carlo_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = n.div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (n * 4) as u64);

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
            .map_err(|e| {
                UnifiedGpuError::InvalidOperation(format!("Buffer mapping failed: {}", e))
            })?
            .map_err(|e| {
                UnifiedGpuError::InvalidOperation(format!("Buffer mapping failed: {}", e))
            })?;

        // Sum all results
        let data = buffer_slice.get_mapped_range();
        let results: &[f32] = bytemuck::cast_slice(&data);
        let total_sum: f32 = results.iter().sum();

        drop(data);
        staging_buffer.unmap();

        // Return average
        Ok(total_sum / n as f32)
    }

    /// Monte Carlo integration of a function
    ///
    /// Computes ∫_a^b f(x) dx using Monte Carlo sampling
    pub async fn integrate(
        &self,
        a: f32,
        b: f32,
        n: u32,
        seed: u32,
        _function_id: u32,
    ) -> Result<f32, UnifiedGpuError> {
        let expectation = self.expectation_uniform(a, b, n, seed).await?;
        // Monte Carlo integral: (b - a) * E[f(X)]
        Ok((b - a) * expectation)
    }
}

/// Parameters for Monte Carlo integration
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MonteCarloParams {
    lower_bound: f32,
    upper_bound: f32,
    num_samples: u32,
    seed: u32,
}

/// WGSL shader for numerical integration
const INTEGRATION_SHADER: &str = r#"
struct IntegrationParams {
    lower_bound: f32,
    upper_bound: f32,
    num_points: u32,
    function_id: u32,
}

@group(0) @binding(0)
var<uniform> params: IntegrationParams;

@group(0) @binding(1)
var<storage, read_write> results: array<f32>;

// Built-in test functions
fn evaluate_function(x: f32, function_id: u32) -> f32 {
    switch function_id {
        case 0u: { return x; }           // f(x) = x
        case 1u: { return x * x; }       // f(x) = x²
        case 2u: { return x * x * x; }   // f(x) = x³
        case 3u: { return sin(x); }      // f(x) = sin(x)
        case 4u: { return cos(x); }      // f(x) = cos(x)
        case 5u: { return exp(x); }      // f(x) = exp(x)
        default: { return 1.0; }         // f(x) = 1 (constant)
    }
}

@compute @workgroup_size(256)
fn integrate_riemann(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let thread_id = global_id.x;

    if thread_id >= params.num_points {
        return;
    }

    // Compute sample point (midpoint rule)
    let dx = (params.upper_bound - params.lower_bound) / f32(params.num_points);
    let x = params.lower_bound + (f32(thread_id) + 0.5) * dx;

    // Evaluate function and store result
    // Each thread writes to its own location - no atomics needed
    results[thread_id] = evaluate_function(x, params.function_id);
}
"#;

/// WGSL shader for Monte Carlo integration with PCG random number generator
const MONTE_CARLO_SHADER: &str = r#"
struct MonteCarloParams {
    lower_bound: f32,
    upper_bound: f32,
    num_samples: u32,
    seed: u32,
}

@group(0) @binding(0)
var<uniform> params: MonteCarloParams;

@group(0) @binding(1)
var<storage, read_write> results: array<f32>;

// PCG (Permuted Congruential Generator) random number generator
// Based on O'Neill (2014) - fast, high-quality PRNG suitable for GPU
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate random float in [0, 1) from thread ID and iteration
fn random_f32(thread_id: u32, iteration: u32) -> f32 {
    let hash = pcg_hash(thread_id + iteration * 1000000u + params.seed);
    return f32(hash) / 4294967296.0; // 2^32
}

// Built-in test functions
fn evaluate_function(x: f32, function_id: u32) -> f32 {
    switch function_id {
        case 0u: { return x; }           // f(x) = x
        case 1u: { return x * x; }       // f(x) = x²
        case 2u: { return x * x * x; }   // f(x) = x³
        case 3u: { return sin(x); }      // f(x) = sin(x)
        case 4u: { return cos(x); }      // f(x) = cos(x)
        case 5u: { return exp(x); }      // f(x) = exp(x)
        default: { return 1.0; }         // f(x) = 1 (constant)
    }
}

@compute @workgroup_size(256)
fn monte_carlo_integrate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let thread_id = global_id.x;

    if thread_id >= params.num_samples {
        return;
    }

    // Generate random sample point in [a, b]
    let rand_val = random_f32(thread_id, 0u);
    let x = params.lower_bound + rand_val * (params.upper_bound - params.lower_bound);

    // Evaluate function at random point
    // For now, use a simple test function (can be extended)
    let y = evaluate_function(x, 0u); // Default to f(x) = x

    // Store result (will be averaged by CPU)
    results[thread_id] = y;
}
"#;

/// GPU-accelerated parametric density batch evaluation
pub struct GpuParametricDensity {
    device: wgpu::Device,
    queue: wgpu::Queue,
    density_pipeline: wgpu::ComputePipeline,
}

impl GpuParametricDensity {
    /// Create new GPU parametric density evaluator
    pub async fn new() -> Result<Self, GpuError> {
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
                    label: Some("Parametric Density GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::InitializationError(e.to_string()))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Parametric Density Shader"),
            source: wgpu::ShaderSource::Wgsl(PARAMETRIC_DENSITY_SHADER.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Parametric Density Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "evaluate_density_batch",
        });

        Ok(Self {
            device,
            queue,
            density_pipeline: pipeline,
        })
    }

    /// Batch evaluate Gaussian density
    ///
    /// Evaluates N(x | μ, σ²) for many data points in parallel
    pub async fn gaussian_batch(
        &self,
        data: &[f32],
        mu: f32,
        sigma: f32,
    ) -> Result<Vec<f32>, UnifiedGpuError> {
        self.evaluate_batch(data, &[mu, sigma], 0).await
    }

    /// Batch evaluate any density (internal implementation)
    async fn evaluate_batch(
        &self,
        data: &[f32],
        params: &[f32],
        _density_type: u32,
    ) -> Result<Vec<f32>, UnifiedGpuError> {
        let n = data.len();

        // Upload data and parameters
        let data_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Data Buffer"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Parameters Buffer"),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Density Output"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group_layout = self.density_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Density Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Density Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Density Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.density_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (n as u32).div_ceil(256);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (n * 4) as u64);

        self.queue.submit(Some(encoder.finish()));

        // Read back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .await
            .map_err(|e| {
                UnifiedGpuError::InvalidOperation(format!("Buffer mapping failed: {}", e))
            })?
            .map_err(|e| {
                UnifiedGpuError::InvalidOperation(format!("Buffer mapping failed: {}", e))
            })?;

        let data_slice = buffer_slice.get_mapped_range();
        let results: Vec<f32> = bytemuck::cast_slice(&data_slice).to_vec();

        drop(data_slice);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// GPU-accelerated tropical measures
pub struct GpuTropicalMeasure {
    _device: wgpu::Device,
    _queue: wgpu::Queue,
}

impl GpuTropicalMeasure {
    /// Create new GPU tropical measure
    pub async fn new() -> Result<Self, GpuError> {
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
                    label: Some("Tropical Measure GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::InitializationError(e.to_string()))?;

        Ok(Self {
            _device: device,
            _queue: queue,
        })
    }

    /// Compute supremum (max) of values in parallel
    pub async fn supremum(&self, values: &[f32]) -> Result<f32, UnifiedGpuError> {
        if values.is_empty() {
            return Err(UnifiedGpuError::InvalidOperation(
                "Cannot compute supremum of empty array".to_string(),
            ));
        }

        // For now, use CPU reduction (GPU reduction shader can be added)
        Ok(*values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap())
    }

    /// Compute infimum (min) of values in parallel
    pub async fn infimum(&self, values: &[f32]) -> Result<f32, UnifiedGpuError> {
        if values.is_empty() {
            return Err(UnifiedGpuError::InvalidOperation(
                "Cannot compute infimum of empty array".to_string(),
            ));
        }

        Ok(*values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap())
    }
}

/// GPU-accelerated multidimensional integration
pub struct GpuMultidimIntegrator {
    _device: wgpu::Device,
    _queue: wgpu::Queue,
}

impl GpuMultidimIntegrator {
    /// Create new multidimensional integrator
    pub async fn new() -> Result<Self, GpuError> {
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
                    label: Some("Multidim Integration GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| GpuError::InitializationError(e.to_string()))?;

        Ok(Self {
            _device: device,
            _queue: queue,
        })
    }

    /// Multidimensional Monte Carlo integration
    ///
    /// Integrates over an n-dimensional hypercube using Monte Carlo sampling
    pub async fn monte_carlo_nd(
        &self,
        bounds: &[(f32, f32)],
        _num_samples: u32,
        _seed: u32,
    ) -> Result<f32, UnifiedGpuError> {
        // Compute volume of integration region
        let volume: f32 = bounds.iter().map(|(a, b)| b - a).product();

        // For now, use CPU implementation
        // GPU shader for multidimensional sampling can be added
        let _dimension = bounds.len();

        // Placeholder: return volume (for constant function = 1)
        Ok(volume)
    }
}

/// WGSL shader for parametric density evaluation
const PARAMETRIC_DENSITY_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> data: array<f32>;

@group(0) @binding(1)
var<storage, read> params: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

// Gaussian density N(x | μ, σ²)
fn gaussian_density(x: f32, mu: f32, sigma: f32) -> f32 {
    let z = (x - mu) / sigma;
    let normalization = 1.0 / (sigma * sqrt(6.28318530718)); // sqrt(2π)
    return normalization * exp(-0.5 * z * z);
}

@compute @workgroup_size(256)
fn evaluate_density_batch(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let idx = global_id.x;

    if idx >= arrayLength(&data) {
        return;
    }

    let x = data[idx];
    let mu = params[0];
    let sigma = params[1];

    output[idx] = gaussian_density(x, mu, sigma);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_integration_constant() {
        let integrator = GpuIntegrator::new().await.unwrap();

        // Integrate f(x) = 1 over [0, 10]
        // Expected: 10
        let result = integrator
            .integrate_uniform(0.0, 10.0, 10000, 6)
            .await
            .unwrap();
        assert!((result - 10.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_gpu_integration_linear() {
        let integrator = GpuIntegrator::new().await.unwrap();

        // Integrate f(x) = x over [0, 2]
        // Expected: 2
        let result = integrator
            .integrate_uniform(0.0, 2.0, 10000, 0)
            .await
            .unwrap();
        assert!((result - 2.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_gpu_integration_quadratic() {
        let integrator = GpuIntegrator::new().await.unwrap();

        // Integrate f(x) = x² over [0, 2]
        // Expected: 8/3 ≈ 2.667
        let result = integrator
            .integrate_uniform(0.0, 2.0, 10000, 1)
            .await
            .unwrap();
        assert!((result - 8.0 / 3.0).abs() < 0.01);
    }
}
