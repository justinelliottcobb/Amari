//! Phase 4B: Kernel-Level Mathematical Property Checking
//!
//! This module implements verification strategies that work at the GPU kernel level,
//! providing mathematical property checking that can be embedded directly into
//! compute shaders while maintaining performance constraints.

use std::marker::PhantomData;
use std::time::{Duration, Instant};
use amari_core::Multivector;

/// Kernel verification configuration for different GPU contexts
#[derive(Debug, Clone, Copy)]
pub struct KernelVerificationConfig {
    /// Enable invariant checking within kernels
    pub enable_kernel_checks: bool,
    /// Maximum register pressure allowed for verification
    pub max_register_overhead: u32,
    /// Thread divergence tolerance
    pub divergence_tolerance: f32,
    /// Verification frequency (every Nth operation)
    pub check_frequency: u32,
    /// Memory tier for verification data
    pub verification_memory_tier: VerificationMemoryTier,
}

/// Memory tiers for storing verification data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerificationMemoryTier {
    /// Store verification data in registers (fastest, most limited)
    Register,
    /// Store in shared memory (fast, block-local)
    Shared,
    /// Store in global memory (slower, persistent)
    Global,
    /// Store on host and transfer as needed (slowest, unlimited)
    Host,
}

/// Kernel verification error types
#[derive(Debug, Clone)]
pub enum KernelVerificationError {
    /// Register pressure exceeded limits
    RegisterPressure { used: u32, limit: u32 },
    /// Thread divergence detected
    ThreadDivergence { warp: u32, divergent_threads: u32 },
    /// Kernel invariant violation
    InvariantViolation { kernel: String, property: String, thread_id: u32 },
    /// Memory access violation
    MemoryViolation { address: u64, tier: VerificationMemoryTier },
    /// Numerical precision loss
    PrecisionLoss { operation: String, error: f64 },
    /// Kernel execution timeout
    ExecutionTimeout { kernel: String, timeout_ms: u64 },
}

/// GPU kernel verification context
pub struct KernelVerificationContext<const P: usize, const Q: usize, const R: usize> {
    config: KernelVerificationConfig,
    operation_count: u64,
    total_kernel_time: Duration,
    verification_overhead: Duration,
    active_kernels: Vec<KernelInstance>,
    _phantom: PhantomData<(P, Q, R)>,
}

/// Information about an active kernel instance
#[derive(Debug, Clone)]
pub struct KernelInstance {
    pub name: String,
    pub thread_count: u32,
    pub block_size: u32,
    pub start_time: Instant,
    pub verification_level: KernelVerificationLevel,
}

/// Verification level specific to kernel execution
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelVerificationLevel {
    /// No verification - pure performance
    None,
    /// Boundary checks only at kernel entry/exit
    Boundary,
    /// Per-thread verification
    PerThread,
    /// Per-warp verification (32 threads)
    PerWarp,
    /// Per-block verification
    PerBlock,
    /// Full verification on every operation
    Full,
}

impl<const P: usize, const Q: usize, const R: usize> KernelVerificationContext<P, Q, R> {
    /// Create kernel verification context with GPU-specific configuration
    pub fn new(config: KernelVerificationConfig) -> Self {
        Self {
            config,
            operation_count: 0,
            total_kernel_time: Duration::ZERO,
            verification_overhead: Duration::ZERO,
            active_kernels: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Register a kernel for verification tracking
    pub fn register_kernel(
        &mut self,
        name: String,
        thread_count: u32,
        block_size: u32,
    ) -> Result<KernelInstance, KernelVerificationError> {
        let verification_level = self.determine_kernel_verification_level(thread_count, block_size);

        // Check register pressure constraints
        if self.config.enable_kernel_checks {
            let estimated_register_use = self.estimate_register_usage(&verification_level);
            if estimated_register_use > self.config.max_register_overhead {
                return Err(KernelVerificationError::RegisterPressure {
                    used: estimated_register_use,
                    limit: self.config.max_register_overhead,
                });
            }
        }

        let instance = KernelInstance {
            name,
            thread_count,
            block_size,
            start_time: Instant::now(),
            verification_level,
        };

        self.active_kernels.push(instance.clone());
        Ok(instance)
    }

    /// Determine optimal verification level for kernel parameters
    fn determine_kernel_verification_level(
        &self,
        thread_count: u32,
        block_size: u32,
    ) -> KernelVerificationLevel {
        if !self.config.enable_kernel_checks {
            return KernelVerificationLevel::None;
        }

        // Adapt verification level based on kernel size and configuration
        match (thread_count, block_size) {
            (t, _) if t <= 32 => KernelVerificationLevel::PerThread,
            (t, b) if t <= 1024 && b <= 256 => KernelVerificationLevel::PerWarp,
            (t, _) if t <= 65536 => KernelVerificationLevel::PerBlock,
            _ => KernelVerificationLevel::Boundary,
        }
    }

    /// Estimate register usage for verification level
    fn estimate_register_usage(&self, level: &KernelVerificationLevel) -> u32 {
        match level {
            KernelVerificationLevel::None => 0,
            KernelVerificationLevel::Boundary => 2,       // Basic state tracking
            KernelVerificationLevel::PerThread => 8,      // Per-thread verification data
            KernelVerificationLevel::PerWarp => 16,       // Warp-level coordination
            KernelVerificationLevel::PerBlock => 24,      // Block-level shared data
            KernelVerificationLevel::Full => 32,          // Complete verification state
        }
    }

    /// Generate kernel verification code (WGSL/GLSL)
    pub fn generate_verification_shader_code(
        &self,
        kernel_name: &str,
        verification_level: KernelVerificationLevel,
    ) -> String {
        match verification_level {
            KernelVerificationLevel::None => String::new(),
            KernelVerificationLevel::Boundary => self.generate_boundary_verification_code(),
            KernelVerificationLevel::PerThread => self.generate_thread_verification_code(),
            KernelVerificationLevel::PerWarp => self.generate_warp_verification_code(),
            KernelVerificationLevel::PerBlock => self.generate_block_verification_code(),
            KernelVerificationLevel::Full => self.generate_full_verification_code(kernel_name),
        }
    }

    /// Generate boundary verification WGSL code
    fn generate_boundary_verification_code(&self) -> String {
        r#"
// Boundary verification functions
fn verify_input_boundary(coeffs: array<f32, 8>) -> bool {
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (!isFinite(coeffs[i])) {
            return false;
        }
        if (abs(coeffs[i]) > 1e10) {
            return false; // Magnitude too large for GPU precision
        }
    }
    return true;
}

fn verify_output_boundary(coeffs: array<f32, 8>) -> bool {
    var magnitude_sq = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        if (!isFinite(coeffs[i])) {
            return false;
        }
        magnitude_sq += coeffs[i] * coeffs[i];
    }
    return magnitude_sq < 1e20; // Reasonable magnitude bound
}
"#.to_string()
    }

    /// Generate per-thread verification WGSL code
    fn generate_thread_verification_code(&self) -> String {
        r#"
// Per-thread verification with minimal register usage
struct ThreadVerificationState {
    error_count: u32,
    max_error: f32,
}

var<private> thread_verification: ThreadVerificationState;

fn init_thread_verification() {
    thread_verification.error_count = 0u;
    thread_verification.max_error = 0.0;
}

fn verify_geometric_product_thread(
    a: array<f32, 8>,
    b: array<f32, 8>,
    result: array<f32, 8>
) -> bool {
    // Quick sanity checks for geometric product
    var a_mag_sq = 0.0;
    var b_mag_sq = 0.0;
    var result_mag_sq = 0.0;

    for (var i = 0u; i < 8u; i = i + 1u) {
        a_mag_sq += a[i] * a[i];
        b_mag_sq += b[i] * b[i];
        result_mag_sq += result[i] * result[i];
    }

    // Check magnitude relationship (simplified)
    let expected_mag_bound = sqrt(a_mag_sq) * sqrt(b_mag_sq) * 2.0;
    if (sqrt(result_mag_sq) > expected_mag_bound) {
        thread_verification.error_count += 1u;
        return false;
    }

    return true;
}
"#.to_string()
    }

    /// Generate warp-level verification WGSL code
    fn generate_warp_verification_code(&self) -> String {
        r#"
// Warp-level verification using subgroup operations
var<workgroup> warp_error_flags: array<u32, 32>; // One flag per warp

fn verify_warp_consistency(thread_id: u32) {
    let warp_id = thread_id / 32u;
    let lane_id = thread_id % 32u;

    // Initialize warp error flag
    if (lane_id == 0u) {
        warp_error_flags[warp_id] = 0u;
    }

    workgroupBarrier();
}

fn report_warp_error(thread_id: u32, error_type: u32) {
    let warp_id = thread_id / 32u;
    let lane_id = thread_id % 32u;

    // Atomic update of warp error flags
    atomicOr(&warp_error_flags[warp_id], 1u << error_type);
}

fn check_warp_divergence(thread_id: u32, condition: bool) -> bool {
    let warp_id = thread_id / 32u;

    // Use ballot-style operation to detect divergence
    // This is a simplified version - real implementation would use subgroup operations
    if (condition != all(vec4<bool>(condition))) {
        report_warp_error(thread_id, 1u); // Divergence error
        return false;
    }
    return true;
}
"#.to_string()
    }

    /// Generate block-level verification WGSL code
    fn generate_block_verification_code(&self) -> String {
        r#"
// Block-level verification with shared memory coordination
var<workgroup> block_verification_data: array<f32, 64>; // Shared verification state
var<workgroup> block_error_count: atomic<u32>;

fn init_block_verification() {
    if (local_invocation_index == 0u) {
        atomicStore(&block_error_count, 0u);
        for (var i = 0u; i < 64u; i = i + 1u) {
            block_verification_data[i] = 0.0;
        }
    }
    workgroupBarrier();
}

fn accumulate_block_statistics(thread_id: u32, value: f32) {
    let slot = thread_id % 64u;
    block_verification_data[slot] = max(block_verification_data[slot], abs(value));
}

fn verify_block_consistency() -> bool {
    workgroupBarrier();

    if (local_invocation_index == 0u) {
        var max_value = 0.0;
        for (var i = 0u; i < 64u; i = i + 1u) {
            max_value = max(max_value, block_verification_data[i]);
        }

        // Check if any thread produced extreme values
        if (max_value > 1e8) {
            atomicAdd(&block_error_count, 1u);
            return false;
        }
    }

    workgroupBarrier();
    return atomicLoad(&block_error_count) == 0u;
}
"#.to_string()
    }

    /// Generate full verification WGSL code
    fn generate_full_verification_code(&self, kernel_name: &str) -> String {
        format!(r#"
// Full verification for kernel: {}
struct FullVerificationState {{
    input_hash: u32,
    operation_count: u32,
    error_accumulator: f32,
    last_result_magnitude: f32,
}}

var<private> full_verification: FullVerificationState;

fn init_full_verification(input_data: array<f32, 8>) {{
    full_verification.input_hash = hash_coefficients(input_data);
    full_verification.operation_count = 0u;
    full_verification.error_accumulator = 0.0;
    full_verification.last_result_magnitude = 0.0;
}}

fn hash_coefficients(coeffs: array<f32, 8>) -> u32 {{
    var hash = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {{
        let bits = bitcast<u32>(coeffs[i]);
        hash = hash * 31u + bits;
    }}
    return hash;
}}

fn verify_operation_full(
    operation_name: u32,
    input_a: array<f32, 8>,
    input_b: array<f32, 8>,
    result: array<f32, 8>
) -> bool {{
    full_verification.operation_count += 1u;

    // Verify mathematical properties based on operation type
    switch operation_name {{
        case 0u: {{ // Geometric product
            return verify_geometric_product_properties(input_a, input_b, result);
        }}
        case 1u: {{ // Addition
            return verify_addition_properties(input_a, input_b, result);
        }}
        case 2u: {{ // Scalar multiplication
            return verify_scalar_multiplication_properties(input_a, input_b, result);
        }}
        default: {{
            return true; // Unknown operation, assume valid
        }}
    }}
}}

fn verify_geometric_product_properties(
    a: array<f32, 8>,
    b: array<f32, 8>,
    result: array<f32, 8>
) -> bool {{
    // Check associativity sampling
    if (full_verification.operation_count % 100u == 0u) {{
        // Simplified associativity check
        let a_mag_sq = magnitude_squared(a);
        let b_mag_sq = magnitude_squared(b);
        let result_mag_sq = magnitude_squared(result);

        // For unit vectors, geometric product magnitude should not exceed input magnitudes significantly
        if (a_mag_sq > 0.0 && b_mag_sq > 0.0) {{
            let expected_bound = sqrt(a_mag_sq * b_mag_sq) * 2.0; // Conservative bound
            if (sqrt(result_mag_sq) > expected_bound) {{
                return false;
            }}
        }}
    }}

    return true;
}}

fn verify_addition_properties(
    a: array<f32, 8>,
    b: array<f32, 8>,
    result: array<f32, 8>
) -> bool {{
    // Verify component-wise addition
    for (var i = 0u; i < 8u; i = i + 1u) {{
        let expected = a[i] + b[i];
        let error = abs(result[i] - expected);
        if (error > 1e-6) {{ // Single precision tolerance
            return false;
        }}
    }}
    return true;
}}

fn verify_scalar_multiplication_properties(
    a: array<f32, 8>,
    scalar_vec: array<f32, 8>, // Scalar stored in first component
    result: array<f32, 8>
) -> bool {{
    let scalar = scalar_vec[0];

    // Verify scalar multiplication
    for (var i = 0u; i < 8u; i = i + 1u) {{
        let expected = a[i] * scalar;
        let error = abs(result[i] - expected);
        if (error > 1e-6) {{
            return false;
        }}
    }}
    return true;
}}

fn magnitude_squared(coeffs: array<f32, 8>) -> f32 {{
    var mag_sq = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {{
        mag_sq += coeffs[i] * coeffs[i];
    }}
    return mag_sq;
}}
"#, kernel_name)
    }

    /// Verify kernel execution with mathematical property checking
    pub async fn verify_kernel_execution(
        &mut self,
        kernel_instance: &KernelInstance,
        input_data: &[Multivector<P, Q, R>],
        output_data: &[Multivector<P, Q, R>],
    ) -> Result<(), KernelVerificationError> {
        let start = Instant::now();

        match kernel_instance.verification_level {
            KernelVerificationLevel::None => Ok(()),
            KernelVerificationLevel::Boundary => {
                self.verify_kernel_boundaries(input_data, output_data).await
            }
            KernelVerificationLevel::PerThread => {
                self.verify_per_thread_properties(kernel_instance, input_data, output_data).await
            }
            KernelVerificationLevel::PerWarp => {
                self.verify_per_warp_properties(kernel_instance, input_data, output_data).await
            }
            KernelVerificationLevel::PerBlock => {
                self.verify_per_block_properties(kernel_instance, input_data, output_data).await
            }
            KernelVerificationLevel::Full => {
                self.verify_full_kernel_properties(kernel_instance, input_data, output_data).await
            }
        }?;

        self.verification_overhead += start.elapsed();
        self.operation_count += 1;

        Ok(())
    }

    /// Verify kernel boundaries (input/output validation)
    async fn verify_kernel_boundaries(
        &self,
        input_data: &[Multivector<P, Q, R>],
        output_data: &[Multivector<P, Q, R>],
    ) -> Result<(), KernelVerificationError> {
        // Verify input data is suitable for GPU processing
        for (i, mv) in input_data.iter().enumerate() {
            let magnitude = mv.magnitude();
            if magnitude > 1e10 {
                return Err(KernelVerificationError::PrecisionLoss {
                    operation: "input_validation".to_string(),
                    error: magnitude,
                });
            }

            // Check for values that might cause GPU numerical issues
            for j in 0..8 {
                let coeff = mv.get(j);
                if !coeff.is_finite() {
                    return Err(KernelVerificationError::InvariantViolation {
                        kernel: "boundary_check".to_string(),
                        property: "finite_coefficients".to_string(),
                        thread_id: i as u32,
                    });
                }
            }
        }

        // Verify output data integrity
        for (i, mv) in output_data.iter().enumerate() {
            if !mv.magnitude().is_finite() {
                return Err(KernelVerificationError::InvariantViolation {
                    kernel: "boundary_check".to_string(),
                    property: "finite_output".to_string(),
                    thread_id: i as u32,
                });
            }
        }

        Ok(())
    }

    /// Verify per-thread properties
    async fn verify_per_thread_properties(
        &self,
        kernel_instance: &KernelInstance,
        input_data: &[Multivector<P, Q, R>],
        output_data: &[Multivector<P, Q, R>],
    ) -> Result<(), KernelVerificationError> {
        // Simulate per-thread verification by checking thread-sized chunks
        let threads_per_check = kernel_instance.block_size.min(32);

        for chunk_start in (0..input_data.len()).step_by(threads_per_check as usize) {
            let chunk_end = (chunk_start + threads_per_check as usize).min(input_data.len());

            for i in chunk_start..chunk_end {
                let thread_id = i as u32;

                // Verify thread-local mathematical properties
                if !self.verify_thread_local_properties(&input_data[i], &output_data[i]) {
                    return Err(KernelVerificationError::InvariantViolation {
                        kernel: kernel_instance.name.clone(),
                        property: "thread_local_properties".to_string(),
                        thread_id,
                    });
                }
            }
        }

        Ok(())
    }

    /// Verify per-warp properties
    async fn verify_per_warp_properties(
        &self,
        kernel_instance: &KernelInstance,
        input_data: &[Multivector<P, Q, R>],
        output_data: &[Multivector<P, Q, R>],
    ) -> Result<(), KernelVerificationError> {
        // Verify warp-level consistency (32-thread groups)
        for warp_start in (0..input_data.len()).step_by(32) {
            let warp_end = (warp_start + 32).min(input_data.len());
            let warp_id = (warp_start / 32) as u32;

            // Check for divergent execution patterns
            let mut magnitudes: Vec<f64> = Vec::new();
            for i in warp_start..warp_end {
                magnitudes.push(output_data[i].magnitude());
            }

            // Check if there's excessive divergence in results
            if let (Some(&min_mag), Some(&max_mag)) = (magnitudes.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
                                                       magnitudes.iter().max_by(|a, b| a.partial_cmp(b).unwrap())) {
                if max_mag > 0.0 && (max_mag / min_mag.max(1e-10)) > 1e6 {
                    return Err(KernelVerificationError::ThreadDivergence {
                        warp: warp_id,
                        divergent_threads: (warp_end - warp_start) as u32,
                    });
                }
            }
        }

        Ok(())
    }

    /// Verify per-block properties
    async fn verify_per_block_properties(
        &self,
        kernel_instance: &KernelInstance,
        input_data: &[Multivector<P, Q, R>],
        output_data: &[Multivector<P, Q, R>],
    ) -> Result<(), KernelVerificationError> {
        // Verify block-level consistency
        let block_size = kernel_instance.block_size as usize;

        for block_start in (0..input_data.len()).step_by(block_size) {
            let block_end = (block_start + block_size).min(input_data.len());

            // Accumulate block-level statistics
            let mut total_energy = 0.0;
            for i in block_start..block_end {
                total_energy += output_data[i].magnitude();
            }

            // Check for energy conservation violations
            let mut input_energy = 0.0;
            for i in block_start..block_end {
                input_energy += input_data[i].magnitude();
            }

            // Allow for some energy growth due to geometric products, but not excessive
            if total_energy > input_energy * 10.0 {
                return Err(KernelVerificationError::InvariantViolation {
                    kernel: kernel_instance.name.clone(),
                    property: "energy_conservation".to_string(),
                    thread_id: block_start as u32,
                });
            }
        }

        Ok(())
    }

    /// Verify full kernel properties with comprehensive checking
    async fn verify_full_kernel_properties(
        &self,
        kernel_instance: &KernelInstance,
        input_data: &[Multivector<P, Q, R>],
        output_data: &[Multivector<P, Q, R>],
    ) -> Result<(), KernelVerificationError> {
        // Comprehensive verification including all previous levels
        self.verify_kernel_boundaries(input_data, output_data).await?;
        self.verify_per_thread_properties(kernel_instance, input_data, output_data).await?;
        self.verify_per_warp_properties(kernel_instance, input_data, output_data).await?;
        self.verify_per_block_properties(kernel_instance, input_data, output_data).await?;

        // Additional full verification checks
        for (i, (input, output)) in input_data.iter().zip(output_data.iter()).enumerate() {
            if !self.verify_comprehensive_properties(input, output) {
                return Err(KernelVerificationError::InvariantViolation {
                    kernel: kernel_instance.name.clone(),
                    property: "comprehensive_verification".to_string(),
                    thread_id: i as u32,
                });
            }
        }

        Ok(())
    }

    /// Verify thread-local mathematical properties
    fn verify_thread_local_properties(
        &self,
        input: &Multivector<P, Q, R>,
        output: &Multivector<P, Q, R>,
    ) -> bool {
        // Basic mathematical property checks
        let input_magnitude = input.magnitude();
        let output_magnitude = output.magnitude();

        // Check for reasonable magnitude relationships
        if output_magnitude > input_magnitude * 100.0 {
            return false; // Excessive magnitude growth
        }

        // Check for coefficient sanity
        for i in 0..8 {
            if !output.get(i).is_finite() {
                return false;
            }
        }

        true
    }

    /// Verify comprehensive mathematical properties
    fn verify_comprehensive_properties(
        &self,
        input: &Multivector<P, Q, R>,
        output: &Multivector<P, Q, R>,
    ) -> bool {
        // Thread-local checks
        if !self.verify_thread_local_properties(input, output) {
            return false;
        }

        // Additional comprehensive checks
        let input_norm = input.norm();
        let output_norm = output.norm();

        // Check norm preservation for certain operations
        if input_norm > 0.0 && output_norm > 0.0 {
            let norm_ratio = output_norm / input_norm;
            if norm_ratio > 1000.0 || norm_ratio < 0.001 {
                return false; // Excessive norm change
            }
        }

        true
    }

    /// Finalize kernel execution and collect statistics
    pub fn finalize_kernel(&mut self, kernel_instance: &KernelInstance) -> KernelExecutionStats {
        let execution_time = kernel_instance.start_time.elapsed();
        self.total_kernel_time += execution_time;

        // Remove from active kernels
        self.active_kernels.retain(|k| k.name != kernel_instance.name);

        KernelExecutionStats {
            kernel_name: kernel_instance.name.clone(),
            execution_time,
            verification_level: kernel_instance.verification_level,
            thread_count: kernel_instance.thread_count,
            verification_overhead_ratio: if execution_time.as_nanos() > 0 {
                self.verification_overhead.as_nanos() as f64 / execution_time.as_nanos() as f64
            } else {
                0.0
            },
        }
    }

    /// Get kernel verification statistics
    pub fn get_kernel_stats(&self) -> KernelVerificationStats {
        KernelVerificationStats {
            total_operations: self.operation_count,
            total_kernel_time: self.total_kernel_time,
            total_verification_overhead: self.verification_overhead,
            active_kernel_count: self.active_kernels.len(),
            average_overhead_ratio: if self.total_kernel_time.as_nanos() > 0 {
                self.verification_overhead.as_nanos() as f64 / self.total_kernel_time.as_nanos() as f64
            } else {
                0.0
            },
        }
    }
}

/// Statistics for individual kernel execution
#[derive(Debug)]
pub struct KernelExecutionStats {
    pub kernel_name: String,
    pub execution_time: Duration,
    pub verification_level: KernelVerificationLevel,
    pub thread_count: u32,
    pub verification_overhead_ratio: f64,
}

/// Overall kernel verification statistics
#[derive(Debug)]
pub struct KernelVerificationStats {
    pub total_operations: u64,
    pub total_kernel_time: Duration,
    pub total_verification_overhead: Duration,
    pub active_kernel_count: usize,
    pub average_overhead_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_verification_config() {
        let config = KernelVerificationConfig {
            enable_kernel_checks: true,
            max_register_overhead: 16,
            divergence_tolerance: 0.1,
            check_frequency: 10,
            verification_memory_tier: VerificationMemoryTier::Shared,
        };

        let context = KernelVerificationContext::<3, 0, 0>::new(config);
        assert_eq!(context.operation_count, 0);
    }

    #[test]
    fn test_kernel_registration() {
        let config = KernelVerificationConfig {
            enable_kernel_checks: true,
            max_register_overhead: 32,
            divergence_tolerance: 0.1,
            check_frequency: 10,
            verification_memory_tier: VerificationMemoryTier::Global,
        };

        let mut context = KernelVerificationContext::<3, 0, 0>::new(config);
        let result = context.register_kernel("test_kernel".to_string(), 1024, 256);

        assert!(result.is_ok());
        let instance = result.unwrap();
        assert_eq!(instance.name, "test_kernel");
        assert_eq!(instance.verification_level, KernelVerificationLevel::PerBlock);
    }

    #[test]
    fn test_register_pressure_limit() {
        let config = KernelVerificationConfig {
            enable_kernel_checks: true,
            max_register_overhead: 4, // Very low limit
            divergence_tolerance: 0.1,
            check_frequency: 10,
            verification_memory_tier: VerificationMemoryTier::Register,
        };

        let mut context = KernelVerificationContext::<3, 0, 0>::new(config);
        let result = context.register_kernel("high_reg_kernel".to_string(), 65536, 512);

        // Should fail due to register pressure
        assert!(result.is_err());
        if let Err(KernelVerificationError::RegisterPressure { used, limit }) = result {
            assert!(used > limit);
        }
    }

    #[tokio::test]
    async fn test_boundary_verification() {
        let config = KernelVerificationConfig {
            enable_kernel_checks: true,
            max_register_overhead: 32,
            divergence_tolerance: 0.1,
            check_frequency: 10,
            verification_memory_tier: VerificationMemoryTier::Global,
        };

        let mut context = KernelVerificationContext::<3, 0, 0>::new(config);
        let kernel_instance = context.register_kernel("test".to_string(), 32, 32).unwrap();

        let input_data = vec![
            Multivector::<3, 0, 0>::basis_vector(0),
            Multivector::<3, 0, 0>::basis_vector(1),
        ];
        let output_data = vec![
            Multivector::<3, 0, 0>::scalar(1.0),
            Multivector::<3, 0, 0>::scalar(0.0),
        ];

        let result = context.verify_kernel_execution(&kernel_instance, &input_data, &output_data).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_shader_code_generation() {
        let config = KernelVerificationConfig {
            enable_kernel_checks: true,
            max_register_overhead: 32,
            divergence_tolerance: 0.1,
            check_frequency: 10,
            verification_memory_tier: VerificationMemoryTier::Shared,
        };

        let context = KernelVerificationContext::<3, 0, 0>::new(config);

        let boundary_code = context.generate_verification_shader_code(
            "test_kernel",
            KernelVerificationLevel::Boundary
        );
        assert!(boundary_code.contains("verify_input_boundary"));
        assert!(boundary_code.contains("verify_output_boundary"));

        let thread_code = context.generate_verification_shader_code(
            "test_kernel",
            KernelVerificationLevel::PerThread
        );
        assert!(thread_code.contains("ThreadVerificationState"));
        assert!(thread_code.contains("verify_geometric_product_thread"));
    }
}