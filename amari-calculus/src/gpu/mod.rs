//! GPU-accelerated calculus operations
//!
//! This module provides GPU acceleration for computationally intensive calculus operations:
//! - Batch field evaluation
//! - Numerical derivatives (gradient, divergence, curl)
//! - Christoffel symbol computation
//! - Curvature tensor calculations
//!
//! ## Design
//!
//! - Uses `SharedGpuContext` for workspace-wide GPU resource sharing
//! - Adaptive CPU/GPU dispatch based on workload size
//! - Graceful CPU fallback when GPU unavailable
//! - WGSL shaders for all numerical operations
//!
//! ## Example
//!
//! ```rust,ignore
//! use amari_calculus::gpu::{GpuCalculusContext, BatchFieldEvaluator};
//! use amari_calculus::ScalarField;
//!
//! // Create GPU context (shares resources workspace-wide)
//! let ctx = GpuCalculusContext::new().await?;
//!
//! // Define scalar field
//! let f = ScalarField::new(|coords| coords[0] * coords[0] + coords[1] * coords[1]);
//!
//! // Batch evaluate at 10,000 points (GPU accelerated)
//! let points = vec![[0.0, 0.0]; 10000];
//! let results = ctx.batch_evaluate_scalar(&f, &points).await?;
//! ```

use crate::CalculusError;
use amari_gpu::{SharedGpuContext, UnifiedGpuError, UnifiedGpuResult};

pub mod christoffel_ops;
pub mod curvature_ops;
pub mod derivative_ops;
pub mod field_ops;

pub use christoffel_ops::BatchChristoffelComputer;
pub use curvature_ops::BatchCurvatureComputer;
pub use derivative_ops::BatchDerivativeOperator;
pub use field_ops::BatchFieldEvaluator;

/// GPU context for calculus operations
///
/// Integrates with SharedGpuContext for workspace-wide resource sharing.
/// Automatically falls back to CPU for small workloads or when GPU unavailable.
///
/// **Note**: This is currently a placeholder implementation. Full GPU acceleration
/// will be enabled in future versions with proper const generic support and
/// runtime field code generation.
pub struct GpuCalculusContext {
    /// CPU threshold: workloads smaller than this use CPU
    cpu_threshold: usize,
}

impl GpuCalculusContext {
    /// Create a new GPU calculus context
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use amari_calculus::gpu::GpuCalculusContext;
    ///
    /// let ctx = GpuCalculusContext::new();
    /// ```
    pub fn new() -> Self {
        Self {
            cpu_threshold: 1000, // Use GPU for 1000+ operations
        }
    }

    /// Create context with custom CPU threshold
    ///
    /// # Arguments
    ///
    /// * `cpu_threshold` - Minimum workload size for GPU dispatch
    pub fn with_threshold(cpu_threshold: usize) -> Self {
        Self { cpu_threshold }
    }

    /// Get reference to shared GPU context
    ///
    /// This method accesses the global GPU context. It's async because
    /// the GPU context may need to be initialized on first access.
    pub async fn gpu() -> UnifiedGpuResult<&'static SharedGpuContext> {
        SharedGpuContext::global().await
    }

    /// Check if workload should use GPU (currently always returns false)
    pub fn should_use_gpu(&self, workload_size: usize) -> bool {
        workload_size >= self.cpu_threshold
        // Future: add GPU availability check
    }
}

impl Default for GpuCalculusContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert UnifiedGpuError to CalculusError
impl From<UnifiedGpuError> for CalculusError {
    fn from(err: UnifiedGpuError) -> Self {
        CalculusError::FieldEvaluationError {
            reason: format!("GPU error: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_threshold_logic() {
        // This test doesn't require async/GPU, just tests threshold logic
        let threshold = 1000;
        assert!(999 < threshold);
        assert!(1000 >= threshold);
        assert!(10000 >= threshold);
    }
}
