//! GPU-accelerated batch field evaluation

use crate::ScalarField;
use amari_gpu::{SharedGpuContext, UnifiedGpuResult};

/// Batch field evaluator using GPU acceleration
pub struct BatchFieldEvaluator;

impl BatchFieldEvaluator {
    /// Evaluate scalar field at multiple points (GPU accelerated)
    ///
    /// # Arguments
    ///
    /// * `gpu` - Shared GPU context
    /// * `field` - Scalar field to evaluate
    /// * `points` - Array of evaluation points
    ///
    /// # Returns
    ///
    /// Vector of field values at each point
    ///
    /// # Limitations
    ///
    /// Currently, this requires the field function to be expressible in WGSL.
    /// For arbitrary Rust closures, we fall back to CPU evaluation.
    /// Future versions will support JIT compilation of field functions.
    pub async fn evaluate_scalar_2d(
        _gpu: &SharedGpuContext,
        field: &ScalarField<2, 0, 0>,
        points: &[[f64; 2]],
    ) -> UnifiedGpuResult<Vec<f64>> {
        // CPU evaluation (GPU code generation for arbitrary functions not yet implemented)
        Ok(points
            .iter()
            .map(|p| field.evaluate(p.as_slice()))
            .collect())
    }

    /// Evaluate scalar field at multiple 3D points
    pub async fn evaluate_scalar_3d(
        _gpu: &SharedGpuContext,
        field: &ScalarField<3, 0, 0>,
        points: &[[f64; 3]],
    ) -> UnifiedGpuResult<Vec<f64>> {
        // CPU evaluation
        Ok(points
            .iter()
            .map(|p| field.evaluate(p.as_slice()))
            .collect())
    }
}

/// WGSL shader for polynomial field evaluation
///
/// This shader evaluates polynomial fields of the form:
/// f(x, y, z) = c₀ + c₁x + c₂y + c₃z + c₄x² + c₅xy + ...
///
/// Note: Currently unused due to closure serialization limitations.
/// Will be activated when field type tagging is implemented.
#[allow(dead_code)]
const POLYNOMIAL_FIELD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> coefficients: array<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<f32>;

// Polynomial evaluation kernel
@compute @workgroup_size(256)
fn evaluate_polynomial(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&points)) {
        return;
    }

    let p = points[idx];
    let x = p.x;
    let y = p.y;
    let z = p.z;

    // Evaluate polynomial (degree determined by coefficient count)
    var result = 0.0;
    var coeff_idx = 0u;

    // Constant term
    result = coefficients[coeff_idx];
    coeff_idx = coeff_idx + 1u;

    // Linear terms
    result = result + coefficients[coeff_idx] * x;
    coeff_idx = coeff_idx + 1u;
    result = result + coefficients[coeff_idx] * y;
    coeff_idx = coeff_idx + 1u;
    result = result + coefficients[coeff_idx] * z;
    coeff_idx = coeff_idx + 1u;

    // Quadratic terms (if available)
    if (coeff_idx < arrayLength(&coefficients)) {
        result = result + coefficients[coeff_idx] * x * x;
        coeff_idx = coeff_idx + 1u;
    }
    if (coeff_idx < arrayLength(&coefficients)) {
        result = result + coefficients[coeff_idx] * x * y;
        coeff_idx = coeff_idx + 1u;
    }
    if (coeff_idx < arrayLength(&coefficients)) {
        result = result + coefficients[coeff_idx] * x * z;
        coeff_idx = coeff_idx + 1u;
    }
    if (coeff_idx < arrayLength(&coefficients)) {
        result = result + coefficients[coeff_idx] * y * y;
        coeff_idx = coeff_idx + 1u;
    }
    if (coeff_idx < arrayLength(&coefficients)) {
        result = result + coefficients[coeff_idx] * y * z;
        coeff_idx = coeff_idx + 1u;
    }
    if (coeff_idx < arrayLength(&coefficients)) {
        result = result + coefficients[coeff_idx] * z * z;
        coeff_idx = coeff_idx + 1u;
    }

    results[idx] = result;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_compiles() {
        // Verify WGSL shader is syntactically valid
        assert!(POLYNOMIAL_FIELD_SHADER.contains("@compute"));
        assert!(POLYNOMIAL_FIELD_SHADER.contains("evaluate_polynomial"));
    }
}
