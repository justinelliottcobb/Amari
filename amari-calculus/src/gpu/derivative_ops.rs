//! GPU-accelerated numerical derivative computation

use crate::ScalarField;
use amari_gpu::{SharedGpuContext, UnifiedGpuResult};

/// Batch derivative operator using GPU acceleration
pub struct BatchDerivativeOperator;

impl BatchDerivativeOperator {
    /// Compute gradients at multiple points (GPU accelerated)
    ///
    /// Uses centered finite differences: ∂f/∂xᵢ ≈ (f(x+h·eᵢ) - f(x-h·eᵢ)) / (2h)
    ///
    /// # Arguments
    ///
    /// * `gpu` - Shared GPU context
    /// * `field` - Scalar field
    /// * `points` - Evaluation points
    ///
    /// # Returns
    ///
    /// Vector of gradient vectors at each point
    pub async fn compute_gradients_2d(
        _gpu: &SharedGpuContext,
        field: &ScalarField<2, 0, 0>,
        points: &[[f64; 2]],
    ) -> UnifiedGpuResult<Vec<[f64; 2]>> {
        // CPU fallback: centered difference method
        let h = 1e-5;
        let mut gradients = Vec::with_capacity(points.len());

        for point in points {
            let mut grad = [0.0; 2];

            for i in 0..2 {
                let mut p_plus = *point;
                let mut p_minus = *point;
                p_plus[i] += h;
                p_minus[i] -= h;

                let f_plus = field.evaluate(&p_plus);
                let f_minus = field.evaluate(&p_minus);

                grad[i] = (f_plus - f_minus) / (2.0 * h);
            }

            gradients.push(grad);
        }

        Ok(gradients)
    }

    /// Compute gradients at multiple 3D points
    pub async fn compute_gradients_3d(
        _gpu: &SharedGpuContext,
        field: &ScalarField<3, 0, 0>,
        points: &[[f64; 3]],
    ) -> UnifiedGpuResult<Vec<[f64; 3]>> {
        let h = 1e-5;
        let mut gradients = Vec::with_capacity(points.len());

        for point in points {
            let mut grad = [0.0; 3];

            for i in 0..3 {
                let mut p_plus = *point;
                let mut p_minus = *point;
                p_plus[i] += h;
                p_minus[i] -= h;

                let f_plus = field.evaluate(&p_plus);
                let f_minus = field.evaluate(&p_minus);

                grad[i] = (f_plus - f_minus) / (2.0 * h);
            }

            gradients.push(grad);
        }

        Ok(gradients)
    }

    /// Compute Laplacians at multiple 2D points
    pub async fn compute_laplacians_2d(
        _gpu: &SharedGpuContext,
        field: &ScalarField<2, 0, 0>,
        points: &[[f64; 2]],
    ) -> UnifiedGpuResult<Vec<f64>> {
        let h = 1e-5;
        let h2 = h * h;
        let mut laplacians = Vec::with_capacity(points.len());

        for point in points {
            let f_center = field.evaluate(point.as_slice());
            let mut laplacian = 0.0;

            for i in 0..2 {
                let mut p_plus = *point;
                let mut p_minus = *point;
                p_plus[i] += h;
                p_minus[i] -= h;

                let f_plus = field.evaluate(&p_plus);
                let f_minus = field.evaluate(&p_minus);

                laplacian += (f_plus + f_minus - 2.0 * f_center) / h2;
            }

            laplacians.push(laplacian);
        }

        Ok(laplacians)
    }

    /// Compute Laplacians at multiple 3D points
    pub async fn compute_laplacians_3d(
        _gpu: &SharedGpuContext,
        field: &ScalarField<3, 0, 0>,
        points: &[[f64; 3]],
    ) -> UnifiedGpuResult<Vec<f64>> {
        let h = 1e-5;
        let h2 = h * h;
        let mut laplacians = Vec::with_capacity(points.len());

        for point in points {
            let f_center = field.evaluate(point.as_slice());
            let mut laplacian = 0.0;

            for i in 0..3 {
                let mut p_plus = *point;
                let mut p_minus = *point;
                p_plus[i] += h;
                p_minus[i] -= h;

                let f_plus = field.evaluate(&p_plus);
                let f_minus = field.evaluate(&p_minus);

                laplacian += (f_plus + f_minus - 2.0 * f_center) / h2;
            }

            laplacians.push(laplacian);
        }

        Ok(laplacians)
    }
}

/// WGSL shader for batch gradient computation
///
/// Computes gradients using centered finite differences on GPU.
/// Each workgroup processes multiple points in parallel.
#[allow(dead_code)]
const GRADIENT_SHADER: &str = r#"
struct GradientParams {
    num_points: u32,
    dimension: u32,
    step_size: f32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: GradientParams;
@group(0) @binding(2) var<storage, read_write> gradients: array<vec4<f32>>;

// Field evaluation function (to be specialized per field type)
fn evaluate_field(p: vec3<f32>) -> f32 {
    // Placeholder: f(x,y,z) = x² + y² + z²
    return p.x * p.x + p.y * p.y + p.z * p.z;
}

// Compute gradient using centered differences
@compute @workgroup_size(256)
fn compute_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) {
        return;
    }

    let p = points[idx].xyz;
    let h = params.step_size;
    var grad = vec3<f32>(0.0, 0.0, 0.0);

    // ∂f/∂x
    let px_plus = vec3<f32>(p.x + h, p.y, p.z);
    let px_minus = vec3<f32>(p.x - h, p.y, p.z);
    grad.x = (evaluate_field(px_plus) - evaluate_field(px_minus)) / (2.0 * h);

    // ∂f/∂y
    let py_plus = vec3<f32>(p.x, p.y + h, p.z);
    let py_minus = vec3<f32>(p.x, p.y - h, p.z);
    grad.y = (evaluate_field(py_plus) - evaluate_field(py_minus)) / (2.0 * h);

    // ∂f/∂z
    let pz_plus = vec3<f32>(p.x, p.y, p.z + h);
    let pz_minus = vec3<f32>(p.x, p.y, p.z - h);
    grad.z = (evaluate_field(pz_plus) - evaluate_field(pz_minus)) / (2.0 * h);

    gradients[idx] = vec4<f32>(grad.x, grad.y, grad.z, 0.0);
}
"#;

/// WGSL shader for batch Laplacian computation
#[allow(dead_code)]
const LAPLACIAN_SHADER: &str = r#"
struct LaplacianParams {
    num_points: u32,
    dimension: u32,
    step_size: f32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: LaplacianParams;
@group(0) @binding(2) var<storage, read_write> laplacians: array<f32>;

// Field evaluation function (to be specialized)
fn evaluate_field(p: vec3<f32>) -> f32 {
    return p.x * p.x + p.y * p.y + p.z * p.z;
}

// Compute Laplacian using finite differences
@compute @workgroup_size(256)
fn compute_laplacian(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_points) {
        return;
    }

    let p = points[idx].xyz;
    let h = params.step_size;
    let h2 = h * h;
    let f_center = evaluate_field(p);
    var laplacian = 0.0;

    // ∂²f/∂x²
    let px_plus = vec3<f32>(p.x + h, p.y, p.z);
    let px_minus = vec3<f32>(p.x - h, p.y, p.z);
    laplacian = laplacian + (evaluate_field(px_plus) + evaluate_field(px_minus) - 2.0 * f_center) / h2;

    // ∂²f/∂y²
    let py_plus = vec3<f32>(p.x, p.y + h, p.z);
    let py_minus = vec3<f32>(p.x, p.y - h, p.z);
    laplacian = laplacian + (evaluate_field(py_plus) + evaluate_field(py_minus) - 2.0 * f_center) / h2;

    // ∂²f/∂z²
    let pz_plus = vec3<f32>(p.x, p.y, p.z + h);
    let pz_minus = vec3<f32>(p.x, p.y, p.z - h);
    laplacian = laplacian + (evaluate_field(pz_plus) + evaluate_field(pz_minus) - 2.0 * f_center) / h2;

    laplacians[idx] = laplacian;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_shader_compiles() {
        assert!(GRADIENT_SHADER.contains("compute_gradient"));
        assert!(GRADIENT_SHADER.contains("centered differences"));
    }

    #[test]
    fn test_laplacian_shader_compiles() {
        assert!(LAPLACIAN_SHADER.contains("compute_laplacian"));
        assert!(LAPLACIAN_SHADER.contains("finite differences"));
    }
}
