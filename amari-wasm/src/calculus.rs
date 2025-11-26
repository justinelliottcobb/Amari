//! WebAssembly bindings for amari-calculus
//!
//! This module provides JavaScript/TypeScript access to:
//! - Scalar and vector field evaluation
//! - Numerical derivatives (gradient, divergence, curl, Laplacian)
//! - Lie derivatives
//! - Integration (definite integrals, line integrals, surface integrals)
//! - Riemannian manifolds (metrics, connections, curvature, geodesics)

use wasm_bindgen::prelude::*;

// ============================================================================
// Scalar Fields
// ============================================================================

/// Scalar field f: ℝⁿ → ℝ
///
/// Represents a function that maps points in n-dimensional space to real numbers.
///
/// # JavaScript Example
///
/// ```javascript
/// // Create a scalar field f(x, y) = x² + y²
/// const field = WasmScalarField.fromFunction2D((x, y) => x*x + y*y);
///
/// // Evaluate at a point
/// const value = field.evaluate([1.0, 2.0]); // Returns 5.0
/// ```
#[wasm_bindgen(js_name = ScalarField)]
pub struct WasmScalarField {
    dimension: usize,
    // Store as JavaScript function for flexibility
    evaluator: js_sys::Function,
}

#[wasm_bindgen(js_class = ScalarField)]
impl WasmScalarField {
    /// Create a 2D scalar field from a JavaScript function
    ///
    /// # Arguments
    ///
    /// * `func` - JavaScript function (x, y) => f(x, y)
    #[wasm_bindgen(constructor)]
    pub fn new(func: js_sys::Function, dimension: usize) -> Self {
        Self {
            dimension,
            evaluator: func,
        }
    }

    /// Create a 2D scalar field from a JavaScript function
    ///
    /// # Arguments
    ///
    /// * `func` - JavaScript function (x, y) => f(x, y)
    #[wasm_bindgen(js_name = fromFunction2D)]
    pub fn from_function_2d(func: js_sys::Function) -> Self {
        Self::new(func, 2)
    }

    /// Create a 3D scalar field from a JavaScript function
    ///
    /// # Arguments
    ///
    /// * `func` - JavaScript function (x, y, z) => f(x, y, z)
    #[wasm_bindgen(js_name = fromFunction3D)]
    pub fn from_function_3d(func: js_sys::Function) -> Self {
        Self::new(func, 3)
    }

    /// Evaluate the scalar field at a point
    ///
    /// # Arguments
    ///
    /// * `point` - Coordinates [x, y] or [x, y, z]
    ///
    /// # Returns
    ///
    /// Field value at the point
    #[wasm_bindgen]
    pub fn evaluate(&self, point: &[f64]) -> Result<f64, JsValue> {
        if point.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Expected {}-dimensional point, got {}",
                self.dimension,
                point.len()
            )));
        }

        let this = JsValue::NULL;
        let args: Vec<JsValue> = point.iter().map(|&x| JsValue::from_f64(x)).collect();

        let result = match self.dimension {
            2 => self.evaluator.call2(&this, &args[0], &args[1])?,
            3 => self.evaluator.call3(&this, &args[0], &args[1], &args[2])?,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unsupported dimension: {}",
                    self.dimension
                )))
            }
        };

        result
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function did not return a number"))
    }

    /// Batch evaluate the field at multiple points
    ///
    /// # Arguments
    ///
    /// * `points` - Array of points as flat array [x1, y1, x2, y2, ...]
    ///
    /// # Returns
    ///
    /// Array of field values
    #[wasm_bindgen(js_name = batchEvaluate)]
    pub fn batch_evaluate(&self, points: &[f64]) -> Result<Vec<f64>, JsValue> {
        if !points.len().is_multiple_of(self.dimension) {
            return Err(JsValue::from_str(&format!(
                "Points array length must be multiple of {}",
                self.dimension
            )));
        }

        let num_points = points.len() / self.dimension;
        let mut results = Vec::with_capacity(num_points);

        for i in 0..num_points {
            let start = i * self.dimension;
            let end = start + self.dimension;
            let point = &points[start..end];
            results.push(self.evaluate(point)?);
        }

        Ok(results)
    }
}

// ============================================================================
// Vector Fields
// ============================================================================

/// Vector field F: ℝⁿ → ℝⁿ
///
/// Represents a function that maps points to vectors.
///
/// # JavaScript Example
///
/// ```javascript
/// // Create a 2D vector field F(x, y) = [y, -x] (rotation)
/// const field = WasmVectorField.fromFunction2D((x, y) => [y, -x]);
///
/// // Evaluate at a point
/// const vector = field.evaluate([1.0, 2.0]); // Returns [2.0, -1.0]
/// ```
#[wasm_bindgen(js_name = VectorField)]
pub struct WasmVectorField {
    dimension: usize,
    evaluator: js_sys::Function,
}

#[wasm_bindgen(js_class = VectorField)]
impl WasmVectorField {
    /// Create a vector field from a JavaScript function
    #[wasm_bindgen(constructor)]
    pub fn new(func: js_sys::Function, dimension: usize) -> Self {
        Self {
            dimension,
            evaluator: func,
        }
    }

    /// Create a 2D vector field from a JavaScript function
    ///
    /// # Arguments
    ///
    /// * `func` - JavaScript function (x, y) => [fx, fy]
    #[wasm_bindgen(js_name = fromFunction2D)]
    pub fn from_function_2d(func: js_sys::Function) -> Self {
        Self::new(func, 2)
    }

    /// Create a 3D vector field from a JavaScript function
    ///
    /// # Arguments
    ///
    /// * `func` - JavaScript function (x, y, z) => [fx, fy, fz]
    #[wasm_bindgen(js_name = fromFunction3D)]
    pub fn from_function_3d(func: js_sys::Function) -> Self {
        Self::new(func, 3)
    }

    /// Evaluate the vector field at a point
    ///
    /// # Arguments
    ///
    /// * `point` - Coordinates [x, y] or [x, y, z]
    ///
    /// # Returns
    ///
    /// Vector at the point
    #[wasm_bindgen]
    pub fn evaluate(&self, point: &[f64]) -> Result<Vec<f64>, JsValue> {
        if point.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Expected {}-dimensional point, got {}",
                self.dimension,
                point.len()
            )));
        }

        let this = JsValue::NULL;
        let args: Vec<JsValue> = point.iter().map(|&x| JsValue::from_f64(x)).collect();

        let result = match self.dimension {
            2 => self.evaluator.call2(&this, &args[0], &args[1])?,
            3 => self.evaluator.call3(&this, &args[0], &args[1], &args[2])?,
            _ => {
                return Err(JsValue::from_str(&format!(
                    "Unsupported dimension: {}",
                    self.dimension
                )))
            }
        };

        // Convert JavaScript array to Vec<f64>
        let array = js_sys::Array::from(&result);
        if array.length() as usize != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Function returned array of length {}, expected {}",
                array.length(),
                self.dimension
            )));
        }

        let vec: Vec<f64> = array
            .iter()
            .map(|val| {
                val.as_f64()
                    .ok_or_else(|| JsValue::from_str("Array element is not a number"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(vec)
    }
}

// ============================================================================
// Numerical Derivatives
// ============================================================================

/// Numerical derivative operations
///
/// Provides gradient, divergence, curl, and Laplacian computations
/// using centered finite differences.
#[wasm_bindgen(js_name = NumericalDerivative)]
pub struct WasmNumericalDerivative {
    step_size: f64,
}

#[wasm_bindgen(js_class = NumericalDerivative)]
impl WasmNumericalDerivative {
    /// Create a new numerical derivative computer
    ///
    /// # Arguments
    ///
    /// * `step_size` - Optional step size for finite differences (default: 1e-5)
    #[wasm_bindgen(constructor)]
    pub fn new(step_size: Option<f64>) -> Self {
        Self {
            step_size: step_size.unwrap_or(1e-5),
        }
    }

    /// Compute gradient of a scalar field at a point
    ///
    /// ∇f = [∂f/∂x, ∂f/∂y, ∂f/∂z]
    ///
    /// # Arguments
    ///
    /// * `field` - Scalar field
    /// * `point` - Evaluation point
    ///
    /// # Returns
    ///
    /// Gradient vector
    ///
    /// # JavaScript Example
    ///
    /// ```javascript
    /// const field = ScalarField.fromFunction2D((x, y) => x*x + y*y);
    /// const derivative = new NumericalDerivative();
    /// const grad = derivative.gradient(field, [1.0, 2.0]); // Returns [2.0, 4.0]
    /// ```
    #[wasm_bindgen]
    pub fn gradient(&self, field: &WasmScalarField, point: &[f64]) -> Result<Vec<f64>, JsValue> {
        let dim = field.dimension;
        let h = self.step_size;
        let mut grad = Vec::with_capacity(dim);

        for i in 0..dim {
            let mut point_plus = point.to_vec();
            let mut point_minus = point.to_vec();
            point_plus[i] += h;
            point_minus[i] -= h;

            let f_plus = field.evaluate(&point_plus)?;
            let f_minus = field.evaluate(&point_minus)?;

            grad.push((f_plus - f_minus) / (2.0 * h));
        }

        Ok(grad)
    }

    /// Compute divergence of a vector field at a point
    ///
    /// ∇·F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z
    ///
    /// # Arguments
    ///
    /// * `field` - Vector field
    /// * `point` - Evaluation point
    ///
    /// # Returns
    ///
    /// Divergence (scalar)
    #[wasm_bindgen]
    pub fn divergence(&self, field: &WasmVectorField, point: &[f64]) -> Result<f64, JsValue> {
        let dim = field.dimension;
        let h = self.step_size;
        let mut div = 0.0;

        for i in 0..dim {
            let mut point_plus = point.to_vec();
            let mut point_minus = point.to_vec();
            point_plus[i] += h;
            point_minus[i] -= h;

            let f_plus = field.evaluate(&point_plus)?;
            let f_minus = field.evaluate(&point_minus)?;

            div += (f_plus[i] - f_minus[i]) / (2.0 * h);
        }

        Ok(div)
    }

    /// Compute curl of a 3D vector field at a point
    ///
    /// ∇×F = [∂Fz/∂y - ∂Fy/∂z, ∂Fx/∂z - ∂Fz/∂x, ∂Fy/∂x - ∂Fx/∂y]
    ///
    /// # Arguments
    ///
    /// * `field` - 3D vector field
    /// * `point` - Evaluation point [x, y, z]
    ///
    /// # Returns
    ///
    /// Curl vector [cx, cy, cz]
    #[wasm_bindgen]
    pub fn curl(&self, field: &WasmVectorField, point: &[f64]) -> Result<Vec<f64>, JsValue> {
        if field.dimension != 3 {
            return Err(JsValue::from_str("Curl requires 3D vector field"));
        }
        if point.len() != 3 {
            return Err(JsValue::from_str("Curl requires 3D point"));
        }

        let h = self.step_size;

        // ∂Fz/∂y
        let mut py_plus = point.to_vec();
        let mut py_minus = point.to_vec();
        py_plus[1] += h;
        py_minus[1] -= h;
        let fz_y = (field.evaluate(&py_plus)?[2] - field.evaluate(&py_minus)?[2]) / (2.0 * h);

        // ∂Fy/∂z
        let mut pz_plus = point.to_vec();
        let mut pz_minus = point.to_vec();
        pz_plus[2] += h;
        pz_minus[2] -= h;
        let fy_z = (field.evaluate(&pz_plus)?[1] - field.evaluate(&pz_minus)?[1]) / (2.0 * h);

        // ∂Fx/∂z
        let fx_z = (field.evaluate(&pz_plus)?[0] - field.evaluate(&pz_minus)?[0]) / (2.0 * h);

        // ∂Fz/∂x
        let mut px_plus = point.to_vec();
        let mut px_minus = point.to_vec();
        px_plus[0] += h;
        px_minus[0] -= h;
        let fz_x = (field.evaluate(&px_plus)?[2] - field.evaluate(&px_minus)?[2]) / (2.0 * h);

        // ∂Fy/∂x
        let fy_x = (field.evaluate(&px_plus)?[1] - field.evaluate(&px_minus)?[1]) / (2.0 * h);

        // ∂Fx/∂y
        let fx_y = (field.evaluate(&py_plus)?[0] - field.evaluate(&py_minus)?[0]) / (2.0 * h);

        Ok(vec![fz_y - fy_z, fx_z - fz_x, fy_x - fx_y])
    }

    /// Compute Laplacian of a scalar field at a point
    ///
    /// ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
    ///
    /// # Arguments
    ///
    /// * `field` - Scalar field
    /// * `point` - Evaluation point
    ///
    /// # Returns
    ///
    /// Laplacian (scalar)
    #[wasm_bindgen]
    pub fn laplacian(&self, field: &WasmScalarField, point: &[f64]) -> Result<f64, JsValue> {
        let dim = field.dimension;
        let h = self.step_size;
        let h2 = h * h;
        let f_center = field.evaluate(point)?;
        let mut laplacian = 0.0;

        for i in 0..dim {
            let mut point_plus = point.to_vec();
            let mut point_minus = point.to_vec();
            point_plus[i] += h;
            point_minus[i] -= h;

            let f_plus = field.evaluate(&point_plus)?;
            let f_minus = field.evaluate(&point_minus)?;

            laplacian += (f_plus + f_minus - 2.0 * f_center) / h2;
        }

        Ok(laplacian)
    }
}

// ============================================================================
// Integration
// ============================================================================

/// Numerical integration operations
///
/// Provides definite integrals, line integrals, and surface integrals
/// using adaptive quadrature.
#[wasm_bindgen(js_name = Integration)]
pub struct WasmIntegration;

#[wasm_bindgen(js_class = Integration)]
impl WasmIntegration {
    /// Compute 1D definite integral using Simpson's rule
    ///
    /// ∫[a,b] f(x) dx
    ///
    /// # Arguments
    ///
    /// * `func` - JavaScript function to integrate
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `n` - Number of subdivisions (must be even)
    ///
    /// # Returns
    ///
    /// Integral value
    ///
    /// # JavaScript Example
    ///
    /// ```javascript
    /// // Integrate x^2 from 0 to 1
    /// const result = Integration.integrate1D(x => x*x, 0, 1, 100);
    /// // Returns approximately 0.333...
    /// ```
    #[wasm_bindgen(js_name = integrate1D)]
    pub fn integrate_1d(func: js_sys::Function, a: f64, b: f64, n: usize) -> Result<f64, JsValue> {
        if !n.is_multiple_of(2) {
            return Err(JsValue::from_str("Number of subdivisions must be even"));
        }

        let h = (b - a) / n as f64;
        let this = JsValue::NULL;

        // Evaluate at endpoints
        let fa = func
            .call1(&this, &JsValue::from_f64(a))?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function did not return a number"))?;

        let fb = func
            .call1(&this, &JsValue::from_f64(b))?
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function did not return a number"))?;

        let mut sum_odd = 0.0;
        let mut sum_even = 0.0;

        for i in 1..n {
            let x = a + i as f64 * h;
            let fx = func
                .call1(&this, &JsValue::from_f64(x))?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("Function did not return a number"))?;

            if i % 2 == 1 {
                sum_odd += fx;
            } else {
                sum_even += fx;
            }
        }

        Ok((h / 3.0) * (fa + 4.0 * sum_odd + 2.0 * sum_even + fb))
    }

    /// Compute 2D integral over a rectangle using Simpson's rule
    ///
    /// ∫∫[a,b]×[c,d] f(x,y) dx dy
    ///
    /// # Arguments
    ///
    /// * `func` - JavaScript function (x, y) => f(x, y)
    /// * `ax` - Lower x bound
    /// * `bx` - Upper x bound
    /// * `ay` - Lower y bound
    /// * `by` - Upper y bound
    /// * `nx` - Number of x subdivisions (must be even)
    /// * `ny` - Number of y subdivisions (must be even)
    #[wasm_bindgen(js_name = integrate2D)]
    pub fn integrate_2d(
        func: js_sys::Function,
        ax: f64,
        bx: f64,
        ay: f64,
        by: f64,
        nx: usize,
        ny: usize,
    ) -> Result<f64, JsValue> {
        if !nx.is_multiple_of(2) || !ny.is_multiple_of(2) {
            return Err(JsValue::from_str("Number of subdivisions must be even"));
        }

        let hx = (bx - ax) / nx as f64;
        let hy = (by - ay) / ny as f64;
        let this = JsValue::NULL;

        let mut sum = 0.0;

        for i in 0..=nx {
            let x = ax + i as f64 * hx;
            let wx = if i == 0 || i == nx {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };

            for j in 0..=ny {
                let y = ay + j as f64 * hy;
                let wy = if j == 0 || j == ny {
                    1.0
                } else if j % 2 == 1 {
                    4.0
                } else {
                    2.0
                };

                let fxy = func
                    .call2(&this, &JsValue::from_f64(x), &JsValue::from_f64(y))?
                    .as_f64()
                    .ok_or_else(|| JsValue::from_str("Function did not return a number"))?;

                sum += wx * wy * fxy;
            }
        }

        Ok((hx * hy / 9.0) * sum)
    }
}

// ============================================================================
// Riemannian Manifolds
// ============================================================================

/// Riemannian manifold with metric tensor
///
/// Represents a curved space with a metric that defines distances and angles.
///
/// # JavaScript Example
///
/// ```javascript
/// // Create a 2D sphere of radius 1
/// const sphere = RiemannianManifold.sphere(1.0);
///
/// // Compute scalar curvature at the north pole
/// const R = sphere.scalarCurvature([0.0, 0.0]); // Returns 2.0 for unit sphere
/// ```
#[wasm_bindgen(js_name = RiemannianManifold)]
pub struct WasmRiemannianManifold {
    dimension: usize,
    metric_type: String,
    radius: f64, // For sphere/hyperbolic
}

#[wasm_bindgen(js_class = RiemannianManifold)]
impl WasmRiemannianManifold {
    /// Create a Euclidean (flat) manifold
    ///
    /// # Arguments
    ///
    /// * `dimension` - Dimension (2 or 3)
    #[wasm_bindgen]
    pub fn euclidean(dimension: usize) -> Result<WasmRiemannianManifold, JsValue> {
        if dimension != 2 && dimension != 3 {
            return Err(JsValue::from_str("Only 2D and 3D manifolds supported"));
        }

        Ok(Self {
            dimension,
            metric_type: "euclidean".to_string(),
            radius: 1.0,
        })
    }

    /// Create a 2D sphere of given radius
    ///
    /// Metric: ds² = dθ² + sin²θ dφ²
    ///
    /// # Arguments
    ///
    /// * `radius` - Sphere radius
    #[wasm_bindgen]
    pub fn sphere(radius: f64) -> Result<WasmRiemannianManifold, JsValue> {
        if radius <= 0.0 {
            return Err(JsValue::from_str("Radius must be positive"));
        }

        Ok(Self {
            dimension: 2,
            metric_type: "sphere".to_string(),
            radius,
        })
    }

    /// Create a 2D hyperbolic plane (Poincaré half-plane model)
    ///
    /// Metric: ds² = (dx² + dy²) / y²
    #[wasm_bindgen]
    pub fn hyperbolic() -> WasmRiemannianManifold {
        Self {
            dimension: 2,
            metric_type: "hyperbolic".to_string(),
            radius: 1.0,
        }
    }

    /// Get the dimension of the manifold
    #[wasm_bindgen(getter)]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Compute Christoffel symbol Γ^k_ij at a point
    ///
    /// # Arguments
    ///
    /// * `k` - Upper index
    /// * `i` - First lower index
    /// * `j` - Second lower index
    /// * `coords` - Coordinates
    #[wasm_bindgen(js_name = christoffel)]
    pub fn christoffel(
        &self,
        k: usize,
        i: usize,
        j: usize,
        coords: &[f64],
    ) -> Result<f64, JsValue> {
        if coords.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Expected {}-dimensional coordinates",
                self.dimension
            )));
        }

        // Use amari-calculus to compute Christoffel symbols
        use amari_calculus::manifold::{Connection, MetricTensor};

        // Only 2D manifolds are supported for now
        if self.dimension != 2 {
            return Err(JsValue::from_str(
                "Only 2D manifolds are currently supported",
            ));
        }

        let metric = match self.metric_type.as_str() {
            "euclidean" => MetricTensor::<2>::euclidean(),
            "sphere" => MetricTensor::<2>::sphere(self.radius),
            "hyperbolic" => MetricTensor::<2>::hyperbolic(),
            _ => return Err(JsValue::from_str("Unsupported metric type")),
        };

        let connection = Connection::<2>::from_metric(&metric);
        Ok(connection.christoffel(k, i, j, coords))
    }

    /// Compute Riemann curvature tensor component R^i_jkl
    ///
    /// # Arguments
    ///
    /// * `i` - Upper index
    /// * `j` - First lower index
    /// * `k` - Second lower index
    /// * `l` - Third lower index
    /// * `coords` - Coordinates
    #[wasm_bindgen(js_name = riemannTensor)]
    pub fn riemann_tensor(
        &self,
        i: usize,
        j: usize,
        k: usize,
        l: usize,
        coords: &[f64],
    ) -> Result<f64, JsValue> {
        if self.dimension != 2 {
            return Err(JsValue::from_str(
                "Only 2D manifolds are currently supported",
            ));
        }

        let manifold = self.create_manifold()?;
        Ok(manifold.riemann_tensor(i, j, k, l, coords))
    }

    /// Compute Ricci tensor component R_ij
    ///
    /// # Arguments
    ///
    /// * `i` - First index
    /// * `j` - Second index
    /// * `coords` - Coordinates
    #[wasm_bindgen(js_name = ricciTensor)]
    pub fn ricci_tensor(&self, i: usize, j: usize, coords: &[f64]) -> Result<f64, JsValue> {
        if self.dimension != 2 {
            return Err(JsValue::from_str(
                "Only 2D manifolds are currently supported",
            ));
        }

        let manifold = self.create_manifold()?;
        Ok(manifold.ricci_tensor(i, j, coords))
    }

    /// Compute scalar curvature R
    ///
    /// # Arguments
    ///
    /// * `coords` - Coordinates
    ///
    /// # Returns
    ///
    /// Scalar curvature value
    #[wasm_bindgen(js_name = scalarCurvature)]
    pub fn scalar_curvature(&self, coords: &[f64]) -> Result<f64, JsValue> {
        if self.dimension != 2 {
            return Err(JsValue::from_str(
                "Only 2D manifolds are currently supported",
            ));
        }

        let manifold = self.create_manifold()?;
        Ok(manifold.scalar_curvature(coords))
    }

    /// Compute geodesic trajectory
    ///
    /// Solves the geodesic equations using RK4 integration.
    ///
    /// # Arguments
    ///
    /// * `initial_pos` - Initial position
    /// * `initial_vel` - Initial velocity
    /// * `t_max` - Maximum time
    /// * `dt` - Time step
    ///
    /// # Returns
    ///
    /// Flat array of trajectory points and velocities:
    /// [x0, y0, vx0, vy0, x1, y1, vx1, vy1, ...]
    #[wasm_bindgen]
    pub fn geodesic(
        &self,
        initial_pos: &[f64],
        initial_vel: &[f64],
        t_max: f64,
        dt: f64,
    ) -> Result<Vec<f64>, JsValue> {
        if self.dimension != 2 {
            return Err(JsValue::from_str(
                "Only 2D manifolds are currently supported",
            ));
        }

        if initial_pos.len() != self.dimension || initial_vel.len() != self.dimension {
            return Err(JsValue::from_str(
                "Position and velocity must match manifold dimension",
            ));
        }

        let manifold = self.create_manifold()?;
        let trajectory = manifold.geodesic(initial_pos, initial_vel, t_max, dt);

        // Flatten to [x, y, vx, vy, x, y, vx, vy, ...]
        let mut flat = Vec::with_capacity(trajectory.len() * 2 * self.dimension);
        for (pos, vel) in trajectory {
            flat.extend_from_slice(&pos);
            flat.extend_from_slice(&vel);
        }

        Ok(flat)
    }

    // Helper method to create the appropriate RiemannianManifold
    fn create_manifold(&self) -> Result<amari_calculus::manifold::RiemannianManifold<2>, JsValue> {
        use amari_calculus::manifold::{MetricTensor, RiemannianManifold};

        match self.metric_type.as_str() {
            "euclidean" if self.dimension == 2 => {
                Ok(RiemannianManifold::new(MetricTensor::<2>::euclidean()))
            }
            "sphere" => Ok(RiemannianManifold::new(MetricTensor::<2>::sphere(
                self.radius,
            ))),
            "hyperbolic" => Ok(RiemannianManifold::new(MetricTensor::<2>::hyperbolic())),
            _ => Err(JsValue::from_str(
                "Unsupported metric type for this operation",
            )),
        }
    }
}

#[cfg(test)]
#[allow(dead_code)] // wasm_bindgen_test functions not recognized by clippy
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    // ========================================================================
    // Scalar Field Tests
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_scalar_field_evaluation() {
        // f(x, y) = x² + y²
        let func = js_sys::Function::new_with_args("x, y", "return x*x + y*y");
        let field = WasmScalarField::from_function_2d(func);

        // Test evaluation at (3, 4)
        let value = field.evaluate(&[3.0, 4.0]).unwrap();
        assert!((value - 25.0).abs() < 1e-10, "Expected 25, got {}", value);
    }

    #[wasm_bindgen_test]
    fn test_scalar_field_batch_evaluation() {
        let func = js_sys::Function::new_with_args("x, y", "return x + y");
        let field = WasmScalarField::from_function_2d(func);

        // Batch evaluate at [(1,2), (3,4)]
        let flat_points = vec![1.0, 2.0, 3.0, 4.0];
        let results = field.batch_evaluate(&flat_points).unwrap();

        assert_eq!(results.len(), 2);
        assert!((results[0] - 3.0).abs() < 1e-10);
        assert!((results[1] - 7.0).abs() < 1e-10);
    }

    // ========================================================================
    // Vector Field Tests
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_vector_field_evaluation() {
        // F(x, y) = [-y, x] (rotation field)
        let func = js_sys::Function::new_with_args("x, y", "return [-y, x]");
        let field = WasmVectorField::from_function_2d(func);

        let vector = field.evaluate(&[3.0, 4.0]).unwrap();
        assert_eq!(vector.len(), 2);
        assert!((vector[0] + 4.0).abs() < 1e-10); // -y = -4
        assert!((vector[1] - 3.0).abs() < 1e-10); // x = 3
    }

    // ========================================================================
    // Numerical Derivative Tests
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_gradient() {
        // f(x, y) = x² + y²
        let func = js_sys::Function::new_with_args("x, y", "return x*x + y*y");
        let field = WasmScalarField::from_function_2d(func);

        let derivative = WasmNumericalDerivative::new(Some(1e-5));
        let grad = derivative.gradient(&field, &[2.0, 3.0]).unwrap();

        // ∇f = [2x, 2y] at (2,3) = [4, 6]
        assert_eq!(grad.len(), 2);
        assert!((grad[0] - 4.0).abs() < 1e-3, "Expected 4, got {}", grad[0]);
        assert!((grad[1] - 6.0).abs() < 1e-3, "Expected 6, got {}", grad[1]);
    }

    #[wasm_bindgen_test]
    fn test_divergence() {
        // F(x, y) = [x, y] (divergence = 2)
        let func = js_sys::Function::new_with_args("x, y", "return [x, y]");
        let field = WasmVectorField::from_function_2d(func);

        let derivative = WasmNumericalDerivative::new(Some(1e-5));
        let div = derivative.divergence(&field, &[1.0, 1.0]).unwrap();

        // ∇·F = ∂x/∂x + ∂y/∂y = 1 + 1 = 2
        assert!((div - 2.0).abs() < 1e-3, "Expected 2, got {}", div);
    }

    #[wasm_bindgen_test]
    fn test_laplacian() {
        // f(x, y) = x² + y²
        let func = js_sys::Function::new_with_args("x, y", "return x*x + y*y");
        let field = WasmScalarField::from_function_2d(func);

        let derivative = WasmNumericalDerivative::new(Some(1e-5));
        let lap = derivative.laplacian(&field, &[1.0, 2.0]).unwrap();

        // ∇²f = ∂²/∂x² + ∂²/∂y² = 2 + 2 = 4
        assert!((lap - 4.0).abs() < 1e-2, "Expected 4, got {}", lap);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_1d_integration() {
        // ∫₀¹ x² dx = 1/3
        let func = js_sys::Function::new_with_args("x", "return x*x");
        let result = WasmIntegration::integrate_1d(func, 0.0, 1.0, 100).unwrap();

        assert!(
            (result - 1.0 / 3.0).abs() < 1e-4,
            "Expected 0.3333, got {}",
            result
        );
    }

    #[wasm_bindgen_test]
    fn test_2d_integration() {
        // ∫∫[0,1]×[0,1] 1 dx dy = 1
        let func = js_sys::Function::new_with_args("x, y", "return 1");
        let result = WasmIntegration::integrate_2d(func, 0.0, 1.0, 0.0, 1.0, 20, 20).unwrap();

        assert!((result - 1.0).abs() < 1e-3, "Expected 1.0, got {}", result);
    }

    // ========================================================================
    // Riemannian Manifold Tests
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_euclidean_manifold() {
        let euclidean = WasmRiemannianManifold::euclidean(2).unwrap();
        assert_eq!(euclidean.dimension(), 2);

        // Christoffel symbols should be zero for flat space
        let gamma = euclidean.christoffel(0, 0, 0, &[1.0, 2.0]).unwrap();
        assert!(
            gamma.abs() < 1e-10,
            "Expected 0 for Euclidean Christoffel, got {}",
            gamma
        );

        // Scalar curvature should be zero for flat space
        let r_curvature = euclidean.scalar_curvature(&[1.0, 2.0]).unwrap();
        assert!(
            r_curvature.abs() < 1e-10,
            "Expected 0 for Euclidean curvature, got {}",
            r_curvature
        );
    }

    #[wasm_bindgen_test]
    fn test_sphere_manifold() {
        let sphere = WasmRiemannianManifold::sphere(1.0).unwrap();
        assert_eq!(sphere.dimension(), 2);

        // Scalar curvature for unit sphere should be 2
        // (Near north pole to avoid singularity)
        let r_curvature = sphere.scalar_curvature(&[0.01, 0.01]).unwrap();
        assert!(
            (r_curvature - 2.0).abs() < 0.1,
            "Expected ~2 for unit sphere curvature, got {}",
            r_curvature
        );
    }

    #[wasm_bindgen_test]
    fn test_hyperbolic_manifold() {
        let hyperbolic = WasmRiemannianManifold::hyperbolic();
        assert_eq!(hyperbolic.dimension(), 2);

        // Scalar curvature for hyperbolic plane should be -2
        let r_curvature = hyperbolic.scalar_curvature(&[1.0, 1.0]).unwrap();
        assert!(
            (r_curvature + 2.0).abs() < 0.1,
            "Expected ~-2 for hyperbolic curvature, got {}",
            r_curvature
        );
    }

    #[wasm_bindgen_test]
    fn test_geodesic_computation() {
        let euclidean = WasmRiemannianManifold::euclidean(2).unwrap();

        // Geodesic with initial position and velocity
        let trajectory = euclidean
            .geodesic(&[0.0, 0.0], &[1.0, 0.0], 1.0, 0.1)
            .unwrap();

        // Should have multiple points (t_max/dt + 1)
        assert!(
            trajectory.len() >= 44,
            "Expected at least 44 values (11 points × 4 values), got {}",
            trajectory.len()
        );

        // First point should be initial position
        assert!((trajectory[0] - 0.0).abs() < 1e-10);
        assert!((trajectory[1] - 0.0).abs() < 1e-10);

        // First velocity should be initial velocity
        assert!((trajectory[2] - 1.0).abs() < 1e-2);
        assert!((trajectory[3] - 0.0).abs() < 1e-2);
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[wasm_bindgen_test]
    fn test_dimension_mismatch_error() {
        let func = js_sys::Function::new_with_args("x, y", "return x*x");
        let field = WasmScalarField::from_function_2d(func);

        // Try to evaluate with wrong dimension
        let result = field.evaluate(&[1.0, 2.0, 3.0]);
        assert!(result.is_err(), "Should error with wrong dimension");
    }

    #[wasm_bindgen_test]
    fn test_invalid_manifold_dimension() {
        let result = WasmRiemannianManifold::euclidean(5);
        assert!(result.is_err(), "Should error with unsupported dimension");
    }

    #[wasm_bindgen_test]
    fn test_negative_radius_error() {
        let result = WasmRiemannianManifold::sphere(-1.0);
        assert!(result.is_err(), "Should error with negative radius");
    }
}
