//! WASM bindings for functional analysis on multivector spaces
//!
//! Provides WebAssembly bindings for:
//! - Hilbert spaces over Clifford algebras
//! - Linear operators and matrix representations
//! - Spectral decomposition and eigenvalue computation
//! - Sobolev spaces with weak derivatives

use amari_functional::{
    operator::MatrixOperator,
    sobolev::poincare_constant_estimate,
    space::{Domain, InnerProductSpace, MultivectorHilbertSpace, NormedSpace, VectorSpace},
    spectral::{spectral_decompose, SpectralDecomposition},
    BoundedOperator, LinearOperator,
};
use wasm_bindgen::prelude::*;

/// WASM wrapper for MultivectorHilbertSpace over Cl(2,0,0)
///
/// Provides finite-dimensional Hilbert space operations on multivectors.
#[wasm_bindgen]
pub struct WasmHilbertSpace {
    inner: MultivectorHilbertSpace<2, 0, 0>,
}

#[wasm_bindgen]
impl WasmHilbertSpace {
    /// Create a new Hilbert space Cl(2,0,0) ~ R^4
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: MultivectorHilbertSpace::new(),
        }
    }

    /// Get the dimension of the space (4 for Cl(2,0,0))
    #[wasm_bindgen(js_name = dimension)]
    pub fn dimension(&self) -> usize {
        self.inner.dimension().unwrap_or(4)
    }

    /// Get the Clifford algebra signature (p, q, r)
    #[wasm_bindgen(js_name = signature)]
    pub fn signature(&self) -> Vec<usize> {
        let (p, q, r) = self.inner.signature();
        vec![p, q, r]
    }

    /// Create a multivector from coefficients
    ///
    /// # Arguments
    /// * `coefficients` - Array of 4 coefficients [scalar, e1, e2, e12]
    #[wasm_bindgen(js_name = fromCoefficients)]
    pub fn from_coefficients(&self, coefficients: &[f64]) -> Result<Vec<f64>, JsValue> {
        if coefficients.len() != 4 {
            return Err(JsValue::from_str(
                "Cl(2,0,0) requires exactly 4 coefficients",
            ));
        }
        self.inner
            .from_coefficients(coefficients)
            .map(|mv| mv.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Compute inner product <x, y>
    #[wasm_bindgen(js_name = innerProduct)]
    pub fn inner_product(&self, x: &[f64], y: &[f64]) -> Result<f64, JsValue> {
        if x.len() != 4 || y.len() != 4 {
            return Err(JsValue::from_str("Both vectors must have 4 coefficients"));
        }
        let mv_x = self
            .inner
            .from_coefficients(x)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mv_y = self
            .inner
            .from_coefficients(y)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.inner.inner_product(&mv_x, &mv_y))
    }

    /// Compute the norm ||x||
    #[wasm_bindgen(js_name = norm)]
    pub fn norm(&self, x: &[f64]) -> Result<f64, JsValue> {
        if x.len() != 4 {
            return Err(JsValue::from_str("Vector must have 4 coefficients"));
        }
        let mv = self
            .inner
            .from_coefficients(x)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.inner.norm(&mv))
    }

    /// Compute distance d(x, y) = ||x - y||
    #[wasm_bindgen(js_name = distance)]
    pub fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, JsValue> {
        if x.len() != 4 || y.len() != 4 {
            return Err(JsValue::from_str("Both vectors must have 4 coefficients"));
        }
        let mv_x = self
            .inner
            .from_coefficients(x)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mv_y = self
            .inner
            .from_coefficients(y)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.inner.distance(&mv_x, &mv_y))
    }

    /// Normalize a vector to unit length
    #[wasm_bindgen(js_name = normalize)]
    pub fn normalize(&self, x: &[f64]) -> Result<Vec<f64>, JsValue> {
        if x.len() != 4 {
            return Err(JsValue::from_str("Vector must have 4 coefficients"));
        }
        let mv = self
            .inner
            .from_coefficients(x)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        self.inner
            .normalize(&mv)
            .map(|n| n.to_vec())
            .ok_or_else(|| JsValue::from_str("Cannot normalize zero vector"))
    }

    /// Orthogonal projection of x onto y
    #[wasm_bindgen(js_name = project)]
    pub fn project(&self, x: &[f64], y: &[f64]) -> Result<Vec<f64>, JsValue> {
        if x.len() != 4 || y.len() != 4 {
            return Err(JsValue::from_str("Both vectors must have 4 coefficients"));
        }
        let mv_x = self
            .inner
            .from_coefficients(x)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mv_y = self
            .inner
            .from_coefficients(y)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let proj = self.inner.project(&mv_x, &mv_y);
        Ok(proj.to_vec())
    }

    /// Check if two vectors are orthogonal
    #[wasm_bindgen(js_name = isOrthogonal)]
    pub fn is_orthogonal(&self, x: &[f64], y: &[f64], tolerance: f64) -> Result<bool, JsValue> {
        if x.len() != 4 || y.len() != 4 {
            return Err(JsValue::from_str("Both vectors must have 4 coefficients"));
        }
        let mv_x = self
            .inner
            .from_coefficients(x)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mv_y = self
            .inner
            .from_coefficients(y)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(self.inner.are_orthogonal(&mv_x, &mv_y, tolerance))
    }
}

impl Default for WasmHilbertSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// WASM wrapper for matrix operators on Hilbert spaces
///
/// Represents bounded linear operators as matrices.
#[wasm_bindgen]
pub struct WasmMatrixOperator {
    inner: MatrixOperator<2, 0, 0>,
}

#[wasm_bindgen]
impl WasmMatrixOperator {
    /// Create a matrix operator from a flattened row-major matrix
    ///
    /// # Arguments
    /// * `entries` - 16 entries for a 4x4 matrix in row-major order
    #[wasm_bindgen(constructor)]
    pub fn new(entries: &[f64]) -> Result<WasmMatrixOperator, JsValue> {
        if entries.len() != 16 {
            return Err(JsValue::from_str("Matrix requires 16 entries (4x4)"));
        }
        MatrixOperator::new(entries.to_vec(), 4, 4)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create the identity operator
    #[wasm_bindgen(js_name = identity)]
    pub fn identity() -> WasmMatrixOperator {
        Self {
            inner: MatrixOperator::identity(),
        }
    }

    /// Create a zero operator
    #[wasm_bindgen(js_name = zero)]
    pub fn zero() -> WasmMatrixOperator {
        Self {
            inner: MatrixOperator::zeros(),
        }
    }

    /// Create a diagonal matrix from diagonal entries
    #[wasm_bindgen(js_name = diagonal)]
    pub fn diagonal(entries: &[f64]) -> Result<WasmMatrixOperator, JsValue> {
        if entries.len() != 4 {
            return Err(JsValue::from_str("Diagonal requires 4 entries"));
        }
        MatrixOperator::diagonal(entries)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create a scaling operator (λI)
    #[wasm_bindgen(js_name = scaling)]
    pub fn scaling(lambda: f64) -> WasmMatrixOperator {
        Self {
            inner: MatrixOperator::diagonal(&[lambda, lambda, lambda, lambda]).unwrap(),
        }
    }

    /// Apply the operator to a vector: T(x)
    #[wasm_bindgen(js_name = apply)]
    pub fn apply(&self, x: &[f64]) -> Result<Vec<f64>, JsValue> {
        if x.len() != 4 {
            return Err(JsValue::from_str("Vector must have 4 coefficients"));
        }
        let mv = amari_core::Multivector::<2, 0, 0>::from_slice(x);
        self.inner
            .apply(&mv)
            .map(|result| result.to_vec())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the matrix entries as a flat array (row-major)
    #[wasm_bindgen(js_name = getEntries)]
    pub fn get_entries(&self) -> Vec<f64> {
        let mut entries = Vec::with_capacity(16);
        for i in 0..4 {
            for j in 0..4 {
                entries.push(self.inner.get(i, j));
            }
        }
        entries
    }

    /// Get a single matrix entry at (i, j)
    #[wasm_bindgen(js_name = getEntry)]
    pub fn get_entry(&self, i: usize, j: usize) -> Result<f64, JsValue> {
        if i >= 4 || j >= 4 {
            return Err(JsValue::from_str("Index out of bounds (0-3)"));
        }
        Ok(self.inner.get(i, j))
    }

    /// Compute operator norm ||T||
    #[wasm_bindgen(js_name = operatorNorm)]
    pub fn operator_norm(&self) -> f64 {
        BoundedOperator::<amari_core::Multivector<2, 0, 0>, amari_core::Multivector<2, 0, 0>, _>::operator_norm(&self.inner)
    }

    /// Check if the matrix is symmetric
    #[wasm_bindgen(js_name = isSymmetric)]
    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        self.inner.is_symmetric(tolerance)
    }

    /// Add two operators
    #[wasm_bindgen(js_name = add)]
    pub fn add(&self, other: &WasmMatrixOperator) -> Result<WasmMatrixOperator, JsValue> {
        self.inner
            .add(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Compose two operators (matrix multiplication)
    #[wasm_bindgen(js_name = compose)]
    pub fn compose(&self, other: &WasmMatrixOperator) -> Result<WasmMatrixOperator, JsValue> {
        self.inner
            .multiply(&other.inner)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Scale operator by scalar
    #[wasm_bindgen(js_name = scale)]
    pub fn scale(&self, lambda: f64) -> WasmMatrixOperator {
        Self {
            inner: self.inner.scale(lambda),
        }
    }

    /// Compute the transpose
    #[wasm_bindgen(js_name = transpose)]
    pub fn transpose(&self) -> WasmMatrixOperator {
        Self {
            inner: self.inner.transpose(),
        }
    }

    /// Compute the trace
    #[wasm_bindgen(js_name = trace)]
    pub fn trace(&self) -> f64 {
        self.inner.trace()
    }
}

/// WASM wrapper for spectral decomposition
///
/// Provides eigenvalue decomposition for symmetric matrices.
#[wasm_bindgen]
pub struct WasmSpectralDecomposition {
    inner: SpectralDecomposition<2, 0, 0>,
}

#[wasm_bindgen]
impl WasmSpectralDecomposition {
    /// Compute spectral decomposition of a symmetric matrix
    ///
    /// # Arguments
    /// * `matrix` - The symmetric matrix operator to decompose
    /// * `max_iterations` - Maximum iterations for eigenvalue computation
    /// * `tolerance` - Convergence tolerance
    #[wasm_bindgen(js_name = compute)]
    pub fn compute(
        matrix: &WasmMatrixOperator,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<WasmSpectralDecomposition, JsValue> {
        spectral_decompose(&matrix.inner, max_iterations, tolerance)
            .map(|inner| Self { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get the eigenvalues
    #[wasm_bindgen(js_name = eigenvalues)]
    pub fn eigenvalues(&self) -> Vec<f64> {
        self.inner.eigenvalues().iter().map(|ev| ev.value).collect()
    }

    /// Get the eigenvectors as a flattened array
    ///
    /// Returns 4 eigenvectors of 4 components each (16 total)
    #[wasm_bindgen(js_name = eigenvectors)]
    pub fn eigenvectors(&self) -> Vec<f64> {
        self.inner
            .eigenvectors()
            .iter()
            .flat_map(|ev| ev.to_vec())
            .collect()
    }

    /// Check if the decomposition is complete
    #[wasm_bindgen(js_name = isComplete)]
    pub fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    /// Get the spectral radius (largest |eigenvalue|)
    #[wasm_bindgen(js_name = spectralRadius)]
    pub fn spectral_radius(&self) -> f64 {
        self.inner.spectral_radius()
    }

    /// Get the condition number
    #[wasm_bindgen(js_name = conditionNumber)]
    pub fn condition_number(&self) -> Option<f64> {
        self.inner.condition_number()
    }

    /// Check if the operator is positive definite
    #[wasm_bindgen(js_name = isPositiveDefinite)]
    pub fn is_positive_definite(&self) -> bool {
        self.inner.is_positive_definite()
    }

    /// Check if the operator is positive semi-definite
    #[wasm_bindgen(js_name = isPositiveSemidefinite)]
    pub fn is_positive_semidefinite(&self) -> bool {
        self.inner.is_positive_semidefinite()
    }

    /// Apply the reconstructed operator T = Σᵢ λᵢ Pᵢ to a vector
    #[wasm_bindgen(js_name = apply)]
    pub fn apply(&self, x: &[f64]) -> Result<Vec<f64>, JsValue> {
        if x.len() != 4 {
            return Err(JsValue::from_str("Vector must have 4 coefficients"));
        }
        let mv = amari_core::Multivector::<2, 0, 0>::from_slice(x);
        Ok(self.inner.apply(&mv).to_vec())
    }

    /// Apply f(T) = Σᵢ f(λᵢ) Pᵢ to a vector using functional calculus
    ///
    /// # Arguments
    /// * `f` - JavaScript function to apply to eigenvalues
    /// * `x` - Input vector
    #[wasm_bindgen(js_name = applyFunction)]
    pub fn apply_function(&self, f: &js_sys::Function, x: &[f64]) -> Result<Vec<f64>, JsValue> {
        if x.len() != 4 {
            return Err(JsValue::from_str("Vector must have 4 coefficients"));
        }

        let mv = amari_core::Multivector::<2, 0, 0>::from_slice(x);

        // Convert JS function to Rust closure
        let this = JsValue::null();
        let result = self.inner.apply_function(
            |lambda| {
                let lambda_js = JsValue::from_f64(lambda);
                f.call1(&this, &lambda_js)
                    .ok()
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            },
            &mv,
        );

        Ok(result.to_vec())
    }
}

/// WASM wrapper for Sobolev spaces H^k
///
/// Provides function spaces with weak derivatives.
#[wasm_bindgen]
pub struct WasmSobolevSpace {
    order: usize,
    lower: f64,
    upper: f64,
    quadrature_points: usize,
}

#[wasm_bindgen]
impl WasmSobolevSpace {
    /// Create a Sobolev space H^k over an interval [a, b]
    ///
    /// # Arguments
    /// * `order` - The Sobolev regularity (1 for H^1, 2 for H^2)
    /// * `lower` - Lower bound of the interval
    /// * `upper` - Upper bound of the interval
    #[wasm_bindgen(constructor)]
    pub fn new(order: usize, lower: f64, upper: f64) -> Result<WasmSobolevSpace, JsValue> {
        if order == 0 {
            return Err(JsValue::from_str("Sobolev order must be at least 1"));
        }
        if upper <= lower {
            return Err(JsValue::from_str("Upper bound must be > lower bound"));
        }
        Ok(Self {
            order,
            lower,
            upper,
            quadrature_points: 32,
        })
    }

    /// Create H^1 over the unit interval [0, 1]
    #[wasm_bindgen(js_name = h1UnitInterval)]
    pub fn h1_unit_interval() -> WasmSobolevSpace {
        Self {
            order: 1,
            lower: 0.0,
            upper: 1.0,
            quadrature_points: 32,
        }
    }

    /// Create H^2 over the unit interval [0, 1]
    #[wasm_bindgen(js_name = h2UnitInterval)]
    pub fn h2_unit_interval() -> WasmSobolevSpace {
        Self {
            order: 2,
            lower: 0.0,
            upper: 1.0,
            quadrature_points: 32,
        }
    }

    /// Set the number of quadrature points
    #[wasm_bindgen(js_name = setQuadraturePoints)]
    pub fn set_quadrature_points(&mut self, n: usize) {
        self.quadrature_points = n;
    }

    /// Get the Sobolev order k
    #[wasm_bindgen(js_name = order)]
    pub fn get_order(&self) -> usize {
        self.order
    }

    /// Get the domain bounds [lower, upper]
    #[wasm_bindgen(js_name = bounds)]
    pub fn bounds(&self) -> Vec<f64> {
        vec![self.lower, self.upper]
    }

    /// Compute the Poincare constant estimate for the domain
    ///
    /// For [a,b], this is (b-a)/π
    #[wasm_bindgen(js_name = poincareConstant)]
    pub fn poincare_constant(&self) -> f64 {
        let domain = Domain::interval(self.lower, self.upper);
        poincare_constant_estimate(&domain)
    }

    /// Compute the H^k norm of a function using numerical integration
    ///
    /// # Arguments
    /// * `f` - JavaScript function f(x) to evaluate
    /// * `df` - JavaScript function f'(x) (first derivative)
    #[wasm_bindgen(js_name = h1Norm)]
    pub fn h1_norm(&self, f: &js_sys::Function, df: &js_sys::Function) -> Result<f64, JsValue> {
        let h = (self.upper - self.lower) / self.quadrature_points as f64;
        let this = JsValue::null();

        let mut sum = 0.0;

        for i in 0..self.quadrature_points {
            let x = self.lower + (i as f64 + 0.5) * h;
            let x_js = JsValue::from_f64(x);

            // Evaluate f(x)
            let fx = f
                .call1(&this, &x_js)?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("f must return a number"))?;

            // Evaluate f'(x)
            let dfx = df
                .call1(&this, &x_js)?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("df must return a number"))?;

            // ||f||²_{H^1} = ∫ |f|² + |f'|² dx
            sum += fx * fx + dfx * dfx;
        }

        Ok((sum * h).sqrt())
    }

    /// Compute the H^1 seminorm |f|_{H^1} = ||f'||_{L^2}
    #[wasm_bindgen(js_name = h1Seminorm)]
    pub fn h1_seminorm(&self, df: &js_sys::Function) -> Result<f64, JsValue> {
        let h = (self.upper - self.lower) / self.quadrature_points as f64;
        let this = JsValue::null();

        let mut sum = 0.0;

        for i in 0..self.quadrature_points {
            let x = self.lower + (i as f64 + 0.5) * h;
            let x_js = JsValue::from_f64(x);

            let dfx = df
                .call1(&this, &x_js)?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("df must return a number"))?;

            sum += dfx * dfx;
        }

        Ok((sum * h).sqrt())
    }

    /// Compute L^2 norm of a function
    #[wasm_bindgen(js_name = l2Norm)]
    pub fn l2_norm(&self, f: &js_sys::Function) -> Result<f64, JsValue> {
        let h = (self.upper - self.lower) / self.quadrature_points as f64;
        let this = JsValue::null();

        let mut sum = 0.0;

        for i in 0..self.quadrature_points {
            let x = self.lower + (i as f64 + 0.5) * h;
            let x_js = JsValue::from_f64(x);

            let fx = f
                .call1(&this, &x_js)?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("f must return a number"))?;

            sum += fx * fx;
        }

        Ok((sum * h).sqrt())
    }

    /// Compute L^2 inner product <f, g>
    #[wasm_bindgen(js_name = l2InnerProduct)]
    pub fn l2_inner_product(
        &self,
        f: &js_sys::Function,
        g: &js_sys::Function,
    ) -> Result<f64, JsValue> {
        let h = (self.upper - self.lower) / self.quadrature_points as f64;
        let this = JsValue::null();

        let mut sum = 0.0;

        for i in 0..self.quadrature_points {
            let x = self.lower + (i as f64 + 0.5) * h;
            let x_js = JsValue::from_f64(x);

            let fx = f
                .call1(&this, &x_js)?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("f must return a number"))?;

            let gx = g
                .call1(&this, &x_js)?
                .as_f64()
                .ok_or_else(|| JsValue::from_str("g must return a number"))?;

            sum += fx * gx;
        }

        Ok(sum * h)
    }
}

/// Power method for computing dominant eigenvalue
///
/// # Arguments
/// * `matrix` - The matrix operator
/// * `initial` - Initial guess (optional, uses random if not provided)
/// * `max_iterations` - Maximum iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// [eigenvalue, eigenvector...] flattened array
#[wasm_bindgen(js_name = powerMethod)]
pub fn power_method(
    matrix: &WasmMatrixOperator,
    initial: Option<Vec<f64>>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Vec<f64>, JsValue> {
    let init = initial.map(|v| amari_core::Multivector::<2, 0, 0>::from_slice(&v));
    amari_functional::spectral::power_method(
        &matrix.inner,
        init.as_ref(),
        max_iterations,
        tolerance,
    )
    .map(|pair| {
        let mut result = vec![pair.eigenvalue.value];
        result.extend(pair.eigenvector.to_vec());
        result
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Inverse iteration for computing eigenvalue near a shift
///
/// # Arguments
/// * `matrix` - The matrix operator
/// * `shift` - Value near the desired eigenvalue
/// * `initial` - Initial guess (optional)
/// * `max_iterations` - Maximum iterations
/// * `tolerance` - Convergence tolerance
///
/// # Returns
/// [eigenvalue, eigenvector...] flattened array
#[wasm_bindgen(js_name = inverseIteration)]
pub fn inverse_iteration(
    matrix: &WasmMatrixOperator,
    shift: f64,
    initial: Option<Vec<f64>>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Vec<f64>, JsValue> {
    let init = initial.map(|v| amari_core::Multivector::<2, 0, 0>::from_slice(&v));
    amari_functional::spectral::inverse_iteration(
        &matrix.inner,
        shift,
        init.as_ref(),
        max_iterations,
        tolerance,
    )
    .map(|pair| {
        let mut result = vec![pair.eigenvalue.value];
        result.extend(pair.eigenvector.to_vec());
        result
    })
    .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute all eigenvalues of a symmetric matrix
#[wasm_bindgen(js_name = computeEigenvalues)]
pub fn compute_eigenvalues(
    matrix: &WasmMatrixOperator,
    max_iterations: usize,
    tolerance: f64,
) -> Result<Vec<f64>, JsValue> {
    amari_functional::spectral::compute_eigenvalues(&matrix.inner, max_iterations, tolerance)
        .map(|evs| evs.iter().map(|ev| ev.value).collect())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
