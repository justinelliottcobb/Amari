//! WASM bindings for measure theory and Lebesgue integration
//!
//! Provides WebAssembly bindings for:
//! - Lebesgue measure on ℝⁿ
//! - Counting measure on discrete sets
//! - Probability measures
//! - Basic integration

use wasm_bindgen::prelude::*;

/// WASM wrapper for Lebesgue measure
///
/// The Lebesgue measure generalizes the notion of length, area, and volume
/// to higher dimensions and more complex sets.
#[wasm_bindgen]
pub struct WasmLebesgueMeasure {
    dimension: usize,
}

#[wasm_bindgen]
impl WasmLebesgueMeasure {
    /// Create a new Lebesgue measure for the specified dimension
    ///
    /// # Arguments
    /// * `dimension` - The dimension of the space (1 for length, 2 for area, 3 for volume, etc.)
    #[wasm_bindgen(constructor)]
    pub fn new(dimension: usize) -> Result<WasmLebesgueMeasure, JsValue> {
        if dimension == 0 {
            return Err(JsValue::from_str("Dimension must be at least 1"));
        }
        Ok(Self { dimension })
    }

    /// Get the dimension of this measure
    #[wasm_bindgen(js_name = getDimension)]
    pub fn get_dimension(&self) -> usize {
        self.dimension
    }

    /// Compute the measure of an interval [a, b]
    ///
    /// For 1D: returns length (b - a)
    /// For higher dimensions: returns the product of interval lengths
    #[wasm_bindgen(js_name = measureInterval)]
    pub fn measure_interval(&self, lower: &[f64], upper: &[f64]) -> Result<f64, JsValue> {
        if lower.len() != self.dimension || upper.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Expected {} coordinates, got {} lower and {} upper",
                self.dimension,
                lower.len(),
                upper.len()
            )));
        }

        let mut volume = 1.0;
        for i in 0..self.dimension {
            if upper[i] < lower[i] {
                return Err(JsValue::from_str(&format!(
                    "Upper bound must be >= lower bound in dimension {}",
                    i
                )));
            }
            volume *= upper[i] - lower[i];
        }

        Ok(volume)
    }

    /// Compute the measure of a box (hyper-rectangle) with given side lengths
    #[wasm_bindgen(js_name = measureBox)]
    pub fn measure_box(&self, sides: &[f64]) -> Result<f64, JsValue> {
        if sides.len() != self.dimension {
            return Err(JsValue::from_str(&format!(
                "Expected {} side lengths, got {}",
                self.dimension,
                sides.len()
            )));
        }

        let volume = sides.iter().product();
        Ok(volume)
    }
}

/// WASM wrapper for counting measure
///
/// The counting measure assigns to each set the number of elements it contains.
#[wasm_bindgen]
pub struct WasmCountingMeasure {}

#[wasm_bindgen]
impl WasmCountingMeasure {
    /// Create a new counting measure
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {}
    }

    /// Measure a finite set (returns its cardinality)
    ///
    /// # Arguments
    /// * `set_size` - The number of elements in the set
    #[wasm_bindgen(js_name = measureFiniteSet)]
    pub fn measure_finite_set(&self, set_size: usize) -> f64 {
        set_size as f64
    }

    /// Check if a set is measurable under counting measure
    /// (all sets are measurable under counting measure)
    #[wasm_bindgen(js_name = isMeasurable)]
    pub fn is_measurable(&self) -> bool {
        true
    }
}

/// WASM wrapper for probability measures
///
/// A probability measure assigns total measure 1 to the entire space.
#[wasm_bindgen]
pub struct WasmProbabilityMeasure {
    description: String,
}

#[wasm_bindgen]
impl WasmProbabilityMeasure {
    /// Create a new uniform probability measure on [0, 1]
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            description: "Uniform[0,1]".to_string(),
        }
    }

    /// Create a uniform probability measure on [a, b]
    #[wasm_bindgen(js_name = uniform)]
    pub fn uniform(a: f64, b: f64) -> Result<WasmProbabilityMeasure, JsValue> {
        if b <= a {
            return Err(JsValue::from_str("Upper bound must be > lower bound"));
        }
        Ok(Self {
            description: format!("Uniform[{},{}]", a, b),
        })
    }

    /// Get a description of this probability measure
    #[wasm_bindgen(js_name = getDescription)]
    pub fn get_description(&self) -> String {
        self.description.clone()
    }

    /// Compute P(X ∈ [a, b]) for uniform distribution
    #[wasm_bindgen(js_name = probabilityInterval)]
    pub fn probability_interval(&self, a: f64, b: f64, lower: f64, upper: f64) -> f64 {
        // Clamp interval [a, b] to [lower, upper]
        let clamped_a = a.max(lower);
        let clamped_b = b.min(upper);

        if clamped_b <= clamped_a {
            return 0.0;
        }

        (clamped_b - clamped_a) / (upper - lower)
    }
}

/// Integration methods available in WASM
#[wasm_bindgen]
pub enum WasmIntegrationMethod {
    /// Riemann sum approximation
    Riemann,
    /// Monte Carlo integration
    MonteCarlo,
    /// Trapezoidal rule
    Trapezoidal,
    /// Simpson's rule
    Simpson,
    /// Adaptive quadrature
    Adaptive,
}

/// Integrate a JavaScript function over an interval
///
/// This function provides numerical integration capabilities to JavaScript.
///
/// # Arguments
/// * `f` - JavaScript function to integrate (must accept a number and return a number)
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `points` - Number of sample points to use
/// * `method` - Integration method to use
///
/// # Returns
/// Approximate value of the integral ∫_a^b f(x) dx
#[wasm_bindgen(js_name = integrate)]
pub fn wasm_integrate(
    f: &js_sys::Function,
    a: f64,
    b: f64,
    points: usize,
    method: WasmIntegrationMethod,
) -> Result<f64, JsValue> {
    if points == 0 {
        return Err(JsValue::from_str("Number of points must be > 0"));
    }

    if b < a {
        return Err(JsValue::from_str("Upper bound must be >= lower bound"));
    }

    match method {
        WasmIntegrationMethod::Riemann => integrate_riemann(f, a, b, points),
        WasmIntegrationMethod::MonteCarlo => integrate_monte_carlo(f, a, b, points),
        WasmIntegrationMethod::Trapezoidal => integrate_trapezoidal(f, a, b, points),
        WasmIntegrationMethod::Simpson => integrate_simpson(f, a, b, points),
        WasmIntegrationMethod::Adaptive => integrate_adaptive(f, a, b, points),
    }
}

/// Riemann sum integration
fn integrate_riemann(f: &js_sys::Function, a: f64, b: f64, points: usize) -> Result<f64, JsValue> {
    let dx = (b - a) / points as f64;
    let mut sum = 0.0;

    for i in 0..points {
        let x = a + (i as f64 + 0.5) * dx;
        let this = JsValue::null();
        let x_val = JsValue::from_f64(x);
        let result = f.call1(&this, &x_val)?;
        let y = result
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function must return a number"))?;
        sum += y;
    }

    Ok(sum * dx)
}

/// Monte Carlo integration
fn integrate_monte_carlo(
    f: &js_sys::Function,
    a: f64,
    b: f64,
    samples: usize,
) -> Result<f64, JsValue> {
    let mut sum = 0.0;
    let mut rng = fastrand::Rng::new();

    for _ in 0..samples {
        let x = a + rng.f64() * (b - a);
        let this = JsValue::null();
        let x_val = JsValue::from_f64(x);
        let result = f.call1(&this, &x_val)?;
        let y = result
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function must return a number"))?;
        sum += y;
    }

    let average = sum / samples as f64;
    Ok((b - a) * average)
}

/// Trapezoidal rule integration
fn integrate_trapezoidal(
    f: &js_sys::Function,
    a: f64,
    b: f64,
    points: usize,
) -> Result<f64, JsValue> {
    let h = (b - a) / points as f64;

    // Evaluate at endpoints
    let this = JsValue::null();
    let f_a = f
        .call1(&this, &JsValue::from_f64(a))?
        .as_f64()
        .ok_or_else(|| JsValue::from_str("Function must return a number"))?;
    let f_b = f
        .call1(&this, &JsValue::from_f64(b))?
        .as_f64()
        .ok_or_else(|| JsValue::from_str("Function must return a number"))?;

    let mut sum = (f_a + f_b) / 2.0;

    // Evaluate at interior points
    for i in 1..points {
        let x = a + (i as f64) * h;
        let result = f.call1(&this, &JsValue::from_f64(x))?;
        let y = result
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function must return a number"))?;
        sum += y;
    }

    Ok(h * sum)
}

/// Simpson's rule integration
fn integrate_simpson(
    f: &js_sys::Function,
    a: f64,
    b: f64,
    intervals: usize,
) -> Result<f64, JsValue> {
    // Simpson's rule requires even number of intervals
    let n = if intervals % 2 == 0 {
        intervals
    } else {
        intervals + 1
    };
    let h = (b - a) / n as f64;

    let this = JsValue::null();

    // Evaluate at endpoints
    let f_a = f
        .call1(&this, &JsValue::from_f64(a))?
        .as_f64()
        .ok_or_else(|| JsValue::from_str("Function must return a number"))?;
    let f_b = f
        .call1(&this, &JsValue::from_f64(b))?
        .as_f64()
        .ok_or_else(|| JsValue::from_str("Function must return a number"))?;

    let mut sum = f_a + f_b;

    // Apply Simpson's coefficients
    for i in 1..n {
        let x = a + (i as f64) * h;
        let result = f.call1(&this, &JsValue::from_f64(x))?;
        let y = result
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function must return a number"))?;

        if i % 2 == 0 {
            sum += 2.0 * y;
        } else {
            sum += 4.0 * y;
        }
    }

    Ok((h / 3.0) * sum)
}

/// Adaptive quadrature integration
fn integrate_adaptive(
    f: &js_sys::Function,
    a: f64,
    b: f64,
    max_evals: usize,
) -> Result<f64, JsValue> {
    let tolerance = 1e-6;
    let max_depth = 10;

    adaptive_quad_recursive(f, a, b, tolerance, max_depth, &mut 0, max_evals)
}

fn adaptive_quad_recursive(
    f: &js_sys::Function,
    a: f64,
    b: f64,
    tolerance: f64,
    depth: usize,
    evals: &mut usize,
    max_evals: usize,
) -> Result<f64, JsValue> {
    if depth == 0 || *evals >= max_evals {
        return integrate_simpson(f, a, b, 4);
    }

    // Compute integral over whole interval
    let whole = integrate_simpson(f, a, b, 4)?;
    *evals += 5;

    // Compute integral over two halves
    let mid = (a + b) / 2.0;
    let left = integrate_simpson(f, a, mid, 4)?;
    let right = integrate_simpson(f, mid, b, 4)?;
    *evals += 10;

    let halves = left + right;

    // Check if subdivision is needed
    if (whole - halves).abs() < 15.0 * tolerance {
        Ok(halves + (halves - whole) / 15.0)
    } else {
        let left_result =
            adaptive_quad_recursive(f, a, mid, tolerance / 2.0, depth - 1, evals, max_evals)?;
        let right_result =
            adaptive_quad_recursive(f, mid, b, tolerance / 2.0, depth - 1, evals, max_evals)?;
        Ok(left_result + right_result)
    }
}

/// Compute expectation E[f(X)] for uniform distribution on [a, b]
///
/// # Arguments
/// * `f` - JavaScript function (must accept a number and return a number)
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `samples` - Number of Monte Carlo samples
#[wasm_bindgen(js_name = expectation)]
pub fn wasm_expectation(
    f: &js_sys::Function,
    a: f64,
    b: f64,
    samples: usize,
) -> Result<f64, JsValue> {
    if samples == 0 {
        return Err(JsValue::from_str("Number of samples must be > 0"));
    }

    let mut sum = 0.0;
    let dx = (b - a) / samples as f64;

    for i in 0..samples {
        let x = a + (i as f64 + 0.5) * dx;
        let this = JsValue::null();
        let x_val = JsValue::from_f64(x);
        let result = f.call1(&this, &x_val)?;
        let y = result
            .as_f64()
            .ok_or_else(|| JsValue::from_str("Function must return a number"))?;
        sum += y;
    }

    Ok(sum / samples as f64)
}

impl Default for WasmCountingMeasure {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WasmProbabilityMeasure {
    fn default() -> Self {
        Self::new()
    }
}

/// Parametric probability density families
#[wasm_bindgen]
pub struct WasmParametricDensity {
    family_type: String,
}

#[wasm_bindgen]
impl WasmParametricDensity {
    /// Create a Gaussian density N(μ, σ²)
    #[wasm_bindgen(js_name = gaussian)]
    pub fn gaussian() -> WasmParametricDensity {
        Self {
            family_type: "gaussian".to_string(),
        }
    }

    /// Create an Exponential density Exp(λ)
    #[wasm_bindgen(js_name = exponential)]
    pub fn exponential() -> WasmParametricDensity {
        Self {
            family_type: "exponential".to_string(),
        }
    }

    /// Create a Cauchy density Cauchy(x₀, γ)
    #[wasm_bindgen(js_name = cauchy)]
    pub fn cauchy() -> WasmParametricDensity {
        Self {
            family_type: "cauchy".to_string(),
        }
    }

    /// Create a Laplace density Laplace(μ, b)
    #[wasm_bindgen(js_name = laplace)]
    pub fn laplace() -> WasmParametricDensity {
        Self {
            family_type: "laplace".to_string(),
        }
    }

    /// Evaluate density at point x with parameters
    ///
    /// # Arguments
    /// * `x` - Point to evaluate
    /// * `params` - Parameters (Gaussian: [μ, σ], Exponential: [λ], etc.)
    #[wasm_bindgen(js_name = evaluate)]
    pub fn evaluate(&self, x: f64, params: &[f64]) -> Result<f64, JsValue> {
        match self.family_type.as_str() {
            "gaussian" => {
                if params.len() != 2 {
                    return Err(JsValue::from_str("Gaussian requires 2 parameters [μ, σ]"));
                }
                let mu = params[0];
                let sigma = params[1];
                if sigma <= 0.0 {
                    return Err(JsValue::from_str("σ must be positive"));
                }
                let z = (x - mu) / sigma;
                let normalization = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
                Ok(normalization * (-0.5 * z * z).exp())
            }
            "exponential" => {
                if params.len() != 1 {
                    return Err(JsValue::from_str("Exponential requires 1 parameter [λ]"));
                }
                let lambda = params[0];
                if lambda <= 0.0 {
                    return Err(JsValue::from_str("λ must be positive"));
                }
                if x < 0.0 {
                    Ok(0.0)
                } else {
                    Ok(lambda * (-lambda * x).exp())
                }
            }
            "cauchy" => {
                if params.len() != 2 {
                    return Err(JsValue::from_str("Cauchy requires 2 parameters [x₀, γ]"));
                }
                let x0 = params[0];
                let gamma = params[1];
                if gamma <= 0.0 {
                    return Err(JsValue::from_str("γ must be positive"));
                }
                let normalization = 1.0 / (std::f64::consts::PI * gamma);
                let term = ((x - x0) / gamma).powi(2);
                Ok(normalization / (1.0 + term))
            }
            "laplace" => {
                if params.len() != 2 {
                    return Err(JsValue::from_str("Laplace requires 2 parameters [μ, b]"));
                }
                let mu = params[0];
                let b = params[1];
                if b <= 0.0 {
                    return Err(JsValue::from_str("b must be positive"));
                }
                Ok((1.0 / (2.0 * b)) * (-(x - mu).abs() / b).exp())
            }
            _ => Err(JsValue::from_str("Unknown density family")),
        }
    }

    /// Compute log-density log p(x|θ)
    #[wasm_bindgen(js_name = logDensity)]
    pub fn log_density(&self, x: f64, params: &[f64]) -> Result<f64, JsValue> {
        let density = self.evaluate(x, params)?;
        if density <= 0.0 {
            Ok(f64::NEG_INFINITY)
        } else {
            Ok(density.ln())
        }
    }

    /// Compute numerical gradient ∇_θ p(x|θ)
    #[wasm_bindgen(js_name = gradient)]
    pub fn gradient(&self, x: f64, params: &[f64]) -> Result<Vec<f64>, JsValue> {
        let epsilon = 1e-7;
        let mut gradient = Vec::with_capacity(params.len());

        let f = self.evaluate(x, params)?;

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;
            let f_plus = self.evaluate(x, &params_plus)?;
            gradient.push((f_plus - f) / epsilon);
        }

        Ok(gradient)
    }

    /// Compute Fisher information matrix from data samples
    #[wasm_bindgen(js_name = fisherInformation)]
    pub fn fisher_information(&self, data: &[f64], params: &[f64]) -> Result<Vec<f64>, JsValue> {
        let n = params.len();
        let mut fisher = vec![0.0; n * n];

        for &x in data {
            let (_log_p, score) = self.compute_score(x, params)?;

            for i in 0..n {
                for j in 0..n {
                    fisher[i * n + j] += score[i] * score[j];
                }
            }
        }

        // Normalize by sample size
        for val in fisher.iter_mut() {
            *val /= data.len() as f64;
        }

        Ok(fisher)
    }

    /// Compute score function: ∂/∂θ log p(x|θ)
    fn compute_score(&self, x: f64, params: &[f64]) -> Result<(f64, Vec<f64>), JsValue> {
        let epsilon = 1e-7;
        let log_p = self.log_density(x, params)?;
        let mut score = Vec::with_capacity(params.len());

        for i in 0..params.len() {
            let mut params_plus = params.to_vec();
            params_plus[i] += epsilon;
            let log_p_plus = self.log_density(x, &params_plus)?;
            score.push((log_p_plus - log_p) / epsilon);
        }

        Ok((log_p, score))
    }
}

/// Tropical (max-plus) algebra operations for optimization
#[wasm_bindgen]
pub struct WasmTropicalMeasure;

#[wasm_bindgen]
impl WasmTropicalMeasure {
    /// Create a new tropical measure
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    /// Compute tropical supremum (maximum) of function over sample points
    ///
    /// Returns the maximum value and the point where it occurs
    #[wasm_bindgen(js_name = supremum)]
    pub fn supremum(&self, f: &js_sys::Function, points: &[f64]) -> Result<Vec<f64>, JsValue> {
        if points.is_empty() {
            return Err(JsValue::from_str("Sample points must not be empty"));
        }

        let this = JsValue::null();
        let mut max_value = f64::NEG_INFINITY;
        let mut max_point = points[0];

        for &x in points {
            let result = f.call1(&this, &JsValue::from_f64(x))?;
            let y = result
                .as_f64()
                .ok_or_else(|| JsValue::from_str("Function must return a number"))?;

            if y > max_value {
                max_value = y;
                max_point = x;
            }
        }

        Ok(vec![max_value, max_point])
    }

    /// Compute tropical infimum (minimum) of function over sample points
    ///
    /// Returns the minimum value and the point where it occurs
    #[wasm_bindgen(js_name = infimum)]
    pub fn infimum(&self, f: &js_sys::Function, points: &[f64]) -> Result<Vec<f64>, JsValue> {
        if points.is_empty() {
            return Err(JsValue::from_str("Sample points must not be empty"));
        }

        let this = JsValue::null();
        let mut min_value = f64::INFINITY;
        let mut min_point = points[0];

        for &x in points {
            let result = f.call1(&this, &JsValue::from_f64(x))?;
            let y = result
                .as_f64()
                .ok_or_else(|| JsValue::from_str("Function must return a number"))?;

            if y < min_value {
                min_value = y;
                min_point = x;
            }
        }

        Ok(vec![min_value, min_point])
    }

    /// Tropical integration (supremum over region)
    #[wasm_bindgen(js_name = tropicalIntegrate)]
    pub fn tropical_integrate(
        &self,
        f: &js_sys::Function,
        a: f64,
        b: f64,
        samples: usize,
    ) -> Result<f64, JsValue> {
        let mut points = Vec::with_capacity(samples);
        for i in 0..samples {
            let t = i as f64 / (samples - 1) as f64;
            points.push(a + t * (b - a));
        }

        let result = self.supremum(f, &points)?;
        Ok(result[0])
    }
}

impl Default for WasmTropicalMeasure {
    fn default() -> Self {
        Self::new()
    }
}

/// Fisher-Riemannian geometry on statistical manifolds
#[wasm_bindgen]
pub struct WasmFisherMeasure {
    density: WasmParametricDensity,
}

#[wasm_bindgen]
impl WasmFisherMeasure {
    /// Create Fisher measure from a parametric density
    #[wasm_bindgen(js_name = fromDensity)]
    pub fn from_density(density: WasmParametricDensity) -> Self {
        Self { density }
    }

    /// Compute the Fisher information metric at parameter point θ
    #[wasm_bindgen(js_name = fisherMetric)]
    pub fn fisher_metric(&self, data: &[f64], params: &[f64]) -> Result<Vec<f64>, JsValue> {
        self.density.fisher_information(data, params)
    }

    /// Compute the Riemannian volume element √det(g(θ))
    #[wasm_bindgen(js_name = volumeElement)]
    pub fn volume_element(&self, data: &[f64], params: &[f64]) -> Result<f64, JsValue> {
        let fisher = self.fisher_metric(data, params)?;
        let det = self.determinant(&fisher, params.len())?;

        if det < 0.0 {
            return Err(JsValue::from_str("Fisher matrix has negative determinant"));
        }

        Ok(det.sqrt())
    }

    /// Compute determinant (supports up to 3x3 matrices)
    fn determinant(&self, matrix: &[f64], n: usize) -> Result<f64, JsValue> {
        match n {
            1 => Ok(matrix[0]),
            2 => Ok(matrix[0] * matrix[3] - matrix[1] * matrix[2]),
            3 => Ok(matrix[0] * matrix[4] * matrix[8]
                + matrix[1] * matrix[5] * matrix[6]
                + matrix[2] * matrix[3] * matrix[7]
                - matrix[2] * matrix[4] * matrix[6]
                - matrix[1] * matrix[3] * matrix[8]
                - matrix[0] * matrix[5] * matrix[7]),
            _ => Err(JsValue::from_str(
                "Determinant only supported for matrices up to 3x3",
            )),
        }
    }
}

/// Compute KL divergence D_KL(P||Q) between two distributions
///
/// # Arguments
/// * `p_density` - First distribution
/// * `q_density` - Second distribution
/// * `p_params` - Parameters for P
/// * `q_params` - Parameters for Q
/// * `sample_points` - Points to evaluate at
#[wasm_bindgen(js_name = klDivergence)]
pub fn kl_divergence(
    p_density: &WasmParametricDensity,
    q_density: &WasmParametricDensity,
    p_params: &[f64],
    q_params: &[f64],
    sample_points: &[f64],
) -> Result<f64, JsValue> {
    let mut sum = 0.0;

    for &x in sample_points {
        let p = p_density.evaluate(x, p_params)?;
        let q = q_density.evaluate(x, q_params)?;

        if p > 0.0 && q > 0.0 {
            sum += p * (p / q).ln();
        }
    }

    // Approximate integral via average
    Ok(sum / sample_points.len() as f64)
}
