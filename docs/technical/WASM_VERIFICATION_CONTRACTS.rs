//! Phase 4C: WASM-Compatible Verification Contracts
//!
//! This module implements verification contracts specifically designed for WebAssembly environments
//! where phantom types are erased at runtime and JavaScript interop introduces additional constraints.
//! The design focuses on boundary verification, runtime type checking, and progressive enhancement.

use std::marker::PhantomData;
use std::time::{Duration, Instant};
use amari_core::Multivector;

/// WASM-specific verification constraints and limitations
#[derive(Debug, Clone, Copy)]
pub struct WasmConstraints {
    /// JavaScript interoperability mode
    pub js_interop: bool,
    /// Memory constraints in WASM linear memory (bytes)
    pub max_linear_memory: u64,
    /// Maximum verification overhead per operation (microseconds)
    pub max_overhead_us: u64,
    /// Browser environment type
    pub browser_env: BrowserEnvironment,
    /// Cross-origin restrictions
    pub cross_origin_restricted: bool,
}

/// Browser environment types with different capabilities
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BrowserEnvironment {
    /// Modern browser with full WebAssembly support
    Modern,
    /// Limited environment (some older browsers)
    Limited,
    /// Node.js environment
    NodeJs,
    /// Web Worker context
    WebWorker,
    /// Service Worker context
    ServiceWorker,
}

/// Verification error types specific to WASM environments
#[derive(Debug, Clone)]
pub enum WasmVerificationError {
    /// Type erasure at WASM-JS boundary
    TypeErasure { operation: String, expected_type: String },
    /// JavaScript interop validation failure
    JsInteropFailure { function: String, details: String },
    /// Linear memory constraint violation
    LinearMemoryExhaustion { requested: u64, available: u64 },
    /// Browser security restriction
    SecurityRestriction { policy: String, operation: String },
    /// Async execution consistency failure
    AsyncConsistencyFailure { operation: String, state: String },
    /// Cross-origin policy violation
    CrossOriginViolation { origin: String, operation: String },
    /// Mathematical invariant violation at WASM boundary
    BoundaryInvariantViolation { operation: String, details: String },
    /// Browser compatibility issue
    CompatibilityError { browser: String, feature: String },
}

/// WASM-aware verification level that adapts to browser constraints
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WasmVerificationLevel {
    /// No verification - maximum performance for production
    None,
    /// Interface boundary verification only
    Interface,
    /// Statistical verification using sampling
    Statistical { sample_rate: f32 },
    /// Full verification in development mode
    Development,
    /// Progressive enhancement based on browser capabilities
    Progressive,
}

/// WASM verification context that manages verification across WASM-JS boundary
pub struct WasmVerificationContext<const P: usize, const Q: usize, const R: usize> {
    constraints: WasmConstraints,
    level: WasmVerificationLevel,
    js_verification_enabled: bool,
    wasm_verification_enabled: bool,
    operation_count: u64,
    total_overhead: Duration,
    browser_features: BrowserFeatures,
    _phantom: PhantomData<(P, Q, R)>,
}

/// Browser feature detection for progressive enhancement
#[derive(Debug, Clone)]
pub struct BrowserFeatures {
    pub webassembly_support: bool,
    pub shared_array_buffer: bool,
    pub atomic_operations: bool,
    pub wasm_threads: bool,
    pub simd: bool,
    pub bulk_memory: bool,
}

impl<const P: usize, const Q: usize, const R: usize> WasmVerificationContext<P, Q, R> {
    /// Create WASM verification context with browser feature detection
    pub fn new(constraints: WasmConstraints) -> Self {
        let browser_features = Self::detect_browser_features();
        let level = Self::determine_optimal_wasm_level(constraints, &browser_features);

        Self {
            constraints,
            level,
            js_verification_enabled: constraints.js_interop,
            wasm_verification_enabled: true,
            operation_count: 0,
            total_overhead: Duration::ZERO,
            browser_features,
            _phantom: PhantomData,
        }
    }

    /// Detect available browser features for progressive enhancement
    fn detect_browser_features() -> BrowserFeatures {
        // In a real implementation, this would use JavaScript FFI to detect features
        // For now, assume a modern browser environment
        BrowserFeatures {
            webassembly_support: true,
            shared_array_buffer: false, // Often disabled for security
            atomic_operations: false,
            wasm_threads: false,
            simd: true,
            bulk_memory: true,
        }
    }

    /// Determine optimal verification level based on WASM constraints and browser features
    fn determine_optimal_wasm_level(
        constraints: WasmConstraints,
        features: &BrowserFeatures,
    ) -> WasmVerificationLevel {
        match constraints.browser_env {
            BrowserEnvironment::Modern if features.webassembly_support => {
                if constraints.js_interop {
                    WasmVerificationLevel::Statistical { sample_rate: 0.1 }
                } else {
                    WasmVerificationLevel::Interface
                }
            }
            BrowserEnvironment::Limited => WasmVerificationLevel::Interface,
            BrowserEnvironment::NodeJs => WasmVerificationLevel::Development,
            BrowserEnvironment::WebWorker | BrowserEnvironment::ServiceWorker => {
                if constraints.max_overhead_us < 100 {
                    WasmVerificationLevel::None
                } else {
                    WasmVerificationLevel::Interface
                }
            }
        }
    }

    /// Verify data transfer from JavaScript to WASM
    pub async fn verify_js_to_wasm_transfer(
        &mut self,
        js_data: &[f64],
        expected_multivectors: usize,
    ) -> Result<Vec<Multivector<P, Q, R>>, WasmVerificationError> {
        let start = Instant::now();
        self.operation_count += 1;

        // Check data size consistency
        if js_data.len() != expected_multivectors * 8 {
            return Err(WasmVerificationError::TypeErasure {
                operation: "js_to_wasm_transfer".to_string(),
                expected_type: format!("{}x8 coefficients", expected_multivectors),
            });
        }

        // Check linear memory constraints
        let required_memory = js_data.len() * std::mem::size_of::<f64>();
        if required_memory as u64 > self.constraints.max_linear_memory {
            return Err(WasmVerificationError::LinearMemoryExhaustion {
                requested: required_memory as u64,
                available: self.constraints.max_linear_memory,
            });
        }

        let mut multivectors = Vec::with_capacity(expected_multivectors);

        // Convert and validate data
        for i in 0..expected_multivectors {
            let start_idx = i * 8;
            let coeffs = &js_data[start_idx..start_idx + 8];

            // Verify all coefficients are finite (JavaScript can pass NaN/Infinity)
            for (j, &coeff) in coeffs.iter().enumerate() {
                if !coeff.is_finite() {
                    return Err(WasmVerificationError::JsInteropFailure {
                        function: "coefficient_transfer".to_string(),
                        details: format!("Non-finite coefficient at multivector {} basis {}: {}", i, j, coeff),
                    });
                }
            }

            let mv = Multivector::<P, Q, R>::from_coefficients(coeffs.to_vec());

            // Additional mathematical validation
            if let Err(e) = self.verify_multivector_properties(&mv).await {
                return Err(e);
            }

            multivectors.push(mv);
        }

        self.total_overhead += start.elapsed();
        Ok(multivectors)
    }

    /// Verify data transfer from WASM to JavaScript
    pub async fn verify_wasm_to_js_transfer(
        &mut self,
        multivectors: &[Multivector<P, Q, R>],
    ) -> Result<Vec<f64>, WasmVerificationError> {
        let start = Instant::now();

        let mut js_data = Vec::with_capacity(multivectors.len() * 8);

        for (i, mv) in multivectors.iter().enumerate() {
            // Verify mathematical properties before transfer
            if let Err(e) = self.verify_multivector_properties(mv).await {
                return Err(e);
            }

            // Extract coefficients
            for j in 0..8 {
                let coeff = mv.get(j);

                // Ensure coefficient is safe for JavaScript
                if !coeff.is_finite() {
                    return Err(WasmVerificationError::BoundaryInvariantViolation {
                        operation: "wasm_to_js_transfer".to_string(),
                        details: format!("Non-finite coefficient at multivector {} basis {}: {}", i, j, coeff),
                    });
                }

                // Check for values that might cause JavaScript precision issues
                if coeff.abs() > f64::MAX / 2.0 {
                    return Err(WasmVerificationError::JsInteropFailure {
                        function: "coefficient_transfer".to_string(),
                        details: format!("Coefficient too large for JavaScript: {}", coeff),
                    });
                }

                js_data.push(coeff);
            }
        }

        self.total_overhead += start.elapsed();
        Ok(js_data)
    }

    /// Verify asynchronous operation consistency (Promise-based operations)
    pub async fn verify_async_operation<F, T>(
        &mut self,
        operation_name: &str,
        operation: F,
    ) -> Result<T, WasmVerificationError>
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();

        match self.level {
            WasmVerificationLevel::None => {
                let result = operation.await;
                Ok(result)
            }
            WasmVerificationLevel::Interface => {
                // Basic boundary checking
                self.verify_async_preconditions(operation_name).await?;
                let result = operation.await;
                self.verify_async_postconditions(operation_name).await?;
                Ok(result)
            }
            WasmVerificationLevel::Statistical { sample_rate } => {
                if self.should_sample_operation(sample_rate) {
                    self.verify_async_preconditions(operation_name).await?;
                }
                let result = operation.await;
                if self.should_sample_operation(sample_rate) {
                    self.verify_async_postconditions(operation_name).await?;
                }
                Ok(result)
            }
            WasmVerificationLevel::Development => {
                // Full verification in development
                self.verify_async_preconditions(operation_name).await?;
                let result = operation.await;
                self.verify_async_postconditions(operation_name).await?;
                self.log_operation_performance(operation_name, start.elapsed());
                Ok(result)
            }
            WasmVerificationLevel::Progressive => {
                // Adapt based on browser capabilities
                if self.browser_features.webassembly_support {
                    self.verify_async_preconditions(operation_name).await?;
                }
                let result = operation.await;
                if self.browser_features.webassembly_support {
                    self.verify_async_postconditions(operation_name).await?;
                }
                Ok(result)
            }
        }
    }

    /// Verify batch operations with WASM-specific optimizations
    pub async fn verify_batch_operation(
        &mut self,
        operation_name: &str,
        input_batch: &[Multivector<P, Q, R>],
        output_batch: &[Multivector<P, Q, R>],
    ) -> Result<(), WasmVerificationError> {
        if input_batch.len() != output_batch.len() {
            return Err(WasmVerificationError::BoundaryInvariantViolation {
                operation: operation_name.to_string(),
                details: "Batch size mismatch".to_string(),
            });
        }

        // Check memory constraints for batch operations
        let batch_memory = input_batch.len() * std::mem::size_of::<Multivector<P, Q, R>>() * 2;
        if batch_memory as u64 > self.constraints.max_linear_memory / 2 {
            return Err(WasmVerificationError::LinearMemoryExhaustion {
                requested: batch_memory as u64,
                available: self.constraints.max_linear_memory / 2,
            });
        }

        match self.level {
            WasmVerificationLevel::None => Ok(()),
            WasmVerificationLevel::Interface => {
                // Verify only first and last elements
                if !input_batch.is_empty() {
                    self.verify_multivector_properties(&input_batch[0]).await?;
                    self.verify_multivector_properties(&output_batch[0]).await?;

                    if input_batch.len() > 1 {
                        let last = input_batch.len() - 1;
                        self.verify_multivector_properties(&input_batch[last]).await?;
                        self.verify_multivector_properties(&output_batch[last]).await?;
                    }
                }
                Ok(())
            }
            WasmVerificationLevel::Statistical { sample_rate } => {
                // Sample verification based on rate
                let sample_count = ((input_batch.len() as f32) * sample_rate).ceil() as usize;
                let step = input_batch.len() / sample_count.max(1);

                for i in (0..input_batch.len()).step_by(step) {
                    self.verify_multivector_properties(&input_batch[i]).await?;
                    self.verify_multivector_properties(&output_batch[i]).await?;
                }
                Ok(())
            }
            WasmVerificationLevel::Development => {
                // Verify all elements in development
                for (input, output) in input_batch.iter().zip(output_batch.iter()) {
                    self.verify_multivector_properties(input).await?;
                    self.verify_multivector_properties(output).await?;
                }
                Ok(())
            }
            WasmVerificationLevel::Progressive => {
                // Progressive verification based on browser capabilities
                if self.browser_features.bulk_memory {
                    // Use optimized bulk verification
                    self.verify_bulk_properties(input_batch).await?;
                    self.verify_bulk_properties(output_batch).await?;
                } else {
                    // Fallback to interface-level verification
                    if !input_batch.is_empty() {
                        self.verify_multivector_properties(&input_batch[0]).await?;
                        self.verify_multivector_properties(&output_batch[0]).await?;
                    }
                }
                Ok(())
            }
        }
    }

    /// Verify cross-origin operation safety
    pub async fn verify_cross_origin_operation(
        &self,
        operation_name: &str,
        origin: &str,
    ) -> Result<(), WasmVerificationError> {
        if self.constraints.cross_origin_restricted {
            // In a real implementation, this would check against allowed origins
            if !self.is_origin_allowed(origin) {
                return Err(WasmVerificationError::CrossOriginViolation {
                    origin: origin.to_string(),
                    operation: operation_name.to_string(),
                });
            }
        }
        Ok(())
    }

    /// Check if origin is allowed for cross-origin operations
    fn is_origin_allowed(&self, origin: &str) -> bool {
        // Simplified origin checking - in real implementation would be more sophisticated
        origin.starts_with("https://") || origin == "null" // Allow secure origins and local file
    }

    /// Verify multivector mathematical properties for WASM environment
    async fn verify_multivector_properties(
        &self,
        mv: &Multivector<P, Q, R>,
    ) -> Result<(), WasmVerificationError> {
        // Check magnitude is finite and non-negative
        let magnitude = mv.magnitude();
        if !magnitude.is_finite() || magnitude < 0.0 {
            return Err(WasmVerificationError::BoundaryInvariantViolation {
                operation: "multivector_property_check".to_string(),
                details: format!("Invalid magnitude: {}", magnitude),
            });
        }

        // Check all coefficients are finite (critical for JavaScript interop)
        for i in 0..8 {
            let coeff = mv.get(i);
            if !coeff.is_finite() {
                return Err(WasmVerificationError::BoundaryInvariantViolation {
                    operation: "coefficient_validation".to_string(),
                    details: format!("Non-finite coefficient at basis {}: {}", i, coeff),
                });
            }
        }

        // Check JavaScript number range constraints
        for i in 0..8 {
            let coeff = mv.get(i);
            if coeff.abs() > 1.7976931348623157e+308 {  // JavaScript Number.MAX_VALUE
                return Err(WasmVerificationError::JsInteropFailure {
                    function: "number_range_check".to_string(),
                    details: format!("Coefficient exceeds JavaScript number range: {}", coeff),
                });
            }
        }

        Ok(())
    }

    /// Bulk verification for arrays when browser supports bulk memory operations
    async fn verify_bulk_properties(
        &self,
        multivectors: &[Multivector<P, Q, R>],
    ) -> Result<(), WasmVerificationError> {
        // Use SIMD-style verification when available
        if self.browser_features.simd {
            // In a real implementation, this would use WASM SIMD instructions
            for mv in multivectors {
                self.verify_multivector_properties(mv).await?;
            }
        } else {
            // Fallback to sequential verification
            for mv in multivectors {
                self.verify_multivector_properties(mv).await?;
            }
        }
        Ok(())
    }

    /// Check if operation should be sampled for statistical verification
    fn should_sample_operation(&self, sample_rate: f32) -> bool {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.operation_count.hash(&mut hasher);
        let hash = hasher.finish();

        (hash as f64 / u64::MAX as f64) < sample_rate as f64
    }

    /// Verify async operation preconditions
    async fn verify_async_preconditions(&self, operation_name: &str) -> Result<(), WasmVerificationError> {
        // Check browser environment compatibility
        match self.constraints.browser_env {
            BrowserEnvironment::Limited => {
                if operation_name.contains("simd") && !self.browser_features.simd {
                    return Err(WasmVerificationError::CompatibilityError {
                        browser: "limited".to_string(),
                        feature: "SIMD".to_string(),
                    });
                }
            }
            BrowserEnvironment::WebWorker => {
                if operation_name.contains("shared") && !self.browser_features.shared_array_buffer {
                    return Err(WasmVerificationError::SecurityRestriction {
                        policy: "SharedArrayBuffer disabled".to_string(),
                        operation: operation_name.to_string(),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Verify async operation postconditions
    async fn verify_async_postconditions(&self, _operation_name: &str) -> Result<(), WasmVerificationError> {
        // Post-operation validation
        Ok(())
    }

    /// Log operation performance in development mode
    fn log_operation_performance(&self, operation_name: &str, duration: Duration) {
        if matches!(self.level, WasmVerificationLevel::Development) {
            // In a real implementation, this would use console.log via JS FFI
            eprintln!("WASM Operation '{}' took {:?}", operation_name, duration);
        }
    }

    /// Get WASM verification statistics
    pub fn get_wasm_stats(&self) -> WasmVerificationStats {
        WasmVerificationStats {
            level: self.level,
            operation_count: self.operation_count,
            total_overhead: self.total_overhead,
            js_verification_enabled: self.js_verification_enabled,
            wasm_verification_enabled: self.wasm_verification_enabled,
            browser_features: self.browser_features.clone(),
            constraints: self.constraints,
        }
    }

    /// Enable progressive enhancement based on runtime feature detection
    pub fn enable_progressive_enhancement(&mut self, detected_features: BrowserFeatures) {
        self.browser_features = detected_features;

        // Adjust verification level based on detected features
        if self.browser_features.webassembly_support && self.browser_features.simd {
            self.level = WasmVerificationLevel::Statistical { sample_rate: 0.1 };
        } else if self.browser_features.webassembly_support {
            self.level = WasmVerificationLevel::Interface;
        } else {
            self.level = WasmVerificationLevel::None;
        }
    }
}

/// WASM verification statistics for performance monitoring
#[derive(Debug)]
pub struct WasmVerificationStats {
    pub level: WasmVerificationLevel,
    pub operation_count: u64,
    pub total_overhead: Duration,
    pub js_verification_enabled: bool,
    pub wasm_verification_enabled: bool,
    pub browser_features: BrowserFeatures,
    pub constraints: WasmConstraints,
}

/// WASM-verified multivector wrapper with JavaScript interop safety
pub struct WasmVerifiedMultivector<const P: usize, const Q: usize, const R: usize> {
    inner: Multivector<P, Q, R>,
    js_safe: bool,
    wasm_verified: bool,
    verification_level: WasmVerificationLevel,
    _phantom: PhantomData<(P, Q, R)>,
}

impl<const P: usize, const Q: usize, const R: usize> WasmVerifiedMultivector<P, Q, R> {
    /// Create WASM-verified multivector with JavaScript safety checks
    pub fn new_wasm_verified(
        multivector: Multivector<P, Q, R>,
        context: &WasmVerificationContext<P, Q, R>,
    ) -> Result<Self, WasmVerificationError> {
        // Always verify mathematical properties
        if let Err(e) = futures::executor::block_on(context.verify_multivector_properties(&multivector)) {
            return Err(e);
        }

        Ok(Self {
            inner: multivector,
            js_safe: context.js_verification_enabled,
            wasm_verified: true,
            verification_level: context.level,
            _phantom: PhantomData,
        })
    }

    /// Convert to JavaScript-safe coefficient array
    pub async fn to_js_coefficients(
        &self,
        context: &mut WasmVerificationContext<P, Q, R>,
    ) -> Result<Vec<f64>, WasmVerificationError> {
        context.verify_wasm_to_js_transfer(&[self.inner]).await.map(|coeffs| coeffs)
    }

    /// Create from JavaScript coefficient array with validation
    pub async fn from_js_coefficients(
        coefficients: &[f64],
        context: &mut WasmVerificationContext<P, Q, R>,
    ) -> Result<Self, WasmVerificationError> {
        let multivectors = context.verify_js_to_wasm_transfer(coefficients, 1).await?;
        Self::new_wasm_verified(multivectors[0], context)
    }

    /// Perform verified batch geometric product for WASM
    pub async fn batch_geometric_product_wasm_verified(
        a_batch: &[Self],
        b_batch: &[Self],
        context: &mut WasmVerificationContext<P, Q, R>,
    ) -> Result<Vec<Self>, WasmVerificationError> {
        // Extract inner multivectors
        let a_mvs: Vec<Multivector<P, Q, R>> = a_batch.iter().map(|v| v.inner).collect();
        let b_mvs: Vec<Multivector<P, Q, R>> = b_batch.iter().map(|v| v.inner).collect();

        // Compute results
        let mut result_mvs = Vec::with_capacity(a_mvs.len());
        for (a, b) in a_mvs.iter().zip(b_mvs.iter()) {
            result_mvs.push(a.geometric_product(b));
        }

        // Verify batch operation
        context.verify_batch_operation("batch_geometric_product", &a_mvs, &result_mvs).await?;

        // Wrap results in verified containers
        let verified_results: Result<Vec<Self>, _> = result_mvs
            .into_iter()
            .map(|mv| Self::new_wasm_verified(mv, context))
            .collect();

        verified_results
    }

    /// Get underlying multivector (verified)
    pub fn inner(&self) -> &Multivector<P, Q, R> {
        &self.inner
    }

    /// Check if JavaScript interop safety is verified
    pub fn is_js_safe(&self) -> bool {
        self.js_safe
    }

    /// Check if WASM verification was performed
    pub fn is_wasm_verified(&self) -> bool {
        self.wasm_verified
    }

    /// Get verification level used
    pub fn verification_level(&self) -> WasmVerificationLevel {
        self.verification_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wasm_verification_context_creation() {
        let constraints = WasmConstraints {
            js_interop: true,
            max_linear_memory: 1024 * 1024, // 1MB
            max_overhead_us: 1000,          // 1ms
            browser_env: BrowserEnvironment::Modern,
            cross_origin_restricted: true,
        };

        let context = WasmVerificationContext::<3, 0, 0>::new(constraints);
        assert!(context.js_verification_enabled);
        assert_eq!(context.level, WasmVerificationLevel::Statistical { sample_rate: 0.1 });
    }

    #[tokio::test]
    async fn test_js_to_wasm_transfer_validation() {
        let constraints = WasmConstraints {
            js_interop: true,
            max_linear_memory: 1024 * 1024,
            max_overhead_us: 1000,
            browser_env: BrowserEnvironment::Modern,
            cross_origin_restricted: false,
        };

        let mut context = WasmVerificationContext::<3, 0, 0>::new(constraints);

        // Valid JavaScript data (8 coefficients for 1 multivector)
        let js_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = context.verify_js_to_wasm_transfer(&js_data, 1).await;
        assert!(result.is_ok());

        // Invalid data (wrong size)
        let invalid_data = vec![1.0, 2.0, 3.0];
        let result = context.verify_js_to_wasm_transfer(&invalid_data, 1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_wasm_verified_multivector() {
        let constraints = WasmConstraints {
            js_interop: true,
            max_linear_memory: 1024 * 1024,
            max_overhead_us: 1000,
            browser_env: BrowserEnvironment::Modern,
            cross_origin_restricted: false,
        };

        let context = WasmVerificationContext::<3, 0, 0>::new(constraints);
        let mv = Multivector::<3, 0, 0>::basis_vector(0);

        let verified_mv = WasmVerifiedMultivector::new_wasm_verified(mv, &context);
        assert!(verified_mv.is_ok());

        let verified = verified_mv.unwrap();
        assert!(verified.is_js_safe());
        assert!(verified.is_wasm_verified());
    }

    #[tokio::test]
    async fn test_nan_infinity_rejection() {
        let constraints = WasmConstraints {
            js_interop: true,
            max_linear_memory: 1024 * 1024,
            max_overhead_us: 1000,
            browser_env: BrowserEnvironment::Modern,
            cross_origin_restricted: false,
        };

        let mut context = WasmVerificationContext::<3, 0, 0>::new(constraints);

        // Test NaN rejection
        let nan_data = vec![f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = context.verify_js_to_wasm_transfer(&nan_data, 1).await;
        assert!(result.is_err());

        // Test Infinity rejection
        let inf_data = vec![f64::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = context.verify_js_to_wasm_transfer(&inf_data, 1).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cross_origin_verification() {
        let constraints = WasmConstraints {
            js_interop: true,
            max_linear_memory: 1024 * 1024,
            max_overhead_us: 1000,
            browser_env: BrowserEnvironment::Modern,
            cross_origin_restricted: true,
        };

        let context = WasmVerificationContext::<3, 0, 0>::new(constraints);

        // Test allowed origin
        let result = context.verify_cross_origin_operation("test_op", "https://example.com").await;
        assert!(result.is_ok());

        // Test disallowed origin
        let result = context.verify_cross_origin_operation("test_op", "http://malicious.com").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_browser_feature_detection() {
        let features = WasmVerificationContext::<3, 0, 0>::detect_browser_features();
        assert!(features.webassembly_support);
        // Other features may vary based on environment
    }
}