//! WASM bindings for GA-native optical field operations.
//!
//! Provides WebAssembly bindings for Lee hologram encoding and Vector Symbolic
//! Architecture (VSA) operations using geometric algebra rotor fields.
//!
//! # JavaScript Usage
//!
//! ```javascript
//! import {
//!     WasmOpticalRotorField,
//!     WasmBinaryHologram,
//!     WasmGeometricLeeEncoder,
//!     WasmOpticalFieldAlgebra,
//!     WasmOpticalCodebook
//! } from 'amari-wasm';
//!
//! // Create a random rotor field
//! const field = WasmOpticalRotorField.random(64, 64, 12345n);
//!
//! // Encode to binary hologram
//! const encoder = WasmGeometricLeeEncoder.withFrequency(64, 64, 0.25);
//! const hologram = encoder.encode(field);
//!
//! // VSA operations
//! const algebra = new WasmOpticalFieldAlgebra(64, 64);
//! const bound = algebra.bind(field1, field2);
//! const similarity = algebra.similarity(field1, field2);
//! ```

use amari_holographic::optical::{
    BinaryHologram, CodebookConfig, GeometricLeeEncoder, LeeEncoderConfig, OpticalCodebook,
    OpticalFieldAlgebra, OpticalRotorField, SymbolId, TropicalOpticalAlgebra,
};
use wasm_bindgen::prelude::*;

/// WASM wrapper for OpticalRotorField.
///
/// Represents an optical wavefront as a grid of rotors in Cl(2,0).
/// Each point has phase (rotor angle) and amplitude components.
#[wasm_bindgen]
pub struct WasmOpticalRotorField {
    inner: OpticalRotorField,
}

#[wasm_bindgen]
impl WasmOpticalRotorField {
    /// Create from phase and amplitude arrays.
    ///
    /// # Arguments
    /// * `phase` - Phase values in radians (length = width * height)
    /// * `amplitude` - Amplitude values (length = width * height)
    /// * `width` - Grid width
    /// * `height` - Grid height
    #[wasm_bindgen(constructor)]
    pub fn new(
        phase: &[f32],
        amplitude: &[f32],
        width: usize,
        height: usize,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        let expected = width * height;
        if phase.len() != expected || amplitude.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Array lengths ({}, {}) don't match dimensions {}x{}={}",
                phase.len(),
                amplitude.len(),
                width,
                height,
                expected
            )));
        }

        Ok(Self {
            inner: OpticalRotorField::new(phase.to_vec(), amplitude.to_vec(), (width, height)),
        })
    }

    /// Create from phase array with uniform amplitude of 1.0.
    #[wasm_bindgen(js_name = fromPhase)]
    pub fn from_phase(
        phase: &[f32],
        width: usize,
        height: usize,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        let expected = width * height;
        if phase.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Phase length {} doesn't match dimensions {}x{}={}",
                phase.len(),
                width,
                height,
                expected
            )));
        }

        Ok(Self {
            inner: OpticalRotorField::from_phase(phase.to_vec(), (width, height)),
        })
    }

    /// Create with random phase (deterministic from seed).
    #[wasm_bindgen]
    pub fn random(width: usize, height: usize, seed: u64) -> WasmOpticalRotorField {
        Self {
            inner: OpticalRotorField::random((width, height), seed),
        }
    }

    /// Create uniform field (constant phase and amplitude).
    #[wasm_bindgen]
    pub fn uniform(
        phase: f32,
        amplitude: f32,
        width: usize,
        height: usize,
    ) -> WasmOpticalRotorField {
        Self {
            inner: OpticalRotorField::uniform(phase, amplitude, (width, height)),
        }
    }

    /// Create identity field (phase = 0, amplitude = 1).
    #[wasm_bindgen]
    pub fn identity(width: usize, height: usize) -> WasmOpticalRotorField {
        Self {
            inner: OpticalRotorField::identity((width, height)),
        }
    }

    /// Get grid width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.dimensions().0
    }

    /// Get grid height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.dimensions().1
    }

    /// Get total number of points.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Get phase at a point (in radians, range [-π, π]).
    #[wasm_bindgen(js_name = phaseAt)]
    pub fn phase_at(&self, x: usize, y: usize) -> f32 {
        self.inner.phase_at(x, y)
    }

    /// Get amplitude at a point.
    #[wasm_bindgen(js_name = amplitudeAt)]
    pub fn amplitude_at(&self, x: usize, y: usize) -> f32 {
        self.inner.amplitude_at(x, y)
    }

    /// Get all scalar (cos φ) components.
    #[wasm_bindgen(js_name = getScalars)]
    pub fn get_scalars(&self) -> Vec<f32> {
        self.inner.scalars().to_vec()
    }

    /// Get all bivector (sin φ) components.
    #[wasm_bindgen(js_name = getBivectors)]
    pub fn get_bivectors(&self) -> Vec<f32> {
        self.inner.bivectors().to_vec()
    }

    /// Get all amplitude components.
    #[wasm_bindgen(js_name = getAmplitudes)]
    pub fn get_amplitudes(&self) -> Vec<f32> {
        self.inner.amplitudes().to_vec()
    }

    /// Compute total energy (sum of squared amplitudes).
    #[wasm_bindgen(js_name = totalEnergy)]
    pub fn total_energy(&self) -> f32 {
        self.inner.total_energy()
    }

    /// Create a normalized copy (total energy = 1).
    #[wasm_bindgen]
    pub fn normalized(&self) -> WasmOpticalRotorField {
        Self {
            inner: self.inner.normalized(),
        }
    }

    /// Clone this field.
    #[wasm_bindgen(js_name = clone)]
    pub fn clone_field(&self) -> WasmOpticalRotorField {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// WASM wrapper for BinaryHologram.
///
/// Bit-packed binary pattern for DMD display, the output of Lee encoding.
#[wasm_bindgen]
pub struct WasmBinaryHologram {
    inner: BinaryHologram,
}

#[wasm_bindgen]
impl WasmBinaryHologram {
    /// Create from boolean array (as u8: 0 = false, non-zero = true).
    #[wasm_bindgen(constructor)]
    pub fn new(pattern: &[u8], width: usize, height: usize) -> Result<WasmBinaryHologram, JsValue> {
        let expected = width * height;
        if pattern.len() != expected {
            return Err(JsValue::from_str(&format!(
                "Pattern length {} doesn't match dimensions {}x{}={}",
                pattern.len(),
                width,
                height,
                expected
            )));
        }

        let bools: Vec<bool> = pattern.iter().map(|&b| b != 0).collect();
        Ok(Self {
            inner: BinaryHologram::from_bools(&bools, (width, height)),
        })
    }

    /// Create an all-zeros hologram.
    #[wasm_bindgen]
    pub fn zeros(width: usize, height: usize) -> WasmBinaryHologram {
        Self {
            inner: BinaryHologram::zeros((width, height)),
        }
    }

    /// Create an all-ones hologram.
    #[wasm_bindgen]
    pub fn ones(width: usize, height: usize) -> WasmBinaryHologram {
        Self {
            inner: BinaryHologram::ones((width, height)),
        }
    }

    /// Get grid width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.dimensions().0
    }

    /// Get grid height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.dimensions().1
    }

    /// Get total number of pixels.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Get pixel value at (x, y).
    #[wasm_bindgen]
    pub fn get(&self, x: usize, y: usize) -> bool {
        self.inner.get(x, y)
    }

    /// Set pixel value at (x, y).
    #[wasm_bindgen]
    pub fn set(&mut self, x: usize, y: usize, value: bool) {
        self.inner.set(x, y, value);
    }

    /// Toggle pixel at (x, y).
    #[wasm_bindgen]
    pub fn toggle(&mut self, x: usize, y: usize) {
        self.inner.toggle(x, y);
    }

    /// Get packed binary data (for hardware interface).
    #[wasm_bindgen(js_name = asBytes)]
    pub fn as_bytes(&self) -> Vec<u8> {
        self.inner.as_bytes().to_vec()
    }

    /// Count of "on" pixels.
    #[wasm_bindgen]
    pub fn popcount(&self) -> usize {
        self.inner.popcount()
    }

    /// Fill factor (fraction of "on" pixels, 0 to 1).
    #[wasm_bindgen(js_name = fillFactor)]
    pub fn fill_factor(&self) -> f32 {
        self.inner.fill_factor()
    }

    /// Compute Hamming distance between two holograms.
    #[wasm_bindgen(js_name = hammingDistance)]
    pub fn hamming_distance(&self, other: &WasmBinaryHologram) -> usize {
        self.inner.hamming_distance(&other.inner)
    }

    /// Compute normalized Hamming distance (0 to 1).
    #[wasm_bindgen(js_name = normalizedHammingDistance)]
    pub fn normalized_hamming_distance(&self, other: &WasmBinaryHologram) -> f32 {
        self.inner.normalized_hamming_distance(&other.inner)
    }

    /// XOR two holograms.
    #[wasm_bindgen]
    pub fn xor(&self, other: &WasmBinaryHologram) -> WasmBinaryHologram {
        Self {
            inner: self.inner.xor(&other.inner),
        }
    }

    /// Create an inverted copy.
    #[wasm_bindgen]
    pub fn inverted(&self) -> WasmBinaryHologram {
        Self {
            inner: self.inner.inverted(),
        }
    }

    /// Convert to boolean array (as u8: 0 = false, 1 = true).
    #[wasm_bindgen(js_name = toBools)]
    pub fn to_bools(&self) -> Vec<u8> {
        self.inner.to_bools().iter().map(|&b| b as u8).collect()
    }
}

/// WASM wrapper for GeometricLeeEncoder.
///
/// Encodes optical rotor fields to binary holograms using Lee's method.
#[wasm_bindgen]
pub struct WasmGeometricLeeEncoder {
    inner: GeometricLeeEncoder,
}

#[wasm_bindgen]
impl WasmGeometricLeeEncoder {
    /// Create encoder with configuration.
    ///
    /// # Arguments
    /// * `width` - Grid width
    /// * `height` - Grid height
    /// * `carrier_frequency` - Carrier frequency in cycles per pixel
    /// * `carrier_angle` - Carrier direction angle (radians, 0 = horizontal)
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        carrier_frequency: f32,
        carrier_angle: f32,
    ) -> WasmGeometricLeeEncoder {
        let config =
            LeeEncoderConfig::with_angle((width, height), carrier_frequency, carrier_angle);
        Self {
            inner: GeometricLeeEncoder::new(config),
        }
    }

    /// Create encoder with horizontal carrier (angle = 0).
    #[wasm_bindgen(js_name = withFrequency)]
    pub fn with_frequency(
        width: usize,
        height: usize,
        carrier_frequency: f32,
    ) -> WasmGeometricLeeEncoder {
        Self {
            inner: GeometricLeeEncoder::with_frequency((width, height), carrier_frequency),
        }
    }

    /// Encode a rotor field to binary hologram.
    #[wasm_bindgen]
    pub fn encode(&self, field: &WasmOpticalRotorField) -> WasmBinaryHologram {
        WasmBinaryHologram {
            inner: self.inner.encode(&field.inner),
        }
    }

    /// Compute the modulated rotor field (before thresholding).
    #[wasm_bindgen]
    pub fn modulate(&self, field: &WasmOpticalRotorField) -> WasmOpticalRotorField {
        WasmOpticalRotorField {
            inner: self.inner.modulate(&field.inner),
        }
    }

    /// Theoretical diffraction efficiency for the given field.
    #[wasm_bindgen(js_name = theoreticalEfficiency)]
    pub fn theoretical_efficiency(&self, field: &WasmOpticalRotorField) -> f32 {
        self.inner.theoretical_efficiency(&field.inner)
    }

    /// Get carrier frequency.
    #[wasm_bindgen(getter, js_name = carrierFrequency)]
    pub fn carrier_frequency(&self) -> f32 {
        self.inner.config().carrier_frequency
    }

    /// Get carrier angle.
    #[wasm_bindgen(getter, js_name = carrierAngle)]
    pub fn carrier_angle(&self) -> f32 {
        self.inner.config().carrier_angle
    }

    /// Get encoder width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.dimensions().0
    }

    /// Get encoder height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.dimensions().1
    }
}

/// WASM wrapper for OpticalFieldAlgebra.
///
/// Provides VSA operations on rotor fields: bind, bundle, similarity, inverse.
#[wasm_bindgen]
pub struct WasmOpticalFieldAlgebra {
    inner: OpticalFieldAlgebra,
}

#[wasm_bindgen]
impl WasmOpticalFieldAlgebra {
    /// Create new algebra instance for fields of the given dimensions.
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> WasmOpticalFieldAlgebra {
        Self {
            inner: OpticalFieldAlgebra::new((width, height)),
        }
    }

    /// Get algebra width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.dimensions().0
    }

    /// Get algebra height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.dimensions().1
    }

    /// Create identity field (phase = 0, amplitude = 1).
    #[wasm_bindgen]
    pub fn identity(&self) -> WasmOpticalRotorField {
        WasmOpticalRotorField {
            inner: self.inner.identity(),
        }
    }

    /// Create random field.
    #[wasm_bindgen]
    pub fn random(&self, seed: u64) -> WasmOpticalRotorField {
        WasmOpticalRotorField {
            inner: self.inner.random(seed),
        }
    }

    /// Bind two fields (pointwise rotor product).
    ///
    /// Creates an association: bound = A ⊗ B.
    /// Unbind with: B ≈ inverse(A) ⊗ bound.
    #[wasm_bindgen]
    pub fn bind(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if a.inner.dimensions() != self.inner.dimensions()
            || b.inner.dimensions() != self.inner.dimensions()
        {
            return Err(JsValue::from_str(
                "Field dimensions must match algebra dimensions",
            ));
        }
        Ok(WasmOpticalRotorField {
            inner: self.inner.bind(&a.inner, &b.inner),
        })
    }

    /// Bundle multiple fields (weighted superposition).
    ///
    /// # Arguments
    /// * `fields` - Flattened array of field data (scalars, bivectors, amplitudes for each field)
    /// * `weights` - Weights for each field
    /// * `field_count` - Number of fields
    #[wasm_bindgen]
    pub fn bundle(
        &self,
        fields: Vec<WasmOpticalRotorField>,
        weights: &[f32],
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if fields.is_empty() {
            return Err(JsValue::from_str("Cannot bundle empty field list"));
        }
        if weights.len() != fields.len() {
            return Err(JsValue::from_str("Weights length must match fields length"));
        }

        let owned_fields: Vec<OpticalRotorField> = fields.into_iter().map(|f| f.inner).collect();
        Ok(WasmOpticalRotorField {
            inner: self.inner.bundle(&owned_fields, weights),
        })
    }

    /// Bundle with uniform weights (1/n).
    #[wasm_bindgen(js_name = bundleUniform)]
    pub fn bundle_uniform(
        &self,
        fields: Vec<WasmOpticalRotorField>,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if fields.is_empty() {
            return Err(JsValue::from_str("Cannot bundle empty field list"));
        }

        let owned_fields: Vec<OpticalRotorField> = fields.into_iter().map(|f| f.inner).collect();
        Ok(WasmOpticalRotorField {
            inner: self.inner.bundle_uniform(&owned_fields),
        })
    }

    /// Compute similarity between two fields.
    ///
    /// Returns normalized inner product.
    /// Range: [-1, 1], where 1 = identical phase.
    #[wasm_bindgen]
    pub fn similarity(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
    ) -> Result<f32, JsValue> {
        if a.inner.dimensions() != b.inner.dimensions() {
            return Err(JsValue::from_str("Field dimensions must match"));
        }
        Ok(self.inner.similarity(&a.inner, &b.inner))
    }

    /// Compute inverse field (rotor reverse).
    ///
    /// For rotors: R^(-1) = R^† = cos(φ) - sin(φ)·e₁₂
    #[wasm_bindgen]
    pub fn inverse(&self, field: &WasmOpticalRotorField) -> WasmOpticalRotorField {
        WasmOpticalRotorField {
            inner: self.inner.inverse(&field.inner),
        }
    }

    /// Unbind operation: retrieve associated value.
    ///
    /// Given `bound = bind(key, value)`, calling `unbind(key, bound)`
    /// returns (approximately) `value`.
    #[wasm_bindgen]
    pub fn unbind(
        &self,
        key: &WasmOpticalRotorField,
        bound: &WasmOpticalRotorField,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if key.inner.dimensions() != self.inner.dimensions()
            || bound.inner.dimensions() != self.inner.dimensions()
        {
            return Err(JsValue::from_str(
                "Field dimensions must match algebra dimensions",
            ));
        }
        Ok(WasmOpticalRotorField {
            inner: self.inner.unbind(&key.inner, &bound.inner),
        })
    }

    /// Scale a field's amplitude by a constant factor.
    #[wasm_bindgen]
    pub fn scale(&self, field: &WasmOpticalRotorField, factor: f32) -> WasmOpticalRotorField {
        WasmOpticalRotorField {
            inner: self.inner.scale(&field.inner, factor),
        }
    }

    /// Add a constant phase to all pixels.
    #[wasm_bindgen(js_name = addPhase)]
    pub fn add_phase(&self, field: &WasmOpticalRotorField, phase: f32) -> WasmOpticalRotorField {
        WasmOpticalRotorField {
            inner: self.inner.add_phase(&field.inner, phase),
        }
    }

    /// Compute the average phase (circular mean) of a field.
    #[wasm_bindgen(js_name = meanPhase)]
    pub fn mean_phase(&self, field: &WasmOpticalRotorField) -> f32 {
        self.inner.mean_phase(&field.inner)
    }

    /// Compute phase variance (circular variance) of a field.
    #[wasm_bindgen(js_name = phaseVariance)]
    pub fn phase_variance(&self, field: &WasmOpticalRotorField) -> f32 {
        self.inner.phase_variance(&field.inner)
    }
}

/// WASM wrapper for OpticalCodebook.
///
/// Maps symbols to deterministically-generated rotor fields.
#[wasm_bindgen]
pub struct WasmOpticalCodebook {
    inner: OpticalCodebook,
}

#[wasm_bindgen]
impl WasmOpticalCodebook {
    /// Create a new codebook.
    ///
    /// # Arguments
    /// * `width` - Field grid width
    /// * `height` - Field grid height
    /// * `base_seed` - Base seed for deterministic generation
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, base_seed: u64) -> WasmOpticalCodebook {
        let config = CodebookConfig::new((width, height), base_seed);
        Self {
            inner: OpticalCodebook::new(config),
        }
    }

    /// Register a symbol with auto-generated seed.
    #[wasm_bindgen]
    pub fn register(&mut self, symbol: &str) {
        self.inner.register(SymbolId::new(symbol));
    }

    /// Register a symbol with specific seed.
    #[wasm_bindgen(js_name = registerWithSeed)]
    pub fn register_with_seed(&mut self, symbol: &str, seed: u64) {
        self.inner.register_with_seed(SymbolId::new(symbol), seed);
    }

    /// Register multiple symbols at once.
    #[wasm_bindgen(js_name = registerAll)]
    pub fn register_all(&mut self, symbols: Vec<String>) {
        let ids: Vec<SymbolId> = symbols.into_iter().map(SymbolId::new).collect();
        self.inner.register_all(ids);
    }

    /// Get or generate field for a symbol.
    #[wasm_bindgen]
    pub fn get(&mut self, symbol: &str) -> Option<WasmOpticalRotorField> {
        self.inner
            .get(&SymbolId::new(symbol))
            .map(|f| WasmOpticalRotorField { inner: f.clone() })
    }

    /// Generate field without caching.
    #[wasm_bindgen]
    pub fn generate(&self, symbol: &str) -> Option<WasmOpticalRotorField> {
        self.inner
            .generate(&SymbolId::new(symbol))
            .map(|f| WasmOpticalRotorField { inner: f })
    }

    /// Check if a symbol is registered.
    #[wasm_bindgen]
    pub fn contains(&self, symbol: &str) -> bool {
        self.inner.contains(&SymbolId::new(symbol))
    }

    /// Get the seed for a registered symbol.
    #[wasm_bindgen(js_name = getSeed)]
    pub fn get_seed(&self, symbol: &str) -> Option<u64> {
        self.inner.get_seed(&SymbolId::new(symbol))
    }

    /// Number of registered symbols.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// Check if codebook is empty.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get all registered symbol names.
    #[wasm_bindgen]
    pub fn symbols(&self) -> Vec<String> {
        self.inner
            .symbols()
            .map(|s| s.as_str().to_string())
            .collect()
    }

    /// Clear the field cache (seeds retained).
    #[wasm_bindgen(js_name = clearCache)]
    pub fn clear_cache(&mut self) {
        self.inner.clear_cache();
    }

    /// Remove a symbol from the codebook.
    #[wasm_bindgen]
    pub fn remove(&mut self, symbol: &str) -> bool {
        self.inner.remove(&SymbolId::new(symbol))
    }
}

/// WASM wrapper for TropicalOpticalAlgebra.
///
/// Tropical (min, +) operations on optical fields for attractor dynamics.
#[wasm_bindgen]
pub struct WasmTropicalOpticalAlgebra {
    inner: TropicalOpticalAlgebra,
}

#[wasm_bindgen]
impl WasmTropicalOpticalAlgebra {
    /// Create new tropical algebra instance for fields of the given dimensions.
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> WasmTropicalOpticalAlgebra {
        Self {
            inner: TropicalOpticalAlgebra::new((width, height)),
        }
    }

    /// Get algebra width.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.dimensions().0
    }

    /// Get algebra height.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.dimensions().1
    }

    /// Tropical addition: pointwise minimum of phase magnitudes.
    ///
    /// For each pixel, selects the rotor with smaller absolute phase.
    #[wasm_bindgen(js_name = tropicalAdd)]
    pub fn tropical_add(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if a.inner.dimensions() != b.inner.dimensions() {
            return Err(JsValue::from_str("Field dimensions must match"));
        }
        Ok(WasmOpticalRotorField {
            inner: self.inner.tropical_add(&a.inner, &b.inner),
        })
    }

    /// Tropical maximum: pointwise maximum of phase magnitudes.
    #[wasm_bindgen(js_name = tropicalMax)]
    pub fn tropical_max(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if a.inner.dimensions() != b.inner.dimensions() {
            return Err(JsValue::from_str("Field dimensions must match"));
        }
        Ok(WasmOpticalRotorField {
            inner: self.inner.tropical_max(&a.inner, &b.inner),
        })
    }

    /// Tropical multiplication: binding operation (phase addition).
    #[wasm_bindgen(js_name = tropicalMul)]
    pub fn tropical_mul(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if a.inner.dimensions() != b.inner.dimensions() {
            return Err(JsValue::from_str("Field dimensions must match"));
        }
        Ok(WasmOpticalRotorField {
            inner: self.inner.tropical_mul(&a.inner, &b.inner),
        })
    }

    /// Compute phase distance between two fields.
    ///
    /// Returns the sum of absolute phase differences.
    #[wasm_bindgen(js_name = phaseDistance)]
    pub fn phase_distance(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
    ) -> Result<f32, JsValue> {
        if a.inner.dimensions() != b.inner.dimensions() {
            return Err(JsValue::from_str("Field dimensions must match"));
        }
        Ok(self.inner.phase_distance(&a.inner, &b.inner))
    }

    /// Compute normalized phase distance (average per pixel).
    #[wasm_bindgen(js_name = normalizedPhaseDistance)]
    pub fn normalized_phase_distance(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
    ) -> Result<f32, JsValue> {
        if a.inner.dimensions() != b.inner.dimensions() {
            return Err(JsValue::from_str("Field dimensions must match"));
        }
        Ok(self.inner.normalized_phase_distance(&a.inner, &b.inner))
    }

    /// Soft tropical addition using logsumexp-style smoothing.
    ///
    /// # Arguments
    /// * `a`, `b` - Fields to combine
    /// * `beta` - Temperature parameter (large = hard min, small = soft average)
    #[wasm_bindgen(js_name = softTropicalAdd)]
    pub fn soft_tropical_add(
        &self,
        a: &WasmOpticalRotorField,
        b: &WasmOpticalRotorField,
        beta: f32,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if a.inner.dimensions() != b.inner.dimensions() {
            return Err(JsValue::from_str("Field dimensions must match"));
        }
        Ok(WasmOpticalRotorField {
            inner: self.inner.soft_tropical_add(&a.inner, &b.inner, beta),
        })
    }

    /// Iterate attractor dynamics until convergence.
    ///
    /// # Arguments
    /// * `initial` - Starting state
    /// * `attractors` - Set of attractor fields
    /// * `max_iterations` - Maximum iterations
    /// * `tolerance` - Convergence tolerance
    ///
    /// Returns (final_state, iterations_taken).
    #[wasm_bindgen(js_name = attractorConverge)]
    pub fn attractor_converge(
        &self,
        initial: &WasmOpticalRotorField,
        attractors: Vec<WasmOpticalRotorField>,
        max_iterations: usize,
        tolerance: f32,
    ) -> Result<WasmOpticalRotorField, JsValue> {
        if attractors.is_empty() {
            return Err(JsValue::from_str("Attractors list cannot be empty"));
        }

        let owned_attractors: Vec<OpticalRotorField> =
            attractors.into_iter().map(|f| f.inner).collect();
        let (result, _iterations) = self.inner.attractor_converge(
            &initial.inner,
            &owned_attractors,
            max_iterations,
            tolerance,
        );
        Ok(WasmOpticalRotorField { inner: result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_rotor_field_creation() {
        let field = WasmOpticalRotorField::random(32, 32, 42);
        assert_eq!(field.width(), 32);
        assert_eq!(field.height(), 32);
        assert_eq!(field.length(), 1024);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_hologram_creation() {
        let zeros = WasmBinaryHologram::zeros(16, 16);
        assert_eq!(zeros.popcount(), 0);

        let ones = WasmBinaryHologram::ones(16, 16);
        assert_eq!(ones.popcount(), 256);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_lee_encoding() {
        let field = WasmOpticalRotorField::uniform(0.0, 0.5, 32, 32);
        let encoder = WasmGeometricLeeEncoder::with_frequency(32, 32, 0.25);
        let hologram = encoder.encode(&field);

        assert_eq!(hologram.width(), 32);
        assert_eq!(hologram.height(), 32);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_field_algebra() {
        let algebra = WasmOpticalFieldAlgebra::new(16, 16);
        let field1 = WasmOpticalRotorField::random(16, 16, 42);
        let field2 = WasmOpticalRotorField::random(16, 16, 43);

        let bound = algebra.bind(&field1, &field2).unwrap();
        assert_eq!(bound.width(), 16);

        let sim = algebra.similarity(&field1, &field1).unwrap();
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_codebook() {
        let mut codebook = WasmOpticalCodebook::new(32, 32, 12345);
        codebook.register("AGENT");
        codebook.register("ACTION");

        assert_eq!(codebook.length(), 2);
        assert!(codebook.contains("AGENT"));
        assert!(!codebook.contains("UNKNOWN"));

        let field = codebook.get("AGENT").unwrap();
        assert_eq!(field.width(), 32);
    }

    #[wasm_bindgen_test]
    #[allow(dead_code)]
    fn test_tropical_algebra() {
        let tropical = WasmTropicalOpticalAlgebra::new(8, 8);
        let field1 = WasmOpticalRotorField::random(8, 8, 1);
        let field2 = WasmOpticalRotorField::random(8, 8, 2);

        let result = tropical.tropical_add(&field1, &field2).unwrap();
        assert_eq!(result.width(), 8);
    }
}
