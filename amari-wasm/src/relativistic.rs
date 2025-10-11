//! WebAssembly bindings for relativistic physics
//!
//! This module provides WASM bindings for the amari-relativistic crate,
//! enabling relativistic physics calculations in web browsers for
//! spacecraft orbital mechanics and plasma physics applications.

use amari_relativistic::{
    constants::{C, EARTH_MASS, G, SOLAR_MASS},
    geodesic::{GeodesicIntegrator, Metric},
    particle::RelativisticParticle,
    schwarzschild::SchwarzschildMetric,
    spacetime::{FourVelocity, SpacetimeVector},
};
use js_sys::Array;
use nalgebra::Vector3;
use wasm_bindgen::prelude::*;

/// WASM wrapper for spacetime vectors
#[wasm_bindgen]
pub struct WasmSpacetimeVector {
    inner: SpacetimeVector,
}

#[wasm_bindgen]
impl WasmSpacetimeVector {
    /// Create a new spacetime vector with components (ct, x, y, z)
    #[wasm_bindgen(constructor)]
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: SpacetimeVector::new(t, x, y, z),
        }
    }

    /// Create a timelike vector
    #[wasm_bindgen]
    pub fn timelike(t: f64) -> Self {
        Self {
            inner: SpacetimeVector::timelike(t),
        }
    }

    /// Create a spacelike vector
    #[wasm_bindgen]
    pub fn spacelike(x: f64, y: f64, z: f64) -> Self {
        Self {
            inner: SpacetimeVector::spacelike(x, y, z),
        }
    }

    /// Get temporal component
    #[wasm_bindgen(getter)]
    pub fn t(&self) -> f64 {
        self.inner.time()
    }

    /// Get x component
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f64 {
        self.inner.x()
    }

    /// Get y component
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.inner.y()
    }

    /// Get z component
    #[wasm_bindgen(getter)]
    pub fn z(&self) -> f64 {
        self.inner.z()
    }

    /// Compute Minkowski inner product with another spacetime vector
    #[wasm_bindgen]
    pub fn minkowski_dot(&self, other: &WasmSpacetimeVector) -> f64 {
        self.inner.minkowski_dot(&other.inner)
    }

    /// Compute Minkowski norm squared
    #[wasm_bindgen]
    pub fn norm_squared(&self) -> f64 {
        self.inner.minkowski_norm_squared()
    }

    /// Check if vector is timelike (massive particle)
    #[wasm_bindgen]
    pub fn is_timelike(&self) -> bool {
        self.inner.is_timelike()
    }

    /// Check if vector is spacelike
    #[wasm_bindgen]
    pub fn is_spacelike(&self) -> bool {
        self.inner.is_spacelike()
    }

    /// Check if vector is null (lightlike)
    #[wasm_bindgen]
    pub fn is_null(&self) -> bool {
        self.inner.is_null()
    }

    /// Get string representation
    #[wasm_bindgen]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        format!(
            "SpacetimeVector({:.6}, {:.6}, {:.6}, {:.6})",
            self.t(),
            self.x(),
            self.y(),
            self.z()
        )
    }
}

/// WASM wrapper for four-velocity
#[wasm_bindgen]
pub struct WasmFourVelocity {
    inner: FourVelocity,
}

#[wasm_bindgen]
impl WasmFourVelocity {
    /// Create four-velocity from 3-velocity components
    #[wasm_bindgen]
    pub fn from_velocity(vx: f64, vy: f64, vz: f64) -> Result<WasmFourVelocity, JsValue> {
        let velocity = Vector3::new(vx, vy, vz);
        let four_velocity = FourVelocity::from_velocity(velocity);

        Ok(Self {
            inner: four_velocity,
        })
    }

    /// Get Lorentz factor γ
    #[wasm_bindgen]
    pub fn gamma(&self) -> f64 {
        self.inner.gamma()
    }

    /// Get rapidity
    #[wasm_bindgen]
    pub fn rapidity(&self) -> f64 {
        self.inner.rapidity()
    }

    /// Get spatial velocity magnitude
    #[wasm_bindgen]
    pub fn spatial_velocity_magnitude(&self) -> f64 {
        self.inner.velocity().magnitude()
    }

    /// Get as spacetime vector
    #[wasm_bindgen]
    pub fn as_spacetime_vector(&self) -> WasmSpacetimeVector {
        WasmSpacetimeVector {
            inner: self.inner.as_spacetime_vector().clone(),
        }
    }

    /// Check if normalized (u·u = c²)
    #[wasm_bindgen]
    pub fn is_normalized(&self) -> bool {
        let norm_sq = self.inner.as_spacetime_vector().minkowski_norm_squared();
        let c_sq = C * C;
        (norm_sq - c_sq).abs() < 1e-8
    }

    /// Get string representation
    #[wasm_bindgen]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        format!(
            "FourVelocity(γ={:.6}, v={:.6}c)",
            self.gamma(),
            self.spatial_velocity_magnitude() / C
        )
    }
}

/// WASM wrapper for relativistic particles
#[wasm_bindgen]
pub struct WasmRelativisticParticle {
    inner: RelativisticParticle,
}

#[wasm_bindgen]
impl WasmRelativisticParticle {
    /// Create a new relativistic particle
    #[wasm_bindgen(constructor)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x: f64,
        y: f64,
        z: f64, // Position
        vx: f64,
        vy: f64,
        vz: f64,     // Velocity
        spin: f64,   // Spin
        mass: f64,   // Rest mass
        charge: f64, // Electric charge
    ) -> Result<WasmRelativisticParticle, JsValue> {
        let position = Vector3::new(x, y, z);
        let velocity = Vector3::new(vx, vy, vz);

        match RelativisticParticle::new(position, velocity, spin, mass, charge) {
            Ok(particle) => Ok(Self { inner: particle }),
            Err(e) => Err(JsValue::from_str(&format!(
                "Failed to create particle: {}",
                e
            ))),
        }
    }

    /// Create particle with specified energy
    #[wasm_bindgen]
    #[allow(clippy::too_many_arguments)]
    pub fn with_energy(
        x: f64,
        y: f64,
        z: f64, // Position
        direction_x: f64,
        direction_y: f64,
        direction_z: f64,    // Direction
        kinetic_energy: f64, // Kinetic energy
        mass: f64,           // Rest mass
        charge: f64,         // Electric charge
    ) -> Result<WasmRelativisticParticle, JsValue> {
        let position = Vector3::new(x, y, z);
        let direction = Vector3::new(direction_x, direction_y, direction_z);

        match RelativisticParticle::with_energy(position, direction, kinetic_energy, mass, charge) {
            Ok(particle) => Ok(Self { inner: particle }),
            Err(e) => Err(JsValue::from_str(&format!(
                "Failed to create particle: {}",
                e
            ))),
        }
    }

    /// Get position as spacetime vector
    #[wasm_bindgen]
    pub fn position_4d(&self) -> WasmSpacetimeVector {
        WasmSpacetimeVector {
            inner: self.inner.position.clone(),
        }
    }

    /// Get 3D position components
    #[wasm_bindgen]
    pub fn position_3d(&self) -> Array {
        let pos = self.inner.position_3d();
        let array = Array::new();
        array.push(&JsValue::from_f64(pos.x));
        array.push(&JsValue::from_f64(pos.y));
        array.push(&JsValue::from_f64(pos.z));
        array
    }

    /// Get four-velocity
    #[wasm_bindgen]
    pub fn four_velocity(&self) -> WasmFourVelocity {
        WasmFourVelocity {
            inner: self.inner.four_velocity.clone(),
        }
    }

    /// Get rest mass
    #[wasm_bindgen]
    pub fn mass(&self) -> f64 {
        self.inner.mass
    }

    /// Get electric charge
    #[wasm_bindgen]
    pub fn charge(&self) -> f64 {
        self.inner.charge
    }

    /// Get total energy
    #[wasm_bindgen]
    pub fn total_energy(&self) -> f64 {
        self.inner.total_energy()
    }

    /// Get kinetic energy
    #[wasm_bindgen]
    pub fn kinetic_energy(&self) -> f64 {
        self.inner.kinetic_energy()
    }

    /// Get momentum magnitude
    #[wasm_bindgen]
    pub fn momentum_magnitude(&self) -> f64 {
        self.inner.momentum()
    }

    /// Get string representation
    #[wasm_bindgen]
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let pos = self.inner.position_3d();
        format!(
            "RelativisticParticle(pos=[{:.3}, {:.3}, {:.3}], E={:.3e}, m={:.3e})",
            pos.x,
            pos.y,
            pos.z,
            self.total_energy(),
            self.mass()
        )
    }
}

/// WASM wrapper for Schwarzschild metric
#[wasm_bindgen]
pub struct WasmSchwarzschildMetric {
    inner: SchwarzschildMetric,
}

#[wasm_bindgen]
impl WasmSchwarzschildMetric {
    /// Create Schwarzschild metric for the Sun
    #[wasm_bindgen]
    pub fn sun() -> Self {
        Self {
            inner: SchwarzschildMetric::sun(),
        }
    }

    /// Create Schwarzschild metric for Earth
    #[wasm_bindgen]
    pub fn earth() -> Self {
        Self {
            inner: SchwarzschildMetric::new(EARTH_MASS, Vector3::zeros()),
        }
    }

    /// Create Schwarzschild metric for custom mass
    #[wasm_bindgen]
    pub fn from_mass(mass: f64) -> Self {
        Self {
            inner: SchwarzschildMetric::new(mass, Vector3::zeros()),
        }
    }

    /// Get Schwarzschild radius
    #[wasm_bindgen]
    pub fn schwarzschild_radius(&self) -> f64 {
        self.inner.schwarzschild_radius
    }

    /// Get central mass
    #[wasm_bindgen]
    pub fn mass(&self) -> f64 {
        self.inner.mass
    }

    /// Check for singularity at given position
    #[wasm_bindgen]
    pub fn has_singularity(&self, position: &WasmSpacetimeVector) -> bool {
        <SchwarzschildMetric as Metric>::has_singularity(&self.inner, &position.inner)
    }

    /// Compute effective potential for circular orbits
    #[wasm_bindgen]
    pub fn effective_potential(&self, r: f64, angular_momentum: f64) -> f64 {
        amari_relativistic::schwarzschild::effective_potential(r, angular_momentum, self.inner.mass)
    }
}

/// Trajectory point for particle propagation
#[wasm_bindgen]
pub struct WasmTrajectoryPoint {
    /// Time coordinate
    pub time: f64,
    /// Position vector
    position: WasmSpacetimeVector,
}

#[wasm_bindgen]
impl WasmTrajectoryPoint {
    /// Get position
    #[wasm_bindgen(getter)]
    pub fn position(&self) -> WasmSpacetimeVector {
        WasmSpacetimeVector {
            inner: self.position.inner.clone(),
        }
    }
}

/// WASM wrapper for geodesic integration
#[wasm_bindgen]
pub struct WasmGeodesicIntegrator {
    inner: GeodesicIntegrator,
}

#[wasm_bindgen]
impl WasmGeodesicIntegrator {
    /// Create integrator with Schwarzschild metric
    #[wasm_bindgen]
    pub fn with_schwarzschild(metric: &WasmSchwarzschildMetric) -> Self {
        let metric_box = Box::new(metric.inner.clone());
        Self {
            inner: GeodesicIntegrator::with_metric(metric_box),
        }
    }

    /// Propagate particle through spacetime
    #[wasm_bindgen]
    pub fn propagate_particle(
        &mut self,
        particle: &mut WasmRelativisticParticle,
        integration_time: f64,
        time_step: f64,
    ) -> Result<Array, JsValue> {
        match amari_relativistic::particle::propagate_relativistic(
            &mut particle.inner,
            &mut self.inner,
            integration_time,
            time_step,
        ) {
            Ok(trajectory) => {
                let js_array = Array::new();
                for (time, position, _velocity) in trajectory {
                    let spacetime_pos = SpacetimeVector::from_position_and_time(position, time);
                    let point = WasmTrajectoryPoint {
                        time,
                        position: WasmSpacetimeVector {
                            inner: spacetime_pos,
                        },
                    };
                    let js_point = js_sys::Object::new();
                    js_sys::Reflect::set(&js_point, &"time".into(), &JsValue::from(point.time))
                        .unwrap();
                    js_sys::Reflect::set(
                        &js_point,
                        &"position".into(),
                        &JsValue::from(point.position),
                    )
                    .unwrap();
                    js_array.push(&js_point);
                }
                Ok(js_array)
            }
            Err(e) => Err(JsValue::from_str(&format!("Propagation failed: {}", e))),
        }
    }
}

/// Physical constants
#[wasm_bindgen]
pub struct WasmRelativisticConstants;

#[wasm_bindgen]
impl WasmRelativisticConstants {
    /// Speed of light in vacuum (m/s)
    #[wasm_bindgen(getter)]
    pub fn speed_of_light() -> f64 {
        C
    }

    /// Gravitational constant (m³/kg·s²)
    #[wasm_bindgen(getter)]
    pub fn gravitational_constant() -> f64 {
        G
    }

    /// Solar mass (kg)
    #[wasm_bindgen(getter)]
    pub fn solar_mass() -> f64 {
        SOLAR_MASS
    }

    /// Earth mass (kg)
    #[wasm_bindgen(getter)]
    pub fn earth_mass() -> f64 {
        EARTH_MASS
    }
}

/// Calculate light deflection angle for photon grazing massive object
#[wasm_bindgen]
pub fn light_deflection_angle(impact_parameter: f64, mass: f64) -> f64 {
    amari_relativistic::particle::light_deflection_angle(impact_parameter, mass)
}

/// Convert velocity to Lorentz factor
#[wasm_bindgen]
pub fn velocity_to_gamma(velocity_magnitude: f64) -> Result<f64, JsValue> {
    if velocity_magnitude >= C {
        return Err(JsValue::from_str(
            "Velocity must be less than speed of light",
        ));
    }

    let beta = velocity_magnitude / C;
    Ok(1.0 / (1.0 - beta * beta).sqrt())
}

/// Convert Lorentz factor to velocity
#[wasm_bindgen]
pub fn gamma_to_velocity(gamma: f64) -> Result<f64, JsValue> {
    if gamma < 1.0 {
        return Err(JsValue::from_str("Lorentz factor must be >= 1"));
    }

    let beta = (1.0 - 1.0 / (gamma * gamma)).sqrt();
    Ok(beta * C)
}

/// Validate that this module loaded correctly
#[wasm_bindgen]
pub fn validate_relativistic_module() -> bool {
    true
}
