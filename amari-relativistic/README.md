# amari-relativistic

Relativistic physics using geometric algebra for spacetime calculations.

## Overview

`amari-relativistic` implements relativistic physics using the spacetime algebra Cl(1,3), providing tools for special and general relativity calculations. The crate handles Lorentz transformations, geodesic integration, particle dynamics, and gravitational fields with optional high-precision arithmetic for spacecraft-grade calculations.

## Features

- **Spacetime Algebra**: Full Cl(1,3) implementation for Minkowski spacetime
- **Lorentz Transformations**: Boosts and rotations via rotors
- **Geodesic Integration**: Velocity Verlet method for curved spacetime
- **Schwarzschild Metric**: Spherically symmetric gravitational fields
- **Particle Dynamics**: Relativistic particle propagation
- **High-Precision**: Optional extended precision for orbital mechanics
- **Phantom Types**: Compile-time verification of relativistic invariants

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-relativistic = "0.12"
```

### Feature Flags

```toml
[dependencies]
# Default: std + phantom-types + high-precision
amari-relativistic = "0.12"

# With serialization
amari-relativistic = { version = "0.12", features = ["serde-support"] }

# Native high-precision (rug/GMP backend)
amari-relativistic = { version = "0.12", features = ["native-precision"] }

# WASM-compatible high-precision (dashu backend)
amari-relativistic = { version = "0.12", features = ["wasm-precision"] }
```

## Quick Start

### Lorentz Boost

```rust
use amari_relativistic::prelude::*;

// Create a velocity (0.8c in the x-direction)
let velocity = Vector3::new(0.8, 0.0, 0.0);

// Create the corresponding Lorentz boost
let boost = LorentzBoost::from_velocity(&velocity);

// Transform a 4-vector
let event = FourVector::new(1.0, 0.0, 0.0, 0.0); // (t, x, y, z)
let boosted = boost.apply(&event);
```

### Orbital Mechanics

```rust
use amari_relativistic::prelude::*;

// Create gravitational field (Earth)
let earth = schwarzschild::SchwarzschildMetric::earth();
let mut integrator = geodesic::GeodesicIntegrator::with_metric(Box::new(earth));

// Spacecraft at 400 km altitude
let altitude = 400e3;
let earth_radius = 6.371e6;
let position = Vector3::new(earth_radius + altitude, 0.0, 0.0);
let orbital_velocity = Vector3::new(0.0, 7.67e3, 0.0);

// Create particle
let mut spacecraft = particle::RelativisticParticle::new(
    position,
    orbital_velocity,
    0.0,    // charge
    1000.0, // mass (kg)
    0.0,    // charge
)?;

// Propagate orbit
let trajectory = particle::propagate_relativistic(
    &mut spacecraft,
    &mut integrator,
    5580.0, // orbital period (s)
    60.0,   // time step (s)
)?;
```

## Mathematical Background

### Spacetime Algebra Cl(1,3)

The Minkowski metric signature (+,-,-,-):

- γ₀² = +1 (timelike)
- γ₁² = γ₂² = γ₃² = -1 (spacelike)

### Lorentz Transformations

Boosts and rotations are represented as rotors:

```
Λ = exp(-φ/2 · e₀₁)  // Boost in x-direction
R = exp(-θ/2 · e₁₂)  // Rotation in xy-plane
```

Apply transformation via sandwich product:
```
v' = Λ v Λ†
```

### Schwarzschild Metric

For a mass M at the origin:

```
ds² = (1 - rs/r)dt² - (1 - rs/r)⁻¹dr² - r²dΩ²
```

where rs = 2GM/c² is the Schwarzschild radius.

### Geodesic Equation

Particle motion in curved spacetime:

```
d²xᵘ/dτ² + Γᵘᵥρ (dxᵥ/dτ)(dxρ/dτ) = 0
```

## Key Types

### FourVector

Spacetime 4-vectors:

```rust
let position = FourVector::new(t, x, y, z);
let momentum = FourVector::from_3momentum(mass, velocity);
```

### LorentzBoost / LorentzRotation

Lorentz transformations as rotors:

```rust
let boost = LorentzBoost::from_velocity(&v);
let rotation = LorentzRotation::from_axis_angle(&axis, angle);
let combined = boost.compose(&rotation);
```

### SchwarzschildMetric

Gravitational field:

```rust
let earth = SchwarzschildMetric::earth();
let sun = SchwarzschildMetric::new(solar_mass);
let black_hole = SchwarzschildMetric::new(1e6 * solar_mass);
```

### RelativisticParticle

Particles in spacetime:

```rust
let particle = RelativisticParticle::new(
    position, velocity, charge, mass, spin
)?;
```

## Modules

| Module | Description |
|--------|-------------|
| `lorentz` | Lorentz boosts and rotations |
| `geodesic` | Geodesic integration in curved spacetime |
| `schwarzschild` | Schwarzschild metric implementation |
| `particle` | Relativistic particle dynamics |
| `precision` | High-precision arithmetic types |
| `verified` | Phantom type verification |

## Precision Backends

| Backend | Use Case | Dependencies |
|---------|----------|--------------|
| `high-precision` | Default, WASM-compatible | dashu (pure Rust) |
| `native-precision` | Maximum performance | rug (GMP/MPFR) |
| `wasm-precision` | WebAssembly targets | dashu |

## Applications

- **Spacecraft Navigation**: High-precision orbital mechanics
- **GPS Corrections**: Relativistic time dilation calculations
- **Astrophysics**: Black hole and neutron star simulations
- **Particle Physics**: Relativistic collisions and decays
- **Gravitational Waves**: Spacetime perturbations

## Physical Constants

The crate provides standard physical constants:

```rust
use amari_relativistic::constants::*;

let c = SPEED_OF_LIGHT;        // 299792458 m/s
let G = GRAVITATIONAL_CONSTANT; // 6.67430e-11 m³/(kg·s²)
let M_earth = EARTH_MASS;      // 5.972e24 kg
```

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
