//! Physical constants for relativistic physics computations
//!
//! This module provides fundamental physical constants in SI units for use in
//! relativistic physics calculations. All values are from CODATA 2018.

/// Gravitational constant in m³/(kg·s²)
///
/// **Value**: 6.67430 × 10⁻¹¹ m³/(kg·s²)
/// **Reference**: CODATA 2018
pub const G: f64 = 6.67430e-11;

/// Speed of light in vacuum in m/s
///
/// **Value**: 299,792,458 m/s (exact by definition)
/// **Reference**: SI unit definition
pub const C: f64 = 299792458.0;

/// Elementary charge in Coulombs
///
/// **Value**: 1.602176634 × 10⁻¹⁹ C (exact by definition)
/// **Reference**: SI unit definition
pub const E_CHARGE: f64 = 1.602176634e-19;

/// Atomic mass unit in kg
///
/// **Value**: 1.66053906660 × 10⁻²⁷ kg
/// **Reference**: CODATA 2018
pub const AMU: f64 = 1.66053906660e-27;

/// Solar mass in kg
///
/// **Value**: 1.98892 × 10³⁰ kg
/// **Reference**: IAU 2015 nominal solar mass parameter
pub const SOLAR_MASS: f64 = 1.98892e30;

/// Solar radius in meters
///
/// **Value**: 6.957 × 10⁸ m
/// **Reference**: IAU 2015 nominal solar radius
pub const SOLAR_RADIUS: f64 = 6.957e8;

/// Earth mass in kg
///
/// **Value**: 5.9722 × 10²⁴ kg
/// **Reference**: CODATA 2018
pub const EARTH_MASS: f64 = 5.9722e24;

/// Earth radius in meters (mean radius)
///
/// **Value**: 6.371 × 10⁶ m
/// **Reference**: IERS Conventions 2010
pub const EARTH_RADIUS: f64 = 6.371e6;

/// Astronomical unit in meters
///
/// **Value**: 1.495978707 × 10¹¹ m (exact by definition)
/// **Reference**: IAU 2012 definition
pub const AU: f64 = 1.495978707e11;

/// Planck constant in J·s
///
/// **Value**: 6.62607015 × 10⁻³⁴ J·s (exact by definition)
/// **Reference**: SI unit definition
pub const H_PLANCK: f64 = 6.62607015e-34;

/// Reduced Planck constant (ℏ = h/2π) in J·s
///
/// **Value**: 1.054571817 × 10⁻³⁴ J·s
pub const H_BAR: f64 = H_PLANCK / (2.0 * core::f64::consts::PI);

/// Boltzmann constant in J/K
///
/// **Value**: 1.380649 × 10⁻²³ J/K (exact by definition)
/// **Reference**: SI unit definition
pub const K_BOLTZMANN: f64 = 1.380649e-23;

/// Electron rest mass in kg
///
/// **Value**: 9.1093837015 × 10⁻³¹ kg
/// **Reference**: CODATA 2018
pub const ELECTRON_MASS: f64 = 9.1093837015e-31;

/// Proton rest mass in kg
///
/// **Value**: 1.67262192369 × 10⁻²⁷ kg
/// **Reference**: CODATA 2018
pub const PROTON_MASS: f64 = 1.67262192369e-27;

/// Neutron rest mass in kg
///
/// **Value**: 1.67492749804 × 10⁻²⁷ kg
/// **Reference**: CODATA 2018
pub const NEUTRON_MASS: f64 = 1.67492749804e-27;

/// Common atomic masses in kg (for convenience)
pub mod atomic_masses {
    use super::AMU;

    /// Hydrogen-1 atomic mass in kg
    pub const HYDROGEN: f64 = 1.007825 * AMU;

    /// Carbon-12 atomic mass in kg (exactly 12 AMU by definition)
    pub const CARBON_12: f64 = 12.0 * AMU;

    /// Iron-56 atomic mass in kg
    pub const IRON_56: f64 = 55.934937 * AMU;

    /// Gold-197 atomic mass in kg
    pub const GOLD_197: f64 = 196.966570 * AMU;
}

/// Energy conversion factors
pub mod energy {
    use super::{C, E_CHARGE};

    /// Electron volt in Joules
    pub const EV: f64 = E_CHARGE;

    /// Kiloelectron volt in Joules
    pub const KEV: f64 = 1e3 * EV;

    /// Megaelectron volt in Joules
    pub const MEV: f64 = 1e6 * EV;

    /// Gigaelectron volt in Joules
    pub const GEV: f64 = 1e9 * EV;

    /// Rest mass energy of electron in Joules
    pub const ELECTRON_REST_ENERGY: f64 = super::ELECTRON_MASS * C * C;

    /// Rest mass energy of proton in Joules
    pub const PROTON_REST_ENERGY: f64 = super::PROTON_MASS * C * C;
}

/// Time conversion factors
pub mod time {
    /// Seconds per minute
    pub const MINUTE: f64 = 60.0;

    /// Seconds per hour
    pub const HOUR: f64 = 60.0 * MINUTE;

    /// Seconds per day
    pub const DAY: f64 = 24.0 * HOUR;

    /// Seconds per year (Julian year)
    pub const YEAR: f64 = 365.25 * DAY;
}

/// Distance conversion factors
pub mod distance {
    use super::AU;

    /// Kilometers in meters
    pub const KM: f64 = 1e3;

    /// Light-year in meters
    pub const LIGHT_YEAR: f64 = super::C * super::time::YEAR;

    /// Parsec in meters
    pub const PARSEC: f64 = 648000.0 / core::f64::consts::PI * AU;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_derived_constants() {
        // Test that derived constants are consistent
        assert_relative_eq!(
            H_BAR,
            H_PLANCK / (2.0 * core::f64::consts::PI),
            epsilon = 1e-15
        );

        // Test energy conversions
        assert_relative_eq!(energy::KEV, 1000.0 * energy::EV, epsilon = 1e-15);
        assert_relative_eq!(energy::MEV, 1e6 * energy::EV, epsilon = 1e-15);

        // Test time conversions
        assert_relative_eq!(time::HOUR, 3600.0, epsilon = 1e-15);
        assert_relative_eq!(time::DAY, 86400.0, epsilon = 1e-15);
    }

    #[test]
    fn test_physical_relationships() {
        // Test that c² is reasonable for mass-energy equivalence
        let c_squared = C * C;

        // Electron rest mass energy should be ~0.511 MeV
        let electron_energy_mev = ELECTRON_MASS * c_squared / energy::MEV;
        assert!((electron_energy_mev - 0.511).abs() < 0.001);

        // Proton rest mass energy should be ~938 MeV
        let proton_energy_mev = PROTON_MASS * c_squared / energy::MEV;
        assert!((proton_energy_mev - 938.3).abs() < 1.0);
    }
}
