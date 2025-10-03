//! Electromagnetic Fields with Geometric Algebra
//!
//! This example demonstrates how geometric algebra naturally unifies
//! electric and magnetic fields into a single electromagnetic field
//! multivector, making Maxwell's equations more elegant and intuitive.

use amari_core::{Multivector, Vector, Bivector};
use std::f64::consts::PI;

type Cl3 = Multivector<3, 0, 0>;

/// Electromagnetic field represented as a multivector in geometric algebra
/// F = E + I·B where E is electric field (vector) and B is magnetic field (bivector)
#[derive(Debug, Clone)]
pub struct ElectromagneticField {
    /// Electric field as a vector (grade 1)
    pub electric_field: Vector<3, 0, 0>,
    /// Magnetic field as a bivector (grade 2)
    pub magnetic_field: Bivector<3, 0, 0>,
}

impl ElectromagneticField {
    /// Create a new electromagnetic field
    pub fn new(e_field: [f64; 3], b_field: [f64; 3]) -> Self {
        Self {
            electric_field: Vector::from_components(e_field[0], e_field[1], e_field[2]),
            magnetic_field: Bivector::from_components(b_field[0], b_field[1], b_field[2]),
        }
    }

    /// Get the complete electromagnetic field multivector F = E + I·B
    pub fn field_multivector(&self) -> Cl3 {
        // In geometric algebra, we represent F = E + I·B where I is the pseudoscalar
        // For 3D Euclidean space, this becomes F = E + B (bivector represents I·B)
        self.electric_field.mv.add(&self.magnetic_field.mv)
    }

    /// Calculate the electromagnetic energy density u = (ε₀E² + B²/μ₀)/2
    pub fn energy_density(&self, epsilon_0: f64, mu_0: f64) -> f64 {
        let e_squared = self.electric_field.magnitude_squared();
        let b_squared = self.magnetic_field.magnitude_squared();
        0.5 * (epsilon_0 * e_squared + b_squared / mu_0)
    }

    /// Calculate the Poynting vector S = (1/μ₀) E × B for energy flow
    pub fn poynting_vector(&self, mu_0: f64) -> Vector<3, 0, 0> {
        // In geometric algebra: S = (1/μ₀) E ∧ B gives the Poynting bivector
        // Then we take the dual to get the vector
        let poynting_bivector = self.electric_field.outer_product(&self.magnetic_field.mv);
        Vector::from_components(
            poynting_bivector.get(6) / mu_0,  // yz component -> x
            -poynting_bivector.get(5) / mu_0, // xz component -> -y
            poynting_bivector.get(3) / mu_0,  // xy component -> z
        )
    }

    /// Apply Lorentz transformation (boost) with velocity v
    pub fn lorentz_transform(&self, velocity: Vector<3, 0, 0>, c: f64) -> Self {
        let v_mag = velocity.magnitude();
        if v_mag >= c {
            panic!("Velocity must be less than speed of light");
        }

        let gamma = 1.0 / (1.0 - (v_mag / c).powi(2)).sqrt();
        let v_unit = velocity.normalize().unwrap_or(Vector::zero());

        // Parallel and perpendicular components of fields
        let e_parallel = self.electric_field.project_onto(&v_unit);
        let e_perp = self.electric_field.sub(&e_parallel);

        let b_parallel = self.magnetic_field.project_onto_vector(&v_unit);
        let b_perp = self.magnetic_field.sub(&b_parallel);

        // Transformed fields (relativistic electromagnetism)
        let e_prime_parallel = e_parallel;
        let e_prime_perp = e_perp.scale(gamma).add(
            &Vector::from_multivector(&velocity.outer_product(&self.magnetic_field.mv)).scale(gamma)
        );

        let b_prime_parallel = b_parallel;
        let b_prime_perp = b_perp.scale(gamma).sub(
            &Bivector::from_multivector(&velocity.outer_product(&self.electric_field.mv).scale(1.0 / (c * c)))
        );

        Self {
            electric_field: e_prime_parallel.add(&e_prime_perp),
            magnetic_field: b_prime_parallel.add(&b_prime_perp),
        }
    }
}

/// Point charge creating electromagnetic field
#[derive(Debug, Clone)]
pub struct PointCharge {
    pub position: Vector<3, 0, 0>,
    pub velocity: Vector<3, 0, 0>,
    pub charge: f64,
    pub mass: f64,
}

impl PointCharge {
    /// Calculate the electromagnetic field at a given point due to this charge
    pub fn field_at_point(&self, point: Vector<3, 0, 0>, time: f64, c: f64, epsilon_0: f64) -> ElectromagneticField {
        let r = point.sub(&self.position);
        let r_mag = r.magnitude();

        if r_mag < 1e-10 {
            return ElectromagneticField::new([0.0; 3], [0.0; 3]);
        }

        let r_unit = r.normalize().unwrap();

        // Coulomb's law for electric field
        let k = 1.0 / (4.0 * PI * epsilon_0);
        let e_magnitude = k * self.charge / (r_mag * r_mag);
        let electric_field = r_unit.scale(e_magnitude);

        // Magnetic field from moving charge (Biot-Savart law)
        let velocity_cross_r = self.velocity.outer_product(&r);
        let b_magnitude = (self.charge / (4.0 * PI * epsilon_0 * c * c)) / (r_mag * r_mag);
        let magnetic_field = Bivector::from_multivector(&velocity_cross_r.scale(b_magnitude));

        ElectromagneticField {
            electric_field,
            magnetic_field,
        }
    }

    /// Update position using Lorentz force
    pub fn update_motion(&mut self, field: &ElectromagneticField, dt: f64, c: f64) {
        // Lorentz force: F = q(E + v × B)
        let v_cross_b = self.velocity.outer_product(&field.magnetic_field.mv);
        let magnetic_force = Vector::from_components(
            v_cross_b.get(6),   // yz -> x
            -v_cross_b.get(5),  // xz -> -y
            v_cross_b.get(3),   // xy -> z
        );

        let total_force = field.electric_field.add(&magnetic_force).scale(self.charge);

        // Update velocity and position
        let acceleration = total_force.scale(1.0 / self.mass);
        self.velocity = self.velocity.add(&acceleration.scale(dt));
        self.position = self.position.add(&self.velocity.scale(dt));

        // Ensure velocity doesn't exceed speed of light
        let v_mag = self.velocity.magnitude();
        if v_mag > 0.95 * c {
            self.velocity = self.velocity.scale(0.95 * c / v_mag);
        }
    }
}

/// Simulate electromagnetic wave propagation
fn electromagnetic_wave_simulation() {
    println!("=== Electromagnetic Wave Simulation ===");
    println!("Demonstrating wave propagation using geometric algebra\n");

    let c = 299792458.0; // Speed of light
    let frequency = 1e9; // 1 GHz
    let wavelength = c / frequency;
    let omega = 2.0 * PI * frequency;

    println!("Wave parameters:");
    println!("Frequency: {:.2e} Hz", frequency);
    println!("Wavelength: {:.3} m", wavelength);
    println!("Angular frequency: {:.2e} rad/s\n", omega);

    println!("Position\tTime\tElectric Field\t\tMagnetic Field\t\tPoynting Vector");
    println!("(m)\t\t(ns)\t(V/m)\t\t\t(T)\t\t\t(W/m²)");
    println!("{:-<100}", "");

    let epsilon_0 = 8.854e-12;
    let mu_0 = 4.0 * PI * 1e-7;

    for i in 0..20 {
        let z = i as f64 * wavelength / 10.0; // Position along propagation
        let t = i as f64 * 1e-9; // Time in nanoseconds

        // Plane wave: E = E₀ cos(kz - ωt) x̂, B = (E₀/c) cos(kz - ωt) ŷ
        let k = 2.0 * PI / wavelength;
        let phase = k * z - omega * t;
        let amplitude = 1.0;

        let e_field = [amplitude * phase.cos(), 0.0, 0.0];
        let b_field = [0.0, amplitude * phase.cos() / c, 0.0];

        let em_field = ElectromagneticField::new(e_field, b_field);
        let poynting = em_field.poynting_vector(mu_0);

        println!("{:.2}\t\t{:.1}\t({:.3}, {:.3}, {:.3})\t({:.2e}, {:.2e}, {:.2e})\t({:.2e}, {:.2e}, {:.2e})",
            z,
            t * 1e9,
            e_field[0], e_field[1], e_field[2],
            b_field[0], b_field[1], b_field[2],
            poynting.x(), poynting.y(), poynting.z()
        );
    }

    println!("\nThe Poynting vector shows energy flow in the +z direction");
    println!("E and B fields are perpendicular and in phase");
    println!("This demonstrates electromagnetic wave propagation");
}

/// Simulate charged particle motion in electromagnetic field
fn charged_particle_simulation() {
    println!("\n=== Charged Particle in EM Field ===");
    println!("Cyclotron motion in uniform magnetic field\n");

    let mut electron = PointCharge {
        position: Vector::from_components(0.0, 0.0, 0.0),
        velocity: Vector::from_components(1e6, 0.0, 0.0), // 1 km/s initial velocity
        charge: -1.602e-19, // Elementary charge
        mass: 9.109e-31,    // Electron mass
    };

    // Uniform magnetic field in z-direction
    let magnetic_field = ElectromagneticField::new(
        [0.0, 0.0, 0.0],        // No electric field
        [0.0, 0.0, 0.01]        // 0.01 Tesla magnetic field
    );

    let c = 299792458.0;
    let dt = 1e-11; // 10 ps timestep
    let steps = 1000;

    // Calculate cyclotron frequency
    let cyclotron_freq = electron.charge.abs() * 0.01 / electron.mass;
    let period = 2.0 * PI / cyclotron_freq;

    println!("Cyclotron frequency: {:.2e} rad/s", cyclotron_freq);
    println!("Cyclotron period: {:.2e} s\n", period);

    println!("Time\t\tPosition\t\t\tVelocity\t\t\tRadius");
    println!("(ns)\t\t(mm)\t\t\t\t(km/s)\t\t\t\t(mm)");
    println!("{:-<80}", "");

    for step in 0..steps {
        let time = step as f64 * dt;

        electron.update_motion(&magnetic_field, dt, c);

        if step % 100 == 0 {
            let pos = electron.position;
            let vel = electron.velocity;
            let radius = pos.magnitude();

            println!("{:.2}\t\t({:.3}, {:.3}, {:.3})\t\t({:.1}, {:.1}, {:.1})\t\t{:.3}",
                time * 1e9,
                pos.x() * 1000.0, pos.y() * 1000.0, pos.z() * 1000.0,
                vel.x() * 1e-3, vel.y() * 1e-3, vel.z() * 1e-3,
                radius * 1000.0
            );
        }
    }

    println!("\nThe electron follows a helical path in the magnetic field");
    println!("demonstrating the natural cyclotron motion");
}

/// Demonstrate field transformation under relativistic motion
fn relativistic_field_transformation() {
    println!("\n=== Relativistic Field Transformation ===");
    println!("How electromagnetic fields transform under Lorentz boosts\n");

    let c = 299792458.0;

    // Initial field in rest frame
    let initial_field = ElectromagneticField::new(
        [1000.0, 0.0, 0.0],     // 1 kV/m electric field in x-direction
        [0.0, 0.001, 0.0]       // 1 mT magnetic field in y-direction
    );

    println!("Initial field in rest frame:");
    println!("E = ({:.1}, {:.1}, {:.1}) V/m",
        initial_field.electric_field.x(),
        initial_field.electric_field.y(),
        initial_field.electric_field.z());
    println!("B = ({:.4}, {:.4}, {:.4}) T\n",
        initial_field.magnetic_field.xy(),
        initial_field.magnetic_field.xz(),
        initial_field.magnetic_field.yz());

    let velocities = [0.1, 0.3, 0.5, 0.7, 0.9];

    println!("Velocity\tTransformed E-field\t\tTransformed B-field");
    println!("(c)\t\t(V/m)\t\t\t\t(T)");
    println!("{:-<70}", "");

    for &v_fraction in &velocities {
        let velocity = Vector::from_components(v_fraction * c, 0.0, 0.0);

        let transformed_field = initial_field.lorentz_transform(velocity, c);

        println!("{:.1}\t\t({:.1}, {:.1}, {:.1})\t\t({:.4}, {:.4}, {:.4})",
            v_fraction,
            transformed_field.electric_field.x(),
            transformed_field.electric_field.y(),
            transformed_field.electric_field.z(),
            transformed_field.magnetic_field.xy(),
            transformed_field.magnetic_field.xz(),
            transformed_field.magnetic_field.yz()
        );
    }

    println!("\nObservations:");
    println!("• Parallel field components unchanged");
    println!("• Perpendicular components increase with γ factor");
    println!("• Electric and magnetic fields mix under boosts");
    println!("• This demonstrates the unified nature of electromagnetism");
}

fn main() {
    println!("⚡ Electromagnetic Fields with Geometric Algebra");
    println!("=============================================\n");

    println!("This example demonstrates the elegant treatment of");
    println!("electromagnetism using geometric algebra:\n");

    println!("• Electric field E as vector (grade 1)");
    println!("• Magnetic field B as bivector (grade 2)");
    println!("• Combined field F = E + I·B as multivector");
    println!("• Maxwell's equations in unified form");
    println!("• Natural Lorentz transformations");
    println!("• Elegant energy and momentum calculations\n");

    // Run the simulations
    electromagnetic_wave_simulation();
    charged_particle_simulation();
    relativistic_field_transformation();

    println!("\n=== Maxwell's Equations in Geometric Algebra ===");
    println!("∇F = J   (combines Gauss's and Ampère's laws)");
    println!("∇∧F = 0  (combines no magnetic monopoles and Faraday's law)");
    println!("\nwhere F = E + I·B is the electromagnetic field multivector");
    println!("and J is the current density multivector.");

    println!("\n🎓 Educational Value:");
    println!("Geometric algebra reveals the deep unity of electricity");
    println!("and magnetism, making relativistic electromagnetism");
    println!("natural and intuitive rather than mysterious.");
}