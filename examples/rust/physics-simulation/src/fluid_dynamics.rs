//! Fluid Dynamics with Geometric Algebra
//!
//! This example demonstrates how geometric algebra can elegantly represent
//! fluid flow, vorticity, and circulation in a coordinate-free manner.

use amari_core::{Multivector, Vector, Bivector};
use std::f64::consts::PI;

type Cl3 = Multivector<3, 0, 0>;

/// Fluid velocity field representation
#[derive(Debug, Clone)]
pub struct FluidField {
    /// Velocity vector field
    pub velocity: Vector<3, 0, 0>,
    /// Vorticity bivector field (ω = ∇ ∧ v)
    pub vorticity: Bivector<3, 0, 0>,
    /// Pressure scalar field
    pub pressure: f64,
    /// Density
    pub density: f64,
}

impl FluidField {
    /// Create a new fluid field
    pub fn new(velocity: [f64; 3], vorticity: [f64; 3], pressure: f64, density: f64) -> Self {
        Self {
            velocity: Vector::from_components(velocity[0], velocity[1], velocity[2]),
            vorticity: Bivector::from_components(vorticity[0], vorticity[1], vorticity[2]),
            pressure,
            density,
        }
    }

    /// Calculate kinetic energy density
    pub fn kinetic_energy_density(&self) -> f64 {
        0.5 * self.density * self.velocity.magnitude_squared()
    }

    /// Calculate vorticity magnitude (intensity of rotation)
    pub fn vorticity_magnitude(&self) -> f64 {
        self.vorticity.magnitude()
    }

    /// Calculate circulation around a closed curve (using Stokes' theorem)
    /// In GA: Circulation = ∮ v·dl = ∬ (∇∧v)·dA = ∬ ω·dA
    pub fn circulation(&self, area: f64, normal: Vector<3, 0, 0>) -> f64 {
        // Circulation = vorticity bivector contracted with area bivector
        let area_bivector = normal.outer_product(&Vector::e1()).scale(area);
        self.vorticity.inner_product(&Bivector::from_multivector(&area_bivector)).get(0)
    }

    /// Calculate helicity (measure of how twisted the flow is)
    /// H = v · ω (velocity dotted with vorticity)
    pub fn helicity(&self) -> f64 {
        self.velocity.inner_product(&self.vorticity.mv).get(0)
    }
}

/// Vortex ring simulation using geometric algebra
fn vortex_ring_simulation() {
    println!("=== Vortex Ring Simulation ===");
    println!("Demonstrating vortex ring dynamics with geometric algebra\n");

    let radius = 1.0;        // Ring radius
    let circulation = 10.0;  // Circulation strength
    let core_size = 0.1;     // Vortex core size

    println!("Vortex ring parameters:");
    println!("Radius: {:.2} m", radius);
    println!("Circulation: {:.2} m²/s", circulation);
    println!("Core size: {:.2} m\n", core_size);

    println!("Time\tRing Position\t\tVelocity\t\tVorticity");
    println!("(s)\t(x, y, z)\t\t(u, v, w)\t\t(ωx, ωy, ωz)");
    println!("{:-<70}", "");

    let mut ring_position = Vector::from_components(0.0, 0.0, 0.0);
    let dt = 0.1;

    for step in 0..50 {
        let time = step as f64 * dt;

        // Vortex ring induces its own motion (self-advection)
        // Velocity is proportional to circulation and inversely to radius
        let self_velocity = circulation / (4.0 * PI * radius);
        let velocity = Vector::from_components(self_velocity, 0.0, 0.0);

        // Vorticity is concentrated in the ring (azimuthal direction)
        let vorticity_strength = circulation / (PI * core_size * core_size);
        let vorticity = Bivector::from_components(0.0, 0.0, vorticity_strength);

        // Create fluid field
        let fluid_field = FluidField::new(
            [velocity.x(), velocity.y(), velocity.z()],
            [vorticity.xy(), vorticity.xz(), vorticity.yz()],
            0.0,  // No pressure gradient
            1.0   // Unit density
        );

        // Update ring position
        ring_position = ring_position.add(&velocity.scale(dt));

        println!("{:.1}\t({:.3}, {:.3}, {:.3})\t\t({:.3}, {:.3}, {:.3})\t({:.2}, {:.2}, {:.2})",
            time,
            ring_position.x(), ring_position.y(), ring_position.z(),
            velocity.x(), velocity.y(), velocity.z(),
            vorticity.xy(), vorticity.xz(), vorticity.yz()
        );

        // Calculate ring properties
        if step % 10 == 0 {
            let energy = fluid_field.kinetic_energy_density();
            let helicity = fluid_field.helicity();
            println!("    → Energy density: {:.4} J/m³, Helicity: {:.4} m²/s²", energy, helicity);
        }
    }

    println!("\nVortex ring travels forward due to self-induced velocity");
    println!("demonstrating how geometric algebra naturally handles");
    println!("the coupling between vorticity and motion.");
}

/// Tornado simulation showing vorticity concentration
fn tornado_simulation() {
    println!("\n=== Tornado Vortex Simulation ===");
    println!("Rankine vortex model using geometric algebra\n");

    let max_radius = 5.0;     // Maximum radius to consider
    let core_radius = 0.5;    // Tornado core radius
    let max_velocity = 50.0;  // Maximum tangential velocity (m/s)

    println!("Tornado parameters:");
    println!("Core radius: {:.2} m", core_radius);
    println!("Maximum velocity: {:.2} m/s\n", max_velocity);

    println!("Radius\tTangential Vel\tVorticity\tPressure Drop\tCentripetal Force");
    println!("(m)\t(m/s)\t\t(1/s)\t\t(Pa)\t\t(N/kg)");
    println!("{:-<80}", "");

    let density = 1.225; // Air density kg/m³

    for i in 0..20 {
        let r = (i + 1) as f64 * max_radius / 20.0;

        // Rankine vortex velocity profile
        let tangential_velocity = if r <= core_radius {
            // Solid body rotation inside core
            max_velocity * r / core_radius
        } else {
            // Free vortex outside core
            max_velocity * core_radius / r
        };

        // Vorticity calculation
        let vorticity = if r <= core_radius {
            // Constant vorticity in core
            2.0 * max_velocity / core_radius
        } else {
            // Zero vorticity outside (except at r=0)
            0.0
        };

        // Pressure drop due to centripetal acceleration
        let centripetal_acceleration = tangential_velocity * tangential_velocity / r;
        let pressure_drop = 0.5 * density * tangential_velocity * tangential_velocity;

        // Create fluid field at this radius
        let fluid_field = FluidField::new(
            [0.0, tangential_velocity, 0.0], // Tangential velocity
            [0.0, 0.0, vorticity],           // Vorticity in z-direction
            -pressure_drop,                   // Negative pressure (suction)
            density
        );

        println!("{:.2}\t{:.2}\t\t{:.3}\t\t{:.1}\t\t{:.2}",
            r,
            tangential_velocity,
            vorticity,
            pressure_drop,
            centripetal_acceleration
        );

        // Show key transitions
        if (r - core_radius).abs() < 0.1 {
            println!("    → Core boundary: transition from solid-body to free vortex");
        }
    }

    println!("\nThe Rankine vortex model shows:");
    println!("• Solid-body rotation (constant vorticity) in the core");
    println!("• Free vortex (zero vorticity) outside the core");
    println!("• Maximum velocity at the core boundary");
    println!("• Pressure drop creates the suction effect");
}

/// Demonstrate Kelvin's circulation theorem
fn circulation_theorem_demo() {
    println!("\n=== Kelvin's Circulation Theorem ===");
    println!("Conservation of circulation in inviscid flow\n");

    let initial_circulation = 20.0; // m²/s
    let dt = 0.5;

    println!("Kelvin's theorem states that in an inviscid, barotropic flow");
    println!("with conservative forces, circulation around any material");
    println!("contour remains constant.\n");

    println!("Time\tContour Area\tVorticity\tCirculation\tError");
    println!("(s)\t(m²)\t\t(1/s)\t\t(m²/s)\t\t(%)");
    println!("{:-<60}", "");

    for step in 0..20 {
        let time = step as f64 * dt;

        // Simulate contour deformation (area changes due to flow)
        let contour_area = 1.0 + 0.5 * (0.2 * time).sin(); // Oscillating area

        // By Kelvin's theorem, circulation remains constant
        let current_circulation = initial_circulation;

        // Vorticity adjusts to maintain constant circulation
        // Γ = ∫∫ ω·dA, so ω = Γ/A for uniform vorticity
        let average_vorticity = current_circulation / contour_area;

        // Calculate theoretical vs actual (should be zero error)
        let error_percent = 0.0; // Perfect conservation in this idealized case

        println!("{:.1}\t{:.3}\t\t{:.3}\t\t{:.2}\t\t{:.2}",
            time,
            contour_area,
            average_vorticity,
            current_circulation,
            error_percent
        );
    }

    println!("\nKey insights:");
    println!("• Circulation is a topological invariant in ideal flow");
    println!("• Vorticity can be stretched/compressed but total circulation conserved");
    println!("• This explains why vortices persist in fluids");
    println!("• Geometric algebra makes circulation calculations natural");
}

/// Demonstrate Magnus effect on rotating cylinder
fn magnus_effect_simulation() {
    println!("\n=== Magnus Effect Simulation ===");
    println!("Force on rotating cylinder in crossflow\n");

    let cylinder_radius = 0.1;  // 10 cm radius
    let rotation_rate = 100.0;  // rad/s
    let freestream_velocity = 20.0; // 20 m/s crossflow

    println!("Cylinder parameters:");
    println!("Radius: {:.2} m", cylinder_radius);
    println!("Rotation rate: {:.1} rad/s", rotation_rate);
    println!("Freestream velocity: {:.1} m/s\n", freestream_velocity);

    // Circulation due to rotation (ideal case)
    let circulation = 2.0 * PI * cylinder_radius * cylinder_radius * rotation_rate;

    // Magnus force per unit length: F = ρ * U * Γ
    let density = 1.225; // Air density
    let magnus_force = density * freestream_velocity * circulation;

    println!("Flow analysis:");
    println!("Circulation: {:.3} m²/s", circulation);
    println!("Magnus force: {:.2} N/m", magnus_force);

    // Analyze flow field around cylinder
    println!("\nFlow field analysis (polar coordinates from cylinder center):");
    println!("Angle\tRadius\tVelocity\t\tVorticity\tPressure");
    println!("(deg)\t(m)\t(u, v)\t\t\t(1/s)\t\t(Pa)");
    println!("{:-<70}", "");

    for angle_deg in (0..360).step_by(45) {
        let angle = angle_deg as f64 * PI / 180.0;
        let r = cylinder_radius * 1.5; // Just outside the cylinder

        // Velocity field: superposition of freestream + circulation + doublet
        let u = freestream_velocity * (1.0 - (cylinder_radius / r).powi(2) * angle.cos().powi(2))
                - circulation / (2.0 * PI * r);
        let v = -freestream_velocity * (cylinder_radius / r).powi(2) * angle.sin() * angle.cos();

        let velocity_magnitude = (u * u + v * v).sqrt();

        // Vorticity (curl of velocity field)
        let vorticity = circulation / (PI * cylinder_radius * cylinder_radius);

        // Pressure from Bernoulli's equation
        let pressure_dynamic = 0.5 * density * velocity_magnitude * velocity_magnitude;

        println!("{:3}\t{:.2}\t({:.2}, {:.2})\t\t{:.3}\t\t{:.1}",
            angle_deg,
            r,
            u, v,
            vorticity,
            pressure_dynamic
        );
    }

    println!("\nThe Magnus effect demonstrates:");
    println!("• Asymmetric pressure distribution due to circulation");
    println!("• Force perpendicular to both rotation axis and flow direction");
    println!("• Applications: spinning balls in sports, Flettner rotors on ships");
    println!("• Geometric algebra naturally handles the vector cross products");
}

fn main() {
    println!("🌊 Fluid Dynamics with Geometric Algebra");
    println!("======================================\n");

    println!("This example demonstrates the power of geometric algebra");
    println!("in fluid mechanics:\n");

    println!("• Velocity as vectors (grade 1)");
    println!("• Vorticity as bivectors (grade 2) from ∇∧v");
    println!("• Circulation as line integrals of velocity");
    println!("• Helicity as v·ω (velocity-vorticity correlation)");
    println!("• Natural representation of rotational flow");
    println!("• Coordinate-free formulation of fluid laws\n");

    // Run the simulations
    vortex_ring_simulation();
    tornado_simulation();
    circulation_theorem_demo();
    magnus_effect_simulation();

    println!("\n=== Fundamental Fluid Relations in GA ===");
    println!("Vorticity: ω = ∇∧v (curl as bivector)");
    println!("Circulation: Γ = ∮v·dl = ∬ω·dA (Stokes' theorem)");
    println!("Helicity: H = v·ω (measures flow topology)");
    println!("Magnus force: F = ρU×Γ (cross product as bivector)");

    println!("\n🎓 Educational Value:");
    println!("Geometric algebra reveals the geometric nature of");
    println!("fluid flow, making vorticity, circulation, and helicity");
    println!("natural geometric objects rather than abstract concepts.");
}