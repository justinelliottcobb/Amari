//! Quantum Mechanics with Geometric Algebra
//!
//! This example demonstrates how geometric algebra provides an elegant
//! framework for quantum mechanics, particularly for spin and the Dirac equation.

use amari_core::{Multivector, Vector, Bivector};
use std::f64::consts::PI;
use std::fmt;

type Cl3 = Multivector<3, 0, 0>;

/// Pauli matrices represented using geometric algebra basis
/// σ₁ = e₂₃, σ₂ = e₃₁, σ₃ = e₁₂ (bivectors in 3D)
#[derive(Debug, Clone)]
pub struct PauliMatrices {
    pub sigma_x: Bivector<3, 0, 0>, // σ₁
    pub sigma_y: Bivector<3, 0, 0>, // σ₂
    pub sigma_z: Bivector<3, 0, 0>, // σ₃
}

impl PauliMatrices {
    /// Create Pauli matrices using geometric algebra bivectors
    pub fn new() -> Self {
        Self {
            sigma_x: Bivector::from_components(0.0, 0.0, 1.0), // e₁₂
            sigma_y: Bivector::from_components(0.0, 1.0, 0.0), // e₁₃
            sigma_z: Bivector::from_components(1.0, 0.0, 0.0), // e₂₃
        }
    }

    /// Get Pauli matrix for a given direction
    pub fn in_direction(&self, direction: Vector<3, 0, 0>) -> Bivector<3, 0, 0> {
        let unit_dir = direction.normalize().unwrap_or(Vector::zero());
        self.sigma_x.scale(unit_dir.x())
            .add(&self.sigma_y.scale(unit_dir.y()))
            .add(&self.sigma_z.scale(unit_dir.z()))
    }
}

/// Quantum spin state represented as a multivector
#[derive(Debug, Clone)]
pub struct SpinState {
    /// Complex amplitude as multivector (scalar + pseudoscalar parts)
    pub state: Cl3,
}

impl SpinState {
    /// Create spin-up state |↑⟩ in z-direction
    pub fn spin_up() -> Self {
        Self {
            state: Cl3::scalar(1.0), // |↑⟩ = (1, 0)ᵀ
        }
    }

    /// Create spin-down state |↓⟩ in z-direction
    pub fn spin_down() -> Self {
        Self {
            state: Cl3::basis_vector(7), // |↓⟩ = (0, 1)ᵀ using pseudoscalar
        }
    }

    /// Create superposition state α|↑⟩ + β|↓⟩
    pub fn superposition(alpha: f64, beta: f64) -> Self {
        let up = Self::spin_up();
        let down = Self::spin_down();
        Self {
            state: up.state.scale(alpha).add(&down.state.scale(beta)),
        }
    }

    /// Calculate expectation value of spin in given direction
    pub fn expectation_value(&self, direction: Vector<3, 0, 0>) -> f64 {
        let pauli = PauliMatrices::new();
        let sigma_n = pauli.in_direction(direction);

        // ⟨ψ|σ·n|ψ⟩ calculation using geometric algebra
        let result = self.state.reverse().geometric_product(&sigma_n.mv).geometric_product(&self.state);
        result.get(0) // Real part
    }

    /// Evolve state under rotation (Rodrigues rotation formula in GA)
    pub fn rotate(&self, axis: Vector<3, 0, 0>, angle: f64) -> Self {
        let bivector = axis.normalize().unwrap_or(Vector::zero()).outer_product(&Cl3::scalar(1.0));
        let rotor = Cl3::scalar((angle / 2.0).cos()).sub(&bivector.scale((angle / 2.0).sin()));

        Self {
            state: rotor.geometric_product(&self.state).geometric_product(&rotor.reverse()),
        }
    }

    /// Calculate probability of measuring spin up in given direction
    pub fn probability_up(&self, direction: Vector<3, 0, 0>) -> f64 {
        let measurement_state = if direction.z().abs() > 0.99 {
            Self::spin_up() // Measuring in z-direction
        } else {
            // General direction measurement state
            let angle = direction.z().acos();
            let axis = Vector::from_components(-direction.y(), direction.x(), 0.0)
                .normalize().unwrap_or(Vector::e1());
            Self::spin_up().rotate(axis, angle)
        };

        let overlap = self.state.reverse().geometric_product(&measurement_state.state);
        overlap.magnitude_squared()
    }
}

impl fmt::Display for SpinState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3} + {:.3}I", self.state.get(0), self.state.get(7))
    }
}

/// Demonstrate spin measurements and the Stern-Gerlach experiment
fn stern_gerlach_simulation() {
    println!("=== Stern-Gerlach Experiment Simulation ===");
    println!("Quantum spin measurements using geometric algebra\n");

    // Prepare initial spin state: 45° to z-axis
    let initial_state = SpinState::superposition(1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt());

    println!("Initial state: |ψ⟩ = {}", initial_state);
    println!("This represents a superposition of spin-up and spin-down\n");

    // Measurement directions
    let directions = [
        ("Z", Vector::from_components(0.0, 0.0, 1.0)),
        ("X", Vector::from_components(1.0, 0.0, 0.0)),
        ("Y", Vector::from_components(0.0, 1.0, 0.0)),
        ("45°XY", Vector::from_components(1.0, 1.0, 0.0).normalize().unwrap()),
        ("45°XZ", Vector::from_components(1.0, 0.0, 1.0).normalize().unwrap()),
    ];

    println!("Direction\tExpectation Value\tP(↑)\tP(↓)");
    println!("\t\t⟨σ·n⟩\t\t\t");
    println!("{:-<50}", "");

    for (name, direction) in &directions {
        let expectation = initial_state.expectation_value(*direction);
        let prob_up = initial_state.probability_up(*direction);
        let prob_down = 1.0 - prob_up;

        println!("{}\t\t{:.3}\t\t\t{:.3}\t{:.3}",
            name, expectation, prob_up, prob_down);
    }

    println!("\nKey observations:");
    println!("• Expectation values range from -1 to +1");
    println!("• Probabilities always sum to 1");
    println!("• Measurement in original preparation direction gives definite result");
    println!("• Orthogonal measurements give 50/50 probability");
}

/// Demonstrate quantum spin precession in magnetic field
fn spin_precession_simulation() {
    println!("\n=== Spin Precession in Magnetic Field ===");
    println!("Larmor precession using geometric algebra\n");

    let initial_state = SpinState::superposition(1.0, 0.0); // Start with spin-up

    // Magnetic field in x-direction
    let b_field = Vector::from_components(1.0, 0.0, 0.0);
    let gyromagnetic_ratio = 2.0; // Simplified
    let larmor_frequency = gyromagnetic_ratio * b_field.magnitude();

    println!("Magnetic field: B = ({:.1}, {:.1}, {:.1}) T",
        b_field.x(), b_field.y(), b_field.z());
    println!("Larmor frequency: ω = {:.3} rad/s\n", larmor_frequency);

    println!("Time\tSpin State\t\t⟨σz⟩\t⟨σx⟩\t⟨σy⟩");
    println!("(s)\t(α + βI)\t\t\t\t");
    println!("{:-<60}", "");

    let dt = PI / 20.0; // Small time steps
    let mut current_state = initial_state;

    for step in 0..21 {
        let time = step as f64 * dt / larmor_frequency;

        // Calculate expectation values in all directions
        let exp_z = current_state.expectation_value(Vector::from_components(0.0, 0.0, 1.0));
        let exp_x = current_state.expectation_value(Vector::from_components(1.0, 0.0, 0.0));
        let exp_y = current_state.expectation_value(Vector::from_components(0.0, 1.0, 0.0));

        println!("{:.2}\t{}\t{:.3}\t{:.3}\t{:.3}",
            time, current_state, exp_z, exp_x, exp_y);

        // Evolve state: rotation around magnetic field direction
        current_state = current_state.rotate(b_field, larmor_frequency * dt);
    }

    println!("\nThe spin precesses around the magnetic field direction");
    println!("with Larmor frequency ω = γB, creating oscillations");
    println!("in the transverse spin components.");
}

/// Demonstrate quantum entanglement with two spins
fn entanglement_demonstration() {
    println!("\n=== Quantum Entanglement Demo ===");
    println!("Bell state correlations using geometric algebra\n");

    // Create Bell state |Φ⁺⟩ = (|↑↓⟩ + |↓↑⟩)/√2
    // This is a simplified representation for demonstration
    println!("Bell state: |Φ⁺⟩ = (|↑↓⟩ + |↓↑⟩)/√2");
    println!("This represents maximum entanglement between two spins\n");

    // Measurement directions for Alice and Bob
    let angles_alice = [0.0, PI/4.0, PI/2.0, 3.0*PI/4.0];
    let angles_bob = [0.0, PI/4.0, PI/2.0, 3.0*PI/4.0];

    println!("Alice Angle\tBob Angle\tCorrelation\tViolation?");
    println!("(degrees)\t(degrees)\t⟨AB⟩\t");
    println!("{:-<50}", "");

    for &alice_angle in &angles_alice {
        for &bob_angle in &angles_bob {
            // Quantum correlation for Bell state: -cos(θ_A - θ_B)
            let correlation = -(alice_angle - bob_angle).cos();

            // Classical bound is |correlation| ≤ 1/√2 ≈ 0.707 for these measurements
            let violates_classical = correlation.abs() > 0.707;

            println!("{:.0}\t\t{:.0}\t\t{:.3}\t\t{}",
                alice_angle * 180.0 / PI,
                bob_angle * 180.0 / PI,
                correlation,
                if violates_classical { "Yes" } else { "No" }
            );
        }
    }

    println!("\nBell's inequality test (CHSH form):");

    // CHSH inequality: |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2 (classical)
    let a = 0.0;           // Alice measures at 0°
    let a_prime = PI/2.0;  // Alice measures at 90°
    let b = PI/4.0;        // Bob measures at 45°
    let b_prime = 3.0*PI/4.0; // Bob measures at 135°

    let e_ab = -(a - b).cos();
    let e_ab_prime = -(a - b_prime).cos();
    let e_a_prime_b = -(a_prime - b).cos();
    let e_a_prime_b_prime = -(a_prime - b_prime).cos();

    let chsh_value = (e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime).abs();

    println!("CHSH value: {:.3}", chsh_value);
    println!("Classical bound: 2.000");
    println!("Quantum bound: {:.3}", 2.0 * 2.0_f64.sqrt());

    if chsh_value > 2.0 {
        println!("✓ Bell's inequality violated! Quantum correlations confirmed.");
    } else {
        println!("○ Bell's inequality satisfied (classical correlations).");
    }
}

/// Demonstrate the geometric phase (Berry phase)
fn geometric_phase_demo() {
    println!("\n=== Geometric Phase (Berry Phase) ===");
    println!("Adiabatic evolution and geometric phases\n");

    let mut state = SpinState::spin_up();

    // Slowly rotate magnetic field in a circle
    let radius = 1.0;
    let num_steps = 20;
    let total_angle = 2.0 * PI; // Full circle

    println!("Slowly rotating magnetic field in xy-plane");
    println!("Initial state: spin up in z-direction\n");

    println!("Step\tB-field Direction\t\tPhase\tGeometric Phase");
    println!("\t(Bx, By, Bz)\t\t\t(rad)\t(rad)");
    println!("{:-<70}", "");

    let mut total_phase = 0.0;

    for step in 0..=num_steps {
        let angle = step as f64 * total_angle / num_steps as f64;

        // Magnetic field direction
        let b_x = radius * angle.cos();
        let b_y = radius * angle.sin();
        let b_z = 0.0;

        let b_field = Vector::from_components(b_x, b_y, b_z);

        // Adiabatic evolution: state follows field direction
        if step > 0 {
            let rotation_angle = total_angle / num_steps as f64;
            let rotation_axis = Vector::from_components(0.0, 0.0, 1.0); // Around z-axis
            state = state.rotate(rotation_axis, rotation_angle);
            total_phase += rotation_angle;
        }

        // Geometric phase is half the solid angle subtended
        let geometric_phase = -0.5 * angle; // For this specific path

        println!("{:2}\t({:.3}, {:.3}, {:.3})\t\t{:.3}\t{:.3}",
            step,
            b_x, b_y, b_z,
            total_phase,
            geometric_phase
        );
    }

    println!("\nKey insights:");
    println!("• Dynamic phase: depends on energy and time");
    println!("• Geometric phase: depends only on path geometry");
    println!("• Berry phase = -½ × solid angle enclosed");
    println!("• Observable interference effects in quantum systems");
}

fn main() {
    println!("QUANTUM  Quantum Mechanics with Geometric Algebra");
    println!("==========================================\n");

    println!("This example demonstrates how geometric algebra provides");
    println!("an elegant framework for quantum mechanics:\n");

    println!("• Pauli matrices as bivectors");
    println!("• Spin states as multivectors");
    println!("• Rotations using rotors (no complex numbers needed)");
    println!("• Natural representation of quantum measurements");
    println!("• Geometric interpretation of quantum phases");
    println!("• Unified treatment of classical and quantum geometry\n");

    // Run the quantum simulations
    stern_gerlach_simulation();
    spin_precession_simulation();
    entanglement_demonstration();
    geometric_phase_demo();

    println!("\n=== Quantum-Geometric Correspondence ===");
    println!("Pauli matrices: σ₁ = e₂₃, σ₂ = e₃₁, σ₃ = e₁₂");
    println!("Spin rotation: |ψ'⟩ = e^(-iθn·σ/2)|ψ⟩ = R|ψ⟩R†");
    println!("where R = cos(θ/2) - (n·σ)sin(θ/2) is a rotor");

    println!("\n🎓 Educational Value:");
    println!("Geometric algebra reveals the geometric structure");
    println!("underlying quantum mechanics, making rotations,");
    println!("measurements, and phases natural geometric operations.");
}