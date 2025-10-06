//! Rigid Body Dynamics with Geometric Algebra
//!
//! This example demonstrates how geometric algebra naturally represents
//! 3D rotations, angular velocity, and rigid body dynamics without
//! the singularities inherent in Euler angles or quaternions.

use amari_core::{Multivector, Vector, Bivector, rotor::Rotor};
use std::f64::consts::PI;

type Cl3 = Multivector<3, 0, 0>;

/// A rigid body in 3D space using geometric algebra representation
#[derive(Debug, Clone)]
pub struct RigidBody {
    /// Position in 3D space
    pub position: Vector<3, 0, 0>,
    /// Orientation as a rotor (unit multivector)
    pub orientation: Rotor<3, 0, 0>,
    /// Linear velocity
    pub velocity: Vector<3, 0, 0>,
    /// Angular velocity as a bivector
    pub angular_velocity: Bivector<3, 0, 0>,
    /// Mass of the object
    pub mass: f64,
    /// Inertia tensor (simplified as a scalar for this demo)
    pub inertia: f64,
}

impl RigidBody {
    /// Create a new rigid body
    pub fn new(position: [f64; 3], mass: f64, inertia: f64) -> Self {
        Self {
            position: Vector::from_components(position[0], position[1], position[2]),
            orientation: Rotor::identity(),
            velocity: Vector::zero(),
            angular_velocity: Bivector::zero(),
            mass,
            inertia,
        }
    }

    /// Apply a force at a given point relative to the center of mass
    pub fn apply_force_at_point(&mut self, force: Vector<3, 0, 0>, point: Vector<3, 0, 0>, dt: f64) {
        // Linear acceleration: F = ma -> a = F/m
        let linear_acceleration = force.scale(1.0 / self.mass);
        self.velocity = self.velocity.add(&linear_acceleration.scale(dt));

        // Torque from force applied at offset point: τ = r × F
        // In geometric algebra: τ = r ∧ F (outer product gives bivector torque)
        let torque = point.outer_product(&force);

        // Angular acceleration: τ = I·α -> α = τ/I
        let angular_acceleration = torque.scale(1.0 / self.inertia);
        self.angular_velocity = self.angular_velocity.add(&angular_acceleration.scale(dt));
    }

    /// Update the rigid body position and orientation using integration
    pub fn update(&mut self, dt: f64) {
        // Update position: x(t+dt) = x(t) + v·dt
        self.position = self.position.add(&self.velocity.scale(dt));

        // Update orientation using rotor exponential: R(t+dt) = exp(ω·dt/2) * R(t)
        // where ω is the angular velocity bivector
        let rotation_increment = self.angular_velocity.scale(dt / 2.0);
        let rotor_increment = Rotor::from_bivector(&rotation_increment, rotation_increment.magnitude());
        self.orientation = rotor_increment.compose(&self.orientation);
    }

    /// Get the kinetic energy of the rigid body
    pub fn kinetic_energy(&self) -> f64 {
        let translational = 0.5 * self.mass * self.velocity.magnitude_squared();
        let rotational = 0.5 * self.inertia * self.angular_velocity.magnitude_squared();
        translational + rotational
    }

    /// Transform a local point to world coordinates
    pub fn local_to_world(&self, local_point: Vector<3, 0, 0>) -> Vector<3, 0, 0> {
        // Apply rotation then translation: world_point = R * local_point * R† + position
        let rotated = self.orientation.apply(&local_point.mv);
        Vector::from_multivector(&rotated).add(&self.position)
    }
}

/// Simulate a spinning top under gravity
fn spinning_top_simulation() {
    println!("=== Spinning Top Simulation ===");
    println!("Demonstrating precession using geometric algebra\n");

    let mut top = RigidBody::new([0.0, 1.0, 0.0], 1.0, 0.1);

    // Initial angular velocity around the z-axis (spinning)
    top.angular_velocity = Bivector::from_components(0.0, 0.0, 10.0); // Fast spin around vertical

    // Gravity force
    let gravity = Vector::from_components(0.0, -9.81, 0.0);

    // Point where gravity acts (offset from center of mass to cause precession)
    let gravity_point = Vector::from_components(0.0, -0.1, 0.0);

    let dt = 0.01;
    let simulation_time = 2.0;
    let steps = (simulation_time / dt) as usize;

    println!("Time\tPosition\t\tAngular Velocity\t\tEnergy");
    println!("(s)\t(x, y, z)\t\t(ωx, ωy, ωz)\t\t\t(J)");
    println!("{:-<70}", "");

    for step in 0..steps {
        let time = step as f64 * dt;

        // Apply gravitational torque
        let gravitational_force = gravity.scale(top.mass);
        top.apply_force_at_point(gravitational_force, gravity_point, dt);

        // Update dynamics
        top.update(dt);

        // Print status every 0.2 seconds
        if step % 20 == 0 {
            let pos = top.position;
            let ang_vel = top.angular_velocity;
            let energy = top.kinetic_energy();

            println!("{:.2}\t({:.3}, {:.3}, {:.3})\t\t({:.3}, {:.3}, {:.3})\t\t{:.3}",
                time,
                pos.x(), pos.y(), pos.z(),
                ang_vel.xy(), ang_vel.xz(), ang_vel.yz(),
                energy
            );
        }
    }

    println!("\nFinal angular velocity magnitude: {:.3} rad/s",
        top.angular_velocity.magnitude());
    println!("Energy conservation check: {:.6} J", top.kinetic_energy());
}

/// Demonstrate collision between two rigid bodies
fn collision_simulation() {
    println!("\n=== Rigid Body Collision Simulation ===");
    println!("Two spheres colliding elastically\n");

    let mut sphere1 = RigidBody::new([-2.0, 0.0, 0.0], 1.0, 0.4);
    let mut sphere2 = RigidBody::new([2.0, 0.0, 0.0], 1.0, 0.4);

    // Initial velocities (approaching each other)
    sphere1.velocity = Vector::from_components(3.0, 0.0, 0.5);
    sphere2.velocity = Vector::from_components(-2.0, 0.0, -0.3);

    // Initial angular velocities
    sphere1.angular_velocity = Bivector::from_components(0.0, 1.0, 2.0);
    sphere2.angular_velocity = Bivector::from_components(1.0, 0.0, -1.5);

    let dt = 0.01;
    let radius = 0.5; // Sphere radius

    println!("Time\tSphere 1 Position\t\tSphere 2 Position\t\tDistance");
    println!("(s)\t(x, y, z)\t\t\t(x, y, z)\t\t\t(units)");
    println!("{:-<80}", "");

    for step in 0..200 {
        let time = step as f64 * dt;

        // Check for collision
        let separation = sphere2.position.sub(&sphere1.position);
        let distance = separation.magnitude();

        if distance <= 2.0 * radius && distance > 0.0 {
            // Collision detected! Apply elastic collision response
            println!("\n*** COLLISION DETECTED at t = {:.3}s ***", time);

            // Collision normal (unit vector from sphere1 to sphere2)
            let normal = separation.normalize().unwrap();

            // Relative velocity
            let relative_velocity = sphere2.velocity.sub(&sphere1.velocity);

            // Velocity component along normal
            let v_normal = relative_velocity.inner_product(&normal);

            if v_normal.get(0) < 0.0 {  // Objects approaching
                // Elastic collision response (equal masses)
                let impulse_magnitude = 2.0 * v_normal.get(0) / (1.0/sphere1.mass + 1.0/sphere2.mass);
                let impulse = normal.scale(impulse_magnitude);

                sphere1.velocity = sphere1.velocity.add(&impulse.scale(1.0/sphere1.mass));
                sphere2.velocity = sphere2.velocity.sub(&impulse.scale(1.0/sphere2.mass));

                println!("Applied impulse: ({:.3}, {:.3}, {:.3})",
                    impulse.x(), impulse.y(), impulse.z());
            }
        }

        // Update both spheres
        sphere1.update(dt);
        sphere2.update(dt);

        // Print status every 0.1 seconds
        if step % 10 == 0 {
            let pos1 = sphere1.position;
            let pos2 = sphere2.position;

            println!("{:.2}\t({:.3}, {:.3}, {:.3})\t\t({:.3}, {:.3}, {:.3})\t\t{:.3}",
                time,
                pos1.x(), pos1.y(), pos1.z(),
                pos2.x(), pos2.y(), pos2.z(),
                distance
            );
        }
    }

    // Final energy analysis
    let final_energy = sphere1.kinetic_energy() + sphere2.kinetic_energy();
    println!("\nFinal total kinetic energy: {:.6} J", final_energy);
}

/// Demonstrate gyroscopic effects
fn gyroscope_simulation() {
    println!("\n=== Gyroscope Simulation ===");
    println!("Demonstrating gyroscopic precession and nutation\n");

    let mut gyroscope = RigidBody::new([0.0, 0.0, 0.0], 2.0, 0.05);

    // Fast spin around the x-axis (gyroscope axis)
    gyroscope.angular_velocity = Bivector::from_components(100.0, 0.0, 0.0);

    // Apply small external torque around z-axis
    let external_torque = Bivector::from_components(0.0, 0.0, 0.5);

    let dt = 0.001; // Smaller timestep for stability
    let steps = 1000;

    println!("Time\tAngular Velocity (bivector)\t\tPrecession Rate");
    println!("(s)\t(ωyz, ωxz, ωxy)\t\t\t(rad/s)");
    println!("{:-<60}", "");

    let mut previous_orientation = gyroscope.orientation.clone();

    for step in 0..steps {
        let time = step as f64 * dt;

        // Apply external torque
        let angular_acceleration = external_torque.scale(1.0 / gyroscope.inertia);
        gyroscope.angular_velocity = gyroscope.angular_velocity.add(&angular_acceleration.scale(dt));

        // Update orientation
        gyroscope.update(dt);

        // Calculate precession rate (change in orientation)
        let orientation_change = gyroscope.orientation.inverse().compose(&previous_orientation);
        previous_orientation = gyroscope.orientation.clone();

        // Print status every 0.05 seconds
        if step % 50 == 0 {
            let omega = gyroscope.angular_velocity;
            println!("{:.3}\t({:.3}, {:.3}, {:.3})\t\t\t{:.3}",
                time,
                omega.yz(), omega.xz(), omega.xy(),
                omega.magnitude()
            );
        }
    }

    println!("\nGyroscopic effect: High-speed rotation causes the gyroscope");
    println!("to precess around the axis perpendicular to both the spin");
    println!("axis and the applied torque axis.");
}

fn main() {
    println!("PHYSICS Physics Simulation with Geometric Algebra");
    println!("============================================\n");

    println!("This example demonstrates the natural representation of");
    println!("rigid body dynamics using geometric algebra:\n");

    println!("• Rotations as rotors (avoiding gimbal lock)");
    println!("• Angular velocity as bivectors");
    println!("• Torque as bivectors from r ∧ F");
    println!("• Smooth integration without singularities");
    println!("• Natural gyroscopic effects\n");

    // Run the simulations
    spinning_top_simulation();
    collision_simulation();
    gyroscope_simulation();

    println!("\n=== Mathematical Insights ===");
    println!("Geometric algebra provides:");
    println!("1. Unified treatment of rotations and translations");
    println!("2. No gimbal lock or quaternion normalization issues");
    println!("3. Direct geometric interpretation of angular quantities");
    println!("4. Elegant collision detection and response");
    println!("5. Natural representation of gyroscopic effects");

    println!("\n🎓 Educational Value:");
    println!("This demonstrates why geometric algebra is increasingly");
    println!("used in physics engines, robotics, and computer graphics.");
}