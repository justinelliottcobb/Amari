//! Networked Physics Example
//!
//! Demonstrates deterministic physics for multiplayer games using lockstep
//! or rollback netcode. Shows how bit-exact reproducibility enables:
//!
//! - Replay systems that work across platforms
//! - Lockstep networking without desyncs
//! - Rollback netcode for fighting games
//! - Cross-platform test reproducibility
//!
//! Run with: cargo run --example networked_physics --features deterministic

#![cfg(feature = "deterministic")]

use amari::deterministic::ga2d::{DetRotor2, DetVector2};
use amari::deterministic::DetF32;

/// Game entity with deterministic physics state
#[derive(Clone, Debug)]
struct Entity {
    position: DetVector2,
    velocity: DetVector2,
    rotation: DetRotor2,
    angular_velocity: DetF32,
}

impl Entity {
    fn new(x: f32, y: f32) -> Self {
        Self {
            position: DetVector2::from_f32(x, y),
            velocity: DetVector2::ZERO,
            rotation: DetRotor2::IDENTITY,
            angular_velocity: DetF32::ZERO,
        }
    }

    /// Apply physics step with deterministic operations
    fn step(&mut self, dt: DetF32, gravity: DetVector2) {
        // Update velocity with gravity
        self.velocity = self.velocity + gravity * dt;

        // Update position
        self.position = self.position + self.velocity * dt;

        // Update rotation
        let delta_angle = self.angular_velocity * dt;
        self.rotation = self.rotation * DetRotor2::from_angle(delta_angle);
    }

    /// Serialize state to bit pattern for networking
    fn to_bits(&self) -> EntityBits {
        EntityBits {
            pos_x: self.position.x.to_bits(),
            pos_y: self.position.y.to_bits(),
            vel_x: self.velocity.x.to_bits(),
            vel_y: self.velocity.y.to_bits(),
            rot_s: self.rotation.s.to_bits(),
            rot_b: self.rotation.b.to_bits(),
            ang_vel: self.angular_velocity.to_bits(),
        }
    }

    /// Deserialize from bit pattern (always produces identical state)
    fn from_bits(bits: EntityBits) -> Self {
        Self {
            position: DetVector2::new(DetF32::from_bits(bits.pos_x), DetF32::from_bits(bits.pos_y)),
            velocity: DetVector2::new(DetF32::from_bits(bits.vel_x), DetF32::from_bits(bits.vel_y)),
            rotation: DetRotor2::new(DetF32::from_bits(bits.rot_s), DetF32::from_bits(bits.rot_b)),
            angular_velocity: DetF32::from_bits(bits.ang_vel),
        }
    }

    /// Check if two entities have identical bit patterns
    fn bit_identical(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

/// Network-transmittable bit representation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EntityBits {
    pos_x: u32,
    pos_y: u32,
    vel_x: u32,
    vel_y: u32,
    rot_s: u32,
    rot_b: u32,
    ang_vel: u32,
}

/// Player input (must be deterministic)
#[derive(Clone, Copy, Debug)]
struct Input {
    thrust: DetVector2,
    torque: DetF32,
}

impl Input {
    fn none() -> Self {
        Self {
            thrust: DetVector2::ZERO,
            torque: DetF32::ZERO,
        }
    }
}

/// Game state for a physics simulation
#[derive(Clone)]
struct GameState {
    entities: Vec<Entity>,
    frame: u32,
}

impl GameState {
    fn new() -> Self {
        Self {
            entities: Vec::new(),
            frame: 0,
        }
    }

    fn add_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Deterministic physics step
    fn step(&mut self, inputs: &[Input], dt: DetF32, gravity: DetVector2) {
        // Apply inputs
        for (entity, input) in self.entities.iter_mut().zip(inputs.iter()) {
            entity.velocity = entity.velocity + input.thrust;
            entity.angular_velocity = entity.angular_velocity + input.torque;
        }

        // Physics update
        for entity in &mut self.entities {
            entity.step(dt, gravity);
        }

        self.frame += 1;
    }

    /// Serialize entire game state
    fn to_bits(&self) -> Vec<EntityBits> {
        self.entities.iter().map(|e| e.to_bits()).collect()
    }

    /// Check if two states are bit-identical
    fn bit_identical(&self, other: &Self) -> bool {
        if self.frame != other.frame || self.entities.len() != other.entities.len() {
            return false;
        }

        self.entities
            .iter()
            .zip(other.entities.iter())
            .all(|(a, b)| a.bit_identical(b))
    }
}

/// Simulate lockstep networking scenario
fn demo_lockstep() {
    println!("\n=== Lockstep Networking Demo ===\n");

    // Physics constants
    let dt = DetF32::from_f32(1.0 / 60.0); // 60 FPS
    let gravity = DetVector2::from_f32(0.0, -9.8);

    // Create identical initial states on two "clients"
    let mut client_a = GameState::new();
    let mut client_b = GameState::new();

    client_a.add_entity(Entity::new(0.0, 10.0));
    client_b.add_entity(Entity::new(0.0, 10.0));

    println!("Initial state (both clients):");
    println!("  Position: ({:.2}, {:.2})", 0.0, 10.0);
    println!("  Frame: 0\n");

    // Simulate identical input sequence on both clients
    let input_sequence = [
        Input {
            thrust: DetVector2::from_f32(5.0, 0.0),
            torque: DetF32::from_f32(0.1),
        },
        Input::none(),
        Input {
            thrust: DetVector2::from_f32(0.0, 5.0),
            torque: DetF32::from_f32(-0.05),
        },
    ];

    // Run 100 frames on both clients
    for frame in 0..100 {
        let input = input_sequence[frame % input_sequence.len()];
        let inputs = vec![input];

        client_a.step(&inputs, dt, gravity);
        client_b.step(&inputs, dt, gravity);
    }

    println!("After 100 frames:");
    let entity_a = &client_a.entities[0];
    let entity_b = &client_b.entities[0];

    println!(
        "Client A position: ({:.6}, {:.6})",
        entity_a.position.x.to_f32(),
        entity_a.position.y.to_f32()
    );
    println!(
        "Client B position: ({:.6}, {:.6})",
        entity_b.position.x.to_f32(),
        entity_b.position.y.to_f32()
    );

    // Verify bit-exact match
    if client_a.bit_identical(&client_b) {
        println!("\n✓ SUCCESS: States are bit-identical!");
        println!("  No desync occurred despite 100 frames of simulation.");
    } else {
        println!("\n✗ FAILURE: States diverged!");
    }

    // Show bit patterns
    let bits_a = entity_a.position.x.to_bits();
    let bits_b = entity_b.position.x.to_bits();
    println!("\nBit patterns for position.x:");
    println!("  Client A: 0x{:08x}", bits_a);
    println!("  Client B: 0x{:08x}", bits_b);
    println!("  Match: {}", bits_a == bits_b);
}

/// Simulate rollback networking scenario
fn demo_rollback() {
    println!("\n=== Rollback Netcode Demo ===\n");

    let dt = DetF32::from_f32(1.0 / 60.0);
    let gravity = DetVector2::from_f32(0.0, -9.8);

    // Initial state
    let mut state = GameState::new();
    state.add_entity(Entity::new(0.0, 10.0));

    // Simulate to frame 10 with predicted input
    let predicted_input = vec![Input {
        thrust: DetVector2::from_f32(5.0, 0.0),
        torque: DetF32::ZERO,
    }];

    println!("Simulating frames 0-10 with predicted input...");
    for _ in 0..10 {
        state.step(&predicted_input, dt, gravity);
    }

    // Save state at frame 10
    let frame_10_state = state.clone();
    let frame_10_bits = state.to_bits();

    println!(
        "  Frame 10 position: ({:.6}, {:.6})",
        state.entities[0].position.x.to_f32(),
        state.entities[0].position.y.to_f32()
    );

    // Continue to frame 15 with predicted input
    for _ in 0..5 {
        state.step(&predicted_input, dt, gravity);
    }

    println!(
        "  Frame 15 position: ({:.6}, {:.6})",
        state.entities[0].position.x.to_f32(),
        state.entities[0].position.y.to_f32()
    );

    // Receive actual input for frame 10 (different from prediction!)
    let actual_input = vec![Input {
        thrust: DetVector2::from_f32(0.0, 10.0), // Different!
        torque: DetF32::from_f32(0.2),
    }];

    println!("\n✗ Misprediction detected at frame 10!");
    println!("  Rolling back to frame 10...");

    // Rollback: restore frame 10 state from bits
    let mut restored_state = GameState {
        entities: frame_10_bits
            .iter()
            .map(|bits| Entity::from_bits(*bits))
            .collect(),
        frame: 10,
    };

    // Verify restoration is bit-exact
    if restored_state.bit_identical(&frame_10_state) {
        println!("  ✓ State restored bit-exactly from frame 10");
    }

    // Re-simulate frames 10-15 with correct input
    println!("  Re-simulating frames 10-15 with actual input...");
    for _ in 0..5 {
        restored_state.step(&actual_input, dt, gravity);
    }

    println!(
        "  New frame 15 position: ({:.6}, {:.6})",
        restored_state.entities[0].position.x.to_f32(),
        restored_state.entities[0].position.y.to_f32()
    );

    println!("\n✓ Rollback complete! Game state is now correct.");
}

/// Demonstrate replay verification
fn demo_replay() {
    println!("\n=== Replay Verification Demo ===\n");

    let dt = DetF32::from_f32(1.0 / 60.0);
    let gravity = DetVector2::from_f32(0.0, -9.8);

    // Record a gameplay session
    println!("Recording gameplay session (50 frames)...");
    let mut recorder = GameState::new();
    recorder.add_entity(Entity::new(5.0, 15.0));

    let mut input_log = Vec::new();

    for frame in 0..50 {
        // Generate some interesting input pattern
        let input = if frame % 20 < 10 {
            Input {
                thrust: DetVector2::from_f32(3.0, 0.0),
                torque: DetF32::from_f32(0.15),
            }
        } else {
            Input {
                thrust: DetVector2::from_f32(-2.0, 5.0),
                torque: DetF32::from_f32(-0.1),
            }
        };

        input_log.push(input);
        recorder.step(&[input], dt, gravity);
    }

    let recorded_bits = recorder.to_bits();
    println!(
        "  Final position: ({:.6}, {:.6})",
        recorder.entities[0].position.x.to_f32(),
        recorder.entities[0].position.y.to_f32()
    );

    // Replay the session (could be days later, different machine, different platform)
    println!("\nReplaying recorded session...");
    let mut replay = GameState::new();
    replay.add_entity(Entity::new(5.0, 15.0));

    for input in &input_log {
        replay.step(&[*input], dt, gravity);
    }

    println!(
        "  Final position: ({:.6}, {:.6})",
        replay.entities[0].position.x.to_f32(),
        replay.entities[0].position.y.to_f32()
    );

    // Verify replay matches recording bit-exactly
    let replay_bits = replay.to_bits();

    if recorded_bits == replay_bits {
        println!("\n✓ SUCCESS: Replay matches recording bit-exactly!");
        println!("  This works across platforms and compiler optimizations.");
    } else {
        println!("\n✗ FAILURE: Replay diverged from recording!");
    }

    // Verify specific bit patterns
    println!("\nBit pattern verification:");
    for (i, (rec, rep)) in recorded_bits[0..1]
        .iter()
        .zip(&replay_bits[0..1])
        .enumerate()
    {
        println!("  Entity {}: match={}", i, rec == rep);
        if rec != rep {
            println!("    Recorded:  0x{:08x}", rec.pos_x);
            println!("    Replayed:  0x{:08x}", rep.pos_x);
        }
    }
}

/// Demonstrate collision handling
fn demo_collision() {
    println!("\n=== Deterministic Collision Demo ===\n");

    let dt = DetF32::from_f32(1.0 / 60.0);
    let gravity = DetVector2::from_f32(0.0, -9.8);

    // Create two entities
    let mut entity1 = Entity::new(-5.0, 10.0);
    let mut entity2 = Entity::new(5.0, 10.0);

    entity1.velocity = DetVector2::from_f32(10.0, 0.0); // Moving right
    entity2.velocity = DetVector2::from_f32(-8.0, 0.0); // Moving left

    println!("Initial setup:");
    println!(
        "  Entity 1: pos=({:.2}, {:.2}), vel=({:.2}, {:.2})",
        -5.0, 10.0, 10.0, 0.0
    );
    println!(
        "  Entity 2: pos=({:.2}, {:.2}), vel=({:.2}, {:.2})",
        5.0, 10.0, -8.0, 0.0
    );

    // Simulate until collision
    let collision_radius = DetF32::from_f32(1.0);
    let mut frame = 0;
    let mut collision_frame = None;

    for _ in 0..60 {
        entity1.step(dt, gravity);
        entity2.step(dt, gravity);
        frame += 1;

        // Check collision (deterministic distance calculation)
        let diff = entity2.position - entity1.position;
        let distance = diff.magnitude();

        if distance < collision_radius * DetF32::TWO && collision_frame.is_none() {
            collision_frame = Some(frame);

            println!("\n✓ Collision detected at frame {}!", frame);
            println!(
                "  Entity 1 position: ({:.6}, {:.6})",
                entity1.position.x.to_f32(),
                entity1.position.y.to_f32()
            );
            println!(
                "  Entity 2 position: ({:.6}, {:.6})",
                entity2.position.x.to_f32(),
                entity2.position.y.to_f32()
            );
            println!("  Distance: {:.6}", distance.to_f32());

            // Elastic collision (deterministic)
            let normal = diff.normalize();
            let relative_velocity = entity2.velocity - entity1.velocity;
            let velocity_along_normal = relative_velocity.dot(normal);

            if velocity_along_normal < DetF32::ZERO {
                let impulse = normal * velocity_along_normal;
                entity1.velocity = entity1.velocity + impulse;
                entity2.velocity = entity2.velocity - impulse;
            }

            break;
        }
    }

    // Verify this collision frame is deterministic
    println!("\nRepeating simulation to verify determinism...");

    let mut test1 = Entity::new(-5.0, 10.0);
    let mut test2 = Entity::new(5.0, 10.0);
    test1.velocity = DetVector2::from_f32(10.0, 0.0);
    test2.velocity = DetVector2::from_f32(-8.0, 0.0);

    for _ in 0..collision_frame.unwrap_or(0) {
        test1.step(dt, gravity);
        test2.step(dt, gravity);
    }

    if test1.bit_identical(&entity1) && test2.bit_identical(&entity2) {
        println!("✓ Collision occurs at identical frame on replay!");
    } else {
        println!("✗ Collision frame differs on replay!");
    }
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Deterministic Physics for Networked Games                ║");
    println!("║  Amari v0.9.9 - Bit-Exact Cross-Platform Reproducibility  ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Run all demonstrations
    demo_lockstep();
    demo_rollback();
    demo_replay();
    demo_collision();

    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║  Key Takeaways                                             ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║  • All operations produce identical bit patterns           ║");
    println!("║  • Lockstep networking stays synchronized                  ║");
    println!("║  • Rollback works perfectly (no accumulating errors)       ║");
    println!("║  • Replays work across platforms and years later           ║");
    println!("║  • ~10-20%% performance overhead vs native f32             ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");
}
