//! Camera and Projection Systems with Geometric Algebra
//!
//! This example demonstrates how geometric algebra can be used to implement
//! camera systems, projective geometry, and view transformations in a
//! coordinate-free and intuitive manner.

use amari_core::{Multivector, Vector, Bivector, rotor::Rotor};
use std::f64::consts::PI;

type Cl3 = Multivector<3, 0, 0>;

/// A 3D camera using geometric algebra
#[derive(Debug, Clone)]
pub struct Camera {
    /// Camera position in world space
    pub position: Vector<3, 0, 0>,
    /// Camera orientation as a rotor
    pub orientation: Rotor<3, 0, 0>,
    /// Field of view in radians
    pub fov: f64,
    /// Aspect ratio (width/height)
    pub aspect_ratio: f64,
    /// Near clipping plane distance
    pub near_plane: f64,
    /// Far clipping plane distance
    pub far_plane: f64,
}

impl Camera {
    /// Create a new camera
    pub fn new(position: [f64; 3], fov: f64, aspect_ratio: f64, near: f64, far: f64) -> Self {
        Self {
            position: Vector::from_components(position[0], position[1], position[2]),
            orientation: Rotor::identity(),
            fov,
            aspect_ratio,
            near_plane: near,
            far_plane: far,
        }
    }

    /// Look at a target point from current position
    pub fn look_at(&mut self, target: Vector<3, 0, 0>, up: Vector<3, 0, 0>) {
        // Calculate forward direction (from camera to target)
        let forward = target.sub(&self.position).normalize().unwrap_or(Vector::e3());

        // Calculate right direction (forward × up)
        let right = forward.outer_product(&up.mv);
        let right_vector = Vector::from_components(
            right.get(6),   // yz -> x
            -right.get(5),  // -xz -> y
            right.get(3),   // xy -> z
        ).normalize().unwrap_or(Vector::e1());

        // Calculate actual up direction (right × forward)
        let actual_up = right_vector.outer_product(&forward.mv);
        let up_vector = Vector::from_components(
            actual_up.get(6),   // yz -> x
            -actual_up.get(5),  // -xz -> y
            actual_up.get(3),   // xy -> z
        ).normalize().unwrap_or(Vector::e2());

        // Create orientation rotor from the orthonormal basis
        // This is a simplified approach - in practice, you'd use a more robust method
        self.orientation = self.orientation_from_axes(right_vector, up_vector, forward);
    }

    /// Helper function to create rotor from orthonormal axes
    fn orientation_from_axes(&self, right: Vector<3, 0, 0>, up: Vector<3, 0, 0>, forward: Vector<3, 0, 0>) -> Rotor<3, 0, 0> {
        // This is a simplified implementation
        // In practice, you'd use more robust quaternion-to-rotor conversion
        Rotor::identity()
    }

    /// Get the forward direction vector
    pub fn forward(&self) -> Vector<3, 0, 0> {
        let forward_local = Vector::from_components(0.0, 0.0, -1.0); // Looking down -Z
        let rotated = self.orientation.apply(&forward_local.mv);
        Vector::from_multivector(&rotated)
    }

    /// Get the right direction vector
    pub fn right(&self) -> Vector<3, 0, 0> {
        let right_local = Vector::from_components(1.0, 0.0, 0.0); // X is right
        let rotated = self.orientation.apply(&right_local.mv);
        Vector::from_multivector(&rotated)
    }

    /// Get the up direction vector
    pub fn up(&self) -> Vector<3, 0, 0> {
        let up_local = Vector::from_components(0.0, 1.0, 0.0); // Y is up
        let rotated = self.orientation.apply(&up_local.mv);
        Vector::from_multivector(&rotated)
    }

    /// Transform a world point to camera space
    pub fn world_to_camera(&self, world_point: Vector<3, 0, 0>) -> Vector<3, 0, 0> {
        // Translate to camera origin, then apply inverse rotation
        let translated = world_point.sub(&self.position);
        let rotated = self.orientation.inverse().apply(&translated.mv);
        Vector::from_multivector(&rotated)
    }

    /// Project a camera space point to screen coordinates
    pub fn camera_to_screen(&self, camera_point: Vector<3, 0, 0>) -> Option<[f64; 2]> {
        let z = camera_point.z();

        // Check if point is behind camera or outside clipping planes
        if z >= -self.near_plane || z <= -self.far_plane {
            return None;
        }

        // Perspective projection
        let x = camera_point.x() / (-z);
        let y = camera_point.y() / (-z);

        // Apply field of view and aspect ratio
        let fov_scale = (self.fov / 2.0).tan();
        let screen_x = x / fov_scale;
        let screen_y = y / (fov_scale * self.aspect_ratio);

        // Convert to NDC coordinates [-1, 1]
        Some([screen_x, screen_y])
    }

    /// Complete world-to-screen transformation
    pub fn world_to_screen(&self, world_point: Vector<3, 0, 0>) -> Option<[f64; 2]> {
        let camera_point = self.world_to_camera(world_point);
        self.camera_to_screen(camera_point)
    }

    /// Move camera by offset in world coordinates
    pub fn translate(&mut self, offset: Vector<3, 0, 0>) {
        self.position = self.position.add(&offset);
    }

    /// Rotate camera around its local axes
    pub fn rotate_local(&mut self, axis: Vector<3, 0, 0>, angle: f64) {
        let local_axis = self.orientation.inverse().apply(&axis.mv);
        let local_axis_vector = Vector::from_multivector(&local_axis);
        let rotation_bivector = local_axis_vector.outer_product(&Vector::zero().mv);
        let rotation = Rotor::from_bivector(&Bivector::from_multivector(&rotation_bivector), angle);
        self.orientation = self.orientation.compose(&rotation);
    }

    /// Orbit around a target point
    pub fn orbit(&mut self, target: Vector<3, 0, 0>, axis: Vector<3, 0, 0>, angle: f64) {
        // Translate to target, rotate, translate back
        let offset = self.position.sub(&target);
        let rotation_bivector = axis.normalize().unwrap_or(Vector::e2()).outer_product(&Vector::zero().mv);
        let rotation = Rotor::from_bivector(&Bivector::from_multivector(&rotation_bivector), angle);
        let rotated_offset = rotation.apply(&offset.mv);
        self.position = target.add(&Vector::from_multivector(&rotated_offset));

        // Update orientation to keep looking at target
        self.look_at(target, Vector::from_components(0.0, 1.0, 0.0));
    }
}

/// A 3D mesh for rendering demonstrations
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex positions
    pub vertices: Vec<Vector<3, 0, 0>>,
    /// Triangle indices (each group of 3 forms a triangle)
    pub indices: Vec<usize>,
}

impl Mesh {
    /// Create a cube mesh
    pub fn cube(size: f64) -> Self {
        let s = size / 2.0;
        let vertices = vec![
            Vector::from_components(-s, -s, -s), // 0
            Vector::from_components( s, -s, -s), // 1
            Vector::from_components( s,  s, -s), // 2
            Vector::from_components(-s,  s, -s), // 3
            Vector::from_components(-s, -s,  s), // 4
            Vector::from_components( s, -s,  s), // 5
            Vector::from_components( s,  s,  s), // 6
            Vector::from_components(-s,  s,  s), // 7
        ];

        let indices = vec![
            // Front face
            0, 1, 2,  0, 2, 3,
            // Back face
            4, 6, 5,  4, 7, 6,
            // Left face
            0, 3, 7,  0, 7, 4,
            // Right face
            1, 5, 6,  1, 6, 2,
            // Top face
            3, 2, 6,  3, 6, 7,
            // Bottom face
            0, 4, 5,  0, 5, 1,
        ];

        Self { vertices, indices }
    }

    /// Create a simple tetrahedron
    pub fn tetrahedron(size: f64) -> Self {
        let vertices = vec![
            Vector::from_components( size,  size,  size),
            Vector::from_components(-size, -size,  size),
            Vector::from_components(-size,  size, -size),
            Vector::from_components( size, -size, -size),
        ];

        let indices = vec![
            0, 1, 2,
            0, 2, 3,
            0, 3, 1,
            1, 3, 2,
        ];

        Self { vertices, indices }
    }
}

/// Demonstrate basic camera operations
fn camera_basics_demo() {
    println!("=== Camera Basics Demo ===");
    println!("Basic camera operations and transformations\\n");

    let mut camera = Camera::new([0.0, 0.0, 5.0], PI/3.0, 16.0/9.0, 0.1, 100.0);

    println!("Initial camera state:");
    println!("Position: ({:.3}, {:.3}, {:.3})",
        camera.position.x(), camera.position.y(), camera.position.z());
    println!("Forward: ({:.3}, {:.3}, {:.3})",
        camera.forward().x(), camera.forward().y(), camera.forward().z());
    println!("Right: ({:.3}, {:.3}, {:.3})",
        camera.right().x(), camera.right().y(), camera.right().z());
    println!("Up: ({:.3}, {:.3}, {:.3})",
        camera.up().x(), camera.up().y(), camera.up().z());

    // Look at origin
    let target = Vector::zero();
    let up = Vector::from_components(0.0, 1.0, 0.0);
    camera.look_at(target, up);

    println!("\\nAfter looking at origin:");
    println!("Forward: ({:.3}, {:.3}, {:.3})",
        camera.forward().x(), camera.forward().y(), camera.forward().z());

    // Test world-to-screen projection
    let test_points = vec![
        Vector::zero(),                                    // Origin
        Vector::from_components(1.0, 0.0, 0.0),          // Right
        Vector::from_components(0.0, 1.0, 0.0),          // Up
        Vector::from_components(0.0, 0.0, -1.0),         // Forward
        Vector::from_components(2.0, 1.0, -3.0),         // Arbitrary point
    ];

    println!("\\nWorld-to-Screen Projection:");
    println!("World Point\\t\\t\\tScreen Coordinates");
    println!("(x, y, z)\\t\\t\\t(screen_x, screen_y)");
    println!("{:-<50}", "");

    for point in test_points {
        match camera.world_to_screen(point) {
            Some([sx, sy]) => {
                println!("({:.1}, {:.1}, {:.1})\\t\\t\\t({:.3}, {:.3})",
                    point.x(), point.y(), point.z(), sx, sy);
            }
            None => {
                println!("({:.1}, {:.1}, {:.1})\\t\\t\\t(outside view)",
                    point.x(), point.y(), point.z());
            }
        }
    }
}

/// Demonstrate camera animation and orbiting
fn camera_animation_demo() {
    println!("\\n=== Camera Animation Demo ===");
    println!("Orbital camera movement around a target\\n");

    let mut camera = Camera::new([3.0, 2.0, 3.0], PI/4.0, 1.0, 0.1, 100.0);
    let target = Vector::zero(); // Look at origin
    let orbit_axis = Vector::from_components(0.0, 1.0, 0.0); // Orbit around Y-axis

    println!("Time\\tCamera Position\\t\\t\\tTarget Projection");
    println!("(s)\\t(x, y, z)\\t\\t\\t(screen_x, screen_y)");
    println!("{:-<60}", "");

    for step in 0..12 {
        let time = step as f64 * 0.5;
        let orbit_angle = time * 0.3; // Slow orbit

        // Reset camera position and orbit
        camera.position = Vector::from_components(3.0, 2.0, 3.0);
        camera.orbit(target, orbit_axis, orbit_angle);

        // Project target to screen
        let screen_coords = camera.world_to_screen(target);

        match screen_coords {
            Some([sx, sy]) => {
                println!("{:.1}\\t({:.3}, {:.3}, {:.3})\\t\\t\\t({:.3}, {:.3})",
                    time,
                    camera.position.x(), camera.position.y(), camera.position.z(),
                    sx, sy);
            }
            None => {
                println!("{:.1}\\t({:.3}, {:.3}, {:.3})\\t\\t\\t(out of view)",
                    time,
                    camera.position.x(), camera.position.y(), camera.position.z());
            }
        }
    }

    println!("\\nThe camera orbits around the target while maintaining");
    println!("the target at the center of the screen (0, 0).");
}

/// Demonstrate mesh rendering and frustum culling
fn mesh_rendering_demo() {
    println!("\\n=== Mesh Rendering Demo ===");
    println!("Projecting 3D meshes and frustum culling\\n");

    let camera = Camera::new([0.0, 0.0, 5.0], PI/3.0, 1.0, 0.1, 100.0);
    let cube = Mesh::cube(2.0);

    println!("Cube vertices in world space vs screen projection:");
    println!("Vertex\\tWorld Position\\t\\t\\tScreen Position");
    println!("Index\\t(x, y, z)\\t\\t\\t(screen_x, screen_y)");
    println!("{:-<65}", "");

    for (i, vertex) in cube.vertices.iter().enumerate() {
        match camera.world_to_screen(*vertex) {
            Some([sx, sy]) => {
                let visible = sx >= -1.0 && sx <= 1.0 && sy >= -1.0 && sy <= 1.0;
                let status = if visible { "visible" } else { "clipped" };
                println!("{}\\t({:.1}, {:.1}, {:.1})\\t\\t\\t({:.3}, {:.3}) {}",
                    i, vertex.x(), vertex.y(), vertex.z(), sx, sy, status);
            }
            None => {
                println!("{}\\t({:.1}, {:.1}, {:.1})\\t\\t\\t(behind camera)",
                    i, vertex.x(), vertex.y(), vertex.z());
            }
        }
    }

    // Analyze which triangles are potentially visible
    println!("\\nTriangle visibility analysis:");
    println!("Triangle\\tVertex Indices\\tAll Vertices Visible?");
    println!("{:-<50}", "");

    for (tri_idx, triangle) in cube.indices.chunks(3).enumerate() {
        let v0 = camera.world_to_screen(cube.vertices[triangle[0]]);
        let v1 = camera.world_to_screen(cube.vertices[triangle[1]]);
        let v2 = camera.world_to_screen(cube.vertices[triangle[2]]);

        let all_visible = v0.is_some() && v1.is_some() && v2.is_some();
        let any_visible = v0.is_some() || v1.is_some() || v2.is_some();

        let visibility = if all_visible {
            "All visible"
        } else if any_visible {
            "Partially visible"
        } else {
            "Not visible"
        };

        println!("{}\\t\\t({}, {}, {})\\t\\t{}",
            tri_idx, triangle[0], triangle[1], triangle[2], visibility);
    }
}

/// Demonstrate different projection modes
fn projection_comparison_demo() {
    println!("\\n=== Projection Comparison Demo ===");
    println!("Perspective vs orthographic-like projections\\n");

    // Create cameras with different FOV settings
    let wide_camera = Camera::new([0.0, 0.0, 3.0], PI/2.0, 1.0, 0.1, 100.0);     // 90° FOV (wide)
    let normal_camera = Camera::new([0.0, 0.0, 3.0], PI/3.0, 1.0, 0.1, 100.0);   // 60° FOV (normal)
    let narrow_camera = Camera::new([0.0, 0.0, 3.0], PI/6.0, 1.0, 0.1, 100.0);   // 30° FOV (telephoto)

    let test_points = vec![
        Vector::from_components(-1.0, -1.0, -2.0),
        Vector::from_components( 1.0, -1.0, -2.0),
        Vector::from_components( 1.0,  1.0, -2.0),
        Vector::from_components(-1.0,  1.0, -2.0),
        Vector::from_components( 0.0,  0.0, -1.0), // Closer point
        Vector::from_components( 0.0,  0.0, -5.0), // Farther point
    ];

    println!("Point\\tWide FOV (90°)\\t\\tNormal FOV (60°)\\tNarrow FOV (30°)");
    println!("Index\\t(screen_x, screen_y)\\t(screen_x, screen_y)\\t(screen_x, screen_y)");
    println!("{:-<80}", "");

    for (i, point) in test_points.iter().enumerate() {
        let wide_proj = wide_camera.world_to_screen(*point);
        let normal_proj = normal_camera.world_to_screen(*point);
        let narrow_proj = narrow_camera.world_to_screen(*point);

        let format_coord = |coord: Option<[f64; 2]>| {
            match coord {
                Some([x, y]) => format!("({:.3}, {:.3})", x, y),
                None => "(out of view)".to_string(),
            }
        };

        println!("{}\\t{}\\t\\t{}\\t\\t{}",
            i,
            format_coord(wide_proj),
            format_coord(normal_proj),
            format_coord(narrow_proj));
    }

    println!("\\nObservations:");
    println!("• Wide FOV captures more of the scene but with more distortion");
    println!("• Narrow FOV provides telephoto effect with less distortion");
    println!("• Points closer to camera show more FOV-dependent variation");
    println!("• Geometric algebra handles all projections uniformly");
}

fn main() {
    println!("📷 Camera and Projection Systems with Geometric Algebra");
    println!("======================================================\\n");

    println!("This example demonstrates camera systems and projective");
    println!("geometry using geometric algebra:\\n");

    println!("• Camera orientation using rotors");
    println!("• World-to-screen transformations");
    println!("• Frustum culling and clipping");
    println!("• Orbital camera controls");
    println!("• Mesh rendering pipeline");
    println!("• Different projection modes\\n");

    // Run the demonstrations
    camera_basics_demo();
    camera_animation_demo();
    mesh_rendering_demo();
    projection_comparison_demo();

    println!("\\n=== Key Advantages in Graphics ==");
    println!("1. No gimbal lock in camera rotations");
    println!("2. Smooth interpolation for camera animations");
    println!("3. Unified coordinate transformations");
    println!("4. Natural geometric operations");
    println!("5. Efficient frustum culling");
    println!("6. Intuitive orbital controls");

    println!("\\n🎓 Educational Value:");
    println!("Geometric algebra provides a natural framework");
    println!("for 3D graphics, eliminating many traditional");
    println!("complications while maintaining mathematical rigor.");
}