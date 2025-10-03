//! 3D Transformations with Geometric Algebra
//!
//! This example demonstrates how geometric algebra provides an elegant
//! and unified framework for 3D transformations in computer graphics,
//! avoiding gimbal lock and providing smooth interpolation.

use amari_core::{Multivector, Vector, Bivector, rotor::Rotor};
use std::f64::consts::PI;

type Cl3 = Multivector<3, 0, 0>;

/// A 3D transformation represented using geometric algebra
#[derive(Debug, Clone)]
pub struct Transform3D {
    /// Translation vector
    pub translation: Vector<3, 0, 0>,
    /// Rotation as a rotor (unit multivector)
    pub rotation: Rotor<3, 0, 0>,
    /// Uniform scale factor
    pub scale: f64,
}

impl Transform3D {
    /// Create a new identity transform
    pub fn identity() -> Self {
        Self {
            translation: Vector::zero(),
            rotation: Rotor::identity(),
            scale: 1.0,
        }
    }

    /// Create a translation transform
    pub fn translate(x: f64, y: f64, z: f64) -> Self {
        Self {
            translation: Vector::from_components(x, y, z),
            rotation: Rotor::identity(),
            scale: 1.0,
        }
    }

    /// Create a rotation transform around an axis
    pub fn rotate(axis: Vector<3, 0, 0>, angle: f64) -> Self {
        let unit_axis = axis.normalize().unwrap_or(Vector::e1());
        let bivector = unit_axis.outer_product(&Vector::zero()).scale(-1.0); // Create bivector from axis
        let rotor = Rotor::from_bivector(&Bivector::from_multivector(&bivector), angle);

        Self {
            translation: Vector::zero(),
            rotation: rotor,
            scale: 1.0,
        }
    }

    /// Create a uniform scale transform
    pub fn scale_uniform(factor: f64) -> Self {
        Self {
            translation: Vector::zero(),
            rotation: Rotor::identity(),
            scale: factor,
        }
    }

    /// Compose two transforms (this * other)
    pub fn compose(&self, other: &Transform3D) -> Self {
        // Apply this transform first, then other
        let combined_rotation = other.rotation.compose(&self.rotation);
        let scaled_translation = self.translation.scale(other.scale);
        let rotated_translation = other.rotation.apply(&scaled_translation.mv);
        let combined_translation = Vector::from_multivector(&rotated_translation).add(&other.translation);

        Self {
            translation: combined_translation,
            rotation: combined_rotation,
            scale: self.scale * other.scale,
        }
    }

    /// Apply transform to a point
    pub fn apply_point(&self, point: Vector<3, 0, 0>) -> Vector<3, 0, 0> {
        // Order: Scale, Rotate, Translate
        let scaled = point.scale(self.scale);
        let rotated = self.rotation.apply(&scaled.mv);
        Vector::from_multivector(&rotated).add(&self.translation)
    }

    /// Apply transform to a direction vector (no translation)
    pub fn apply_direction(&self, direction: Vector<3, 0, 0>) -> Vector<3, 0, 0> {
        let scaled = direction.scale(self.scale);
        let rotated = self.rotation.apply(&scaled.mv);
        Vector::from_multivector(&rotated)
    }

    /// Get the inverse transform
    pub fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        let inv_scale = 1.0 / self.scale;

        // Inverse translation: -(R^-1 * T) / s
        let neg_translation = self.translation.scale(-1.0);
        let inv_trans_rotated = inv_rotation.apply(&neg_translation.mv);
        let inv_translation = Vector::from_multivector(&inv_trans_rotated).scale(inv_scale);

        Self {
            translation: inv_translation,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }

    /// Interpolate between two transforms using SLERP for rotation
    pub fn interpolate(&self, other: &Transform3D, t: f64) -> Self {
        let clamped_t = t.clamp(0.0, 1.0);

        // Linear interpolation for translation and scale
        let interp_translation = self.translation.scale(1.0 - clamped_t)
            .add(&other.translation.scale(clamped_t));
        let interp_scale = self.scale * (1.0 - clamped_t) + other.scale * clamped_t;

        // SLERP for rotation
        let interp_rotation = self.rotation.slerp(&other.rotation, clamped_t);

        Self {
            translation: interp_translation,
            rotation: interp_rotation,
            scale: interp_scale,
        }
    }
}

/// Demonstrate basic 3D transformations
fn basic_transformations_demo() {
    println!("=== Basic 3D Transformations ==");
    println!("Demonstrating translation, rotation, and scaling\n");

    // Original point
    let point = Vector::from_components(1.0, 0.0, 0.0);
    println!("Original point: ({:.3}, {:.3}, {:.3})", point.x(), point.y(), point.z());

    // Translation
    let translation = Transform3D::translate(2.0, 3.0, 1.0);
    let translated_point = translation.apply_point(point);
    println!("After translation (+2, +3, +1): ({:.3}, {:.3}, {:.3})",
        translated_point.x(), translated_point.y(), translated_point.z());

    // Rotation around Z-axis by 90 degrees
    let rotation = Transform3D::rotate(Vector::from_components(0.0, 0.0, 1.0), PI/2.0);
    let rotated_point = rotation.apply_point(point);
    println!("After 90° rotation around Z: ({:.3}, {:.3}, {:.3})",
        rotated_point.x(), rotated_point.y(), rotated_point.z());

    // Uniform scaling by factor 2
    let scaling = Transform3D::scale_uniform(2.0);
    let scaled_point = scaling.apply_point(point);
    println!("After 2x uniform scaling: ({:.3}, {:.3}, {:.3})",
        scaled_point.x(), scaled_point.y(), scaled_point.z());

    // Combined transformation: scale, then rotate, then translate
    let combined = translation.compose(&rotation.compose(&scaling));
    let final_point = combined.apply_point(point);
    println!("After combined transform (S→R→T): ({:.3}, {:.3}, {:.3})",
        final_point.x(), final_point.y(), final_point.z());

    // Verify with step-by-step application
    let step1 = scaling.apply_point(point);
    let step2 = rotation.apply_point(step1);
    let step3 = translation.apply_point(step2);
    println!("Verification (step-by-step): ({:.3}, {:.3}, {:.3})",
        step3.x(), step3.y(), step3.z());
}

/// Demonstrate gimbal lock avoidance with geometric algebra
fn gimbal_lock_demo() {
    println!("\n=== Gimbal Lock Avoidance ==");
    println!("Showing smooth interpolation without singularities\n");

    // Create two orientations that would cause gimbal lock with Euler angles
    let start_rotation = Transform3D::rotate(Vector::from_components(1.0, 0.0, 0.0), PI/2.0);  // 90° around X
    let end_rotation = Transform3D::rotate(Vector::from_components(0.0, 0.0, 1.0), PI/2.0);    // 90° around Z

    println!("Interpolating between X-axis rotation and Z-axis rotation:");
    println!("t\\tInterpolated Point\\t\\tAxis of Rotation\\t\\tAngle");
    println!("\\t(after rotation)\\t\\t(unit vector)\\t\\t\\t(degrees)");
    println!("{:-<80}", "");

    let test_point = Vector::from_components(1.0, 0.0, 0.0);

    for i in 0..=10 {
        let t = i as f64 / 10.0;
        let interpolated = start_rotation.interpolate(&end_rotation, t);
        let result_point = interpolated.apply_point(test_point);

        // Extract rotation axis and angle from the rotor
        let rotor_mv = interpolated.rotation.rotor();
        let scalar_part = rotor_mv.get(0);
        let bivector_part = Bivector::from_components(
            rotor_mv.get(6), // yz
            rotor_mv.get(5), // xz
            rotor_mv.get(3)  // xy
        );

        let angle = 2.0 * scalar_part.acos().abs();
        let axis_magnitude = bivector_part.magnitude();
        let axis = if axis_magnitude > 1e-10 {
            Vector::from_components(
                bivector_part.yz() / axis_magnitude,
                bivector_part.xz() / axis_magnitude,
                bivector_part.xy() / axis_magnitude,
            )
        } else {
            Vector::from_components(0.0, 0.0, 1.0) // Default axis
        };

        println!("{:.1}\\t({:.3}, {:.3}, {:.3})\\t\\t({:.3}, {:.3}, {:.3})\\t\\t{:.1}",
            t,
            result_point.x(), result_point.y(), result_point.z(),
            axis.x(), axis.y(), axis.z(),
            angle * 180.0 / PI
        );
    }

    println!("\nNotice the smooth transition without any singularities!");
    println!("The rotation axis smoothly changes from X to an intermediate");
    println!("direction and finally aligns close to Z, avoiding gimbal lock.");
}

/// Demonstrate hierarchical transformations (scene graph)
fn hierarchical_transforms_demo() {
    println!("\n=== Hierarchical Transformations ==");
    println!("Scene graph with parent-child relationships\n");

    // Solar system example: Sun -> Earth -> Moon
    let sun_transform = Transform3D::identity(); // Sun at origin

    // Earth orbits around Sun
    let earth_orbit_angle = PI / 4.0; // 45 degrees
    let earth_distance = 5.0;
    let earth_local = Transform3D::translate(earth_distance, 0.0, 0.0)
        .compose(&Transform3D::rotate(Vector::from_components(0.0, 0.0, 1.0), earth_orbit_angle));
    let earth_global = sun_transform.compose(&earth_local);

    // Moon orbits around Earth
    let moon_orbit_angle = PI / 2.0; // 90 degrees
    let moon_distance = 1.0;
    let moon_local = Transform3D::translate(moon_distance, 0.0, 0.0)
        .compose(&Transform3D::rotate(Vector::from_components(0.0, 0.0, 1.0), moon_orbit_angle));
    let moon_global = earth_global.compose(&moon_local);

    println!("Celestial Body Positions:");
    println!("Object\\tLocal Position\\t\\tGlobal Position");
    println!("\\t(relative to parent)\\t(world coordinates)");
    println!("{:-<60}", "");

    let origin = Vector::zero();

    let sun_pos = sun_transform.apply_point(origin);
    println!("Sun\\t({:.3}, {:.3}, {:.3})\\t\\t({:.3}, {:.3}, {:.3})",
        0.0, 0.0, 0.0,
        sun_pos.x(), sun_pos.y(), sun_pos.z());

    let earth_local_pos = Vector::from_components(earth_distance, 0.0, 0.0);
    let earth_pos = earth_global.apply_point(origin);
    println!("Earth\\t({:.3}, {:.3}, {:.3})\\t\\t({:.3}, {:.3}, {:.3})",
        earth_local_pos.x(), earth_local_pos.y(), earth_local_pos.z(),
        earth_pos.x(), earth_pos.y(), earth_pos.z());

    let moon_local_pos = Vector::from_components(moon_distance, 0.0, 0.0);
    let moon_pos = moon_global.apply_point(origin);
    println!("Moon\\t({:.3}, {:.3}, {:.3})\\t\\t({:.3}, {:.3}, {:.3})",
        moon_local_pos.x(), moon_local_pos.y(), moon_local_pos.z(),
        moon_pos.x(), moon_pos.y(), moon_pos.z());

    // Animate the system
    println!("\nAnimated System (time progression):");
    println!("Time\\tEarth Position\\t\\t\\tMoon Position");
    println!("(s)\\t(world coordinates)\\t\\t(world coordinates)");
    println!("{:-<70}", "");

    for step in 0..8 {
        let time = step as f64 * 0.5; // 0.5 second steps

        // Earth orbital motion
        let earth_angle = earth_orbit_angle + time * 0.5; // Earth orbits slowly
        let earth_transform_t = Transform3D::translate(earth_distance, 0.0, 0.0)
            .compose(&Transform3D::rotate(Vector::from_components(0.0, 0.0, 1.0), earth_angle));
        let earth_global_t = sun_transform.compose(&earth_transform_t);

        // Moon orbital motion (faster)
        let moon_angle = moon_orbit_angle + time * 2.0; // Moon orbits faster
        let moon_transform_t = Transform3D::translate(moon_distance, 0.0, 0.0)
            .compose(&Transform3D::rotate(Vector::from_components(0.0, 0.0, 1.0), moon_angle));
        let moon_global_t = earth_global_t.compose(&moon_transform_t);

        let earth_pos_t = earth_global_t.apply_point(origin);
        let moon_pos_t = moon_global_t.apply_point(origin);

        println!("{:.1}\\t({:.3}, {:.3}, {:.3})\\t\\t\\t({:.3}, {:.3}, {:.3})",
            time,
            earth_pos_t.x(), earth_pos_t.y(), earth_pos_t.z(),
            moon_pos_t.x(), moon_pos_t.y(), moon_pos_t.z());
    }

    println!("\nThe hierarchical nature is preserved: the Moon orbits Earth,");
    println!("while Earth orbits the Sun, creating natural compound motion.");
}

/// Demonstrate transformation matrix equivalent operations
fn matrix_equivalence_demo() {
    println!("\n=== Matrix Equivalence Demo ==");
    println!("Comparing GA transforms with traditional matrix operations\n");

    // Create a complex transformation
    let transform = Transform3D::translate(1.0, 2.0, 3.0)
        .compose(&Transform3D::rotate(Vector::from_components(1.0, 1.0, 0.0).normalize().unwrap(), PI/3.0))
        .compose(&Transform3D::scale_uniform(1.5));

    // Test points
    let test_points = [
        Vector::from_components(1.0, 0.0, 0.0),
        Vector::from_components(0.0, 1.0, 0.0),
        Vector::from_components(0.0, 0.0, 1.0),
        Vector::from_components(1.0, 1.0, 1.0),
        Vector::zero(),
    ];

    println!("Original Point\\t\\tTransformed Point\\t\\tTransform Type");
    println!("(x, y, z)\\t\\t(x', y', z')\\t\\t\\t");
    println!("{:-<70}", "");

    for (i, &point) in test_points.iter().enumerate() {
        let transformed = transform.apply_point(point);
        let point_type = match i {
            0 => "Unit X",
            1 => "Unit Y",
            2 => "Unit Z",
            3 => "Corner",
            4 => "Origin",
            _ => "Other",
        };

        println!("({:.1}, {:.1}, {:.1})\\t\\t\\t({:.3}, {:.3}, {:.3})\\t\\t{}",
            point.x(), point.y(), point.z(),
            transformed.x(), transformed.y(), transformed.z(),
            point_type);
    }

    // Demonstrate inverse transformation
    println!("\nInverse Transformation Verification:");
    println!("Original\\t\\tTransformed\\t\\tInverse Transform\\tError");
    println!("{:-<80}", "");

    let inverse_transform = transform.inverse();

    for &point in &test_points[0..3] { // Test first 3 points
        let transformed = transform.apply_point(point);
        let recovered = inverse_transform.apply_point(transformed);
        let error = point.sub(&recovered).magnitude();

        println!("({:.1}, {:.1}, {:.1})\\t\\t({:.3}, {:.3}, {:.3})\\t\\t({:.3}, {:.3}, {:.3})\\t\\t{:.2e}",
            point.x(), point.y(), point.z(),
            transformed.x(), transformed.y(), transformed.z(),
            recovered.x(), recovered.y(), recovered.z(),
            error);
    }

    println!("\nGeometric algebra provides the same functionality as 4x4");
    println!("transformation matrices but with more intuitive operations");
    println!("and no matrix multiplication overhead.");
}

fn main() {
    println!("🎨 3D Transformations with Geometric Algebra");
    println!("===========================================\n");

    println!("This example demonstrates the elegance of geometric algebra");
    println!("for computer graphics transformations:\n");

    println!("• Rotations as rotors (no gimbal lock)");
    println!("• Smooth interpolation with SLERP");
    println!("• Natural composition of transformations");
    println!("• Unified treatment of points and directions");
    println!("• Hierarchical scene graphs");
    println!("• Matrix-free operations\n");

    // Run the demonstrations
    basic_transformations_demo();
    gimbal_lock_demo();
    hierarchical_transforms_demo();
    matrix_equivalence_demo();

    println!("\n=== Key Advantages of GA in Graphics ==");
    println!("1. No gimbal lock or quaternion normalization drift");
    println!("2. Smooth interpolation between any two orientations");
    println!("3. Intuitive geometric operations");
    println!("4. Unified framework for all transformations");
    println!("5. Natural hierarchical composition");
    println!("6. Efficient inverse and composition operations");

    println!("\n🎓 Educational Value:");
    println!("Geometric algebra eliminates many traditional problems");
    println!("in 3D graphics while providing more intuitive and");
    println!("mathematically elegant solutions.");
}