//! Ray Tracing with Geometric Algebra
//!
//! This example demonstrates how geometric algebra provides an elegant
//! framework for ray tracing, with natural representations of rays,
//! geometric intersections, and lighting calculations.

use amari_core::{Multivector, Vector, Bivector};
use std::f64::consts::PI;
use rand::Rng;

type Cl3 = Multivector<3, 0, 0>;

/// A ray in 3D space using geometric algebra
#[derive(Debug, Clone)]
pub struct Ray {
    /// Ray origin
    pub origin: Vector<3, 0, 0>,
    /// Ray direction (should be normalized)
    pub direction: Vector<3, 0, 0>,
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: Vector<3, 0, 0>, direction: Vector<3, 0, 0>) -> Self {
        Self {
            origin,
            direction: direction.normalize().unwrap_or(Vector::from_components(0.0, 0.0, 1.0)),
        }
    }

    /// Get a point along the ray at parameter t
    pub fn at(&self, t: f64) -> Vector<3, 0, 0> {
        self.origin.add(&self.direction.scale(t))
    }

    /// Reflect the ray off a surface with given normal
    pub fn reflect(&self, normal: Vector<3, 0, 0>) -> Ray {
        // Reflection formula: r = d - 2(d·n)n
        let dot_product = self.direction.inner_product(&normal.mv).get(0);
        let reflection_direction = self.direction.sub(&normal.scale(2.0 * dot_product));

        Ray::new(self.origin, reflection_direction)
    }

    /// Refract the ray through a surface (Snell's law)
    pub fn refract(&self, normal: Vector<3, 0, 0>, eta_ratio: f64) -> Option<Ray> {
        let cos_theta = (-self.direction.inner_product(&normal.mv).get(0)).min(1.0);
        let sin_theta_sq = 1.0 - cos_theta * cos_theta;
        let sin_theta_refracted_sq = eta_ratio * eta_ratio * sin_theta_sq;

        // Total internal reflection check
        if sin_theta_refracted_sq > 1.0 {
            return None;
        }

        let cos_theta_refracted = (1.0 - sin_theta_refracted_sq).sqrt();

        // Refracted direction using Snell's law in vector form
        let parallel_component = self.direction.add(&normal.scale(cos_theta)).scale(eta_ratio);
        let perpendicular_component = normal.scale(-cos_theta_refracted);
        let refracted_direction = parallel_component.add(&perpendicular_component);

        Some(Ray::new(self.origin, refracted_direction))
    }
}

/// A sphere for ray intersection testing
#[derive(Debug, Clone)]
pub struct Sphere {
    pub center: Vector<3, 0, 0>,
    pub radius: f64,
    pub material: Material,
}

impl Sphere {
    /// Create a new sphere
    pub fn new(center: [f64; 3], radius: f64, material: Material) -> Self {
        Self {
            center: Vector::from_components(center[0], center[1], center[2]),
            radius,
            material,
        }
    }

    /// Intersect ray with sphere using geometric algebra
    pub fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        // Vector from ray origin to sphere center
        let oc = ray.origin.sub(&self.center);

        // Quadratic equation coefficients for ray-sphere intersection
        // |ray.origin + t * ray.direction - sphere.center|² = radius²
        let a = ray.direction.magnitude_squared();
        let b = 2.0 * oc.inner_product(&ray.direction.mv).get(0);
        let c = oc.magnitude_squared() - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return None; // No intersection
        }

        // Find the closest positive intersection
        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / (2.0 * a);
        let t2 = (-b + sqrt_discriminant) / (2.0 * a);

        let t = if t1 > 1e-6 {
            t1
        } else if t2 > 1e-6 {
            t2
        } else {
            return None; // Both intersections behind ray origin
        };

        let hit_point = ray.at(t);
        let normal = hit_point.sub(&self.center).normalize()
            .unwrap_or(Vector::from_components(0.0, 1.0, 0.0));

        Some(RayHit {
            point: hit_point,
            normal,
            distance: t,
            material: self.material.clone(),
        })
    }
}

/// A plane for ray intersection testing
#[derive(Debug, Clone)]
pub struct Plane {
    pub point: Vector<3, 0, 0>,
    pub normal: Vector<3, 0, 0>,
    pub material: Material,
}

impl Plane {
    /// Create a new plane
    pub fn new(point: [f64; 3], normal: [f64; 3], material: Material) -> Self {
        Self {
            point: Vector::from_components(point[0], point[1], point[2]),
            normal: Vector::from_components(normal[0], normal[1], normal[2])
                .normalize().unwrap_or(Vector::from_components(0.0, 1.0, 0.0)),
            material,
        }
    }

    /// Intersect ray with plane
    pub fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        let denom = self.normal.inner_product(&ray.direction.mv).get(0);

        // Check if ray is parallel to plane
        if denom.abs() < 1e-6 {
            return None;
        }

        let to_plane = self.point.sub(&ray.origin);
        let t = to_plane.inner_product(&self.normal.mv).get(0) / denom;

        if t > 1e-6 {
            let hit_point = ray.at(t);
            Some(RayHit {
                point: hit_point,
                normal: self.normal,
                distance: t,
                material: self.material.clone(),
            })
        } else {
            None
        }
    }
}

/// Material properties for ray tracing
#[derive(Debug, Clone)]
pub struct Material {
    pub color: [f64; 3],           // RGB color
    pub reflectivity: f64,         // 0.0 = no reflection, 1.0 = perfect mirror
    pub transparency: f64,         // 0.0 = opaque, 1.0 = transparent
    pub refractive_index: f64,     // Index of refraction
    pub roughness: f64,            // Surface roughness for diffuse lighting
}

impl Material {
    /// Create a diffuse (matte) material
    pub fn diffuse(color: [f64; 3]) -> Self {
        Self {
            color,
            reflectivity: 0.0,
            transparency: 0.0,
            refractive_index: 1.0,
            roughness: 1.0,
        }
    }

    /// Create a reflective (mirror-like) material
    pub fn reflective(color: [f64; 3], reflectivity: f64) -> Self {
        Self {
            color,
            reflectivity,
            transparency: 0.0,
            refractive_index: 1.0,
            roughness: 0.0,
        }
    }

    /// Create a refractive (glass-like) material
    pub fn refractive(color: [f64; 3], transparency: f64, refractive_index: f64) -> Self {
        Self {
            color,
            reflectivity: 0.1,
            transparency,
            refractive_index,
            roughness: 0.0,
        }
    }
}

/// Result of a ray intersection
#[derive(Debug, Clone)]
pub struct RayHit {
    pub point: Vector<3, 0, 0>,
    pub normal: Vector<3, 0, 0>,
    pub distance: f64,
    pub material: Material,
}

/// A simple light source
#[derive(Debug, Clone)]
pub struct Light {
    pub position: Vector<3, 0, 0>,
    pub color: [f64; 3],
    pub intensity: f64,
}

impl Light {
    /// Create a new light
    pub fn new(position: [f64; 3], color: [f64; 3], intensity: f64) -> Self {
        Self {
            position: Vector::from_components(position[0], position[1], position[2]),
            color,
            intensity,
        }
    }

    /// Calculate illumination at a point using geometric algebra
    pub fn illuminate(&self, point: Vector<3, 0, 0>, normal: Vector<3, 0, 0>) -> [f64; 3] {
        let to_light = self.position.sub(&point);
        let distance = to_light.magnitude();
        let light_direction = to_light.normalize().unwrap_or(Vector::from_components(0.0, 1.0, 0.0));

        // Lambertian diffuse lighting: intensity ∝ cos(θ) = normal · light_direction
        let cos_theta = normal.inner_product(&light_direction.mv).get(0).max(0.0);

        // Inverse square law for distance attenuation
        let attenuation = self.intensity / (1.0 + 0.1 * distance + 0.01 * distance * distance);

        [
            self.color[0] * cos_theta * attenuation,
            self.color[1] * cos_theta * attenuation,
            self.color[2] * cos_theta * attenuation,
        ]
    }
}

/// A simple scene containing objects and lights
#[derive(Debug)]
pub struct Scene {
    pub spheres: Vec<Sphere>,
    pub planes: Vec<Plane>,
    pub lights: Vec<Light>,
}

impl Scene {
    /// Create a new empty scene
    pub fn new() -> Self {
        Self {
            spheres: Vec::new(),
            planes: Vec::new(),
            lights: Vec::new(),
        }
    }

    /// Find the closest intersection along a ray
    pub fn intersect(&self, ray: &Ray) -> Option<RayHit> {
        let mut closest_hit: Option<RayHit> = None;
        let mut closest_distance = f64::INFINITY;

        // Check sphere intersections
        for sphere in &self.spheres {
            if let Some(hit) = sphere.intersect(ray) {
                if hit.distance < closest_distance {
                    closest_distance = hit.distance;
                    closest_hit = Some(hit);
                }
            }
        }

        // Check plane intersections
        for plane in &self.planes {
            if let Some(hit) = plane.intersect(ray) {
                if hit.distance < closest_distance {
                    closest_distance = hit.distance;
                    closest_hit = Some(hit);
                }
            }
        }

        closest_hit
    }

    /// Trace a ray and calculate color
    pub fn trace_ray(&self, ray: &Ray, depth: i32) -> [f64; 3] {
        if depth <= 0 {
            return [0.0, 0.0, 0.0]; // Max recursion depth reached
        }

        match self.intersect(ray) {
            Some(hit) => {
                let mut color = [0.0, 0.0, 0.0];

                // Direct lighting from all light sources
                for light in &self.lights {
                    // Check for shadows
                    let shadow_ray = Ray::new(
                        hit.point.add(&hit.normal.scale(1e-6)), // Offset to avoid self-intersection
                        light.position.sub(&hit.point).normalize().unwrap()
                    );

                    let in_shadow = if let Some(shadow_hit) = self.intersect(&shadow_ray) {
                        shadow_hit.distance < light.position.sub(&hit.point).magnitude()
                    } else {
                        false
                    };

                    if !in_shadow {
                        let illumination = light.illuminate(hit.point, hit.normal);
                        color[0] += hit.material.color[0] * illumination[0] * (1.0 - hit.material.reflectivity);
                        color[1] += hit.material.color[1] * illumination[1] * (1.0 - hit.material.reflectivity);
                        color[2] += hit.material.color[2] * illumination[2] * (1.0 - hit.material.reflectivity);
                    }
                }

                // Reflection
                if hit.material.reflectivity > 0.0 {
                    let reflected_ray = ray.reflect(hit.normal);
                    let reflected_ray_offset = Ray::new(
                        hit.point.add(&hit.normal.scale(1e-6)),
                        reflected_ray.direction
                    );
                    let reflected_color = self.trace_ray(&reflected_ray_offset, depth - 1);
                    color[0] += reflected_color[0] * hit.material.reflectivity;
                    color[1] += reflected_color[1] * hit.material.reflectivity;
                    color[2] += reflected_color[2] * hit.material.reflectivity;
                }

                // Refraction
                if hit.material.transparency > 0.0 {
                    let eta_ratio = 1.0 / hit.material.refractive_index;
                    if let Some(refracted_ray) = ray.refract(hit.normal, eta_ratio) {
                        let refracted_ray_offset = Ray::new(
                            hit.point.sub(&hit.normal.scale(1e-6)),
                            refracted_ray.direction
                        );
                        let refracted_color = self.trace_ray(&refracted_ray_offset, depth - 1);
                        color[0] += refracted_color[0] * hit.material.transparency;
                        color[1] += refracted_color[1] * hit.material.transparency;
                        color[2] += refracted_color[2] * hit.material.transparency;
                    }
                }

                color
            }
            None => {
                // Sky color (simple gradient)
                let t = 0.5 * (ray.direction.y() + 1.0);
                [
                    (1.0 - t) + t * 0.5,
                    (1.0 - t) + t * 0.7,
                    (1.0 - t) + t * 1.0
                ]
            }
        }
    }
}

/// Demonstrate basic ray-sphere intersection
fn ray_intersection_demo() {
    println!("=== Ray Intersection Demo ===");
    println!("Testing ray-sphere intersections using geometric algebra\\n");

    let sphere = Sphere::new([0.0, 0.0, -5.0], 1.0, Material::diffuse([1.0, 0.0, 0.0]));

    let test_rays = vec![
        Ray::new(Vector::zero(), Vector::from_components(0.0, 0.0, -1.0)),  // Direct hit
        Ray::new(Vector::zero(), Vector::from_components(0.5, 0.0, -1.0)),  // Glancing hit
        Ray::new(Vector::zero(), Vector::from_components(1.5, 0.0, -1.0)),  // Miss
        Ray::new(Vector::zero(), Vector::from_components(0.0, 1.0, -1.0)),  // Hit top
        Ray::new(Vector::zero(), Vector::from_components(0.0, 0.0, 1.0)),   // Opposite direction
    ];

    println!("Sphere at (0, 0, -5) with radius 1.0");
    println!("Ray Origin\\t\\tRay Direction\\t\\tHit?\\tDistance\\tHit Point");
    println!("(x, y, z)\\t\\t(x, y, z)\\t\\t\\t\\t\\t(x, y, z)");
    println!("{:-<90}", "");

    for (i, ray) in test_rays.iter().enumerate() {
        match sphere.intersect(ray) {
            Some(hit) => {
                println!("({:.1}, {:.1}, {:.1})\\t\\t({:.1}, {:.1}, {:.1})\\t\\tYes\\t{:.3}\\t\\t({:.3}, {:.3}, {:.3})",
                    ray.origin.x(), ray.origin.y(), ray.origin.z(),
                    ray.direction.x(), ray.direction.y(), ray.direction.z(),
                    hit.distance,
                    hit.point.x(), hit.point.y(), hit.point.z());
            }
            None => {
                println!("({:.1}, {:.1}, {:.1})\\t\\t({:.1}, {:.1}, {:.1})\\t\\tNo\\t-\\t\\t-",
                    ray.origin.x(), ray.origin.y(), ray.origin.z(),
                    ray.direction.x(), ray.direction.y(), ray.direction.z());
            }
        }
    }

    println!("\\nGeometric algebra provides natural vector operations");
    println!("for efficient ray-object intersection calculations.");
}

/// Demonstrate reflection and refraction
fn reflection_refraction_demo() {
    println!("\\n=== Reflection and Refraction Demo ===");
    println!("Demonstrating light behavior at surfaces\\n");

    let surface_normal = Vector::from_components(0.0, 1.0, 0.0); // Upward normal
    let incident_rays = vec![
        Ray::new(Vector::zero(), Vector::from_components(1.0, -1.0, 0.0).normalize().unwrap()),   // 45° angle
        Ray::new(Vector::zero(), Vector::from_components(0.5, -1.0, 0.0).normalize().unwrap()),   // Shallow angle
        Ray::new(Vector::zero(), Vector::from_components(0.0, -1.0, 0.0)),                        // Normal incidence
        Ray::new(Vector::zero(), Vector::from_components(-1.0, -1.0, 0.0).normalize().unwrap()),  // 45° from left
    ];

    println!("Surface normal: (0, 1, 0) (horizontal surface)");
    println!("Glass refractive index: 1.5\\n");

    println!("Incident Ray\\t\\t\\tReflected Ray\\t\\t\\tRefracted Ray");
    println!("Direction (x, y, z)\\t\\tDirection (x, y, z)\\t\\tDirection (x, y, z)");
    println!("{:-<80}", "");

    for ray in incident_rays {
        let reflected = ray.reflect(surface_normal);
        let refracted = ray.refract(surface_normal, 1.0 / 1.5); // Air to glass

        let refracted_str = match refracted {
            Some(ref_ray) => format!("({:.3}, {:.3}, {:.3})",
                ref_ray.direction.x(), ref_ray.direction.y(), ref_ray.direction.z()),
            None => "Total internal reflection".to_string(),
        };

        println!("({:.3}, {:.3}, {:.3})\\t\\t({:.3}, {:.3}, {:.3})\\t\\t{}",
            ray.direction.x(), ray.direction.y(), ray.direction.z(),
            reflected.direction.x(), reflected.direction.y(), reflected.direction.z(),
            refracted_str);
    }

    // Test total internal reflection (glass to air)
    println!("\\nTotal Internal Reflection Test (glass to air, critical angle ≈ 41.8°):");
    let critical_angle = (1.0 / 1.5).asin();
    println!("Critical angle: {:.1}°\\n", critical_angle * 180.0 / PI);

    let tir_rays = vec![
        Ray::new(Vector::zero(), Vector::from_components(0.3, 1.0, 0.0).normalize().unwrap()),  // Below critical
        Ray::new(Vector::zero(), Vector::from_components(0.7, 1.0, 0.0).normalize().unwrap()),  // At critical
        Ray::new(Vector::zero(), Vector::from_components(1.0, 1.0, 0.0).normalize().unwrap()),  // Above critical
    ];

    for ray in tir_rays {
        let angle = ray.direction.inner_product(&surface_normal.mv).get(0).acos() * 180.0 / PI;
        let refracted = ray.refract(surface_normal, 1.5); // Glass to air

        match refracted {
            Some(ref_ray) => {
                println!("Angle: {:.1}°, Refracted: ({:.3}, {:.3}, {:.3})",
                    angle,
                    ref_ray.direction.x(), ref_ray.direction.y(), ref_ray.direction.z());
            }
            None => {
                println!("Angle: {:.1}°, Total internal reflection", angle);
            }
        }
    }
}

/// Demonstrate simple ray tracing scene
fn ray_tracing_demo() {
    println!("\\n=== Simple Ray Tracing Demo ===");
    println!("Rendering a scene with multiple objects and lighting\\n");

    // Create a simple scene
    let mut scene = Scene::new();

    // Add objects
    scene.spheres.push(Sphere::new(
        [-1.0, 0.0, -3.0],
        0.5,
        Material::reflective([0.8, 0.8, 0.9], 0.8)
    ));

    scene.spheres.push(Sphere::new(
        [1.0, 0.0, -4.0],
        0.7,
        Material::refractive([0.9, 0.9, 1.0], 0.9, 1.5)
    ));

    scene.spheres.push(Sphere::new(
        [0.0, -0.5, -2.5],
        0.3,
        Material::diffuse([1.0, 0.2, 0.2])
    ));

    // Add ground plane
    scene.planes.push(Plane::new(
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        Material::diffuse([0.5, 0.5, 0.5])
    ));

    // Add lights
    scene.lights.push(Light::new([2.0, 2.0, -1.0], [1.0, 1.0, 1.0], 2.0));
    scene.lights.push(Light::new([-2.0, 1.0, -2.0], [0.8, 0.8, 1.0], 1.5));

    println!("Scene setup:");
    println!("• {} spheres (reflective, refractive, diffuse)", scene.spheres.len());
    println!("• {} planes (ground)", scene.planes.len());
    println!("• {} lights\\n", scene.lights.len());

    // Simple camera setup
    let camera_position = Vector::zero();
    let image_width = 10;
    let image_height = 6;

    println!("Rendered image ({}x{} characters, 'X' = bright, '.' = dark):", image_width, image_height);

    for y in 0..image_height {
        for x in 0..image_width {
            // Convert pixel coordinates to ray direction
            let u = (x as f64 + 0.5) / image_width as f64;
            let v = (y as f64 + 0.5) / image_height as f64;

            // Map to [-1, 1] range and create ray direction
            let ray_x = (u * 2.0 - 1.0) * 2.0; // Aspect ratio adjustment
            let ray_y = (1.0 - v * 2.0) * 1.2; // Flip Y and adjust FOV
            let ray_z = -1.0;

            let ray_direction = Vector::from_components(ray_x, ray_y, ray_z).normalize().unwrap();
            let ray = Ray::new(camera_position, ray_direction);

            // Trace the ray
            let color = scene.trace_ray(&ray, 3); // Max depth 3

            // Convert color to character
            let brightness = (color[0] + color[1] + color[2]) / 3.0;
            let character = if brightness > 0.8 {
                '#'
            } else if brightness > 0.6 {
                'X'
            } else if brightness > 0.4 {
                'o'
            } else if brightness > 0.2 {
                '.'
            } else {
                ' '
            };

            print!("{}", character);
        }
        println!();
    }

    println!("\\nThis ASCII art shows reflections, refractions, and shadows");
    println!("calculated using geometric algebra ray tracing techniques.");
}

/// Demonstrate Monte Carlo ray tracing for soft shadows
fn monte_carlo_demo() {
    println!("\\n=== Monte Carlo Ray Tracing Demo ===");
    println!("Soft shadows using random sampling\\n");

    let sphere = Sphere::new([0.0, 0.0, -3.0], 0.5, Material::diffuse([1.0, 0.2, 0.2]));
    let light_center = Vector::from_components(1.0, 2.0, -1.0);
    let light_radius = 0.3;

    let sample_point = Vector::from_components(0.0, -0.5, -3.0); // Point below sphere

    println!("Testing soft shadows with area light:");
    println!("Light center: ({:.1}, {:.1}, {:.1}), radius: {:.1}",
        light_center.x(), light_center.y(), light_center.z(), light_radius);
    println!("Sample point: ({:.1}, {:.1}, {:.1})\\n",
        sample_point.x(), sample_point.y(), sample_point.z());

    let num_samples = 16;
    let mut shadow_samples = 0;
    let mut rng = rand::thread_rng();

    println!("Sample\\tLight Position\\t\\t\\tShadow Ray\\t\\tBlocked?");
    println!("\\t(x, y, z)\\t\\t\\t(direction)\\t\\t");
    println!("{:-<70}", "");

    for i in 0..num_samples {
        // Generate random point on light surface (simplified circular sampling)
        let theta = rng.gen::<f64>() * 2.0 * PI;
        let r = rng.gen::<f64>().sqrt() * light_radius;
        let light_offset = Vector::from_components(
            r * theta.cos(),
            0.0,
            r * theta.sin()
        );
        let light_position = light_center.add(&light_offset);

        // Create shadow ray from sample point to light
        let shadow_direction = light_position.sub(&sample_point).normalize().unwrap();
        let shadow_ray = Ray::new(sample_point, shadow_direction);

        // Test intersection with sphere
        let blocked = sphere.intersect(&shadow_ray).is_some();
        if blocked {
            shadow_samples += 1;
        }

        if i < 8 { // Show first 8 samples
            println!("{}\\t({:.3}, {:.3}, {:.3})\\t\\t({:.3}, {:.3}, {:.3})\\t{}",
                i + 1,
                light_position.x(), light_position.y(), light_position.z(),
                shadow_direction.x(), shadow_direction.y(), shadow_direction.z(),
                if blocked { "Yes" } else { "No" });
        }
    }

    let shadow_factor = shadow_samples as f64 / num_samples as f64;
    println!("...\\n");
    println!("Shadow factor: {:.2} ({} of {} samples blocked)", shadow_factor, shadow_samples, num_samples);
    println!("Illumination: {:.1}% (soft shadow)", (1.0 - shadow_factor) * 100.0);

    println!("\\nMonte Carlo sampling provides realistic soft shadows by");
    println!("randomly sampling the area light source, creating natural");
    println!("penumbra effects that match real-world lighting.");
}

fn main() {
    println!("🌟 Ray Tracing with Geometric Algebra");
    println!("=====================================\\n");

    println!("This example demonstrates ray tracing using geometric algebra:\\n");

    println!("• Natural ray representation with origin and direction vectors");
    println!("• Efficient ray-object intersection using vector operations");
    println!("• Reflection and refraction using geometric algebra");
    println!("• Lighting calculations with dot products");
    println!("• Monte Carlo sampling for realistic effects");
    println!("• Unified coordinate transformations\\n");

    // Run the demonstrations
    ray_intersection_demo();
    reflection_refraction_demo();
    ray_tracing_demo();
    monte_carlo_demo();

    println!("\\n=== Advantages of GA in Ray Tracing ==");
    println!("1. Natural vector operations for ray math");
    println!("2. Efficient intersection calculations");
    println!("3. Elegant reflection/refraction formulas");
    println!("4. Unified treatment of geometric operations");
    println!("5. Clear geometric interpretation");
    println!("6. Robust numerical properties");

    println!("\\n🎓 Educational Value:");
    println!("Geometric algebra provides an intuitive mathematical");
    println!("framework for ray tracing, making complex lighting");
    println!("and shading calculations more understandable and efficient.");
}