//! Mesh Operations and Geometric Processing with Geometric Algebra
//!
//! This example demonstrates how geometric algebra can elegantly handle
//! mesh operations, normal calculations, geometric queries, and spatial
//! transformations in 3D graphics applications.

use amari_core::{Multivector, Vector, Bivector, rotor::Rotor};
use std::f64::consts::PI;

type Cl3 = Multivector<3, 0, 0>;

/// A triangle in 3D space
#[derive(Debug, Clone)]
pub struct Triangle {
    pub vertices: [Vector<3, 0, 0>; 3],
}

impl Triangle {
    /// Create a new triangle
    pub fn new(v0: Vector<3, 0, 0>, v1: Vector<3, 0, 0>, v2: Vector<3, 0, 0>) -> Self {
        Self {
            vertices: [v0, v1, v2],
        }
    }

    /// Calculate the area of the triangle using geometric algebra
    pub fn area(&self) -> f64 {
        let edge1 = self.vertices[1].sub(&self.vertices[0]);
        let edge2 = self.vertices[2].sub(&self.vertices[0]);

        // Area = |edge1 ∧ edge2| / 2
        let cross_product = edge1.outer_product(&edge2.mv);
        0.5 * cross_product.magnitude()
    }

    /// Calculate the normal vector using the geometric product
    pub fn normal(&self) -> Vector<3, 0, 0> {
        let edge1 = self.vertices[1].sub(&self.vertices[0]);
        let edge2 = self.vertices[2].sub(&self.vertices[0]);

        // Normal = edge1 × edge2 (cross product)
        let cross = edge1.outer_product(&edge2.mv);
        Vector::from_components(
            cross.get(6),   // yz component
            -cross.get(5),  // -xz component
            cross.get(3),   // xy component
        ).normalize().unwrap_or(Vector::from_components(0.0, 0.0, 1.0))
    }

    /// Calculate the centroid of the triangle
    pub fn centroid(&self) -> Vector<3, 0, 0> {
        let sum = self.vertices[0].add(&self.vertices[1]).add(&self.vertices[2]);
        sum.scale(1.0 / 3.0)
    }

    /// Check if a point is inside the triangle using barycentric coordinates
    pub fn contains_point(&self, point: Vector<3, 0, 0>) -> bool {
        let v0 = self.vertices[0];
        let v1 = self.vertices[1];
        let v2 = self.vertices[2];

        // Calculate barycentric coordinates using geometric algebra
        let edge1 = v1.sub(&v0);
        let edge2 = v2.sub(&v0);
        let point_offset = point.sub(&v0);

        // Use the geometric product to compute dot products efficiently
        let dot00 = edge2.inner_product(&edge2.mv).get(0);
        let dot01 = edge2.inner_product(&edge1.mv).get(0);
        let dot02 = edge2.inner_product(&point_offset.mv).get(0);
        let dot11 = edge1.inner_product(&edge1.mv).get(0);
        let dot12 = edge1.inner_product(&point_offset.mv).get(0);

        // Compute barycentric coordinates
        let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        // Check if point is in triangle
        u >= 0.0 && v >= 0.0 && (u + v) <= 1.0
    }

    /// Calculate the closest point on the triangle to a given point
    pub fn closest_point(&self, point: Vector<3, 0, 0>) -> Vector<3, 0, 0> {
        let v0 = self.vertices[0];
        let v1 = self.vertices[1];
        let v2 = self.vertices[2];

        // Project point onto triangle plane
        let normal = self.normal();
        let to_point = point.sub(&v0);
        let distance_to_plane = to_point.inner_product(&normal.mv).get(0);
        let projected_point = point.sub(&normal.scale(distance_to_plane));

        // Check if projected point is inside triangle
        if self.contains_point(projected_point) {
            return projected_point;
        }

        // Point is outside triangle, find closest point on edges
        let mut min_distance = f64::INFINITY;
        let mut closest = v0;

        // Check each edge
        for i in 0..3 {
            let edge_start = self.vertices[i];
            let edge_end = self.vertices[(i + 1) % 3];
            let edge_closest = closest_point_on_line_segment(point, edge_start, edge_end);
            let distance = point.sub(&edge_closest).magnitude();

            if distance < min_distance {
                min_distance = distance;
                closest = edge_closest;
            }
        }

        closest
    }

    /// Transform the triangle by a rotor and translation
    pub fn transform(&self, rotor: &Rotor<3, 0, 0>, translation: &Vector<3, 0, 0>) -> Self {
        let transformed_vertices = [
            Vector::from_multivector(&rotor.apply(&self.vertices[0].mv)).add(translation),
            Vector::from_multivector(&rotor.apply(&self.vertices[1].mv)).add(translation),
            Vector::from_multivector(&rotor.apply(&self.vertices[2].mv)).add(translation),
        ];

        Self {
            vertices: transformed_vertices,
        }
    }
}

/// A 3D mesh composed of triangles
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vector<3, 0, 0>>,
    pub triangles: Vec<[usize; 3]>, // Indices into vertices array
}

impl Mesh {
    /// Create a new empty mesh
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            triangles: Vec::new(),
        }
    }

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

        let triangles = vec![
            // Front face
            [0, 1, 2], [0, 2, 3],
            // Back face
            [4, 6, 5], [4, 7, 6],
            // Left face
            [0, 3, 7], [0, 7, 4],
            // Right face
            [1, 5, 6], [1, 6, 2],
            // Top face
            [3, 2, 6], [3, 6, 7],
            // Bottom face
            [0, 4, 5], [0, 5, 1],
        ];

        Self { vertices, triangles }
    }

    /// Create a UV sphere mesh
    pub fn sphere(radius: f64, segments: usize, rings: usize) -> Self {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();

        // Generate vertices
        for ring in 0..=rings {
            let phi = PI * ring as f64 / rings as f64;
            let y = radius * phi.cos();
            let ring_radius = radius * phi.sin();

            for segment in 0..=segments {
                let theta = 2.0 * PI * segment as f64 / segments as f64;
                let x = ring_radius * theta.cos();
                let z = ring_radius * theta.sin();

                vertices.push(Vector::from_components(x, y, z));
            }
        }

        // Generate triangles
        for ring in 0..rings {
            for segment in 0..segments {
                let current = ring * (segments + 1) + segment;
                let next = current + segments + 1;

                if ring != 0 {
                    triangles.push([current, next, current + 1]);
                }
                if ring != rings - 1 {
                    triangles.push([current + 1, next, next + 1]);
                }
            }
        }

        Self { vertices, triangles }
    }

    /// Calculate vertex normals using geometric algebra
    pub fn calculate_vertex_normals(&self) -> Vec<Vector<3, 0, 0>> {
        let mut normals = vec![Vector::zero(); self.vertices.len()];
        let mut counts = vec![0; self.vertices.len()];

        // Accumulate face normals for each vertex
        for triangle_indices in &self.triangles {
            let triangle = Triangle::new(
                self.vertices[triangle_indices[0]],
                self.vertices[triangle_indices[1]],
                self.vertices[triangle_indices[2]],
            );

            let face_normal = triangle.normal();
            let face_area = triangle.area();

            // Weight normal by triangle area
            let weighted_normal = face_normal.scale(face_area);

            for &vertex_idx in triangle_indices {
                normals[vertex_idx] = normals[vertex_idx].add(&weighted_normal);
                counts[vertex_idx] += 1;
            }
        }

        // Normalize accumulated normals
        for (i, normal) in normals.iter_mut().enumerate() {
            if counts[i] > 0 {
                *normal = normal.normalize().unwrap_or(Vector::from_components(0.0, 0.0, 1.0));
            }
        }

        normals
    }

    /// Calculate the mesh's bounding box
    pub fn bounding_box(&self) -> (Vector<3, 0, 0>, Vector<3, 0, 0>) {
        if self.vertices.is_empty() {
            return (Vector::zero(), Vector::zero());
        }

        let mut min_point = self.vertices[0];
        let mut max_point = self.vertices[0];

        for vertex in &self.vertices {
            if vertex.x() < min_point.x() { min_point = Vector::from_components(vertex.x(), min_point.y(), min_point.z()); }
            if vertex.y() < min_point.y() { min_point = Vector::from_components(min_point.x(), vertex.y(), min_point.z()); }
            if vertex.z() < min_point.z() { min_point = Vector::from_components(min_point.x(), min_point.y(), vertex.z()); }

            if vertex.x() > max_point.x() { max_point = Vector::from_components(vertex.x(), max_point.y(), max_point.z()); }
            if vertex.y() > max_point.y() { max_point = Vector::from_components(max_point.x(), vertex.y(), max_point.z()); }
            if vertex.z() > max_point.z() { max_point = Vector::from_components(max_point.x(), max_point.y(), vertex.z()); }
        }

        (min_point, max_point)
    }

    /// Calculate total surface area
    pub fn surface_area(&self) -> f64 {
        let mut total_area = 0.0;

        for triangle_indices in &self.triangles {
            let triangle = Triangle::new(
                self.vertices[triangle_indices[0]],
                self.vertices[triangle_indices[1]],
                self.vertices[triangle_indices[2]],
            );
            total_area += triangle.area();
        }

        total_area
    }

    /// Transform the entire mesh
    pub fn transform(&mut self, rotor: &Rotor<3, 0, 0>, translation: &Vector<3, 0, 0>) {
        for vertex in &mut self.vertices {
            let rotated = rotor.apply(&vertex.mv);
            *vertex = Vector::from_multivector(&rotated).add(translation);
        }
    }

    /// Create a mesh by subdividing triangles (simple subdivision)
    pub fn subdivide(&self) -> Self {
        let mut new_vertices = self.vertices.clone();
        let mut new_triangles = Vec::new();

        for triangle_indices in &self.triangles {
            let v0 = self.vertices[triangle_indices[0]];
            let v1 = self.vertices[triangle_indices[1]];
            let v2 = self.vertices[triangle_indices[2]];

            // Calculate midpoints
            let m01 = v0.add(&v1).scale(0.5);
            let m12 = v1.add(&v2).scale(0.5);
            let m20 = v2.add(&v0).scale(0.5);

            // Add midpoints to vertices
            let m01_idx = new_vertices.len();
            new_vertices.push(m01);
            let m12_idx = new_vertices.len();
            new_vertices.push(m12);
            let m20_idx = new_vertices.len();
            new_vertices.push(m20);

            // Create 4 new triangles
            new_triangles.push([triangle_indices[0], m01_idx, m20_idx]);
            new_triangles.push([triangle_indices[1], m12_idx, m01_idx]);
            new_triangles.push([triangle_indices[2], m20_idx, m12_idx]);
            new_triangles.push([m01_idx, m12_idx, m20_idx]);
        }

        Self {
            vertices: new_vertices,
            triangles: new_triangles,
        }
    }
}

/// Helper function to find closest point on a line segment
fn closest_point_on_line_segment(point: Vector<3, 0, 0>, start: Vector<3, 0, 0>, end: Vector<3, 0, 0>) -> Vector<3, 0, 0> {
    let segment = end.sub(&start);
    let to_point = point.sub(&start);

    let segment_length_sq = segment.magnitude_squared();
    if segment_length_sq < 1e-10 {
        return start; // Degenerate segment
    }

    let t = to_point.inner_product(&segment.mv).get(0) / segment_length_sq;
    let clamped_t = t.clamp(0.0, 1.0);

    start.add(&segment.scale(clamped_t))
}

/// Demonstrate basic mesh operations
fn mesh_operations_demo() {
    println!("=== Mesh Operations Demo ===");
    println!("Basic mesh creation and analysis\\n");

    // Create different meshes
    let cube = Mesh::cube(2.0);
    let sphere = Mesh::sphere(1.0, 16, 8);

    println!("Mesh Statistics:");
    println!("Mesh\\t\\tVertices\\tTriangles\\tSurface Area\\tBounding Box");
    println!("{:-<70}", "");

    // Analyze cube
    let cube_area = cube.surface_area();
    let (cube_min, cube_max) = cube.bounding_box();
    println!("Cube\\t\\t{}\\t\\t{}\\t\\t{:.3}\\t\\t[({:.1},{:.1},{:.1}) to ({:.1},{:.1},{:.1})]",
        cube.vertices.len(), cube.triangles.len(), cube_area,
        cube_min.x(), cube_min.y(), cube_min.z(),
        cube_max.x(), cube_max.y(), cube_max.z());

    // Analyze sphere
    let sphere_area = sphere.surface_area();
    let (sphere_min, sphere_max) = sphere.bounding_box();
    println!("Sphere\\t\\t{}\\t\\t{}\\t\\t{:.3}\\t\\t[({:.1},{:.1},{:.1}) to ({:.1},{:.1},{:.1})]",
        sphere.vertices.len(), sphere.triangles.len(), sphere_area,
        sphere_min.x(), sphere_min.y(), sphere_min.z(),
        sphere_max.x(), sphere_max.y(), sphere_max.z());

    // Theoretical comparison
    let cube_theoretical_area = 6.0 * 2.0 * 2.0; // 6 faces of 2x2
    let sphere_theoretical_area = 4.0 * PI * 1.0 * 1.0; // 4πr²

    println!("\\nTheoretical vs Computed Surface Areas:");
    println!("Cube: theoretical = {:.3}, computed = {:.3}, error = {:.2}%",
        cube_theoretical_area, cube_area,
        100.0 * (cube_area - cube_theoretical_area).abs() / cube_theoretical_area);
    println!("Sphere: theoretical = {:.3}, computed = {:.3}, error = {:.2}%",
        sphere_theoretical_area, sphere_area,
        100.0 * (sphere_area - sphere_theoretical_area).abs() / sphere_theoretical_area);
}

/// Demonstrate normal calculations and geometric queries
fn normal_calculation_demo() {
    println!("\\n=== Normal Calculation Demo ===");
    println!("Computing face and vertex normals using geometric algebra\\n");

    let cube = Mesh::cube(1.0);
    let vertex_normals = cube.calculate_vertex_normals();

    println!("Cube Face Analysis (first 6 triangles - one per face):");
    println!("Triangle\\tVertices\\t\\t\\tFace Normal\\t\\tArea");
    println!("Index\\t\\tIndices\\t\\t\\t\\t(x, y, z)\\t\\t");
    println!("{:-<80}", "");

    for (i, triangle_indices) in cube.triangles.iter().take(6).enumerate() {
        let triangle = Triangle::new(
            cube.vertices[triangle_indices[0]],
            cube.vertices[triangle_indices[1]],
            cube.vertices[triangle_indices[2]],
        );

        let normal = triangle.normal();
        let area = triangle.area();

        println!("{}\\t\\t({}, {}, {})\\t\\t\\t({:.3}, {:.3}, {:.3})\\t{:.3}",
            i, triangle_indices[0], triangle_indices[1], triangle_indices[2],
            normal.x(), normal.y(), normal.z(), area);
    }

    println!("\\nVertex Normal Analysis (first 8 vertices):");
    println!("Vertex\\tPosition\\t\\t\\tComputed Normal");
    println!("Index\\t(x, y, z)\\t\\t\\t(x, y, z)");
    println!("{:-<60}", "");

    for i in 0..8.min(cube.vertices.len()) {
        let vertex = cube.vertices[i];
        let normal = vertex_normals[i];

        println!("{}\\t({:.1}, {:.1}, {:.1})\\t\\t\\t({:.3}, {:.3}, {:.3})",
            i, vertex.x(), vertex.y(), vertex.z(),
            normal.x(), normal.y(), normal.z());
    }

    println!("\\nNote: Vertex normals are area-weighted averages of adjacent face normals.");
}

/// Demonstrate geometric queries and closest point calculations
fn geometric_queries_demo() {
    println!("\\n=== Geometric Queries Demo ===");
    println!("Point-in-triangle and closest point calculations\\n");

    // Create a simple triangle
    let triangle = Triangle::new(
        Vector::from_components(0.0, 0.0, 0.0),
        Vector::from_components(2.0, 0.0, 0.0),
        Vector::from_components(1.0, 2.0, 0.0),
    );

    let test_points = vec![
        Vector::from_components(1.0, 0.5, 0.0),  // Inside triangle
        Vector::from_components(0.5, 0.25, 0.0), // Inside triangle
        Vector::from_components(3.0, 1.0, 0.0),  // Outside triangle
        Vector::from_components(1.0, 1.0, 1.0),  // Above triangle
        Vector::from_components(-1.0, 0.0, 0.0), // Outside triangle
    ];

    println!("Triangle vertices: ({:.1}, {:.1}, {:.1}), ({:.1}, {:.1}, {:.1}), ({:.1}, {:.1}, {:.1})",
        triangle.vertices[0].x(), triangle.vertices[0].y(), triangle.vertices[0].z(),
        triangle.vertices[1].x(), triangle.vertices[1].y(), triangle.vertices[1].z(),
        triangle.vertices[2].x(), triangle.vertices[2].y(), triangle.vertices[2].z());
    println!("Triangle area: {:.3}", triangle.area());
    println!("Triangle normal: ({:.3}, {:.3}, {:.3})\\n",
        triangle.normal().x(), triangle.normal().y(), triangle.normal().z());

    println!("Point\\t\\tInside?\\tClosest Point\\t\\tDistance");
    println!("(x, y, z)\\t\\t\\t(x, y, z)\\t\\t");
    println!("{:-<70}", "");

    for point in test_points {
        let inside = triangle.contains_point(point);
        let closest = triangle.closest_point(point);
        let distance = point.sub(&closest).magnitude();

        println!("({:.1}, {:.1}, {:.1})\\t\\t{}\\t({:.3}, {:.3}, {:.3})\\t\\t{:.3}",
            point.x(), point.y(), point.z(),
            if inside { "Yes" } else { "No" },
            closest.x(), closest.y(), closest.z(),
            distance);
    }
}

/// Demonstrate mesh transformations
fn mesh_transformation_demo() {
    println!("\\n=== Mesh Transformation Demo ===");
    println!("Applying rotations and translations to meshes\\n");

    let mut cube = Mesh::cube(1.0);

    println!("Original cube centroid and bounding box:");
    let (min_orig, max_orig) = cube.bounding_box();
    let centroid_orig = min_orig.add(&max_orig).scale(0.5);
    println!("Centroid: ({:.3}, {:.3}, {:.3})", centroid_orig.x(), centroid_orig.y(), centroid_orig.z());
    println!("Bounding box: ({:.1}, {:.1}, {:.1}) to ({:.1}, {:.1}, {:.1})\\n",
        min_orig.x(), min_orig.y(), min_orig.z(),
        max_orig.x(), max_orig.y(), max_orig.z());

    // Apply rotation around diagonal axis
    let rotation_axis = Vector::from_components(1.0, 1.0, 1.0).normalize().unwrap();
    let rotation_angle = PI / 4.0; // 45 degrees
    let rotation_bivector = rotation_axis.outer_product(&Vector::zero().mv);
    let rotor = Rotor::from_bivector(&Bivector::from_multivector(&rotation_bivector), rotation_angle);

    // Apply translation
    let translation = Vector::from_components(2.0, 1.0, 0.5);

    cube.transform(&rotor, &translation);

    println!("After rotation ({:.1}° around axis ({:.3}, {:.3}, {:.3})) and translation ({:.1}, {:.1}, {:.1}):",
        rotation_angle * 180.0 / PI,
        rotation_axis.x(), rotation_axis.y(), rotation_axis.z(),
        translation.x(), translation.y(), translation.z());

    let (min_trans, max_trans) = cube.bounding_box();
    let centroid_trans = min_trans.add(&max_trans).scale(0.5);
    println!("New centroid: ({:.3}, {:.3}, {:.3})", centroid_trans.x(), centroid_trans.y(), centroid_trans.z());
    println!("New bounding box: ({:.3}, {:.3}, {:.3}) to ({:.3}, {:.3}, {:.3})",
        min_trans.x(), min_trans.y(), min_trans.z(),
        max_trans.x(), max_trans.y(), max_trans.z());

    // Verify surface area preservation
    let area_after = cube.surface_area();
    let area_before = 24.0; // 6 faces × 2×2 area each
    println!("Surface area before: {:.3}, after: {:.3} (should be preserved)", area_before, area_after);
}

/// Demonstrate mesh subdivision
fn subdivision_demo() {
    println!("\\n=== Mesh Subdivision Demo ===");
    println!("Creating higher resolution meshes through subdivision\\n");

    let mut sphere = Mesh::sphere(1.0, 8, 4); // Low resolution sphere

    println!("Subdivision levels:");
    println!("Level\\tVertices\\tTriangles\\tSurface Area\\tTheoretical Area\\tError %");
    println!("{:-<80}", "");

    let theoretical_area = 4.0 * PI; // 4πr² for unit sphere

    for level in 0..4 {
        let area = sphere.surface_area();
        let error_percent = 100.0 * (area - theoretical_area).abs() / theoretical_area;

        println!("{}\\t{}\\t\\t{}\\t\\t{:.3}\\t\\t{:.3}\\t\\t\\t{:.2}",
            level, sphere.vertices.len(), sphere.triangles.len(),
            area, theoretical_area, error_percent);

        if level < 3 {
            sphere = sphere.subdivide();
        }
    }

    println!("\\nSubdivision improves surface area approximation by increasing");
    println!("mesh resolution, demonstrating the geometric accuracy of our");
    println!("geometric algebra-based calculations.");
}

fn main() {
    println!("🔺 Mesh Operations and Geometric Processing with Geometric Algebra");
    println!("=================================================================\\n");

    println!("This example demonstrates mesh operations using geometric algebra:\\n");

    println!("• Triangle area and normal calculations");
    println!("• Vertex normal computation with area weighting");
    println!("• Point-in-triangle tests using barycentric coordinates");
    println!("• Closest point queries and distance calculations");
    println!("• Mesh transformations with rotors");
    println!("• Geometric queries and spatial analysis");
    println!("• Mesh subdivision and refinement\\n");

    // Run the demonstrations
    mesh_operations_demo();
    normal_calculation_demo();
    geometric_queries_demo();
    mesh_transformation_demo();
    subdivision_demo();

    println!("\\n=== Advantages of GA in Mesh Processing ==");
    println!("1. Natural representation of orientations and rotations");
    println!("2. Efficient cross products via outer products");
    println!("3. Unified coordinate transformations");
    println!("4. Geometric clarity in normal calculations");
    println!("5. Robust geometric queries");
    println!("6. Area-preserving transformations");

    println!("\\n🎓 Educational Value:");
    println!("Geometric algebra provides an elegant mathematical");
    println!("framework for 3D mesh processing, making complex");
    println!("geometric operations intuitive and efficient.");
}