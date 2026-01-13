//! # Simplicial Complexes Example
//!
//! Demonstrates construction and analysis of simplicial complexes,
//! the fundamental building blocks of computational topology.
//!
//! ## Mathematical Background
//!
//! A simplicial complex K is a collection of simplices (vertices, edges,
//! triangles, tetrahedra, etc.) satisfying:
//! 1. Every face of a simplex in K is in K
//! 2. The intersection of any two simplices is a face of each
//!
//! Run with: `cargo run --bin simplicial_complexes`

use amari_topology::{
    simplex::Simplex,
    complex::SimplicialComplex,
    chain::ChainComplex,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════════════");
    println!("                 SIMPLICIAL COMPLEXES DEMO");
    println!("═══════════════════════════════════════════════════════════════\n");

    // =========================================================================
    // Part 1: Building Simplices
    // =========================================================================
    println!("Part 1: Building Simplices");
    println!("──────────────────────────\n");

    // 0-simplex (vertex)
    let v0 = Simplex::vertex(0);
    let v1 = Simplex::vertex(1);
    let v2 = Simplex::vertex(2);
    let v3 = Simplex::vertex(3);

    println!("0-simplices (vertices): {}, {}, {}, {}", v0, v1, v2, v3);

    // 1-simplex (edge)
    let e01 = Simplex::edge(0, 1);
    let e12 = Simplex::edge(1, 2);
    let e02 = Simplex::edge(0, 2);

    println!("1-simplices (edges): {}, {}, {}", e01, e12, e02);

    // 2-simplex (triangle)
    let t012 = Simplex::triangle(0, 1, 2);
    println!("2-simplex (triangle): {}", t012);

    // 3-simplex (tetrahedron)
    let tet = Simplex::new(vec![0, 1, 2, 3])?;
    println!("3-simplex (tetrahedron): {}", tet);

    // Faces of a simplex
    println!("\nFaces of triangle [0,1,2]:");
    for face in t012.faces() {
        println!("  {}", face);
    }

    println!("\nFaces of tetrahedron [0,1,2,3]:");
    for face in tet.faces() {
        println!("  {} (dim {})", face, face.dimension());
    }

    // =========================================================================
    // Part 2: Building Simplicial Complexes
    // =========================================================================
    println!("\n\nPart 2: Building Simplicial Complexes");
    println!("──────────────────────────────────────\n");

    // Triangle complex
    let mut triangle_complex = SimplicialComplex::new();
    triangle_complex.add_simplex(t012.clone())?;  // Automatically adds all faces

    println!("Triangle complex (adding one 2-simplex):");
    println!("  Vertices: {:?}", triangle_complex.vertices());
    println!("  Edges: {}", triangle_complex.simplices_of_dim(1).len());
    println!("  Triangles: {}", triangle_complex.simplices_of_dim(2).len());
    println!("  f-vector: {:?}", triangle_complex.f_vector());

    // Tetrahedron complex
    let mut tet_complex = SimplicialComplex::new();
    tet_complex.add_simplex(tet.clone())?;

    println!("\nTetrahedron complex (adding one 3-simplex):");
    println!("  f-vector: {:?}", tet_complex.f_vector());
    println!("  (f₀=4 vertices, f₁=6 edges, f₂=4 triangles, f₃=1 tetrahedron)");

    // Euler characteristic
    let euler = triangle_complex.euler_characteristic();
    println!("\nTriangle Euler characteristic: χ = {}", euler);
    println!("  (χ = f₀ - f₁ + f₂ = 3 - 3 + 1 = 1)");

    let euler_tet = tet_complex.euler_characteristic();
    println!("\nTetrahedron Euler characteristic: χ = {}", euler_tet);
    println!("  (χ = f₀ - f₁ + f₂ - f₃ = 4 - 6 + 4 - 1 = 1)");

    // =========================================================================
    // Part 3: Classic Surfaces
    // =========================================================================
    println!("\n\nPart 3: Triangulated Surfaces");
    println!("──────────────────────────────\n");

    // Triangulate a square (two triangles)
    println!("Square (2 triangles):");
    let mut square = SimplicialComplex::new();
    square.add_simplex(Simplex::triangle(0, 1, 2))?;
    square.add_simplex(Simplex::triangle(0, 2, 3))?;
    println!("  f-vector: {:?}", square.f_vector());
    println!("  Euler characteristic: χ = {}", square.euler_characteristic());

    // Octahedron surface (triangulation of sphere)
    println!("\nOctahedron (triangulation of S²):");
    let mut octahedron = SimplicialComplex::new();
    // 6 vertices: 0=top, 1,2,3,4=equator, 5=bottom
    // 8 triangular faces
    octahedron.add_simplex(Simplex::triangle(0, 1, 2))?;
    octahedron.add_simplex(Simplex::triangle(0, 2, 3))?;
    octahedron.add_simplex(Simplex::triangle(0, 3, 4))?;
    octahedron.add_simplex(Simplex::triangle(0, 4, 1))?;
    octahedron.add_simplex(Simplex::triangle(5, 2, 1))?;
    octahedron.add_simplex(Simplex::triangle(5, 3, 2))?;
    octahedron.add_simplex(Simplex::triangle(5, 4, 3))?;
    octahedron.add_simplex(Simplex::triangle(5, 1, 4))?;

    println!("  f-vector: {:?}", octahedron.f_vector());
    println!("  Euler characteristic: χ = {}", octahedron.euler_characteristic());
    println!("  (For S²: χ = 2, confirming our triangulation)");

    // =========================================================================
    // Part 4: Chain Complexes and Boundary Operator
    // =========================================================================
    println!("\n\nPart 4: Chain Complex and Boundary Operator");
    println!("────────────────────────────────────────────\n");

    // Build chain complex from triangle
    let chain_complex = ChainComplex::from_simplicial(&triangle_complex)?;

    println!("Chain complex of triangle:");
    println!("  C₀: {} generators (vertices)", chain_complex.dimension(0));
    println!("  C₁: {} generators (edges)", chain_complex.dimension(1));
    println!("  C₂: {} generators (triangle)", chain_complex.dimension(2));

    // Boundary maps
    println!("\nBoundary operator ∂:");
    println!("  ∂[0,1,2] = [1,2] - [0,2] + [0,1]");
    println!("  ∂[0,1] = [1] - [0]");
    println!("  ∂[0] = 0");

    // Verify ∂² = 0
    println!("\nVerification: ∂² = 0");
    println!("  ∂²[0,1,2] = ∂([1,2] - [0,2] + [0,1])");
    println!("            = ([2]-[1]) - ([2]-[0]) + ([1]-[0])");
    println!("            = [2] - [1] - [2] + [0] + [1] - [0]");
    println!("            = 0  ✓");

    // =========================================================================
    // Part 5: Connectivity
    // =========================================================================
    println!("\n\nPart 5: Connectivity Analysis");
    println!("──────────────────────────────\n");

    // Connected complex
    println!("Triangle complex: {} connected component(s)",
             triangle_complex.connected_components().len());

    // Disconnected complex
    let mut disconnected = SimplicialComplex::new();
    disconnected.add_simplex(Simplex::triangle(0, 1, 2))?;
    disconnected.add_simplex(Simplex::edge(5, 6))?;

    println!("Disconnected complex (triangle + separate edge): {} component(s)",
             disconnected.connected_components().len());

    // =========================================================================
    // Part 6: Skeleton and Star
    // =========================================================================
    println!("\n\nPart 6: Skeleton and Star");
    println!("─────────────────────────\n");

    // k-skeleton
    let one_skeleton = tet_complex.k_skeleton(1);
    println!("1-skeleton of tetrahedron (just vertices and edges):");
    println!("  f-vector: {:?}", one_skeleton.f_vector());

    // Star of a vertex
    let star = tet_complex.star(&Simplex::vertex(0))?;
    println!("\nStar of vertex 0 in tetrahedron:");
    println!("  (All simplices containing vertex 0)");
    println!("  f-vector: {:?}", star.f_vector());

    // Link of a vertex
    let link = tet_complex.link(&Simplex::vertex(0))?;
    println!("\nLink of vertex 0 in tetrahedron:");
    println!("  (Boundary of star minus vertex)");
    println!("  f-vector: {:?}", link.f_vector());
    println!("  (Link is a triangle [1,2,3])");

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                    DEMO COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");

    Ok(())
}
