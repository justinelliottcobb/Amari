//! Performance benchmarks for amari-topology operations
//!
//! Measures critical operations for simplicial complexes and homology.

use amari_topology::{compute_homology, Simplex, SimplicialComplex};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark simplex creation
fn bench_simplex_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex_creation");

    group.bench_function("0_simplex", |b| {
        b.iter(|| black_box(Simplex::new(vec![black_box(0)])))
    });

    group.bench_function("1_simplex", |b| {
        b.iter(|| black_box(Simplex::new(vec![black_box(0), black_box(1)])))
    });

    group.bench_function("2_simplex", |b| {
        b.iter(|| black_box(Simplex::new(vec![black_box(0), black_box(1), black_box(2)])))
    });

    group.bench_function("3_simplex", |b| {
        b.iter(|| {
            black_box(Simplex::new(vec![
                black_box(0),
                black_box(1),
                black_box(2),
                black_box(3),
            ]))
        })
    });

    group.finish();
}

/// Benchmark simplicial complex construction
fn bench_complex_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_construction");

    group.bench_function("empty", |b| b.iter(|| black_box(SimplicialComplex::new())));

    // Triangle
    group.bench_function("triangle", |b| {
        b.iter(|| {
            let mut complex = SimplicialComplex::new();
            complex.add_simplex(Simplex::new(vec![0]));
            complex.add_simplex(Simplex::new(vec![1]));
            complex.add_simplex(Simplex::new(vec![2]));
            complex.add_simplex(Simplex::new(vec![0, 1]));
            complex.add_simplex(Simplex::new(vec![1, 2]));
            complex.add_simplex(Simplex::new(vec![0, 2]));
            complex.add_simplex(Simplex::new(vec![0, 1, 2]));
            black_box(complex)
        })
    });

    // Tetrahedron
    group.bench_function("tetrahedron", |b| {
        b.iter(|| {
            let mut complex = SimplicialComplex::new();
            // Vertices
            for i in 0..4usize {
                complex.add_simplex(Simplex::new(vec![i]));
            }
            // Edges
            for i in 0..4usize {
                for j in i + 1..4usize {
                    complex.add_simplex(Simplex::new(vec![i, j]));
                }
            }
            // Faces
            complex.add_simplex(Simplex::new(vec![0, 1, 2]));
            complex.add_simplex(Simplex::new(vec![0, 1, 3]));
            complex.add_simplex(Simplex::new(vec![0, 2, 3]));
            complex.add_simplex(Simplex::new(vec![1, 2, 3]));
            // Tetrahedron
            complex.add_simplex(Simplex::new(vec![0, 1, 2, 3]));
            black_box(complex)
        })
    });

    group.finish();
}

/// Benchmark homology computation
fn bench_homology(c: &mut Criterion) {
    let mut group = c.benchmark_group("homology");

    // Circle (S^1)
    let mut circle = SimplicialComplex::new();
    let n = 6usize;
    for i in 0..n {
        circle.add_simplex(Simplex::new(vec![i]));
        circle.add_simplex(Simplex::new(vec![i, (i + 1) % n]));
    }

    group.bench_function("circle", |b| {
        b.iter(|| black_box(compute_homology(black_box(&circle))))
    });

    // Sphere (triangulated, hollow)
    let mut sphere = SimplicialComplex::new();
    // Use octahedron as triangulation of S^2
    for i in 0..6usize {
        sphere.add_simplex(Simplex::new(vec![i]));
    }
    // Edges
    let edges: [(usize, usize); 12] = [
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (4, 3),
        (3, 5),
        (5, 2),
    ];
    for (i, j) in edges {
        sphere.add_simplex(Simplex::new(vec![i, j]));
    }
    // Triangular faces
    let faces: [(usize, usize, usize); 8] = [
        (0, 2, 4),
        (0, 4, 3),
        (0, 3, 5),
        (0, 5, 2),
        (1, 2, 4),
        (1, 4, 3),
        (1, 3, 5),
        (1, 5, 2),
    ];
    for (i, j, k) in faces {
        sphere.add_simplex(Simplex::new(vec![i, j, k]));
    }

    group.bench_function("sphere", |b| {
        b.iter(|| black_box(compute_homology(black_box(&sphere))))
    });

    // Torus-like complex (simplified)
    let mut torus = SimplicialComplex::new();
    let n = 4usize;
    let m = 4usize;
    // Vertices
    for i in 0..(n * m) {
        torus.add_simplex(Simplex::new(vec![i]));
    }
    // Edges (grid with wraparound)
    for i in 0..n {
        for j in 0..m {
            let v = i * m + j;
            let v_right = i * m + (j + 1) % m;
            let v_down = ((i + 1) % n) * m + j;
            torus.add_simplex(Simplex::new(vec![v, v_right]));
            torus.add_simplex(Simplex::new(vec![v, v_down]));
        }
    }

    group.bench_function("torus_grid", |b| {
        b.iter(|| black_box(compute_homology(black_box(&torus))))
    });

    group.finish();
}

/// Benchmark simplex operations
fn bench_simplex_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex_operations");

    let simplex = Simplex::new(vec![0, 1, 2, 3]);

    group.bench_function("dimension", |b| b.iter(|| black_box(simplex.dimension())));

    group.bench_function("vertices", |b| b.iter(|| black_box(simplex.vertices())));

    group.bench_function("contains_vertex", |b| {
        b.iter(|| black_box(simplex.contains_vertex(black_box(2))))
    });

    group.finish();
}

criterion_group!(
    topology_benchmarks,
    bench_simplex_creation,
    bench_complex_construction,
    bench_homology,
    bench_simplex_operations,
);

criterion_main!(topology_benchmarks);
