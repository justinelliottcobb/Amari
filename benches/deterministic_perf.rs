//! Performance benchmarks for deterministic physics module
//!
//! Measures overhead of deterministic operations compared to native f32.
//! Expected overhead: 10-20% for most operations.

#![cfg(feature = "deterministic")]

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use amari::deterministic::ga2d::{DetRotor2, DetVector2};
use amari::deterministic::DetF32;

fn bench_scalar_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_operations");

    // Native f32 baseline
    group.bench_function("native_f32_add", |b| {
        b.iter(|| {
            let a = black_box(1.5f32);
            let b = black_box(2.5f32);
            black_box(a + b)
        })
    });

    // Deterministic DetF32
    group.bench_function("det_f32_add", |b| {
        b.iter(|| {
            let a = black_box(DetF32::from_f32(1.5));
            let b = black_box(DetF32::from_f32(2.5));
            black_box(a + b)
        })
    });

    // Native mul
    group.bench_function("native_f32_mul", |b| {
        b.iter(|| {
            let a = black_box(1.5f32);
            let b = black_box(2.5f32);
            black_box(a * b)
        })
    });

    // Deterministic mul
    group.bench_function("det_f32_mul", |b| {
        b.iter(|| {
            let a = black_box(DetF32::from_f32(1.5));
            let b = black_box(DetF32::from_f32(2.5));
            black_box(a * b)
        })
    });

    group.finish();
}

fn bench_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt");

    group.bench_function("native_f32_sqrt", |b| {
        b.iter(|| {
            let x = black_box(12345.679f32);
            black_box(x.sqrt())
        })
    });

    group.bench_function("det_f32_sqrt", |b| {
        b.iter(|| {
            let x = black_box(DetF32::from_f32(12345.679));
            black_box(x.sqrt())
        })
    });

    group.finish();
}

fn bench_trig(c: &mut Criterion) {
    let mut group = c.benchmark_group("trigonometry");

    // Native sin
    group.bench_function("native_f32_sin", |b| {
        b.iter(|| {
            let x = black_box(1.23f32);
            black_box(x.sin())
        })
    });

    // Deterministic sin
    group.bench_function("det_f32_sin", |b| {
        b.iter(|| {
            let x = black_box(DetF32::from_f32(1.23));
            black_box(x.sin())
        })
    });

    // Native cos
    group.bench_function("native_f32_cos", |b| {
        b.iter(|| {
            let x = black_box(1.23f32);
            black_box(x.cos())
        })
    });

    // Deterministic cos
    group.bench_function("det_f32_cos", |b| {
        b.iter(|| {
            let x = black_box(DetF32::from_f32(1.23));
            black_box(x.cos())
        })
    });

    // Native atan2
    group.bench_function("native_f32_atan2", |b| {
        b.iter(|| {
            let y = black_box(3.0f32);
            let x = black_box(4.0f32);
            black_box(y.atan2(x))
        })
    });

    // Deterministic atan2
    group.bench_function("det_f32_atan2", |b| {
        b.iter(|| {
            let y = black_box(DetF32::from_f32(3.0));
            let x = black_box(DetF32::from_f32(4.0));
            black_box(DetF32::atan2(y, x))
        })
    });

    group.finish();
}

fn bench_vector_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    // Deterministic vector addition
    group.bench_function("det_vector_add", |b| {
        b.iter(|| {
            let v1 = black_box(DetVector2::from_f32(1.5, 2.5));
            let v2 = black_box(DetVector2::from_f32(3.5, 4.5));
            black_box(v1 + v2)
        })
    });

    // Deterministic dot product
    group.bench_function("det_vector_dot", |b| {
        b.iter(|| {
            let v1 = black_box(DetVector2::from_f32(1.5, 2.5));
            let v2 = black_box(DetVector2::from_f32(3.5, 4.5));
            black_box(v1.dot(v2))
        })
    });

    // Deterministic magnitude
    group.bench_function("det_vector_magnitude", |b| {
        b.iter(|| {
            let v = black_box(DetVector2::from_f32(3.0, 4.0));
            black_box(v.magnitude())
        })
    });

    // Deterministic normalize
    group.bench_function("det_vector_normalize", |b| {
        b.iter(|| {
            let v = black_box(DetVector2::from_f32(3.0, 4.0));
            black_box(v.normalize())
        })
    });

    group.finish();
}

fn bench_rotor_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotor_operations");

    // Rotor from angle
    group.bench_function("det_rotor_from_angle", |b| {
        b.iter(|| {
            let angle = black_box(DetF32::from_f32(0.75));
            black_box(DetRotor2::from_angle(angle))
        })
    });

    // Rotor composition
    group.bench_function("det_rotor_compose", |b| {
        b.iter(|| {
            let r1 = black_box(DetRotor2::from_angle(DetF32::from_f32(0.5)));
            let r2 = black_box(DetRotor2::from_angle(DetF32::from_f32(0.3)));
            black_box(r1 * r2)
        })
    });

    // Rotor transform vector
    group.bench_function("det_rotor_transform", |b| {
        b.iter(|| {
            let r = black_box(DetRotor2::from_angle(DetF32::from_f32(0.75)));
            let v = black_box(DetVector2::from_f32(1.0, 0.0));
            black_box(r.transform(v))
        })
    });

    group.finish();
}

fn bench_physics_sim(c: &mut Criterion) {
    let mut group = c.benchmark_group("physics_simulation");

    // Simulate 100 physics steps
    group.bench_function("det_physics_100_steps", |b| {
        b.iter(|| {
            let dt = DetF32::from_f32(1.0 / 60.0);
            let gravity = DetVector2::from_f32(0.0, -9.8);
            let angular_vel = DetF32::PI * DetF32::from_f32(0.25);

            let mut position = black_box(DetVector2::from_f32(0.0, 10.0));
            let mut velocity = black_box(DetVector2::from_f32(15.0, 20.0));
            let mut rotation = black_box(DetRotor2::IDENTITY);

            for _ in 0..100 {
                // Update velocity with gravity
                velocity = velocity + gravity * dt;
                // Update position
                position = position + velocity * dt;
                // Update rotation
                let delta_angle = angular_vel * dt;
                rotation = rotation * DetRotor2::from_angle(delta_angle);
            }

            black_box((position, velocity, rotation))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_ops,
    bench_sqrt,
    bench_trig,
    bench_vector_ops,
    bench_rotor_ops,
    bench_physics_sim
);
criterion_main!(benches);
