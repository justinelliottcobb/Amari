//! Integration tests for amari-dynamics
//!
//! These tests verify the complete integration of the dynamical systems
//! functionality across multiple modules.

use amari_core::Multivector;
use amari_dynamics::{
    flow::{DynamicalSystem, HarmonicOscillator},
    phantom::{Autonomous, ContinuousTime, TypedSystem, UnknownChaos, UnknownStability},
    solver::{ODESolver, RungeKutta4},
};

/// Test harmonic oscillator with RK4 over multiple periods
#[test]
fn test_harmonic_oscillator_multiple_periods() {
    let system = HarmonicOscillator::new(1.0);
    let solver = RungeKutta4::new();

    // Initial condition: x = 1, v = 0
    let mut initial = Multivector::<2, 0, 0>::zero();
    initial.set(1, 1.0);
    initial.set(2, 0.0);

    // Integrate for 5 periods
    let periods = 5;
    let period = 2.0 * std::f64::consts::PI;
    let t_final = periods as f64 * period;

    let trajectory = solver.solve(&system, initial, 0.0, t_final, 50000).unwrap();

    // Check periodicity - should return to initial state after each period
    let final_state = trajectory.final_state().unwrap();
    let x_final = final_state.get(1);
    let v_final = final_state.get(2);

    // After 5 complete periods, should be back near (1, 0)
    assert!(
        (x_final - 1.0).abs() < 0.01,
        "After {} periods, x = {} (expected ~1.0)",
        periods,
        x_final
    );
    assert!(
        v_final.abs() < 0.01,
        "After {} periods, v = {} (expected ~0.0)",
        periods,
        v_final
    );
}

/// Test energy conservation in harmonic oscillator
#[test]
fn test_energy_conservation() {
    let omega = 2.0;
    let system = HarmonicOscillator::new(omega);
    let solver = RungeKutta4::new();

    // Initial condition: x = 2, v = 1
    let mut initial = Multivector::<2, 0, 0>::zero();
    initial.set(1, 2.0);
    initial.set(2, 1.0);

    // Energy = 0.5 * v^2 + 0.5 * ω^2 * x^2
    let compute_energy = |state: &Multivector<2, 0, 0>| {
        let x = state.get(1);
        let v = state.get(2);
        0.5 * v * v + 0.5 * omega * omega * x * x
    };

    let initial_energy = compute_energy(&initial);
    let trajectory = solver.solve(&system, initial, 0.0, 20.0, 20000).unwrap();

    // Sample energy at regular intervals
    for (i, (_, state)) in trajectory.iter().enumerate() {
        if i % 1000 == 0 {
            let energy = compute_energy(state);
            let relative_error = (energy - initial_energy).abs() / initial_energy;
            assert!(
                relative_error < 1e-6,
                "Energy drift at step {}: {} vs {} (error: {})",
                i,
                energy,
                initial_energy,
                relative_error
            );
        }
    }
}

/// Test trajectory interpolation
#[test]
fn test_trajectory_interpolation() {
    let system = HarmonicOscillator::new(1.0);
    let solver = RungeKutta4::new();

    let mut initial = Multivector::<2, 0, 0>::zero();
    initial.set(1, 1.0);

    let trajectory = solver.solve(&system, initial, 0.0, 2.0, 100).unwrap();

    // Interpolate at the midpoint
    let t_mid = 1.0;
    let interpolated = trajectory.interpolate(t_mid).unwrap();

    // For cos(t), at t=1, x ≈ cos(1) ≈ 0.5403
    let x = interpolated.get(1);
    assert!(
        (x - 1.0_f64.cos()).abs() < 0.1,
        "Interpolated x at t=1: {} (expected ~{})",
        x,
        1.0_f64.cos()
    );
}

/// Test trajectory resampling
#[test]
fn test_trajectory_resample() {
    let system = HarmonicOscillator::new(1.0);
    let solver = RungeKutta4::new();

    let mut initial = Multivector::<2, 0, 0>::zero();
    initial.set(1, 1.0);

    let trajectory = solver.solve(&system, initial, 0.0, 1.0, 1000).unwrap();

    // Resample to 10 points
    let resampled = trajectory.resample(10).unwrap();

    assert_eq!(resampled.len(), 10);

    // Check that resampled times are uniformly spaced
    let expected_dt = 1.0 / 9.0;
    for i in 1..10 {
        let dt = resampled.times[i] - resampled.times[i - 1];
        assert!(
            (dt - expected_dt).abs() < 1e-10,
            "Non-uniform spacing at index {}: {} vs {}",
            i,
            dt,
            expected_dt
        );
    }
}

/// Test typed system wrapper
#[test]
fn test_typed_system() {
    let system = HarmonicOscillator::new(1.0);

    // Wrap in typed system
    type MyTypedSystem =
        TypedSystem<HarmonicOscillator, Autonomous, ContinuousTime, UnknownStability, UnknownChaos>;

    let typed: MyTypedSystem = TypedSystem::new(system);

    // Check properties
    assert!(typed.is_autonomous());
    assert!(typed.is_continuous());
    assert!(!typed.stability_verified());
    assert!(!typed.chaos_verified());

    // Can still use the inner system
    let mut state = Multivector::<2, 0, 0>::zero();
    state.set(1, 1.0);

    let vf = typed.inner().vector_field(&state).unwrap();
    assert!((vf.get(1) - 0.0).abs() < 1e-10); // dx/dt = v = 0
    assert!((vf.get(2) - (-1.0)).abs() < 1e-10); // dv/dt = -x = -1
}

/// Test Jacobian computation
#[test]
fn test_jacobian() {
    let system = HarmonicOscillator::new(1.0);

    let mut state = Multivector::<2, 0, 0>::zero();
    state.set(1, 1.0);
    state.set(2, 0.5);

    let jacobian = system.jacobian(&state).unwrap();

    // For harmonic oscillator with ω=1:
    // dx/dt = v, dv/dt = -x
    // J = | 0  1 |
    //     |-1  0 |
    // In the full 4x4 matrix (Cl(2,0,0)), we expect:
    // - J[1,2] ≈ 1 (∂(dx/dt)/∂v)
    // - J[2,1] ≈ -1 (∂(dv/dt)/∂x)

    assert_eq!(jacobian.len(), 16); // 4x4 for Cl(2,0,0)

    // Check key entries (using row-major indexing)
    // J[1,2] = jacobian[1*4 + 2] = jacobian[6]
    // J[2,1] = jacobian[2*4 + 1] = jacobian[9]
    assert!(
        (jacobian[6] - 1.0).abs() < 1e-5,
        "J[1,2] = {} (expected 1.0)",
        jacobian[6]
    );
    assert!(
        (jacobian[9] - (-1.0)).abs() < 1e-5,
        "J[2,1] = {} (expected -1.0)",
        jacobian[9]
    );
}
