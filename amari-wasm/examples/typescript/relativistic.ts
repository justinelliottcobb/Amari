/**
 * Relativistic Physics Example using Amari WASM
 *
 * This example demonstrates:
 * - Spacetime vector operations in Minkowski space
 * - Four-velocity calculations and Lorentz factors
 * - Relativistic particle dynamics
 * - Geodesic integration in Schwarzschild spacetime
 * - Orbital mechanics around massive objects
 */

import init, {
    WasmSpacetimeVector,
    WasmFourVelocity,
    WasmRelativisticParticle,
    WasmSchwarzschildMetric,
    WasmGeodesicIntegrator,
    WasmRelativisticConstants,
    light_deflection_angle,
    velocity_to_gamma,
    gamma_to_velocity,
    validate_relativistic_module
} from "../pkg/amari_wasm.js";

async function runRelativisticPhysicsDemo() {
    console.log("🚀 Initializing Amari Relativistic Physics WASM Module...");

    // Initialize the WASM module
    await init();

    // Validate module loaded correctly
    if (!validate_relativistic_module()) {
        console.error("❌ Failed to validate relativistic module");
        return;
    }

    console.log("✅ Relativistic physics module loaded successfully\n");

    // ===== Physical Constants =====
    console.log("📊 Physical Constants:");
    console.log(`Speed of light: ${WasmRelativisticConstants.speed_of_light.toExponential(3)} m/s`);
    console.log(`Gravitational constant: ${WasmRelativisticConstants.gravitational_constant.toExponential(3)} m³/kg·s²`);
    console.log(`Solar mass: ${WasmRelativisticConstants.solar_mass.toExponential(3)} kg`);
    console.log(`Earth mass: ${WasmRelativisticConstants.earth_mass.toExponential(3)} kg\n`);

    // ===== Spacetime Vector Operations =====
    console.log("🌌 Spacetime Vector Operations:");

    // Create spacetime vectors
    const event1 = new WasmSpacetimeVector(1.0, 2.0, 3.0, 4.0);  // (ct, x, y, z)
    const event2 = new WasmSpacetimeVector(2.0, 1.0, 1.0, 1.0);

    console.log(`Event 1: ${event1.to_string()}`);
    console.log(`Event 2: ${event2.to_string()}`);

    // Minkowski inner product
    const interval = event1.minkowski_dot(event2);
    console.log(`Spacetime interval: ${interval.toFixed(6)}`);

    // Check causal structure
    console.log(`Event 1 is timelike: ${event1.is_timelike()}`);
    console.log(`Event 1 is spacelike: ${event1.is_spacelike()}`);
    console.log(`Event 1 is null: ${event1.is_null()}\n`);

    // ===== Lorentz Factors and Velocities =====
    console.log("⚡ Lorentz Factors and Velocities:");

    // Convert between velocity and Lorentz factor
    const testVelocities = [0.1, 0.5, 0.9, 0.99, 0.999];

    for (const beta of testVelocities) {
        const velocity = beta * WasmRelativisticConstants.speed_of_light;
        const gamma = velocity_to_gamma(velocity);
        const reconstructedVelocity = gamma_to_velocity(gamma);

        console.log(`β = ${beta}: γ = ${gamma.toFixed(6)}, reconstructed v = ${(reconstructedVelocity / WasmRelativisticConstants.speed_of_light).toFixed(6)}c`);
    }
    console.log();

    // ===== Four-Velocity Calculations =====
    console.log("🎯 Four-Velocity Calculations:");

    // Create four-velocity from 3-velocity
    const velocity3d = [0.5e8, 0.3e8, 0.1e8];  // m/s
    const fourVelocity = WasmFourVelocity.from_velocity(velocity3d[0], velocity3d[1], velocity3d[2]);

    console.log(`Four-velocity: ${fourVelocity.to_string()}`);
    console.log(`Lorentz factor γ: ${fourVelocity.gamma().toFixed(6)}`);
    console.log(`Rapidity: ${fourVelocity.rapidity().toFixed(6)}`);
    console.log(`Spatial velocity magnitude: ${(fourVelocity.spatial_velocity_magnitude() / 1e8).toFixed(3)} × 10⁸ m/s`);
    console.log(`Is normalized: ${fourVelocity.is_normalized()}\n`);

    // ===== Relativistic Particle Dynamics =====
    console.log("🌟 Relativistic Particle Dynamics:");

    // Create a relativistic particle (electron-like)
    const electronMass = 9.109e-31;  // kg
    const electronCharge = -1.602e-19;  // C

    const particle = new WasmRelativisticParticle(
        1e6, 2e6, 3e6,          // Position (m)
        1e7, 0.5e7, 0.2e7,      // Velocity (m/s)
        0.5,                     // Spin
        electronMass,            // Mass
        electronCharge           // Charge
    );

    console.log(`Particle: ${particle.to_string()}`);
    console.log(`Total energy: ${(particle.total_energy() / 1.602e-19).toExponential(3)} eV`);
    console.log(`Kinetic energy: ${(particle.kinetic_energy() / 1.602e-19).toExponential(3)} eV`);
    console.log(`Momentum magnitude: ${particle.momentum_magnitude().toExponential(3)} kg⋅m/s`);

    const particleFourVel = particle.four_velocity();
    console.log(`Particle γ: ${particleFourVel.gamma().toFixed(6)}\n`);

    // ===== High-Energy Particle =====
    console.log("⚡ High-Energy Particle Example:");

    // Create particle with specific kinetic energy (cosmic ray proton)
    const protonMass = 1.673e-27;  // kg
    const highEnergyParticle = WasmRelativisticParticle.with_energy(
        0, 0, 0,                // Position at origin
        1, 0, 0,                // Direction along x-axis
        1e-11,                  // 10 GeV kinetic energy (J)
        protonMass,             // Proton mass
        1.602e-19               // Proton charge
    );

    console.log(`High-energy particle: ${highEnergyParticle.to_string()}`);
    const highEnergyFourVel = highEnergyParticle.four_velocity();
    console.log(`High-energy γ: ${highEnergyFourVel.gamma().toFixed(2)}`);
    console.log(`High-energy velocity: ${(highEnergyFourVel.spatial_velocity_magnitude() / WasmRelativisticConstants.speed_of_light).toFixed(6)}c\n`);

    // ===== Schwarzschild Spacetime =====
    console.log("🌑 Schwarzschild Spacetime:");

    // Create Schwarzschild metrics
    const sunMetric = WasmSchwarzschildMetric.sun();
    const earthMetric = WasmSchwarzschildMetric.earth();

    console.log(`Sun Schwarzschild radius: ${sunMetric.schwarzschild_radius().toFixed(0)} m`);
    console.log(`Earth Schwarzschild radius: ${earthMetric.schwarzschild_radius().toExponential(3)} m`);

    // Custom black hole (10 solar masses)
    const blackHole = WasmSchwarzschildMetric.from_mass(10 * WasmRelativisticConstants.solar_mass);
    console.log(`10 M☉ black hole radius: ${blackHole.schwarzschild_radius().toFixed(0)} m`);

    // Effective potential for circular orbits
    const orbitRadius = 6 * sunMetric.schwarzschild_radius();  // 6 Rs
    const angularMomentum = 1e42;  // kg⋅m²/s
    const effectivePotential = sunMetric.effective_potential(orbitRadius, angularMomentum);
    console.log(`Effective potential at 6Rs: ${effectivePotential.toExponential(3)}\n`);

    // ===== Light Deflection =====
    console.log("💫 Light Deflection by Massive Objects:");

    // Calculate light deflection for photon grazing the Sun
    const sunRadius = 6.96e8;  // m
    const deflectionAngle = light_deflection_angle(sunRadius, WasmRelativisticConstants.solar_mass);
    const deflectionArcsec = deflectionAngle * 206265;  // Convert to arcseconds

    console.log(`Light deflection by Sun: ${deflectionArcsec.toFixed(3)} arcseconds`);
    console.log(`(Einstein's prediction: 1.75 arcseconds)\n`);

    // ===== Geodesic Integration =====
    console.log("🛸 Geodesic Integration - Spacecraft Trajectory:");

    // Create geodesic integrator for solar system
    const integrator = WasmGeodesicIntegrator.with_schwarzschild(sunMetric);

    // Create spacecraft particle
    const spacecraft = new WasmRelativisticParticle(
        1.5e11, 0, 0,           // Position: 1 AU from Sun
        0, 30000, 0,            // Velocity: ~30 km/s (Earth orbital speed)
        0,                      // No spin
        1000,                   // 1000 kg spacecraft
        0                       // Neutral charge
    );

    console.log(`Initial spacecraft: ${spacecraft.to_string()}`);

    // Propagate for one Earth year
    const yearInSeconds = 365.25 * 24 * 3600;
    const timeStep = yearInSeconds / 1000;  // 1000 steps per year

    try {
        console.log("Integrating spacecraft trajectory...");
        const trajectory = integrator.propagate_particle(spacecraft, yearInSeconds / 4, timeStep);
        console.log(`✅ Computed ${trajectory.length} trajectory points`);

        // Analyze trajectory
        if (trajectory.length > 0) {
            const firstPoint = trajectory[0];
            const lastPoint = trajectory[trajectory.length - 1];

            console.log(`Time span: ${(lastPoint.time / (24 * 3600)).toFixed(1)} days`);

            const finalPos = lastPoint.position;
            const finalDistance = Math.sqrt(finalPos.x * finalPos.x + finalPos.y * finalPos.y + finalPos.z * finalPos.z);
            console.log(`Final distance from Sun: ${(finalDistance / 1.5e11).toFixed(3)} AU`);
        }
    } catch (error) {
        console.error(`Integration failed: ${error}`);
    }

    console.log("\n🎉 Relativistic physics demonstration complete!");
    console.log("This example showed spacetime algebra, particle dynamics,");
    console.log("gravitational effects, and geodesic integration using");
    console.log("the Amari relativistic physics WASM bindings.");
}

// Run the demo when the page loads
if (typeof window !== 'undefined') {
    // Browser environment
    window.addEventListener('load', runRelativisticPhysicsDemo);
} else {
    // Node.js environment
    runRelativisticPhysicsDemo().catch(console.error);
}

export { runRelativisticPhysicsDemo };