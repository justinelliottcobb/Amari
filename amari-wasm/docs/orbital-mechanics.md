# High-Precision Orbital Mechanics (amari-relativistic)

Spacetime algebra (Cl(1,3)) with high-precision arithmetic for spacecraft trajectory calculations in browsers.

## Quick Start

```typescript
import init, {
  WasmSpacetimeVector,
  WasmFourVelocity,
  WasmRelativisticParticle,
  WasmSchwarzschildMetric
} from '@justinelliottcobb/amari-wasm';

async function spacecraftSimulation() {
  await init();

  // Create Earth's gravitational field
  const earth = WasmSchwarzschildMetric.earth();

  // Spacecraft at 400km altitude (ISS orbit)
  const altitude = 400e3;
  const earthRadius = 6.371e6;
  const position = new Float64Array([earthRadius + altitude, 0.0, 0.0]);
  const velocity = new Float64Array([0.0, 7.67e3, 0.0]);

  // Create spacecraft with high-precision arithmetic
  const spacecraft = WasmRelativisticParticle.new(
    position, velocity,
    0.0,    // No charge
    1000.0, // 1000 kg
    0.0     // No magnetic charge
  );

  // Propagate orbit
  const orbitalPeriod = 5580.0; // ~93 minutes
  const trajectory = spacecraft.propagate_trajectory(earth, orbitalPeriod, 60.0);

  console.log(`Trajectory: ${trajectory.length} points`);
  console.log(`Position error: ${spacecraft.position_error()} meters`);

  // Clean up
  earth.free();
  spacecraft.free();
}

spacecraftSimulation();
```

## Features

- **Pure Rust Backend**: dashu-powered arithmetic with no native dependencies
- **Universal Deployment**: Same precision guarantees across desktop, web, and edge
- **Orbital-Grade Tolerance**: Configurable precision for critical trajectory calculations

## Use Cases

- **Spacecraft Trajectory Planning**: High-precision orbital mechanics in web applications
- **Relativistic Physics**: Spacetime algebra simulations
- **Gravitational Modeling**: Schwarzschild metric calculations
