import { Container, Stack, Card, Title, Text, SimpleGrid, Code, Badge } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ExampleCard } from "../components/ExampleCard";

export function Relativistic() {
  // Physical constants
  const C = 299792458; // Speed of light in m/s
  const G = 6.67430e-11; // Gravitational constant
  const SOLAR_MASS = 1.989e30; // kg

  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  // Lorentz factor
  const gamma = (v: number): number => {
    const beta = v / C;
    return 1 / Math.sqrt(1 - beta * beta);
  };

  // Velocity from gamma
  const velocityFromGamma = (g: number): number => {
    return C * Math.sqrt(1 - 1 / (g * g));
  };

  const examples = [
    {
      title: "Lorentz Factor",
      description: "Compute time dilation and length contraction factors",
      category: "Special Relativity",
      code: `// The Lorentz factor γ determines relativistic effects
// γ = 1/√(1 - v²/c²)

const velocity = 0.9 * C;  // 90% speed of light
const gamma = velocity_to_gamma(velocity);

console.log("Velocity:", velocity, "m/s");
console.log("Lorentz factor γ:", gamma);

// Time dilation: Δt' = γΔt
// A clock moving at 0.9c runs 2.29x slower

// Length contraction: L' = L/γ
// A 1 meter rod appears 0.44m long

// Inverse: find velocity from gamma
const v = gamma_to_velocity(gamma);
console.log("Velocity from γ:", v, "m/s");`,
      onRun: simulateExample(() => {
        const velocities = [0.1, 0.5, 0.9, 0.99, 0.999];

        const results = velocities.map(beta => {
          const v = beta * C;
          const g = gamma(v);
          return { beta, gamma: g, timeDilation: g, lengthContraction: 1/g };
        });

        return [
          "Lorentz Factor γ = 1/√(1 - v²/c²)",
          "",
          "v/c      γ        Time Dilation   Length Contraction",
          "─".repeat(55),
          ...results.map(r =>
            `${r.beta.toFixed(3)}    ${r.gamma.toFixed(4).padStart(8)}    ${r.timeDilation.toFixed(4).padStart(8)}x slower    ${r.lengthContraction.toFixed(4).padStart(8)}x shorter`
          ),
          "",
          "At 99.9% c, time runs ~22x slower and lengths contract to ~4.5%"
        ].join('\n');
      })
    },
    {
      title: "Four-Velocity",
      description: "Spacetime velocity vector that maintains constant magnitude",
      category: "Special Relativity",
      code: `// Four-velocity: U = γ(c, vx, vy, vz)
// Its magnitude is always c: |U|² = c²

// Create a four-velocity for motion in x-direction
const vx = 0.8 * C;
const fourVelocity = new WasmFourVelocity(vx, 0, 0);

// Get components
const components = fourVelocity.getComponents();
console.log("U⁰ (time):", components[0]);  // γc
console.log("U¹ (x):", components[1]);     // γvx
console.log("U² (y):", components[2]);     // γvy
console.log("U³ (z):", components[3]);     // γvz

// Verify magnitude
const magnitude = fourVelocity.magnitude();
console.log("Magnitude:", magnitude);  // Should equal c`,
      onRun: simulateExample(() => {
        const vx = 0.8 * C;
        const vy = 0;
        const vz = 0;
        const g = gamma(vx);

        const U = [g * C, g * vx, g * vy, g * vz];

        // Minkowski metric: |U|² = U⁰² - U¹² - U²² - U³²
        const magnitudeSq = U[0]**2 - U[1]**2 - U[2]**2 - U[3]**2;
        const magnitude = Math.sqrt(magnitudeSq);

        return [
          "Four-Velocity: U = γ(c, vx, vy, vz)",
          "",
          `Velocity: vx = ${(vx/C).toFixed(2)}c`,
          `Lorentz factor γ = ${g.toFixed(4)}`,
          "",
          "Four-velocity components:",
          `  U⁰ (time): ${U[0].toExponential(4)} m/s`,
          `  U¹ (x):    ${U[1].toExponential(4)} m/s`,
          `  U² (y):    ${U[2].toExponential(4)} m/s`,
          `  U³ (z):    ${U[3].toExponential(4)} m/s`,
          "",
          `Magnitude |U| = ${magnitude.toExponential(4)} m/s`,
          `Speed of light c = ${C.toExponential(4)} m/s`,
          "",
          "The four-velocity magnitude is invariant and equals c"
        ].join('\n');
      })
    },
    {
      title: "Schwarzschild Metric",
      description: "Spacetime geometry around a non-rotating massive object",
      category: "General Relativity",
      code: `// Schwarzschild metric describes spacetime around a spherical mass
// ds² = -(1-rs/r)c²dt² + (1-rs/r)⁻¹dr² + r²dΩ²

// Schwarzschild radius: rs = 2GM/c²
const mass = SOLAR_MASS;
const metric = new WasmSchwarzschildMetric(mass);

// Get Schwarzschild radius
const rs = metric.schwarzschildRadius();
console.log("Schwarzschild radius:", rs, "meters");
// For the Sun: ~3 km

// Time dilation at radius r
const r = 10 * rs;  // 10 Schwarzschild radii
const timeDilation = metric.timeDilation(r);
console.log("Time dilation at r=10rs:", timeDilation);

// Escape velocity
const escapeVelocity = metric.escapeVelocity(r);
console.log("Escape velocity:", escapeVelocity / C, "c");`,
      onRun: simulateExample(() => {
        const mass = SOLAR_MASS;
        const rs = 2 * G * mass / (C * C);

        const radii = [10, 5, 3, 2, 1.5, 1.1];

        const results = radii.map(factor => {
          const r = factor * rs;
          // Time dilation: √(1 - rs/r)
          const td = Math.sqrt(1 - rs/r);
          // Escape velocity: √(rs*c²/r) = c√(rs/r)
          const vEsc = C * Math.sqrt(rs / r);
          return { factor, timeDilation: td, escapeVelocity: vEsc / C };
        });

        return [
          "Schwarzschild Metric: ds² = -(1-rs/r)c²dt² + ...",
          "",
          `Mass: 1 Solar mass = ${SOLAR_MASS.toExponential(3)} kg`,
          `Schwarzschild radius rs = ${rs.toFixed(0)} m (≈ ${(rs/1000).toFixed(1)} km)`,
          "",
          "Effects at different radii:",
          "r/rs     Time Dilation    Escape Velocity",
          "─".repeat(45),
          ...results.map(r =>
            `${r.factor.toFixed(1).padStart(4)}     ${r.timeDilation.toFixed(4).padStart(10)}      ${r.escapeVelocity.toFixed(4)}c`
          ),
          "",
          "At r = rs (event horizon): time stops, escape velocity = c"
        ].join('\n');
      })
    },
    {
      title: "Light Deflection",
      description: "Gravitational bending of light by massive objects",
      category: "General Relativity",
      code: `// Light bending angle: θ ≈ 4GM/(c²b)
// where b is the impact parameter (closest approach)

// Light grazing the Sun
const mass = SOLAR_MASS;
const impactParameter = 6.96e8;  // Solar radius in meters

const deflection = light_deflection_angle(impactParameter, mass);
console.log("Deflection angle:", deflection, "radians");
console.log("In arcseconds:", deflection * 206265);
// Famous prediction: 1.75 arcseconds

// This was confirmed during the 1919 solar eclipse,
// providing key evidence for General Relativity`,
      onRun: simulateExample(() => {
        const solarRadius = 6.96e8; // meters

        const objects = [
          { name: "Sun", mass: SOLAR_MASS, radius: solarRadius },
          { name: "Earth", mass: 5.97e24, radius: 6.37e6 },
          { name: "Neutron Star", mass: 2 * SOLAR_MASS, radius: 10000 },
          { name: "Black Hole (10 M☉)", mass: 10 * SOLAR_MASS, radius: 2 * G * 10 * SOLAR_MASS / (C*C) * 1.5 }
        ];

        const results = objects.map(obj => {
          // θ = 4GM/(c²b)
          const angle = 4 * G * obj.mass / (C * C * obj.radius);
          const arcsec = angle * 206265;
          return { ...obj, angle, arcsec };
        });

        return [
          "Gravitational Light Deflection: θ = 4GM/(c²b)",
          "",
          "Object           Impact Parameter    Deflection",
          "─".repeat(55),
          ...results.map(r =>
            `${r.name.padEnd(18)} ${(r.radius/1000).toExponential(2).padStart(12)} km    ${r.arcsec < 10 ? r.arcsec.toFixed(4) + '"' : r.angle.toFixed(4) + ' rad'}`
          ),
          "",
          "The Sun's 1.75\" deflection was confirmed in 1919,",
          "validating Einstein's General Relativity"
        ].join('\n');
      })
    },
    {
      title: "Relativistic Particle",
      description: "Track a particle with relativistic mass, energy, and momentum",
      category: "Special Relativity",
      code: `// Create a relativistic particle (electron at 0.99c)
const restMass = 9.109e-31;  // kg (electron)
const velocity = 0.99 * C;

const particle = new WasmRelativisticParticle(
  restMass,
  velocity, 0, 0,  // velocity components
  0, 0, 0          // position
);

// Get relativistic properties
console.log("Rest mass:", particle.getRestMass(), "kg");
console.log("Relativistic mass:", particle.getRelativisticMass(), "kg");
console.log("Kinetic energy:", particle.getKineticEnergy(), "J");
console.log("Total energy:", particle.getTotalEnergy(), "J");
console.log("Momentum:", particle.getMomentum(), "kg·m/s");

// E² = (pc)² + (mc²)²
// At high speeds, E ≈ pc (ultra-relativistic limit)`,
      onRun: simulateExample(() => {
        const restMass = 9.109e-31; // electron mass in kg
        const velocity = 0.99 * C;
        const g = gamma(velocity);

        const relativisticMass = g * restMass;
        const momentum = g * restMass * velocity;
        const restEnergy = restMass * C * C;
        const totalEnergy = g * restMass * C * C;
        const kineticEnergy = totalEnergy - restEnergy;

        // Convert to eV for electron
        const eV = 1.602e-19;
        const MeV = 1e6 * eV;

        return [
          "Relativistic Electron at v = 0.99c",
          "",
          `Rest mass m₀ = ${restMass.toExponential(3)} kg`,
          `Lorentz factor γ = ${g.toFixed(2)}`,
          "",
          "Relativistic Properties:",
          `  Relativistic mass: ${relativisticMass.toExponential(3)} kg (${g.toFixed(1)}x rest mass)`,
          `  Momentum: ${momentum.toExponential(3)} kg·m/s`,
          `  Rest energy: ${(restEnergy/MeV).toFixed(3)} MeV (0.511 MeV)`,
          `  Total energy: ${(totalEnergy/MeV).toFixed(2)} MeV`,
          `  Kinetic energy: ${(kineticEnergy/MeV).toFixed(2)} MeV`,
          "",
          "Energy-momentum relation: E² = (pc)² + (m₀c²)²"
        ].join('\n');
      })
    },
    {
      title: "Relativistic Constants",
      description: "Physical constants used in relativistic calculations",
      category: "Reference",
      code: `const constants = WasmRelativisticConstants.getAll();

console.log("Speed of light c:", constants.c, "m/s");
console.log("Gravitational constant G:", constants.G, "m³/(kg·s²)");
console.log("Solar mass M☉:", constants.solarMass, "kg");
console.log("Earth mass M⊕:", constants.earthMass, "kg");

// Derived quantities
const solarSchwarzschild = 2 * constants.G * constants.solarMass / (constants.c ** 2);
console.log("Solar Schwarzschild radius:", solarSchwarzschild, "m");`,
      onRun: simulateExample(() => {
        const constants = {
          c: C,
          G: G,
          solarMass: SOLAR_MASS,
          earthMass: 5.972e24,
          planckConstant: 6.626e-34,
          electronMass: 9.109e-31,
          protonMass: 1.673e-27
        };

        const solarRs = 2 * G * SOLAR_MASS / (C * C);
        const earthRs = 2 * G * constants.earthMass / (C * C);

        return [
          "Fundamental Constants for Relativity:",
          "",
          `Speed of light c = ${C.toExponential(6)} m/s`,
          `Gravitational constant G = ${G.toExponential(4)} m³/(kg·s²)`,
          "",
          "Astronomical Constants:",
          `  Solar mass M☉ = ${SOLAR_MASS.toExponential(3)} kg`,
          `  Earth mass M⊕ = ${constants.earthMass.toExponential(3)} kg`,
          "",
          "Particle Physics:",
          `  Electron mass = ${constants.electronMass.toExponential(3)} kg`,
          `  Proton mass = ${constants.protonMass.toExponential(3)} kg`,
          "",
          "Derived Quantities:",
          `  Solar Schwarzschild radius = ${(solarRs/1000).toFixed(2)} km`,
          `  Earth Schwarzschild radius = ${(earthRs*1000).toFixed(2)} mm`
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Relativistic Physics</Title>
          <Text size="lg" c="dimmed">
            Special and general relativity with spacetime geometry
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Overview</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The <Code>amari-relativistic</Code> module provides tools for relativistic calculations,
              from special relativity kinematics to general relativistic spacetime metrics.
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Special Relativity</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Lorentz transformations</li>
                  <li>Four-vectors (velocity, momentum)</li>
                  <li>Time dilation and length contraction</li>
                  <li>Relativistic energy-momentum</li>
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">General Relativity</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Schwarzschild metric</li>
                  <li>Gravitational time dilation</li>
                  <li>Light deflection</li>
                  <li>Black hole physics</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Key Equations</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <CodeHighlight
              code={`Special Relativity:
  Lorentz factor:     γ = 1/√(1 - v²/c²)
  Time dilation:      Δt' = γΔt
  Length contraction: L' = L/γ
  Energy-momentum:    E² = (pc)² + (m₀c²)²

General Relativity:
  Schwarzschild radius:  rs = 2GM/c²
  Gravitational time:    dτ = √(1 - rs/r) dt
  Light deflection:      θ = 4GM/(c²b)`}
              language="plaintext"
            />
          </Card.Section>
        </Card>

        <Title order={2}>Interactive Examples</Title>

        <SimpleGrid cols={1} spacing="lg">
          {examples.map((example, i) => (
            <ExampleCard
              key={i}
              title={example.title}
              description={example.description}
              code={example.code}
              onRun={example.onRun}
              badge={<Badge size="sm" variant="light">{example.category}</Badge>}
            />
          ))}
        </SimpleGrid>
      </Stack>
    </Container>
  );
}
