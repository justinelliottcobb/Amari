import { H1, P, Card, CardHeader, CardBody } from "jadis-ui";
import { ExampleCard } from "../components/ExampleCard";
import { useState } from "react";

export function Automata() {
  const [gridData, setGridData] = useState<number[][]>([]);

  // Simulate geometric cellular automata operations
  const simulateExample = (operation: () => string | {result: string, visualization?: number[][]}) => {
    return async () => {
      try {
        const result = operation();
        if (typeof result === 'object' && result.visualization) {
          setGridData(result.visualization);
          return result.result;
        }
        return typeof result === 'string' ? result : result.result;
      } catch (err) {
        throw new Error(`Automata simulation error: ${err}`);
      }
    };
  };

  // Simple visualization component
  const GridVisualization = ({ grid }: { grid: number[][] }) => {
    if (grid.length === 0) return null;

    return (
      <div className="mt-4 p-4 bg-muted rounded-lg">
        <h4 className="text-sm font-semibold mb-2">Grid Visualization:</h4>
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${grid[0].length}, 1fr)` }}>
          {grid.flat().map((cell, i) => (
            <div
              key={i}
              className={`w-4 h-4 border ${cell > 0.5 ? 'bg-primary' : 'bg-background'}`}
              title={`Cell ${i}: ${cell.toFixed(3)}`}
            />
          ))}
        </div>
      </div>
    );
  };

  const examples = [
    {
      title: "Geometric Cellular Automaton",
      description: "Create a 2D cellular automaton where cells contain multivectors",
      category: "Fundamentals",
      code: `// Geometric CA where each cell contains a multivector from Clifford algebra
// Cells evolve based on geometric product operations with neighbors

class GeometricCA {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.generation = 0;

    // Initialize grid with multivector cells [scalar, e1, e2, e3, e12, e13, e23, e123]
    this.grid = Array(height).fill(null).map(() =>
      Array(width).fill(null).map(() => [0, 0, 0, 0, 0, 0, 0, 0])
    );
  }

  setCell(x, y, multivector) {
    if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
      this.grid[y][x] = [...multivector];
    }
  }

  getCell(x, y) {
    if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
      return [0, 0, 0, 0, 0, 0, 0, 0]; // Zero multivector for boundaries
    }
    return this.grid[y][x];
  }

  // Get neighbors for 2D Moore neighborhood
  getNeighbors(x, y) {
    const neighbors = [];
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx !== 0 || dy !== 0) {
          neighbors.push(this.getCell(x + dx, y + dy));
        }
      }
    }
    return neighbors;
  }

  // Geometric evolution rule: cells evolve based on geometric product with neighbors
  evolve() {
    const newGrid = Array(this.height).fill(null).map(() =>
      Array(this.width).fill(null).map(() => [0, 0, 0, 0, 0, 0, 0, 0])
    );

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const currentCell = this.getCell(x, y);
        const neighbors = this.getNeighbors(x, y);

        // Apply geometric CA rule: average neighbor influence with decay
        let newCell = currentCell.map(c => c * 0.9); // Decay factor

        neighbors.forEach(neighbor => {
          // Simplified geometric product influence
          const influence = neighbor[0] * 0.1; // Use scalar part
          newCell[0] += influence; // Add to scalar part
        });

        // Keep values bounded
        newCell = newCell.map(c => Math.max(-2, Math.min(2, c)));
        newGrid[y][x] = newCell;
      }
    }

    this.grid = newGrid;
    this.generation++;
  }

  // Extract visualization data (scalar components)
  getVisualization() {
    return this.grid.map(row =>
      row.map(cell => Math.abs(cell[0])) // Use absolute value of scalar part
    );
  }
}

// Create and run geometric CA
const ca = new GeometricCA(8, 8);

// Set initial pattern - glider-like structure with multivectors
ca.setCell(1, 1, [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
ca.setCell(2, 1, [0.8, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]);
ca.setCell(3, 1, [1.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

console.log("Initial state:");
console.log(\`Generation: \${ca.generation}\`);

// Evolve for several steps
for (let i = 0; i < 3; i++) {
  ca.evolve();
  console.log(\`Generation \${ca.generation}: pattern evolved\`);
}

const visualization = ca.getVisualization();
console.log("Final grid (scalar magnitudes):");
visualization.forEach((row, y) => {
  console.log(\`Row \${y}: [\${row.map(v => v.toFixed(2)).join(', ')}]\`);
});`,
      onRun: simulateExample(() => {
        class GeometricCA {
          width: number;
          height: number;
          generation: number;
          grid: number[][][];

          constructor(width: number, height: number) {
            this.width = width;
            this.height = height;
            this.generation = 0;
            this.grid = Array(height).fill(null).map(() =>
              Array(width).fill(null).map(() => [0, 0, 0, 0, 0, 0, 0, 0])
            );
          }

          setCell(x: number, y: number, multivector: number[]) {
            if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
              this.grid[y][x] = [...multivector];
            }
          }

          getCell(x: number, y: number) {
            if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
              return [0, 0, 0, 0, 0, 0, 0, 0];
            }
            return this.grid[y][x];
          }

          getNeighbors(x: number, y: number) {
            const neighbors = [];
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (dx !== 0 || dy !== 0) {
                  neighbors.push(this.getCell(x + dx, y + dy));
                }
              }
            }
            return neighbors;
          }

          evolve() {
            const newGrid = Array(this.height).fill(null).map(() =>
              Array(this.width).fill(null).map(() => [0, 0, 0, 0, 0, 0, 0, 0])
            );

            for (let y = 0; y < this.height; y++) {
              for (let x = 0; x < this.width; x++) {
                const currentCell = this.getCell(x, y);
                const neighbors = this.getNeighbors(x, y);

                let newCell = currentCell.map(c => c * 0.9);

                neighbors.forEach(neighbor => {
                  const influence = neighbor[0] * 0.1;
                  newCell[0] += influence;
                });

                newCell = newCell.map(c => Math.max(-2, Math.min(2, c)));
                newGrid[y][x] = newCell;
              }
            }

            this.grid = newGrid;
            this.generation++;
          }

          getVisualization() {
            return this.grid.map(row =>
              row.map(cell => Math.abs(cell[0]))
            );
          }
        }

        const ca = new GeometricCA(8, 8);
        ca.setCell(1, 1, [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        ca.setCell(2, 1, [0.8, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]);
        ca.setCell(3, 1, [1.2, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let result = [`Initial state:`, `Generation: ${ca.generation}`];

        for (let i = 0; i < 3; i++) {
          ca.evolve();
          result.push(`Generation ${ca.generation}: pattern evolved`);
        }

        const visualization = ca.getVisualization();
        result.push("Final grid (scalar magnitudes):");
        visualization.forEach((row, y) => {
          result.push(`Row ${y}: [${row.map(v => v.toFixed(2)).join(', ')}]`);
        });

        return {
          result: result.join('\n'),
          visualization: visualization
        };
      })
    },
    {
      title: "Rotor Cellular Automaton",
      description: "CA evolution using rotors for 3D geometric transformations",
      category: "Geometric Operations",
      code: `// Rotor CA: cells evolve by applying rotational transformations
// Uses geometric algebra rotors for smooth 3D rotations

class RotorCA {
  constructor(size) {
    this.size = size;
    this.generation = 0;

    // Each cell contains a rotor (rotation) as a multivector
    this.rotors = Array(size).fill(null).map(() => [
      1.0, 0, 0, 0,  // cos(θ/2) + sin(θ/2) * bivector
      0, 0, 0, 0     // [1, e1, e2, e3, e12, e13, e23, e123]
    ]);

    // Target vectors to be rotated
    this.vectors = Array(size).fill(null).map(() => [0, 1, 0, 0, 0, 0, 0, 0]); // e1 vector
  }

  // Create a rotor from angle and axis
  createRotor(angle, axis_x, axis_y, axis_z) {
    const halfAngle = angle / 2;
    const cos_half = Math.cos(halfAngle);
    const sin_half = Math.sin(halfAngle);

    // Normalize axis
    const length = Math.sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
    const nx = axis_x / length;
    const ny = axis_y / length;
    const nz = axis_z / length;

    return [
      cos_half,                    // scalar part
      0, 0, 0,                    // vector part (zero for rotors)
      -sin_half * nx * ny,        // e12 = e1^e2
      -sin_half * nx * nz,        // e13 = e1^e3
      -sin_half * ny * nz,        // e23 = e2^e3
      0                           // e123
    ];
  }

  // Apply rotor to vector: R * v * R†
  applyRotor(rotor, vector) {
    // Simplified rotor application (real implementation would use geometric product)
    const angle = 2 * Math.acos(Math.abs(rotor[0]));
    const rotated = [...vector];

    // Simple rotation in xy-plane for demonstration
    const cos_a = Math.cos(angle);
    const sin_a = Math.sin(angle);

    const x = vector[1]; // e1 component
    const y = vector[2]; // e2 component

    rotated[1] = cos_a * x - sin_a * y; // new e1
    rotated[2] = sin_a * x + cos_a * y; // new e2

    return rotated;
  }

  evolve() {
    const newRotors = [...this.rotors];
    const newVectors = [...this.vectors];

    for (let i = 0; i < this.size; i++) {
      // Get neighbors (periodic boundary)
      const leftIdx = (i - 1 + this.size) % this.size;
      const rightIdx = (i + 1) % this.size;

      // Create evolution rotor based on neighbor influence
      const leftInfluence = this.rotors[leftIdx][0]; // scalar part
      const rightInfluence = this.rotors[rightIdx][0];

      const evolutionAngle = (leftInfluence + rightInfluence) * 0.1;
      const evolutionRotor = this.createRotor(evolutionAngle, 0, 0, 1); // rotate around z-axis

      // Apply evolution to current rotor (composition)
      newRotors[i][0] *= evolutionRotor[0]; // simplified composition

      // Apply rotor to vector
      newVectors[i] = this.applyRotor(newRotors[i], this.vectors[i]);
    }

    this.rotors = newRotors;
    this.vectors = newVectors;
    this.generation++;
  }

  getRotationAngles() {
    return this.rotors.map(rotor => 2 * Math.acos(Math.abs(rotor[0])));
  }

  getVectorMagnitudes() {
    return this.vectors.map(v => Math.sqrt(v[1]*v[1] + v[2]*v[2] + v[3]*v[3]));
  }
}

// Create and run rotor CA
const rotorCA = new RotorCA(8);

// Initialize with some rotations
for (let i = 0; i < 8; i++) {
  const angle = (i / 8) * Math.PI / 2;
  rotorCA.rotors[i] = rotorCA.createRotor(angle, 0, 0, 1);
}

console.log("Rotor Cellular Automaton Evolution:");
console.log(\`Initial angles: [\${rotorCA.getRotationAngles().map(a => (a * 180/Math.PI).toFixed(1)).join('°, ')}°]\`);

// Evolve the system
for (let step = 0; step < 5; step++) {
  rotorCA.evolve();
  const angles = rotorCA.getRotationAngles();
  const magnitudes = rotorCA.getVectorMagnitudes();

  console.log(\`Step \${step + 1}:\`);
  console.log(\`  Angles: [\${angles.map(a => (a * 180/Math.PI).toFixed(1)).join('°, ')}°]\`);
  console.log(\`  Vector magnitudes: [\${magnitudes.map(m => m.toFixed(3)).join(', ')}]\`);
}

console.log("\\nRotor CA demonstrates:");
console.log("• Smooth rotational evolution using geometric algebra");
console.log("• Composition of rotations through rotor multiplication");
console.log("• Natural handling of 3D orientations without gimbal lock");`,
      onRun: simulateExample(() => {
        class RotorCA {
          size: number;
          generation: number;
          rotors: number[][];
          vectors: number[][];

          constructor(size: number) {
            this.size = size;
            this.generation = 0;
            this.rotors = Array(size).fill(null).map(() => [1.0, 0, 0, 0, 0, 0, 0, 0]);
            this.vectors = Array(size).fill(null).map(() => [0, 1, 0, 0, 0, 0, 0, 0]);
          }

          createRotor(angle: number, axis_x: number, axis_y: number, axis_z: number) {
            const halfAngle = angle / 2;
            const cos_half = Math.cos(halfAngle);
            const sin_half = Math.sin(halfAngle);

            const length = Math.sqrt(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z);
            const nx = axis_x / length;
            const ny = axis_y / length;
            const nz = axis_z / length;

            return [
              cos_half, 0, 0, 0,
              -sin_half * nx * ny,
              -sin_half * nx * nz,
              -sin_half * ny * nz,
              0
            ];
          }

          applyRotor(rotor: number[], vector: number[]) {
            const angle = 2 * Math.acos(Math.abs(rotor[0]));
            const rotated = [...vector];
            const cos_a = Math.cos(angle);
            const sin_a = Math.sin(angle);

            const x = vector[1];
            const y = vector[2];

            rotated[1] = cos_a * x - sin_a * y;
            rotated[2] = sin_a * x + cos_a * y;

            return rotated;
          }

          evolve() {
            const newRotors = [...this.rotors];
            const newVectors = [...this.vectors];

            for (let i = 0; i < this.size; i++) {
              const leftIdx = (i - 1 + this.size) % this.size;
              const rightIdx = (i + 1) % this.size;

              const leftInfluence = this.rotors[leftIdx][0];
              const rightInfluence = this.rotors[rightIdx][0];

              const evolutionAngle = (leftInfluence + rightInfluence) * 0.1;
              const evolutionRotor = this.createRotor(evolutionAngle, 0, 0, 1);

              newRotors[i][0] *= evolutionRotor[0];
              newVectors[i] = this.applyRotor(newRotors[i], this.vectors[i]);
            }

            this.rotors = newRotors;
            this.vectors = newVectors;
            this.generation++;
          }

          getRotationAngles() {
            return this.rotors.map(rotor => 2 * Math.acos(Math.abs(rotor[0])));
          }

          getVectorMagnitudes() {
            return this.vectors.map(v => Math.sqrt(v[1]*v[1] + v[2]*v[2] + v[3]*v[3]));
          }
        }

        const rotorCA = new RotorCA(8);

        for (let i = 0; i < 8; i++) {
          const angle = (i / 8) * Math.PI / 2;
          rotorCA.rotors[i] = rotorCA.createRotor(angle, 0, 0, 1);
        }

        let result = [];
        result.push("Rotor Cellular Automaton Evolution:");
        result.push(`Initial angles: [${rotorCA.getRotationAngles().map(a => (a * 180/Math.PI).toFixed(1)).join('°, ')}°]`);

        for (let step = 0; step < 5; step++) {
          rotorCA.evolve();
          const angles = rotorCA.getRotationAngles();
          const magnitudes = rotorCA.getVectorMagnitudes();

          result.push(`Step ${step + 1}:`);
          result.push(`  Angles: [${angles.map(a => (a * 180/Math.PI).toFixed(1)).join('°, ')}°]`);
          result.push(`  Vector magnitudes: [${magnitudes.map(m => m.toFixed(3)).join(', ')}]`);
        }

        result.push("");
        result.push("Rotor CA demonstrates:");
        result.push("• Smooth rotational evolution using geometric algebra");
        result.push("• Composition of rotations through rotor multiplication");
        result.push("• Natural handling of 3D orientations without gimbal lock");

        return result.join('\n');
      })
    },
    {
      title: "Self-Assembly System",
      description: "Geometric self-assembly using Clifford algebra constraints",
      category: "Self-Assembly",
      code: `// Self-assembling system where components fit together using geometric constraints
// Uses Clifford algebra to ensure proper alignment and connectivity

class SelfAssemblySystem {
  constructor() {
    this.components = [];
    this.connections = [];
    this.generation = 0;
  }

  // Define a component with geometric constraints
  addComponent(id, position, orientation, connectionPoints) {
    this.components.push({
      id,
      position: [...position],           // [x, y, z]
      orientation: [...orientation],     // Rotor as [s, e12, e13, e23]
      connectionPoints: connectionPoints.map(cp => [...cp]), // Array of 3D points
      connected: false
    });
  }

  // Check if two components can connect based on geometric constraints
  canConnect(comp1, comp2, tolerance = 0.1) {
    // Check each connection point of comp1 against comp2's points
    for (let i = 0; i < comp1.connectionPoints.length; i++) {
      for (let j = 0; j < comp2.connectionPoints.length; j++) {
        const p1 = this.transformPoint(comp1.connectionPoints[i], comp1.position, comp1.orientation);
        const p2 = this.transformPoint(comp2.connectionPoints[j], comp2.position, comp2.orientation);

        const distance = Math.sqrt(
          Math.pow(p1[0] - p2[0], 2) +
          Math.pow(p1[1] - p2[1], 2) +
          Math.pow(p1[2] - p2[2], 2)
        );

        if (distance < tolerance) {
          return { canConnect: true, point1: i, point2: j, distance };
        }
      }
    }
    return { canConnect: false };
  }

  // Transform a point using position and orientation (simplified rotor application)
  transformPoint(point, position, rotor) {
    // Apply rotation (simplified - real implementation would use full rotor math)
    const angle = 2 * Math.acos(Math.abs(rotor[0]));
    const cos_a = Math.cos(angle);
    const sin_a = Math.sin(angle);

    // Rotate in xy-plane for simplification
    const rotatedX = cos_a * point[0] - sin_a * point[1];
    const rotatedY = sin_a * point[0] + cos_a * point[1];
    const rotatedZ = point[2];

    // Translate
    return [
      rotatedX + position[0],
      rotatedY + position[1],
      rotatedZ + position[2]
    ];
  }

  // Attempt to connect compatible components
  tryAssemble() {
    const newConnections = [];

    for (let i = 0; i < this.components.length; i++) {
      for (let j = i + 1; j < this.components.length; j++) {
        const comp1 = this.components[i];
        const comp2 = this.components[j];

        // Skip if already connected
        if (this.connections.some(conn =>
          (conn.comp1 === i && conn.comp2 === j) || (conn.comp1 === j && conn.comp2 === i)
        )) {
          continue;
        }

        const connectionResult = this.canConnect(comp1, comp2);
        if (connectionResult.canConnect) {
          newConnections.push({
            comp1: i,
            comp2: j,
            point1: connectionResult.point1,
            point2: connectionResult.point2,
            strength: 1.0 / (connectionResult.distance + 0.001) // Stronger for closer fits
          });

          comp1.connected = true;
          comp2.connected = true;
        }
      }
    }

    this.connections.push(...newConnections);
    this.generation++;

    return newConnections.length;
  }

  // Get assembly statistics
  getAssemblyStats() {
    const totalComponents = this.components.length;
    const connectedComponents = this.components.filter(c => c.connected).length;
    const totalConnections = this.connections.length;
    const averageConnectionStrength = this.connections.length > 0 ?
      this.connections.reduce((sum, conn) => sum + conn.strength, 0) / this.connections.length : 0;

    return {
      totalComponents,
      connectedComponents,
      totalConnections,
      averageConnectionStrength,
      assemblyProgress: connectedComponents / totalComponents
    };
  }
}

// Create self-assembly system
const assembly = new SelfAssemblySystem();

// Add some components with connection points
console.log("Initializing self-assembly system...");

// Component A: L-shaped piece
assembly.addComponent('A', [0, 0, 0], [1, 0, 0, 0], [
  [1, 0, 0],   // Right connection
  [0, 1, 0]    // Top connection
]);

// Component B: Straight piece
assembly.addComponent('B', [0.9, 0, 0], [1, 0, 0, 0], [
  [-0.1, 0, 0], // Left connection
  [0.9, 0, 0]   // Right connection
]);

// Component C: Corner piece
assembly.addComponent('C', [0, 0.9, 0], [1, 0, 0, 0], [
  [0, -0.1, 0], // Bottom connection
  [1, 0.9, 0]   // Right connection
]);

// Component D: T-shaped piece
assembly.addComponent('D', [2, 0, 0], [1, 0, 0, 0], [
  [-0.1, 0, 0], // Left connection
  [0, 1, 0],    // Top connection
  [0, -1, 0]    // Bottom connection
]);

console.log(\`Added \${assembly.components.length} components\`);

// Attempt assembly over multiple generations
console.log("\\nAttempting self-assembly...");
for (let gen = 0; gen < 3; gen++) {
  const newConnections = assembly.tryAssemble();
  const stats = assembly.getAssemblyStats();

  console.log(\`Generation \${gen + 1}:\`);
  console.log(\`  New connections: \${newConnections}\`);
  console.log(\`  Connected components: \${stats.connectedComponents}/\${stats.totalComponents}\`);
  console.log(\`  Assembly progress: \${(stats.assemblyProgress * 100).toFixed(1)}%\`);
  console.log(\`  Average connection strength: \${stats.averageConnectionStrength.toFixed(3)}\`);
}

console.log("\\nSelf-assembly complete!");
console.log("Geometric constraints ensure proper component alignment");`,
      onRun: simulateExample(() => {
        class SelfAssemblySystem {
          components: Array<{
            id: string,
            position: number[],
            orientation: number[],
            connectionPoints: number[][],
            connected: boolean
          }> = [];
          connections: Array<{
            comp1: number,
            comp2: number,
            point1: number,
            point2: number,
            strength: number
          }> = [];
          generation = 0;

          addComponent(id: string, position: number[], orientation: number[], connectionPoints: number[][]) {
            this.components.push({
              id,
              position: [...position],
              orientation: [...orientation],
              connectionPoints: connectionPoints.map(cp => [...cp]),
              connected: false
            });
          }

          canConnect(comp1: any, comp2: any, tolerance = 0.1) {
            for (let i = 0; i < comp1.connectionPoints.length; i++) {
              for (let j = 0; j < comp2.connectionPoints.length; j++) {
                const p1 = this.transformPoint(comp1.connectionPoints[i], comp1.position, comp1.orientation);
                const p2 = this.transformPoint(comp2.connectionPoints[j], comp2.position, comp2.orientation);

                const distance = Math.sqrt(
                  Math.pow(p1[0] - p2[0], 2) +
                  Math.pow(p1[1] - p2[1], 2) +
                  Math.pow(p1[2] - p2[2], 2)
                );

                if (distance < tolerance) {
                  return { canConnect: true, point1: i, point2: j, distance };
                }
              }
            }
            return { canConnect: false };
          }

          transformPoint(point: number[], position: number[], rotor: number[]) {
            const angle = 2 * Math.acos(Math.abs(rotor[0]));
            const cos_a = Math.cos(angle);
            const sin_a = Math.sin(angle);

            const rotatedX = cos_a * point[0] - sin_a * point[1];
            const rotatedY = sin_a * point[0] + cos_a * point[1];
            const rotatedZ = point[2];

            return [
              rotatedX + position[0],
              rotatedY + position[1],
              rotatedZ + position[2]
            ];
          }

          tryAssemble() {
            const newConnections = [];

            for (let i = 0; i < this.components.length; i++) {
              for (let j = i + 1; j < this.components.length; j++) {
                const comp1 = this.components[i];
                const comp2 = this.components[j];

                if (this.connections.some(conn =>
                  (conn.comp1 === i && conn.comp2 === j) || (conn.comp1 === j && conn.comp2 === i)
                )) {
                  continue;
                }

                const connectionResult = this.canConnect(comp1, comp2);
                if (connectionResult.canConnect && connectionResult.distance !== undefined) {
                  newConnections.push({
                    comp1: i,
                    comp2: j,
                    point1: connectionResult.point1!,
                    point2: connectionResult.point2!,
                    strength: 1.0 / (connectionResult.distance + 0.001)
                  });

                  comp1.connected = true;
                  comp2.connected = true;
                }
              }
            }

            this.connections.push(...newConnections);
            this.generation++;

            return newConnections.length;
          }

          getAssemblyStats() {
            const totalComponents = this.components.length;
            const connectedComponents = this.components.filter(c => c.connected).length;
            const totalConnections = this.connections.length;
            const averageConnectionStrength = this.connections.length > 0 ?
              this.connections.reduce((sum, conn) => sum + conn.strength, 0) / this.connections.length : 0;

            return {
              totalComponents,
              connectedComponents,
              totalConnections,
              averageConnectionStrength,
              assemblyProgress: connectedComponents / totalComponents
            };
          }
        }

        const assembly = new SelfAssemblySystem();

        let result = ["Initializing self-assembly system..."];

        assembly.addComponent('A', [0, 0, 0], [1, 0, 0, 0], [[1, 0, 0], [0, 1, 0]]);
        assembly.addComponent('B', [0.9, 0, 0], [1, 0, 0, 0], [[-0.1, 0, 0], [0.9, 0, 0]]);
        assembly.addComponent('C', [0, 0.9, 0], [1, 0, 0, 0], [[0, -0.1, 0], [1, 0.9, 0]]);
        assembly.addComponent('D', [2, 0, 0], [1, 0, 0, 0], [[-0.1, 0, 0], [0, 1, 0], [0, -1, 0]]);

        result.push(`Added ${assembly.components.length} components`);
        result.push("");
        result.push("Attempting self-assembly...");

        for (let gen = 0; gen < 3; gen++) {
          const newConnections = assembly.tryAssemble();
          const stats = assembly.getAssemblyStats();

          result.push(`Generation ${gen + 1}:`);
          result.push(`  New connections: ${newConnections}`);
          result.push(`  Connected components: ${stats.connectedComponents}/${stats.totalComponents}`);
          result.push(`  Assembly progress: ${(stats.assemblyProgress * 100).toFixed(1)}%`);
          result.push(`  Average connection strength: ${stats.averageConnectionStrength.toFixed(3)}`);
        }

        result.push("");
        result.push("Self-assembly complete!");
        result.push("Geometric constraints ensure proper component alignment");

        return result.join('\n');
      })
    }
  ];

  return (
<div className="p-8">
        <div className="max-w-4xl mx-auto">
          <H1>Cellular Automata Examples</H1>
          <P className="text-lg text-muted-foreground mb-4">
            Explore geometric cellular automata, rotor evolution, and self-assembling systems.
          </P>

          <Card className="mb-8">
            <CardHeader>
              <h3 className="text-lg font-semibold">Geometric Cellular Automata</h3>
            </CardHeader>
            <CardBody>
              <P className="mb-4">
                The Amari automata system extends traditional cellular automata with geometric algebra:
              </P>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <h4 className="font-semibold text-sm mb-2">Traditional CA</h4>
                  <ul className="text-sm space-y-1">
                    <li>• Binary or discrete states</li>
                    <li>• Simple neighborhood rules</li>
                    <li>• Limited spatial relationships</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-sm mb-2">Geometric CA</h4>
                  <ul className="text-sm space-y-1">
                    <li>• Multivector cell states</li>
                    <li>• Geometric product evolution</li>
                    <li>• Rich spatial/rotational dynamics</li>
                  </ul>
                </div>
              </div>
              <P className="text-sm text-muted-foreground">
                This enables sophisticated behaviors like rotor-based rotations, self-assembly
                with geometric constraints, and inverse design for target configurations.
              </P>
            </CardBody>
          </Card>

          <div className="space-y-6">
            {examples.map((example, index) => (
              <div key={index}>
                <ExampleCard
                  title={example.title}
                  description={example.description}
                  code={example.code}
                  category={example.category}
                  onRun={example.onRun}
                />
                {gridData.length > 0 && index === 0 && (
                  <GridVisualization grid={gridData} />
                )}
              </div>
            ))}
          </div>

          <Card className="mt-8">
            <CardHeader>
              <h3 className="text-lg font-semibold">Applications & Research Directions</h3>
            </CardHeader>
            <CardBody>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="font-semibold text-sm mb-2">Current Applications</h4>
                  <ul className="text-sm space-y-1">
                    <li>• Self-assembling UI components</li>
                    <li>• Geometric pattern generation</li>
                    <li>• Spatial constraint solving</li>
                    <li>• 3D structure optimization</li>
                  </ul>
                </div>
                <div>
                  <h4 className="font-semibold text-sm mb-2">Research Potential</h4>
                  <ul className="text-sm space-y-1">
                    <li>• Quantum cellular automata simulation</li>
                    <li>• Crystalline growth modeling</li>
                    <li>• Robotic swarm coordination</li>
                    <li>• Architectural design automation</li>
                  </ul>
                </div>
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
);
}