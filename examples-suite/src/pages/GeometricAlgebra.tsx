import { H1, P } from "jadis-ui";
import { ExampleCard } from "../components/ExampleCard";
export function GeometricAlgebra() {
  // Simulate operations for demo purposes
  const simulateExample = (title: string, operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  const examples = [
    {
      title: "Basic Multivector Creation",
      description: "Create and manipulate multivectors in 3D Euclidean space (3,0,0 signature)",
      category: "Fundamentals",
      code: `// Create a 3D Euclidean multivector from coefficients
// [scalar, e1, e2, e3, e12, e13, e23, e123]
const coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
const mv = amari.WasmMultivector.fromCoefficients(coeffs);

// Access individual components
const scalar = mv.getCoefficient(0);
const e1 = mv.getCoefficient(1);
const grade1 = mv.gradeProjection(1);

console.log("Scalar component:", scalar);
console.log("e1 component:", e1);
console.log("Grade 1 coefficients:", grade1.getCoefficients());`,
      onRun: simulateExample("multivector-creation", () => {
        // Simulate multivector operations
        const coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        const scalar = coeffs[0];
        const e1 = coeffs[1];
        const grade1 = [coeffs[1], coeffs[2], coeffs[3]];

        return [
          `Scalar component: ${scalar}`,
          `e1 component: ${e1}`,
          `Grade 1 coefficients: [${grade1.join(', ')}]`
        ].join('\n');
      })
    },
    {
      title: "Geometric Product",
      description: "Compute the geometric product of two multivectors",
      category: "Operations",
      code: `// Create two basis vectors
const v1 = amari.WasmMultivector.basisVector(0); // e1
const v2 = amari.WasmMultivector.basisVector(1); // e2

// Geometric product: e1 * e2 = e1∧e2 (bivector)
const product = v1.geometricProduct(v2);

console.log("v1 coefficients:", v1.getCoefficients());
console.log("v2 coefficients:", v2.getCoefficients());
console.log("Product coefficients:", product.getCoefficients());
console.log("e12 component:", product.getCoefficient(4));`,
      onRun: simulateExample("geometric-product", () => {
        // Simulate geometric product: e1 * e2 = e12
        const v1 = [0, 1, 0, 0, 0, 0, 0, 0]; // e1
        const v2 = [0, 0, 1, 0, 0, 0, 0, 0]; // e2
        const product = [0, 0, 0, 0, 1, 0, 0, 0]; // e12

        return [
          `v1 coefficients: [${v1.join(', ')}]`,
          `v2 coefficients: [${v2.join(', ')}]`,
          `Product coefficients: [${product.join(', ')}]`,
          `e12 component: ${product[4]}`
        ].join('\n');
      })
    },
    {
      title: "Rotor Rotations",
      description: "Create a rotor and use it to rotate a vector",
      category: "Applications",
      code: `// Create a vector to rotate (e1)
const vector = amari.WasmMultivector.basisVector(0);

// Create bivector for rotation plane (e1∧e2)
const e1 = amari.WasmMultivector.basisVector(0);
const e2 = amari.WasmMultivector.basisVector(1);
const bivector = e1.outerProduct(e2);

// Create rotor for 90-degree rotation
const angle = Math.PI / 2; // 90 degrees
const rotor = amari.WasmRotor.fromBivector(bivector, angle);

// Apply rotation
const rotated = rotor.apply(vector);

console.log("Original vector:", vector.getCoefficients());
console.log("Rotated vector:", rotated.getCoefficients());`,
      onRun: simulateExample("rotor-rotation", () => {
        // Simulate 90-degree rotation of e1 → e2
        const original = [0, 1, 0, 0, 0, 0, 0, 0]; // e1
        const rotated = [0, 0, 1, 0, 0, 0, 0, 0];  // e2 (90° rotation)

        return [
          `Original vector: [${original.join(', ')}]`,
          `Rotated vector: [${rotated.join(', ')}]`
        ].join('\n');
      })
    },
    {
      title: "Inner and Outer Products",
      description: "Explore the inner product (contraction) and outer product (wedge)",
      category: "Operations",
      code: `// Create two vectors with mixed components
const coeffs1 = [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]; // 1e1 + 2e2 + 3e3
const coeffs2 = [0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0]; // 4e1 + 5e2 + 6e3

const v1 = amari.WasmMultivector.fromCoefficients(coeffs1);
const v2 = amari.WasmMultivector.fromCoefficients(coeffs2);

// Inner product (dot product for vectors)
const inner = v1.innerProduct(v2);
const innerScalar = inner.getCoefficient(0);

// Outer product (wedge product - produces bivector)
const outer = v1.outerProduct(v2);

console.log("v1:", coeffs1);
console.log("v2:", coeffs2);
console.log("Inner product (scalar):", innerScalar);
console.log("Outer product:", outer.getCoefficients());`,
      onRun: simulateExample("inner-outer-products", () => {
        const coeffs1 = [0, 1, 2, 3, 0, 0, 0, 0];
        const coeffs2 = [0, 4, 5, 6, 0, 0, 0, 0];

        // Inner product: 1*4 + 2*5 + 3*6 = 32
        const innerScalar = 1*4 + 2*5 + 3*6;

        // Outer product bivector components: 1*5-2*4, 1*6-3*4, 2*6-3*5
        const outer = [0, 0, 0, 0, -3, -6, 3, 0];

        return [
          `v1: [${coeffs1.join(', ')}]`,
          `v2: [${coeffs2.join(', ')}]`,
          `Inner product (scalar): ${innerScalar}`,
          `Outer product: [${outer.join(', ')}]`
        ].join('\n');
      })
    }
  ];

  return (
<div className="p-8">
        <div className="max-w-4xl mx-auto">
          <H1>Geometric Algebra Examples</H1>
          <P className="text-lg text-muted-foreground mb-8">
            Explore multivectors, geometric products, and rotors in Clifford algebra with interactive examples.
          </P>

          <div className="space-y-6">
            {examples.map((example, index) => (
              <ExampleCard
                key={index}
                title={example.title}
                description={example.description}
                code={example.code}
                category={example.category}
                onRun={example.onRun}
              />
            ))}
          </div>
        </div>
      </div>
);
}