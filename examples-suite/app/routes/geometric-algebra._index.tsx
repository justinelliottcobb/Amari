import type { MetaFunction } from "@remix-run/node";
import { H1, P } from "jadis";
import { Layout } from "~/components/Layout";
import { ExampleCard } from "~/components/ExampleCard";
import { useState, useEffect } from "react";

export const meta: MetaFunction = () => {
  return [
    { title: "Geometric Algebra Examples - Amari Library" },
    { name: "description", content: "Interactive examples of geometric algebra operations using the Amari library" },
  ];
};

export default function GeometricAlgebra() {
  const [amari, setAmari] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadAmari() {
      try {
        const wasmModule = await import("@amari/core");
        await wasmModule.default();
        setAmari(wasmModule);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load Amari WASM module');
      }
    }

    loadAmari();
  }, []);

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
      onRun: async () => {
        if (!amari) throw new Error("Amari module not loaded");

        const coeffs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        const mv = amari.WasmMultivector.fromCoefficients(coeffs);

        const scalar = mv.getCoefficient(0);
        const e1 = mv.getCoefficient(1);
        const grade1 = mv.gradeProjection(1);

        const result = [
          `Scalar component: ${scalar}`,
          `e1 component: ${e1}`,
          `Grade 1 coefficients: [${grade1.getCoefficients().join(', ')}]`
        ].join('\n');

        return result;
      }
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
      onRun: async () => {
        if (!amari) throw new Error("Amari module not loaded");

        const v1 = amari.WasmMultivector.basisVector(0);
        const v2 = amari.WasmMultivector.basisVector(1);
        const product = v1.geometricProduct(v2);

        const result = [
          `v1 coefficients: [${v1.getCoefficients().join(', ')}]`,
          `v2 coefficients: [${v2.getCoefficients().join(', ')}]`,
          `Product coefficients: [${product.getCoefficients().join(', ')}]`,
          `e12 component: ${product.getCoefficient(4)}`
        ].join('\n');

        return result;
      }
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
      onRun: async () => {
        if (!amari) throw new Error("Amari module not loaded");

        const vector = amari.WasmMultivector.basisVector(0);
        const e1 = amari.WasmMultivector.basisVector(0);
        const e2 = amari.WasmMultivector.basisVector(1);
        const bivector = e1.outerProduct(e2);

        const angle = Math.PI / 2;
        const rotor = amari.WasmRotor.fromBivector(bivector, angle);
        const rotated = rotor.apply(vector);

        return [
          `Original vector: [${vector.getCoefficients().join(', ')}]`,
          `Rotated vector: [${rotated.getCoefficients().join(', ')}]`
        ].join('\n');
      }
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
      onRun: async () => {
        if (!amari) throw new Error("Amari module not loaded");

        const coeffs1 = [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
        const coeffs2 = [0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];

        const v1 = amari.WasmMultivector.fromCoefficients(coeffs1);
        const v2 = amari.WasmMultivector.fromCoefficients(coeffs2);

        const inner = v1.innerProduct(v2);
        const innerScalar = inner.getCoefficient(0);
        const outer = v1.outerProduct(v2);

        return [
          `v1: [${coeffs1.join(', ')}]`,
          `v2: [${coeffs2.join(', ')}]`,
          `Inner product (scalar): ${innerScalar}`,
          `Outer product: [${outer.getCoefficients().join(', ')}]`
        ].join('\n');
      }
    }
  ];

  if (error) {
    return (
      <Layout>
        <div className="p-8">
          <div className="max-w-4xl mx-auto">
            <H1>Geometric Algebra Examples</H1>
            <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 mt-4">
              <P className="text-destructive">Error loading Amari WASM module: {error}</P>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  if (!amari) {
    return (
      <Layout>
        <div className="p-8">
          <div className="max-w-4xl mx-auto">
            <H1>Geometric Algebra Examples</H1>
            <div className="bg-muted rounded-lg p-4 mt-4">
              <P>Loading Amari WASM module...</P>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
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
    </Layout>
  );
}