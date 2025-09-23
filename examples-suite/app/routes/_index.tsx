import type { MetaFunction } from "@remix-run/node";
import { H1, H2, P, Card, CardHeader, CardBody, Button } from "jadis";
import { Layout } from "~/components/Layout";

export const meta: MetaFunction = () => {
  return [
    { title: "Amari Mathematical Computing Library - API Examples" },
    { name: "description", content: "Interactive examples and documentation for the Amari library" },
  ];
};

export default function Index() {
  return (
    <Layout>
      <div className="p-8">
        <div className="max-w-4xl mx-auto">
          <H1>Amari Mathematical Computing Library</H1>
          <P className="text-lg mb-8 text-muted-foreground">
            Interactive API examples and documentation for exotic number systems and algebraic structures
          </P>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <Card>
              <CardHeader>
                <H2>Core Mathematics</H2>
              </CardHeader>
              <CardBody>
                <P className="mb-4">Explore the fundamental algebraic structures that power the Amari library:</P>
                <ul className="space-y-2 text-sm">
                  <li>• Geometric Algebra (Clifford Algebra)</li>
                  <li>• Tropical Algebra (Max-Plus Semiring)</li>
                  <li>• Dual Number Automatic Differentiation</li>
                  <li>• Information Geometry</li>
                </ul>
                <Button href="/geometric-algebra" className="mt-4">
                  Start with Geometric Algebra
                </Button>
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <H2>Advanced Systems</H2>
              </CardHeader>
              <CardBody>
                <P className="mb-4">Discover high-performance computing and integration features:</P>
                <ul className="space-y-2 text-sm">
                  <li>• WebGPU Acceleration</li>
                  <li>• TropicalDualClifford Fusion</li>
                  <li>• Cellular Automata</li>
                  <li>• Edge Computing</li>
                </ul>
                <Button href="/webgpu" className="mt-4">
                  Explore WebGPU
                </Button>
              </CardBody>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <H2>Quick Start</H2>
            </CardHeader>
            <CardBody>
              <P className="mb-4">
                The Amari library provides a unified framework for mathematical computing with exotic number systems.
                Each module is designed to work independently or as part of the integrated fusion system.
              </P>
              <div className="flex gap-4">
                <Button href="/playground">
                  Interactive Playground
                </Button>
                <Button href="/api-reference" variant="outline">
                  API Reference
                </Button>
              </div>
            </CardBody>
          </Card>
        </div>
      </div>
    </Layout>
  );
}