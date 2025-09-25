import { H1, H2, P, Card, CardHeader, CardBody, Button } from "jadis-ui";

export function Home() {
  return (
<div style={{ padding: '2rem' }}>
        <div>
          <H1>Amari Mathematical Computing Library</H1>
          <P style={{ fontSize: '1.125rem', marginBottom: '2rem', opacity: 0.7 }}>
            Interactive API examples and documentation for exotic number systems and algebraic structures
          </P>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem', marginBottom: '2rem' }}>
            <Card>
              <CardHeader>
                <H2>Core Mathematics</H2>
              </CardHeader>
              <CardBody>
                <P style={{ marginBottom: '1rem' }}>Explore the fundamental algebraic structures that power the Amari library:</P>
                <ul style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>
                  <li>• Geometric Algebra (Clifford Algebra)</li>
                  <li>• Tropical Algebra (Max-Plus Semiring)</li>
                  <li>• Dual Number Automatic Differentiation</li>
                  <li>• Information Geometry</li>
                </ul>
                <Button href="/geometric-algebra" style={{ marginTop: '1rem' }}>
                  Start with Geometric Algebra
                </Button>
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <H2>Advanced Systems</H2>
              </CardHeader>
              <CardBody>
                <P style={{ marginBottom: '1rem' }}>Discover high-performance computing and integration features:</P>
                <ul style={{ fontSize: '0.875rem', lineHeight: '1.5' }}>
                  <li>• WebGPU Acceleration</li>
                  <li>• TropicalDualClifford Fusion</li>
                  <li>• Cellular Automata</li>
                  <li>• Edge Computing</li>
                </ul>
                <Button href="/webgpu" style={{ marginTop: '1rem' }}>
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
              <P style={{ marginBottom: '1rem' }}>
                The Amari library provides a unified framework for mathematical computing with exotic number systems.
                Each module is designed to work independently or as part of the integrated fusion system.
              </P>
              <div style={{ display: 'flex', gap: '1rem' }}>
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
);
}