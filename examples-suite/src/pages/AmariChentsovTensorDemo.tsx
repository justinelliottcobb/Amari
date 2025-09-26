import { useState } from "react";
import { H1, P, Card, CardHeader, CardBody, Button, CodeBlock } from "jadis-ui";
import { TensorVisualization } from "../components/TensorVisualization";

export function AmariChentsovTensorDemo() {
  const [tensor, setTensor] = useState<number[][][] | null>(null);
  const [probabilities, setProbabilities] = useState<number[]>([0.5, 0.3, 0.2]);
  const [customProbs, setCustomProbs] = useState<string>("0.5, 0.3, 0.2");

  // Compute Amari-Chentsov tensor
  const computeTensor = (probs: number[]) => {
    const n = probs.length;
    const newTensor = Array(n).fill(null).map(() =>
      Array(n).fill(null).map(() => Array(n).fill(0))
    );

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        for (let k = 0; k < n; k++) {
          let value = 0;

          if (i === j && j === k) {
            // T[i,i,i] = (1-2p_i)/(p_i)²
            value = (1 - 2 * probs[i]) / Math.pow(probs[i], 2);
          } else if (i === j && i !== k) {
            // T[i,i,k] = -1/(p_i * p_k)
            value = -1 / (probs[i] * probs[k]);
          } else if (i === k && i !== j) {
            // T[i,j,i] = -1/(p_i * p_j)
            value = -1 / (probs[i] * probs[j]);
          } else if (j === k && j !== i) {
            // T[i,j,j] = -1/(p_i * p_j)
            value = -1 / (probs[i] * probs[j]);
          } else if (i !== j && j !== k && i !== k) {
            // T[i,j,k] = 1/(p_i * p_j * p_k)
            value = 1 / (probs[i] * probs[j] * probs[k]);
          }

          newTensor[i][j][k] = Math.abs(value) < 1e-12 ? 0 : value;
        }
      }
    }

    return newTensor;
  };

  const handleCompute = () => {
    setTensor(computeTensor(probabilities));
  };

  const handleCustomProbs = () => {
    try {
      const probs = customProbs.split(',').map(p => parseFloat(p.trim()));
      const sum = probs.reduce((a, b) => a + b, 0);

      if (Math.abs(sum - 1.0) > 1e-6) {
        alert("Probabilities must sum to 1.0");
        return;
      }

      if (probs.some(p => p <= 0 || p >= 1)) {
        alert("Each probability must be between 0 and 1");
        return;
      }

      setProbabilities(probs);
      setTensor(computeTensor(probs));
    } catch (e) {
      alert("Invalid probability format. Use comma-separated values like: 0.5, 0.3, 0.2");
    }
  };

  // Example distributions
  const exampleDistributions = [
    { name: "Uniform", probs: [0.333, 0.333, 0.334] },
    { name: "Skewed", probs: [0.7, 0.2, 0.1] },
    { name: "Balanced", probs: [0.4, 0.35, 0.25] },
    { name: "Binary-like", probs: [0.45, 0.45, 0.1] },
  ];

  return (
    <div style={{ padding: '2rem' }}>
      <div style={{ maxWidth: '1536px', margin: '0 auto' }}>
        <H1>Amari-Chentsov Tensor Interactive Demo</H1>
        <P style={{ fontSize: '1.125rem', marginBottom: '1.5rem', opacity: 0.7 }}>
          Explore the fundamental tensor of information geometry with comprehensive visualizations
        </P>

        <Card style={{ marginBottom: '1.5rem' }}>
          <CardHeader>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>What is the Amari-Chentsov Tensor?</h3>
          </CardHeader>
          <CardBody>
            <P style={{ marginBottom: '1rem' }}>
              The Amari-Chentsov tensor is a fundamental 3rd-order tensor in information geometry that completely
              characterizes the geometric structure of statistical manifolds. It's defined as:
            </P>
            <CodeBlock language="math" variant="muted">
              T(∂ᵢ, ∂ⱼ, ∂ₖ) = E[∂ᵢ log p · ∂ⱼ log p · ∂ₖ log p]
            </CodeBlock>
            <P style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
              This tensor determines the α-connections and provides the unique geometric structure that makes
              information geometry powerful for machine learning, statistics, and optimization.
            </P>
          </CardBody>
        </Card>

        <Card style={{ marginBottom: '1.5rem' }}>
          <CardHeader>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>Configure Probability Distribution</h3>
          </CardHeader>
          <CardBody>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div>
                <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                  Example Distributions
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  {exampleDistributions.map(dist => (
                    <Button
                      key={dist.name}
                      onClick={() => {
                        setProbabilities(dist.probs);
                        setCustomProbs(dist.probs.join(', '));
                        setTensor(computeTensor(dist.probs));
                      }}
                      variant="outline"
                      style={{ justifyContent: 'flex-start' }}
                    >
                      <span>{dist.name}: [{dist.probs.join(', ')}]</span>
                    </Button>
                  ))}
                </div>
              </div>

              <div>
                <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                  Custom Distribution
                </h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <input
                    type="text"
                    value={customProbs}
                    onChange={(e) => setCustomProbs(e.target.value)}
                    placeholder="e.g., 0.5, 0.3, 0.2"
                    style={{
                      padding: '0.5rem',
                      border: '1px solid var(--border)',
                      borderRadius: '4px',
                      backgroundColor: 'var(--background)',
                      color: 'var(--foreground)'
                    }}
                  />
                  <Button onClick={handleCustomProbs}>
                    Apply Custom Distribution
                  </Button>
                  <P style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                    Enter comma-separated probabilities that sum to 1.0
                  </P>
                </div>
              </div>
            </div>

            <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'var(--muted)', borderRadius: '4px' }}>
              <P style={{ fontSize: '0.875rem' }}>
                Current Distribution: <strong>[{probabilities.map(p => p.toFixed(3)).join(', ')}]</strong>
              </P>
              {!tensor && (
                <Button onClick={handleCompute} style={{ marginTop: '0.5rem' }}>
                  Compute Amari-Chentsov Tensor
                </Button>
              )}
            </div>
          </CardBody>
        </Card>

        {tensor && (
          <>
            <TensorVisualization tensor={tensor} probabilities={probabilities} />

            <Card style={{ marginTop: '1.5rem' }}>
              <CardHeader>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '600' }}>Mathematical Properties</h3>
              </CardHeader>
              <CardBody>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  <div>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                      Tensor Components
                    </h4>
                    <CodeBlock language="javascript" variant="muted">
{`Diagonal: T[i,i,i] = (1-2pᵢ)/(pᵢ)²
Two equal: T[i,i,k] = -1/(pᵢ·pₖ)
All different: T[i,j,k] = 1/(pᵢ·pⱼ·pₖ)`}
                    </CodeBlock>
                  </div>
                  <div>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                      α-Connection Relationship
                    </h4>
                    <CodeBlock language="javascript" variant="muted">
{`Γᵅ[i,j,k] = (1-α)/2 · T[i,j,k]

α = -1: e-connection (exponential)
α =  0: Levi-Civita connection
α =  1: m-connection (mixture)`}
                    </CodeBlock>
                  </div>
                </div>

                <div style={{ marginTop: '1rem' }}>
                  <h4 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                    Geometric Significance
                  </h4>
                  <ul style={{ fontSize: '0.875rem', listStyle: 'none', padding: 0 }}>
                    <li>• Encodes all statistical curvature information</li>
                    <li>• Defines unique geometric structure of statistical manifolds</li>
                    <li>• Foundation for natural gradient descent optimization</li>
                    <li>• Invariant under reparameterization</li>
                    <li>• Enables efficient computation via coordinate duality</li>
                  </ul>
                </div>
              </CardBody>
            </Card>
          </>
        )}
      </div>
    </div>
  );
}