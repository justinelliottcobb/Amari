import { useState } from "react";
import { Container, Stack, Card, Title, Text, Button, TextInput, SimpleGrid, Code } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
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
    } catch (_e) {
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
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Amari-Chentsov Tensor Interactive Demo</Title>
          <Text size="lg" c="dimmed">
            Explore the fundamental tensor of information geometry with comprehensive visualizations
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">What is the Amari-Chentsov Tensor?</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The Amari-Chentsov tensor is a fundamental 3rd-order tensor in information geometry that completely
              characterizes the geometric structure of statistical manifolds. It's defined as:
            </Text>
            <CodeHighlight
              code="T(∂ᵢ, ∂ⱼ, ∂ₖ) = E[∂ᵢ log p · ∂ⱼ log p · ∂ₖ log p]"
              language="plaintext"
              mb="md"
            />
            <Text size="sm" c="dimmed">
              This tensor determines the α-connections and provides the unique geometric structure that makes
              information geometry powerful for machine learning, statistics, and optimization.
            </Text>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Configure Probability Distribution</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
              <div>
                <Title order={4} size="sm" mb="sm">Example Distributions</Title>
                <Stack gap="xs">
                  {exampleDistributions.map(dist => (
                    <Button
                      key={dist.name}
                      onClick={() => {
                        setProbabilities(dist.probs);
                        setCustomProbs(dist.probs.join(', '));
                        setTensor(computeTensor(dist.probs));
                      }}
                      variant="outline"
                      fullWidth
                      justify="flex-start"
                    >
                      {dist.name}: [{dist.probs.join(', ')}]
                    </Button>
                  ))}
                </Stack>
              </div>

              <div>
                <Title order={4} size="sm" mb="sm">Custom Distribution</Title>
                <Stack gap="xs">
                  <TextInput
                    value={customProbs}
                    onChange={(e) => setCustomProbs(e.target.value)}
                    placeholder="e.g., 0.5, 0.3, 0.2"
                  />
                  <Button onClick={handleCustomProbs}>
                    Apply Custom Distribution
                  </Button>
                  <Text size="xs" c="dimmed">
                    Enter comma-separated probabilities that sum to 1.0
                  </Text>
                </Stack>
              </div>
            </SimpleGrid>

            <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'var(--mantine-color-dark-6)', borderRadius: 'var(--mantine-radius-sm)' }}>
              <Text size="sm">
                Current Distribution: <Code>[{probabilities.map(p => p.toFixed(3)).join(', ')}]</Code>
              </Text>
              {!tensor && (
                <Button onClick={handleCompute} mt="sm">
                  Compute Amari-Chentsov Tensor
                </Button>
              )}
            </div>
          </Card.Section>
        </Card>

        {tensor && (
          <>
            <TensorVisualization tensor={tensor} probabilities={probabilities} />

            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={3} size="h4">Mathematical Properties</Title>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, md: 2 }} spacing="lg">
                  <div>
                    <Title order={4} size="sm" mb="xs">Tensor Components</Title>
                    <CodeHighlight
                      code={`Diagonal: T[i,i,i] = (1-2pᵢ)/(pᵢ)²
Two equal: T[i,i,k] = -1/(pᵢ·pₖ)
All different: T[i,j,k] = 1/(pᵢ·pⱼ·pₖ)`}
                      language="plaintext"
                    />
                  </div>
                  <div>
                    <Title order={4} size="sm" mb="xs">α-Connection Relationship</Title>
                    <CodeHighlight
                      code={`Γᵅ[i,j,k] = (1-α)/2 · T[i,j,k]

α = -1: e-connection (exponential)
α =  0: Levi-Civita connection
α =  1: m-connection (mixture)`}
                      language="plaintext"
                    />
                  </div>
                </SimpleGrid>

                <div style={{ marginTop: '1.5rem' }}>
                  <Title order={4} size="sm" mb="xs">Geometric Significance</Title>
                  <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem' }}>
                    <li>Encodes all statistical curvature information</li>
                    <li>Defines unique geometric structure of statistical manifolds</li>
                    <li>Foundation for natural gradient descent optimization</li>
                    <li>Invariant under reparameterization</li>
                    <li>Enables efficient computation via coordinate duality</li>
                  </Text>
                </div>
              </Card.Section>
            </Card>

            <Card withBorder>
              <Card.Section inheritPadding py="xs" bg="dark.6">
                <Title order={3} size="h4">Applications</Title>
              </Card.Section>
              <Card.Section inheritPadding py="md">
                <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
                  <div>
                    <Title order={4} size="sm" mb="xs">Machine Learning</Title>
                    <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem' }}>
                      <li>Natural gradient descent</li>
                      <li>Fisher-Rao metrics for neural networks</li>
                      <li>Variational inference</li>
                      <li>Information-geometric regularization</li>
                    </Text>
                  </div>
                  <div>
                    <Title order={4} size="sm" mb="xs">Statistics</Title>
                    <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem' }}>
                      <li>Higher-order asymptotics</li>
                      <li>Exponential family analysis</li>
                      <li>Geometric interpretation of estimators</li>
                      <li>Curvature-based model selection</li>
                    </Text>
                  </div>
                </SimpleGrid>
              </Card.Section>
            </Card>
          </>
        )}
      </Stack>
    </Container>
  );
}
