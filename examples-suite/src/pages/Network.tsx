import { Container, Stack, Card, Title, Text, SimpleGrid, Code, Badge } from "@mantine/core";
import { CodeHighlight } from "@mantine/code-highlight";
import { ExampleCard } from "../components/ExampleCard";

export function Network() {
  const simulateExample = (operation: () => string) => {
    return async () => {
      try {
        return operation();
      } catch (err) {
        throw new Error(`Simulation error: ${err}`);
      }
    };
  };

  // Simple graph utilities
  const createRandomGraph = (n: number, p: number): number[][] => {
    const adj: number[][] = Array(n).fill(null).map(() => []);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.random() < p) {
          adj[i].push(j);
          adj[j].push(i);
        }
      }
    }
    return adj;
  };

  const clusteringCoefficient = (adj: number[][], node: number): number => {
    const neighbors = adj[node];
    const k = neighbors.length;
    if (k < 2) return 0;

    let triangles = 0;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        if (adj[neighbors[i]].includes(neighbors[j])) {
          triangles++;
        }
      }
    }
    return (2 * triangles) / (k * (k - 1));
  };

  const examples = [
    {
      title: "Create Geometric Network",
      description: "Build a network with geometric properties attached to nodes and edges",
      category: "Construction",
      code: `// Create a random geometric network
const network = NetworkUtils.createRandomNetwork(
  20,     // number of nodes
  0.3     // connection probability
);

// Get network statistics
const numNodes = network.getNodeCount();
const numEdges = network.getEdgeCount();
const density = network.getDensity();

console.log("Nodes:", numNodes);
console.log("Edges:", numEdges);
console.log("Density:", density);

// Each node can have a multivector position
// Edges have geometric weights from multivector operations`,
      onRun: simulateExample(() => {
        const n = 20;
        const p = 0.3;
        const adj = createRandomGraph(n, p);

        let edgeCount = 0;
        for (const neighbors of adj) {
          edgeCount += neighbors.length;
        }
        edgeCount /= 2; // Each edge counted twice

        const maxEdges = n * (n - 1) / 2;
        const density = edgeCount / maxEdges;

        const degrees = adj.map(neighbors => neighbors.length);
        const avgDegree = degrees.reduce((a, b) => a + b, 0) / n;
        const maxDegree = Math.max(...degrees);
        const minDegree = Math.min(...degrees);

        return [
          `Random Geometric Network (n=${n}, p=${p})`,
          "",
          `Nodes: ${n}`,
          `Edges: ${edgeCount}`,
          `Density: ${density.toFixed(4)} (max edges: ${maxEdges})`,
          "",
          "Degree Statistics:",
          `  Average degree: ${avgDegree.toFixed(2)}`,
          `  Min degree: ${minDegree}`,
          `  Max degree: ${maxDegree}`,
          "",
          `Expected edges: ${(p * maxEdges).toFixed(0)}`
        ].join('\n');
      })
    },
    {
      title: "Clustering Coefficient",
      description: "Measure how clustered the network is (transitivity)",
      category: "Analysis",
      code: `// Clustering coefficient measures triangle density
// C = (# of triangles) / (# of possible triangles)

const network = NetworkUtils.createRandomNetwork(30, 0.4);
const clustering = NetworkUtils.clusteringCoefficient(network);

console.log("Global clustering coefficient:", clustering);

// For random graphs, expected C ≈ p (connection probability)
// For social networks, C is typically much higher

// Small-world networks have high clustering AND short paths`,
      onRun: simulateExample(() => {
        const n = 30;
        const p = 0.4;
        const adj = createRandomGraph(n, p);

        // Compute clustering coefficient for each node
        const localClustering = [];
        for (let i = 0; i < n; i++) {
          localClustering.push(clusteringCoefficient(adj, i));
        }

        const globalClustering = localClustering.reduce((a, b) => a + b, 0) / n;
        const maxClustering = Math.max(...localClustering);
        const nonZeroClustering = localClustering.filter(c => c > 0);

        return [
          `Clustering Analysis (n=${n}, p=${p})`,
          "",
          `Global clustering coefficient: ${globalClustering.toFixed(4)}`,
          `Expected for random graph: ${p.toFixed(4)}`,
          "",
          "Local clustering:",
          `  Max node clustering: ${maxClustering.toFixed(4)}`,
          `  Nodes with C > 0: ${nonZeroClustering.length}/${n}`,
          `  Nodes with C = 0: ${n - nonZeroClustering.length}`,
          "",
          "High clustering indicates community structure"
        ].join('\n');
      })
    },
    {
      title: "Small-World Network",
      description: "Create networks with high clustering and short path lengths",
      category: "Construction",
      code: `// Watts-Strogatz small-world model
// Start with ring lattice, rewire edges with probability β

const network = NetworkUtils.createSmallWorldNetwork(
  30,    // nodes
  4,     // k neighbors on each side in ring
  0.3    // rewiring probability β
);

// Properties:
// - High clustering (from lattice structure)
// - Short average path length (from random shortcuts)
// - Models social networks, neural networks

const clustering = NetworkUtils.clusteringCoefficient(network);
console.log("Clustering:", clustering);`,
      onRun: simulateExample(() => {
        const n = 30;
        const k = 4;
        const beta = 0.3;

        // Create Watts-Strogatz small-world network
        const adj: number[][] = Array(n).fill(null).map(() => []);

        // Start with ring lattice
        for (let i = 0; i < n; i++) {
          for (let j = 1; j <= k; j++) {
            const neighbor = (i + j) % n;
            if (!adj[i].includes(neighbor)) {
              adj[i].push(neighbor);
              adj[neighbor].push(i);
            }
          }
        }

        // Rewire edges with probability beta
        let rewired = 0;
        for (let i = 0; i < n; i++) {
          for (let j = 1; j <= k; j++) {
            if (Math.random() < beta) {
              const oldNeighbor = (i + j) % n;
              let newNeighbor;
              do {
                newNeighbor = Math.floor(Math.random() * n);
              } while (newNeighbor === i || adj[i].includes(newNeighbor));

              // Remove old edge
              adj[i] = adj[i].filter(x => x !== oldNeighbor);
              adj[oldNeighbor] = adj[oldNeighbor].filter(x => x !== i);

              // Add new edge
              adj[i].push(newNeighbor);
              adj[newNeighbor].push(i);
              rewired++;
            }
          }
        }

        // Compute clustering
        let totalClustering = 0;
        for (let i = 0; i < n; i++) {
          totalClustering += clusteringCoefficient(adj, i);
        }
        const avgClustering = totalClustering / n;

        // Ring lattice clustering for comparison
        const ringClustering = 3 * (k - 1) / (2 * (2 * k - 1));

        return [
          `Watts-Strogatz Small-World Network`,
          `Parameters: n=${n}, k=${k}, β=${beta}`,
          "",
          `Edges rewired: ${rewired}`,
          `Average clustering: ${avgClustering.toFixed(4)}`,
          `Ring lattice clustering: ${ringClustering.toFixed(4)}`,
          "",
          "Small-world properties:",
          "  ✓ High clustering (community structure)",
          "  ✓ Short path lengths (6 degrees of separation)",
          "",
          "Applications: social networks, neural circuits"
        ].join('\n');
      })
    },
    {
      title: "Community Detection",
      description: "Find groups of densely connected nodes",
      category: "Analysis",
      code: `// Community detection finds clusters in networks
const network = NetworkUtils.createRandomNetwork(50, 0.2);

// Get communities using label propagation
const communities = network.detectCommunities();

console.log("Number of communities:", communities.length);
for (const community of communities) {
  console.log("Community size:", community.size());
  console.log("Internal density:", community.internalDensity());
}

// Modularity measures quality of partition
const modularity = network.modularity(communities);
console.log("Modularity:", modularity);`,
      onRun: simulateExample(() => {
        // Create network with planted communities
        const communitySize = 10;
        const numCommunities = 3;
        const n = communitySize * numCommunities;
        const pIn = 0.6;  // Within-community probability
        const pOut = 0.05; // Between-community probability

        const adj: number[][] = Array(n).fill(null).map(() => []);

        for (let i = 0; i < n; i++) {
          for (let j = i + 1; j < n; j++) {
            const sameComm = Math.floor(i / communitySize) === Math.floor(j / communitySize);
            const p = sameComm ? pIn : pOut;
            if (Math.random() < p) {
              adj[i].push(j);
              adj[j].push(i);
            }
          }
        }

        // Simple community detection: ground truth
        const communities = [];
        for (let c = 0; c < numCommunities; c++) {
          const nodes = [];
          for (let i = c * communitySize; i < (c + 1) * communitySize; i++) {
            nodes.push(i);
          }

          // Count internal edges
          let internalEdges = 0;
          for (const node of nodes) {
            for (const neighbor of adj[node]) {
              if (nodes.includes(neighbor)) {
                internalEdges++;
              }
            }
          }
          internalEdges /= 2;

          const maxInternal = communitySize * (communitySize - 1) / 2;
          communities.push({
            size: communitySize,
            internalEdges,
            internalDensity: internalEdges / maxInternal
          });
        }

        return [
          "Community Detection (Planted Partition Model)",
          `Network: ${n} nodes, ${numCommunities} communities`,
          `p_in = ${pIn}, p_out = ${pOut}`,
          "",
          "Detected Communities:",
          ...communities.map((c, i) =>
            `  Community ${i + 1}: ${c.size} nodes, density ${c.internalDensity.toFixed(3)}`
          ),
          "",
          "Modularity measures partition quality:",
          "  Q > 0.3 typically indicates significant structure"
        ].join('\n');
      })
    },
    {
      title: "Geometric Edge Weights",
      description: "Use multivector operations to compute edge weights",
      category: "Geometry",
      code: `// Create a network where nodes have multivector positions
const network = new WasmGeometricNetwork();

// Add nodes with multivector positions
for (let i = 0; i < 10; i++) {
  const position = [0, Math.cos(i * 0.6), Math.sin(i * 0.6), 0, 0, 0, 0, 0];
  network.addNode(i, position);
}

// Connect nodes and compute geometric edge weights
// Weight can be based on geometric product, distance, etc.
for (let i = 0; i < 10; i++) {
  const j = (i + 1) % 10;
  const edge = network.addEdge(i, j);
  console.log("Edge weight:", edge.getWeight());
}`,
      onRun: simulateExample(() => {
        const n = 10;

        // Create positions on unit circle (vectors in Cl(3,0,0))
        const positions = [];
        for (let i = 0; i < n; i++) {
          const theta = 2 * Math.PI * i / n;
          positions.push([0, Math.cos(theta), Math.sin(theta), 0, 0, 0, 0, 0]);
        }

        // Compute edge weights as Euclidean distance
        const edges = [];
        for (let i = 0; i < n; i++) {
          const j = (i + 1) % n;
          const dx = positions[i][1] - positions[j][1];
          const dy = positions[i][2] - positions[j][2];
          const dist = Math.sqrt(dx * dx + dy * dy);
          edges.push({ from: i, to: j, weight: dist });
        }

        // Also compute geometric product magnitude for first few edges
        const geoProducts = [];
        for (let i = 0; i < 3; i++) {
          const j = (i + 1) % n;
          // Simplified: just dot product of vector parts
          const dot = positions[i][1] * positions[j][1] + positions[i][2] * positions[j][2];
          geoProducts.push(dot);
        }

        return [
          "Geometric Network with Multivector Positions",
          "",
          `${n} nodes arranged on unit circle`,
          "Positions stored as vectors: [0, x, y, 0, ...]",
          "",
          "Edge weights (Euclidean distance):",
          ...edges.slice(0, 5).map(e =>
            `  Edge ${e.from}→${e.to}: ${e.weight.toFixed(4)}`
          ),
          "",
          "Geometric products (dot of vectors):",
          ...geoProducts.map((g, i) =>
            `  Nodes ${i}·${i+1}: ${g.toFixed(4)}`
          ),
          "",
          "Edge weights can encode geometric relationships"
        ].join('\n');
      })
    },
    {
      title: "Network Flow with Tropical Algebra",
      description: "Shortest paths using tropical (max-plus) operations",
      category: "Algorithms",
      code: `// Tropical algebra for shortest paths
// ⊕ = min, ⊗ = +

const network = new WasmTropicalNetwork();

// Add weighted edges
network.addEdge(0, 1, 3);  // A → B: 3
network.addEdge(0, 2, 1);  // A → C: 1
network.addEdge(1, 3, 2);  // B → D: 2
network.addEdge(2, 3, 4);  // C → D: 4

// Compute shortest paths using tropical matrix multiplication
const distances = network.shortestPaths(0);
console.log("Shortest distances from node 0:", distances);

// Floyd-Warshall with tropical operations
const allPairs = network.allPairsShortestPaths();`,
      onRun: simulateExample(() => {
        // Simple graph
        const n = 4;
        const edges = [
          { from: 0, to: 1, weight: 3 },
          { from: 0, to: 2, weight: 1 },
          { from: 1, to: 2, weight: 1 },
          { from: 1, to: 3, weight: 2 },
          { from: 2, to: 3, weight: 4 }
        ];

        // Initialize distance matrix (tropical: infinity = Infinity)
        const dist: number[][] = Array(n).fill(null).map(() => Array(n).fill(Infinity));
        for (let i = 0; i < n; i++) dist[i][i] = 0;

        for (const e of edges) {
          dist[e.from][e.to] = e.weight;
        }

        // Floyd-Warshall with tropical operations
        for (let k = 0; k < n; k++) {
          for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
              // Tropical: min instead of +, + instead of *
              dist[i][j] = Math.min(dist[i][j], dist[i][k] + dist[k][j]);
            }
          }
        }

        return [
          "Tropical Algebra for Shortest Paths",
          "",
          "Graph edges:",
          ...edges.map(e => `  ${e.from} → ${e.to}: weight ${e.weight}`),
          "",
          "All-pairs shortest distances:",
          "     " + [0, 1, 2, 3].map(j => j.toString().padStart(4)).join(""),
          ...dist.map((row, i) =>
            `  ${i}: ` + row.map(d => d === Infinity ? " ∞" : d.toString().padStart(4)).join("")
          ),
          "",
          "Tropical matrix multiplication: A ⊗ B",
          "  (A⊗B)ᵢⱼ = min_k (Aᵢₖ + Bₖⱼ)"
        ].join('\n');
      })
    }
  ];

  return (
    <Container size="lg" py="xl">
      <Stack gap="lg">
        <div>
          <Title order={1}>Geometric Networks</Title>
          <Text size="lg" c="dimmed">
            Network analysis with geometric algebra node and edge properties
          </Text>
        </div>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Overview</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <Text mb="md">
              The <Code>amari-network</Code> module extends graph theory with geometric algebra,
              allowing nodes and edges to carry multivector data.
            </Text>
            <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
              <div>
                <Title order={4} size="sm" mb="xs">Network Construction</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Random graphs (Erdős-Rényi)</li>
                  <li>Small-world networks (Watts-Strogatz)</li>
                  <li>Geometric networks with positions</li>
                  <li>Tropical networks for optimization</li>
                </Text>
              </div>
              <div>
                <Title order={4} size="sm" mb="xs">Analysis Tools</Title>
                <Text size="sm" c="dimmed" component="ul" style={{ paddingLeft: '1rem', margin: 0 }}>
                  <li>Clustering coefficient</li>
                  <li>Community detection</li>
                  <li>Shortest paths (tropical)</li>
                  <li>Centrality measures</li>
                </Text>
              </div>
            </SimpleGrid>
          </Card.Section>
        </Card>

        <Card withBorder>
          <Card.Section inheritPadding py="xs" bg="dark.6">
            <Title order={3} size="h4">Geometric Networks</Title>
          </Card.Section>
          <Card.Section inheritPadding py="md">
            <CodeHighlight
              code={`Traditional Graph:        Geometric Network:
  Node → ID                 Node → ID + Multivector position
  Edge → (u, v)             Edge → (u, v) + Geometric weight

Edge weight from geometry:
  - Euclidean distance: |p₁ - p₂|
  - Geometric product: p₁ * p₂
  - Inner product: p₁ · p₂

Applications:
  - Spatial networks (roads, neurons)
  - Similarity graphs in ML
  - Physical simulations`}
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
