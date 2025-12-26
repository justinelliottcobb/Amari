import { useState } from "react";
import { Card, Title, Text, Button, Group, Box, Stack, Slider, SimpleGrid } from "@mantine/core";

interface TensorVisualizationProps {
  tensor: number[][][];
  probabilities: number[];
}

export function TensorVisualization({ tensor, probabilities }: TensorVisualizationProps) {
  const [selectedSlice, setSelectedSlice] = useState(0);
  const [visualizationType, setVisualizationType] = useState<'heatmap' | 'surface' | 'network' | 'interactive'>('heatmap');

  const n = probabilities.length;

  // Calculate max value for normalization
  const flatTensor = tensor.flat(2);
  const maxVal = Math.max(...flatTensor.map(Math.abs));

  // Generate color for heat map based on value
  const getColor = (value: number): string => {
    const normalized = Math.abs(value) / maxVal;
    const intensity = Math.round(normalized * 255);
    if (value > 0) {
      return `rgb(${intensity}, 0, 0)`; // Red for positive
    } else {
      return `rgb(0, 0, ${intensity})`; // Blue for negative
    }
  };

  // ASCII character for magnitude
  const getAsciiChar = (magnitude: number): string => {
    if (magnitude > 0.75) return "■";
    if (magnitude > 0.5) return "▓";
    if (magnitude > 0.25) return "▒";
    if (magnitude > 0.1) return "░";
    return "·";
  };

  // Render heat map visualization
  const renderHeatMap = () => (
    <Box ff="monospace">
      <Text size="sm" fw={600} mb="sm">
        Heat Map - Slice k={selectedSlice}
      </Text>
      <SimpleGrid cols={n} spacing={2} mb="md">
        {Array.from({ length: n }).map((_, i) =>
          Array.from({ length: n }).map((_, j) => {
            const value = tensor[i][j][selectedSlice];
            return (
              <Box
                key={`${i}-${j}`}
                style={{
                  width: '60px',
                  height: '60px',
                  backgroundColor: getColor(value),
                  color: 'white',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.75rem',
                  border: '1px solid var(--mantine-color-dark-4)',
                  borderRadius: 'var(--mantine-radius-xs)',
                  position: 'relative'
                }}
                title={`T[${i},${j},${selectedSlice}] = ${value.toFixed(4)}`}
              >
                <span>{value.toFixed(2)}</span>
                <span style={{
                  position: 'absolute',
                  top: '2px',
                  left: '4px',
                  fontSize: '0.6rem',
                  opacity: 0.7
                }}>
                  [{i},{j}]
                </span>
              </Box>
            );
          })
        )}
      </SimpleGrid>
      <Group gap="xs" mt="sm">
        {Array.from({ length: n }).map((_, k) => (
          <Button
            key={k}
            onClick={() => setSelectedSlice(k)}
            variant={selectedSlice === k ? 'filled' : 'outline'}
            size="xs"
          >
            k={k}
          </Button>
        ))}
      </Group>
    </Box>
  );

  // Render 3D surface plot (ASCII representation)
  const render3DSurface = () => {
    const layers: string[] = [];

    for (let k = 0; k < n; k++) {
      layers.push(`Layer k=${k}:`);
      const grid: string[] = [];

      for (let i = 0; i < n; i++) {
        let row = "";
        for (let j = 0; j < n; j++) {
          const magnitude = Math.abs(tensor[i][j][k]) / maxVal;
          const height = Math.round(magnitude * 9);
          row += height.toString() + " ";
        }
        grid.push("  " + row);
      }

      // Add isometric projection effect
      for (let g = 0; g < grid.length; g++) {
        const indent = "  ".repeat(n - k);
        grid[g] = indent + grid[g];
      }

      layers.push(...grid);
      layers.push("");
    }

    return (
      <Box ff="monospace">
        <Text size="sm" fw={600} mb="sm">
          3D Surface Plot (Magnitude as Height: 0-9)
        </Text>
        <Box
          component="pre"
          style={{ fontSize: '0.75rem', lineHeight: '1.2' }}
        >
          {layers.join('\n')}
        </Box>
        <Stack gap={4} mt="sm">
          <Text size="xs" c="dimmed">Higher numbers = larger tensor magnitude</Text>
          <Text size="xs" c="dimmed">Layers stacked in isometric view</Text>
        </Stack>
      </Box>
    );
  };

  // Render network graph visualization
  const renderNetworkGraph = () => {
    const nodes: Array<{id: string, x: number, y: number, value: number}> = [];

    // Create nodes for each tensor position
    const radius = 100;
    let nodeIndex = 0;

    for (let k = 0; k < n; k++) {
      const layerOffset = k * 150;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const angle = (nodeIndex / (n * n)) * 2 * Math.PI;
          nodes.push({
            id: `${i},${j},${k}`,
            x: layerOffset + radius * Math.cos(angle),
            y: 100 + radius * Math.sin(angle),
            value: tensor[i][j][k]
          });
          nodeIndex++;
        }
      }
    }

    // Create edges for significant connections
    const edges: Array<{from: typeof nodes[0], to: typeof nodes[0], value: number}> = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const val = Math.abs(nodes[i].value * nodes[j].value);
        if (val > maxVal * 0.3) { // Only show significant connections
          edges.push({
            from: nodes[i],
            to: nodes[j],
            value: val
          });
        }
      }
    }

    return (
      <Box ff="monospace">
        <Text size="sm" fw={600} mb="sm">
          Network Graph - Component Relationships
        </Text>
        <svg
          width="500"
          height="250"
          style={{
            border: '1px solid var(--mantine-color-dark-4)',
            borderRadius: 'var(--mantine-radius-sm)'
          }}
        >
          {/* Draw edges */}
          {edges.slice(0, 20).map((edge, idx) => (
            <line
              key={idx}
              x1={edge.from.x}
              y1={edge.from.y}
              x2={edge.to.x}
              y2={edge.to.y}
              stroke="var(--mantine-color-dark-3)"
              strokeWidth={edge.value / maxVal}
              opacity={0.3}
            />
          ))}

          {/* Draw nodes */}
          {nodes.map((node, idx) => {
            const size = 5 + Math.abs(node.value) / maxVal * 10;
            return (
              <g key={idx}>
                <circle
                  cx={node.x}
                  cy={node.y}
                  r={size}
                  fill={node.value > 0 ? '#ff4444' : '#4444ff'}
                  opacity={0.8}
                />
                <text
                  x={node.x}
                  y={node.y - size - 2}
                  fontSize="8"
                  fill="var(--mantine-color-dimmed)"
                  textAnchor="middle"
                >
                  {node.id}
                </text>
              </g>
            );
          })}
        </svg>
        <Stack gap={4} mt="sm">
          <Text size="xs" c="dimmed">Red nodes: positive values, Blue nodes: negative values</Text>
          <Text size="xs" c="dimmed">Node size: magnitude, Edge width: interaction strength</Text>
        </Stack>
      </Box>
    );
  };

  // Render interactive exploration
  const InteractiveExplorer = () => {
    const [i, setI] = useState(0);
    const [j, setJ] = useState(0);
    const [k, setK] = useState(0);

    const value = tensor[i][j][k];
    const magnitude = Math.abs(value) / maxVal;

    return (
      <Box ff="monospace">
        <Text size="sm" fw={600} mb="sm">
          Interactive Tensor Explorer
        </Text>

        <Group align="flex-start" gap="xl">
          {/* Controls */}
          <Stack gap="md" style={{ width: '200px' }}>
            <Box>
              <Text size="xs" mb="xs">
                i = {i} (p_i = {probabilities[i].toFixed(3)})
              </Text>
              <Slider
                min={0}
                max={n - 1}
                value={i}
                onChange={setI}
                marks={Array.from({ length: n }, (_, idx) => ({ value: idx }))}
              />
            </Box>

            <Box>
              <Text size="xs" mb="xs">
                j = {j} (p_j = {probabilities[j].toFixed(3)})
              </Text>
              <Slider
                min={0}
                max={n - 1}
                value={j}
                onChange={setJ}
                marks={Array.from({ length: n }, (_, idx) => ({ value: idx }))}
              />
            </Box>

            <Box>
              <Text size="xs" mb="xs">
                k = {k} (p_k = {probabilities[k].toFixed(3)})
              </Text>
              <Slider
                min={0}
                max={n - 1}
                value={k}
                onChange={setK}
                marks={Array.from({ length: n }, (_, idx) => ({ value: idx }))}
              />
            </Box>
          </Stack>

          {/* Display */}
          <Box style={{ flex: 1 }}>
            <Box
              p="md"
              style={{
                border: '2px solid var(--mantine-color-dark-4)',
                borderRadius: 'var(--mantine-radius-md)',
                backgroundColor: value > 0 ? 'rgba(255, 0, 0, 0.1)' : 'rgba(0, 0, 255, 0.1)'
              }}
            >
              <Text size="xl" fw={700} mb="sm">
                T[{i},{j},{k}] = {value.toFixed(6)}
              </Text>

              <Stack gap={4}>
                <Text size="xs" c="dimmed">
                  Magnitude: {magnitude.toFixed(4)} ({(magnitude * 100).toFixed(1)}% of max)
                </Text>
                <Text size="xs" c="dimmed">
                  Visual: {getAsciiChar(magnitude).repeat(Math.round(magnitude * 10))}
                </Text>
              </Stack>

              <Box mt="md">
                <Text size="xs">
                  {i === j && j === k && "Diagonal element (all indices equal)"}
                  {i === j && i !== k && "Two indices equal (i=j≠k)"}
                  {i === k && i !== j && "Two indices equal (i=k≠j)"}
                  {j === k && j !== i && "Two indices equal (j=k≠i)"}
                  {i !== j && j !== k && i !== k && "All indices different"}
                </Text>
              </Box>
            </Box>

            {/* Related values */}
            <Box mt="md">
              <Text size="xs" c="dimmed" mb="xs">Related tensor values:</Text>
              <SimpleGrid cols={3} spacing="xs">
                <Text size="xs">T[{j},{i},{k}] = {tensor[j][i][k].toFixed(4)}</Text>
                <Text size="xs">T[{k},{j},{i}] = {tensor[k][j][i].toFixed(4)}</Text>
                <Text size="xs">T[{i},{k},{j}] = {tensor[i][k][j].toFixed(4)}</Text>
              </SimpleGrid>
            </Box>
          </Box>
        </Group>
      </Box>
    );
  };

  return (
    <Card withBorder mt="md">
      <Card.Section withBorder inheritPadding py="sm">
        <Group justify="space-between" align="center">
          <Title order={4}>Tensor Visualizations</Title>
          <Group gap="xs">
            <Button
              onClick={() => setVisualizationType('heatmap')}
              variant={visualizationType === 'heatmap' ? 'filled' : 'outline'}
              size="xs"
            >
              Heat Map
            </Button>
            <Button
              onClick={() => setVisualizationType('surface')}
              variant={visualizationType === 'surface' ? 'filled' : 'outline'}
              size="xs"
            >
              3D Surface
            </Button>
            <Button
              onClick={() => setVisualizationType('network')}
              variant={visualizationType === 'network' ? 'filled' : 'outline'}
              size="xs"
            >
              Network
            </Button>
            <Button
              onClick={() => setVisualizationType('interactive')}
              variant={visualizationType === 'interactive' ? 'filled' : 'outline'}
              size="xs"
            >
              Interactive
            </Button>
          </Group>
        </Group>
      </Card.Section>
      <Card.Section inheritPadding py="md">
        {visualizationType === 'heatmap' && renderHeatMap()}
        {visualizationType === 'surface' && render3DSurface()}
        {visualizationType === 'network' && renderNetworkGraph()}
        {visualizationType === 'interactive' && <InteractiveExplorer />}
      </Card.Section>
    </Card>
  );
}
