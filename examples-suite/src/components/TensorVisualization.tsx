import { useState } from "react";
import { Card, CardHeader, CardBody, H3, Button, P } from "jadis-ui";

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
    <div style={{ fontFamily: 'monospace' }}>
      <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>
        Heat Map - Slice k={selectedSlice}
      </h4>
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, 60px)`, gap: '2px', marginBottom: '1rem' }}>
        {Array.from({ length: n }).map((_, i) =>
          Array.from({ length: n }).map((_, j) => {
            const value = tensor[i][j][selectedSlice];
            return (
              <div
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
                  border: '1px solid var(--border)',
                  borderRadius: '4px',
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
              </div>
            );
          })
        )}
      </div>
      <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
        {Array.from({ length: n }).map((_, k) => (
          <Button
            key={k}
            onClick={() => setSelectedSlice(k)}
            variant={selectedSlice === k ? 'default' : 'outline'}
            style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
          >
            k={k}
          </Button>
        ))}
      </div>
    </div>
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
      <div style={{ fontFamily: 'monospace' }}>
        <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>
          3D Surface Plot (Magnitude as Height: 0-9)
        </h4>
        <pre style={{ fontSize: '0.75rem', lineHeight: '1.2' }}>
          {layers.join('\n')}
        </pre>
        <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', opacity: 0.7 }}>
          <P>Higher numbers = larger tensor magnitude</P>
          <P>Layers stacked in isometric view</P>
        </div>
      </div>
    );
  };

  // Render network graph visualization
  const renderNetworkGraph = () => {
    const nodes: Array<{id: string, x: number, y: number, value: number}> = [];
    const edges: Array<{from: string, to: string, value: number}> = [];

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
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const val = Math.abs(nodes[i].value * nodes[j].value);
        if (val > maxVal * 0.3) { // Only show significant connections
          edges.push({
            from: nodes[i].id,
            to: nodes[j].id,
            value: val
          });
        }
      }
    }

    return (
      <div style={{ fontFamily: 'monospace' }}>
        <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>
          Network Graph - Component Relationships
        </h4>
        <svg width="500" height="250" style={{ border: '1px solid var(--border)', borderRadius: '4px' }}>
          {/* Draw edges */}
          {edges.slice(0, 20).map((edge, idx) => {
            const fromNode = nodes.find(n => n.id === edge.from);
            const toNode = nodes.find(n => n.id === edge.to);
            if (!fromNode || !toNode) return null;

            return (
              <line
                key={idx}
                x1={fromNode.x}
                y1={fromNode.y}
                x2={toNode.x}
                y2={toNode.y}
                stroke="var(--muted)"
                strokeWidth={edge.value / maxVal}
                opacity={0.3}
              />
            );
          })}

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
                  title={`T[${node.id}] = ${node.value.toFixed(4)}`}
                />
                <text
                  x={node.x}
                  y={node.y - size - 2}
                  fontSize="8"
                  fill="var(--foreground)"
                  textAnchor="middle"
                >
                  {node.id}
                </text>
              </g>
            );
          })}
        </svg>
        <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', opacity: 0.7 }}>
          <P>Red nodes: positive values, Blue nodes: negative values</P>
          <P>Node size: magnitude, Edge width: interaction strength</P>
        </div>
      </div>
    );
  };

  // Render interactive exploration
  const renderInteractive = () => {
    const [i, setI] = useState(0);
    const [j, setJ] = useState(0);
    const [k, setK] = useState(0);

    const value = tensor[i][j][k];
    const magnitude = Math.abs(value) / maxVal;

    return (
      <div style={{ fontFamily: 'monospace' }}>
        <h4 style={{ fontSize: '0.875rem', marginBottom: '0.5rem' }}>
          Interactive Tensor Explorer
        </h4>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '1rem' }}>
          {/* Controls */}
          <div>
            <div style={{ marginBottom: '0.75rem' }}>
              <label style={{ display: 'block', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                i = {i} (p_{i} = {probabilities[i].toFixed(3)})
              </label>
              <input
                type="range"
                min="0"
                max={n - 1}
                value={i}
                onChange={(e) => setI(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div style={{ marginBottom: '0.75rem' }}>
              <label style={{ display: 'block', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                j = {j} (p_{j} = {probabilities[j].toFixed(3)})
              </label>
              <input
                type="range"
                min="0"
                max={n - 1}
                value={j}
                onChange={(e) => setJ(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>

            <div style={{ marginBottom: '0.75rem' }}>
              <label style={{ display: 'block', fontSize: '0.75rem', marginBottom: '0.25rem' }}>
                k = {k} (p_{k} = {probabilities[k].toFixed(3)})
              </label>
              <input
                type="range"
                min="0"
                max={n - 1}
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </div>

          {/* Display */}
          <div>
            <div style={{
              padding: '1rem',
              border: '2px solid var(--border)',
              borderRadius: '8px',
              backgroundColor: value > 0 ? 'rgba(255, 0, 0, 0.1)' : 'rgba(0, 0, 255, 0.1)'
            }}>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                T[{i},{j},{k}] = {value.toFixed(6)}
              </div>

              <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>
                <P>Magnitude: {magnitude.toFixed(4)} ({(magnitude * 100).toFixed(1)}% of max)</P>
                <P>Visual: {getAsciiChar(magnitude).repeat(Math.round(magnitude * 10))}</P>
              </div>

              <div style={{ marginTop: '0.75rem' }}>
                <P style={{ fontSize: '0.7rem' }}>
                  {i === j && j === k && "Diagonal element (all indices equal)"}
                  {i === j && i !== k && "Two indices equal (i=j≠k)"}
                  {i === k && i !== j && "Two indices equal (i=k≠j)"}
                  {j === k && j !== i && "Two indices equal (j=k≠i)"}
                  {i !== j && j !== k && i !== k && "All indices different"}
                </P>
              </div>
            </div>

            {/* Related values */}
            <div style={{ marginTop: '1rem', fontSize: '0.7rem' }}>
              <P>Related tensor values:</P>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem', marginTop: '0.5rem' }}>
                <div>T[{j},{i},{k}] = {tensor[j][i][k].toFixed(4)}</div>
                <div>T[{k},{j},{i}] = {tensor[k][j][i].toFixed(4)}</div>
                <div>T[{i},{k},{j}] = {tensor[i][k][j].toFixed(4)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <Card style={{ marginTop: '1rem' }}>
      <CardHeader>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <H3>Tensor Visualizations</H3>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <Button
              onClick={() => setVisualizationType('heatmap')}
              variant={visualizationType === 'heatmap' ? 'default' : 'outline'}
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            >
              Heat Map
            </Button>
            <Button
              onClick={() => setVisualizationType('surface')}
              variant={visualizationType === 'surface' ? 'default' : 'outline'}
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            >
              3D Surface
            </Button>
            <Button
              onClick={() => setVisualizationType('network')}
              variant={visualizationType === 'network' ? 'default' : 'outline'}
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            >
              Network
            </Button>
            <Button
              onClick={() => setVisualizationType('interactive')}
              variant={visualizationType === 'interactive' ? 'default' : 'outline'}
              style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}
            >
              Interactive
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardBody>
        {visualizationType === 'heatmap' && renderHeatMap()}
        {visualizationType === 'surface' && render3DSurface()}
        {visualizationType === 'network' && renderNetworkGraph()}
        {visualizationType === 'interactive' && renderInteractive()}
      </CardBody>
    </Card>
  );
}