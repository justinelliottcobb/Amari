# v0.14.0 Examples Overhaul Plan

## Overview

Major update to the examples ecosystem:
1. Migrate examples-suite from JadisUI to MantineUI
2. Add missing examples for 13 crates without coverage

## Track 1: MantineUI Migration

### Current State (JadisUI)
- 23 files importing from `jadis-ui`
- 13 page components
- 8 reusable components
- ~8,000 lines of code

### Component Mapping (JadisUI → MantineUI)

| JadisUI | MantineUI | Notes |
|---------|-----------|-------|
| `Grid`, `GridItem` | `Grid`, `Grid.Col` | Similar API |
| `Card`, `CardHeader`, `CardBody` | `Card`, `Card.Section` | Minor API differences |
| `H1`-`H6`, `P` | `Title`, `Text` | Use `order` prop for headings |
| `Code`, `CodeBlock` | `Code`, `Prism` | Prism for syntax highlighting |
| `Button` | `Button` | Nearly identical |
| `Input` | `TextInput` | Renamed |
| `Table`, `TableRow`, `TableCell` | `Table` components | Similar structure |
| `StatusBadge` | `Badge` | Renamed |
| `LoadingSpinner` | `Loader` | Renamed |
| `ProgressBar` | `Progress` | Renamed |
| `ThemeProvider` | `MantineProvider` | Different theme structure |
| `PageLayout`, `Sidebar` | `AppShell`, `AppShell.Navbar` | More structured API |

### Migration Steps

1. **Setup** (Phase 1)
   - [ ] Install `@mantine/core`, `@mantine/hooks`, `@mantine/prism`
   - [ ] Remove `jadis-ui` dependency
   - [ ] Configure MantineProvider in main.tsx
   - [ ] Set up theme (colors, fonts, spacing)

2. **Core Components** (Phase 2)
   - [ ] Migrate Layout.tsx → AppShell
   - [ ] Migrate Navigation.tsx → AppShell.Navbar
   - [ ] Create shared component wrappers if needed

3. **Page Components** (Phase 3)
   - [ ] Home.tsx
   - [ ] GeometricAlgebra.tsx
   - [ ] InformationGeometry.tsx
   - [ ] TropicalAlgebra.tsx
   - [ ] DualNumbers.tsx
   - [ ] Fusion.tsx
   - [ ] Automata.tsx
   - [ ] AmariChentsovTensorDemo.tsx
   - [ ] EnumerativeGeometry.tsx
   - [ ] WebGPU.tsx
   - [ ] Benchmarks.tsx
   - [ ] Playground.tsx
   - [ ] APIReference.tsx

4. **Utility Components** (Phase 4)
   - [ ] ExampleCard.tsx
   - [ ] CodePlayground.tsx
   - [ ] TensorVisualization.tsx
   - [ ] RealTimeDisplay.tsx
   - [ ] LoadingState.tsx
   - [ ] ErrorBoundary.tsx

5. **Cleanup** (Phase 5)
   - [ ] Remove jadis-ui.d.ts
   - [ ] Update package.json
   - [ ] Test all pages
   - [ ] Fix styling issues

---

## Track 2: Missing Crate Examples

### Priority 1: New/Core Crates (v0.13.x additions)

| Crate | Web Example | Rust Example | Status |
|-------|-------------|--------------|--------|
| amari-probabilistic | [ ] | [ ] | New in 0.13.0 |
| amari-holographic | [ ] | [ ] | New in 0.12.3 |
| amari-calculus | [ ] | [ ] | New in 0.11.0 |

### Priority 2: Core Algebras

| Crate | Web Example | Rust Example | Status |
|-------|-------------|--------------|--------|
| amari-dual | [ ] | [ ] | Core autodiff |
| amari-tropical | [ ] | [ ] | Core algebra |

### Priority 3: Domain Crates

| Crate | Web Example | Rust Example | Status |
|-------|-------------|--------------|--------|
| amari-gpu | [ ] | [ ] | GPU acceleration |
| amari-fusion | [ ] | [ ] | Fusion systems |
| amari-relativistic | [ ] | [ ] | Spacetime algebra |
| amari-flynn | [ ] | [ ] | Probabilistic contracts |
| amari-enumerative | [ ] | [ ] | Intersection theory |

### Priority 4: Update Existing

| Crate | Current | Needed |
|-------|---------|--------|
| amari-measure | 5 Rust | Web example |
| amari-network | 5 Rust | Web example |
| amari-optimization | 3 Rust | Web example |
| amari-automata | Web only | Rust example |

---

## Example Content Guidelines

### Web Examples (examples-suite)
Each page should include:
- Brief introduction to the crate's purpose
- Interactive demos with live WASM execution
- Code snippets users can modify
- Visual output where applicable
- Links to API documentation

### Rust Examples (crate/examples/)
Each example should:
- Be self-contained and runnable
- Include doc comments explaining concepts
- Show common use cases
- Demonstrate error handling
- Be referenced in crate README

---

## New Pages for examples-suite

### Probabilistic (amari-probabilistic)
- Gaussian sampling on multivector spaces
- MCMC visualization
- Bayesian inference demo
- SDE path simulation

### Holographic Memory (amari-holographic)
- Symbol binding/unbinding
- Associative recall demo
- Similarity search visualization
- Resonator network cleanup

### Calculus (amari-calculus)
- Gradient/divergence/curl visualization
- Vector field plots
- Manifold geodesics
- Lie derivative demo

### Tropical Algebra (amari-tropical)
- Max-plus operations
- Shortest path visualization
- Tropical matrix powers

### Dual Numbers (amari-dual)
- Automatic differentiation chain
- Gradient computation
- Higher-order derivatives

### GPU Acceleration (amari-gpu)
- Performance comparison charts
- Batch operation demos
- WebGPU capability detection

---

## Timeline Estimate

| Phase | Scope | Effort |
|-------|-------|--------|
| MantineUI Setup | Configure, theme | Small |
| Layout Migration | AppShell, Nav | Medium |
| Page Migration | 13 pages | Large |
| New Examples: Priority 1 | 3 crates | Large |
| New Examples: Priority 2 | 2 crates | Medium |
| New Examples: Priority 3-4 | 5+ crates | Large |
| Testing & Polish | All | Medium |

---

## Dependencies

```json
{
  "@mantine/core": "^7.x",
  "@mantine/hooks": "^7.x",
  "@mantine/code-highlight": "^7.x",
  "@emotion/react": "^11.x"
}
```

## Notes

- MantineUI v7 uses CSS-in-JS with Emotion
- Consider using `@mantine/code-highlight` instead of Prism
- AppShell provides responsive layout out of the box
- Theme customization via `createTheme()`
