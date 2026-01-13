# Amari PureScript Examples

Type-safe PureScript bindings for `amari-wasm`, demonstrating functional programming with geometric algebra and automatic differentiation.

## Overview

This package provides PureScript FFI bindings to the Amari WASM library, enabling:

- Geometric algebra computations in a pure functional style
- Type-safe multivector operations
- Automatic differentiation with dual numbers
- Effect-tracked WASM interop

## Prerequisites

- [PureScript](https://www.purescript.org/) 0.15+
- [Spago](https://github.com/purescript/spago)
- Node.js 18+ (for running)
- The `@justinelliottcobb/amari-wasm` npm package

## Installation

```bash
# Install PureScript and Spago if needed
npm install -g purescript spago

# Install amari-wasm
npm install @justinelliottcobb/amari-wasm

# Build the PureScript project
spago build
```

## Running Examples

```bash
spago run
```

## Module Structure

### `Amari.Wasm`

Core FFI bindings to amari-wasm:

```purescript
import Amari.Wasm as Amari

-- Initialize WASM (required before any operations)
Amari.initAmari :: Aff Unit

-- Multivector operations
Amari.multivectorFromVector :: Array Number -> Effect Multivector
Amari.multivectorAdd :: Multivector -> Multivector -> Effect Multivector
Amari.multivectorMul :: Multivector -> Multivector -> Effect Multivector
Amari.multivectorOuterProduct :: Multivector -> Multivector -> Effect Multivector

-- Dual numbers for automatic differentiation
Amari.dualVariable :: Number -> Effect Dual
Amari.dualSin :: Dual -> Effect Dual
Amari.dualReal :: Dual -> Effect Number  -- Function value
Amari.dualDual :: Dual -> Effect Number  -- Derivative
```

## Examples

### Geometric Algebra Basics

```purescript
import Amari.Wasm as Amari

main = launchAff_ do
  Amari.initAmari
  liftEffect do
    -- Create vectors
    v1 <- Amari.multivectorFromVector [1.0, 0.0, 0.0]
    v2 <- Amari.multivectorFromVector [0.0, 1.0, 0.0]

    -- Geometric product: v1 * v2 = v1 · v2 + v1 ∧ v2
    product <- Amari.multivectorMul v1 v2

    -- Outer product gives bivector (rotation plane)
    bivector <- Amari.multivectorOuterProduct v1 v2

    -- Norm (magnitude)
    norm <- Amari.multivectorNorm v1
    log $ "‖v1‖ = " <> show norm
```

### Automatic Differentiation

```purescript
import Amari.Wasm as Amari

-- Compute f(x) = x² + 2x + 1 and its derivative at x = 3
differentiatePolynomial = do
  -- Create variable x = 3
  x <- Amari.dualVariable 3.0
  two <- Amari.dualFromReal 2.0
  one <- Amari.dualFromReal 1.0

  -- Build expression: x² + 2x + 1
  x2 <- Amari.dualMul x x
  twoX <- Amari.dualMul two x
  sum1 <- Amari.dualAdd x2 twoX
  result <- Amari.dualAdd sum1 one

  -- Extract value and derivative
  value <- Amari.dualReal result      -- f(3) = 16
  derivative <- Amari.dualDual result -- f'(3) = 8

  log $ "f(3) = " <> show value
  log $ "f'(3) = " <> show derivative
```

### Rotor Rotations

```purescript
import Amari.Wasm as Amari

-- Rotate vector by 90° in xy-plane using rotor
rotateVector = do
  let angle = Math.pi / 2.0
  let cosHalf = Math.cos (angle / 2.0)
  let sinHalf = Math.sin (angle / 2.0)

  -- Rotor R = cos(θ/2) - sin(θ/2) * e12
  scalar <- Amari.multivectorFromScalar cosHalf
  bivector <- Amari.multivectorFromBivector [sinHalf, 0.0, 0.0]
  negBivector <- Amari.multivectorScale bivector (-1.0)
  rotor <- Amari.multivectorAdd scalar negBivector

  -- Vector to rotate
  v <- Amari.multivectorFromVector [1.0, 0.0, 0.0]

  -- Apply rotation: v' = R * v * R†
  rotorRev <- Amari.multivectorReverse rotor
  temp <- Amari.multivectorMul rotor v
  rotated <- Amari.multivectorMul temp rotorRev

  -- Result: (0, 1, 0)
  x <- Amari.multivectorGet rotated 1
  y <- Amari.multivectorGet rotated 2
  log $ "Rotated: (" <> show x <> ", " <> show y <> ")"
```

## Type Safety

The PureScript bindings provide several layers of type safety:

1. **Effect Tracking**: All WASM operations are in `Effect` or `Aff`
2. **Opaque Types**: `Multivector` and `Dual` are opaque, preventing invalid states
3. **Result Types**: Fallible operations return `Result a` for explicit error handling

```purescript
-- Safe inverse with explicit error handling
case fromResult =<< Amari.multivectorInverse mv of
  Right inv -> -- Use inverse
  Left err -> -- Handle error
```

## Integration Patterns

### With Aff for Async Operations

```purescript
import Effect.Aff (Aff, launchAff_)
import Effect.Class (liftEffect)

computation :: Aff Unit
computation = do
  Amari.initAmari  -- Async WASM loading
  result <- liftEffect do
    v <- Amari.multivectorFromVector [1.0, 2.0, 3.0]
    Amari.multivectorNorm v
  log $ "Result: " <> show result
```

### With State Monad

```purescript
import Control.Monad.State (State, execState, modify)

type SimState = { position :: Multivector, velocity :: Multivector }

updatePhysics :: Number -> State SimState Unit
updatePhysics dt = do
  modify \s -> s { position = -- updated }
```

## API Reference

### Multivector Construction

| Function | Description |
|----------|-------------|
| `multivectorNew p q r` | Create multivector with signature Cl(p,q,r) |
| `multivectorFromScalar s` | Create from scalar |
| `multivectorFromVector [x,y,z]` | Create from vector components |
| `multivectorFromBivector [xy,xz,yz]` | Create from bivector components |
| `multivectorZero p q r` | Zero multivector |
| `multivectorOne p q r` | Identity (scalar 1) |

### Multivector Operations

| Function | Description |
|----------|-------------|
| `multivectorAdd a b` | Addition |
| `multivectorSub a b` | Subtraction |
| `multivectorMul a b` | Geometric product |
| `multivectorScale mv s` | Scalar multiplication |
| `multivectorOuterProduct a b` | Wedge product |
| `multivectorInnerProduct a b` | Dot product |
| `multivectorReverse mv` | Reversion |
| `multivectorConjugate mv` | Clifford conjugate |
| `multivectorNorm mv` | Magnitude |
| `multivectorNormalize mv` | Unit multivector |
| `multivectorInverse mv` | Inverse (may fail) |

### Dual Numbers

| Function | Description |
|----------|-------------|
| `dualNew r d` | Create with real and dual parts |
| `dualFromReal r` | Constant (derivative = 0) |
| `dualVariable r` | Variable (derivative = 1) |
| `dualAdd a b` | Addition |
| `dualMul a b` | Multiplication |
| `dualSin d` | Sine |
| `dualCos d` | Cosine |
| `dualExp d` | Exponential |
| `dualLog d` | Natural logarithm |
| `dualPow d n` | Power |
| `dualReal d` | Extract real part |
| `dualDual d` | Extract dual (derivative) part |

## License

MIT License - see [LICENSE](../../LICENSE) for details.
