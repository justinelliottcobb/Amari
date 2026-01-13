-- | Main module demonstrating amari-wasm usage in PureScript
module Main where

import Prelude

import Amari.Wasm as Amari
import Data.Either (Either(..))
import Data.Traversable (for_)
import Effect (Effect)
import Effect.Aff (launchAff_)
import Effect.Class (liftEffect)
import Effect.Class.Console (log)

main :: Effect Unit
main = launchAff_ do
  -- Initialize the WASM module
  log "=== Amari PureScript Examples ==="
  log ""
  log "Initializing amari-wasm..."
  Amari.initAmari
  log "WASM module loaded successfully!"
  log ""

  -- Run examples
  liftEffect do
    geometricAlgebraExample
    dualNumberExample
    rotorExample
    bivectorExample

-- | Demonstrate basic geometric algebra operations
geometricAlgebraExample :: Effect Unit
geometricAlgebraExample = do
  log "=== Geometric Algebra Example ==="
  log ""

  -- Create vectors in Cl(3,0,0)
  v1 <- Amari.multivectorFromVector [1.0, 0.0, 0.0]
  v2 <- Amari.multivectorFromVector [0.0, 1.0, 0.0]

  -- Geometric product gives a bivector (rotation plane)
  product <- Amari.multivectorMul v1 v2
  log "v1 = e1, v2 = e2"
  log "v1 * v2 (geometric product):"

  -- Get the bivector component (e12)
  e12 <- Amari.multivectorGet product 3  -- Index 3 is e12 in Cl(3,0,0)
  log $ "  e12 component: " <> show e12

  -- Outer product
  outer <- Amari.multivectorOuterProduct v1 v2
  outerE12 <- Amari.multivectorGet outer 3
  log "v1 ∧ v2 (outer product):"
  log $ "  e12 component: " <> show outerE12

  -- Inner product
  inner <- Amari.multivectorInnerProduct v1 v2
  innerScalar <- Amari.multivectorGet inner 0  -- Scalar component
  log "v1 · v2 (inner product):"
  log $ "  scalar component: " <> show innerScalar

  -- Vector addition
  sum <- Amari.multivectorAdd v1 v2
  sumX <- Amari.multivectorGet sum 1
  sumY <- Amari.multivectorGet sum 2
  log "v1 + v2:"
  log $ "  = (" <> show sumX <> ", " <> show sumY <> ", 0)"

  -- Norm
  normV1 <- Amari.multivectorNorm v1
  log $ "‖v1‖ = " <> show normV1

  log ""

-- | Demonstrate automatic differentiation with dual numbers
dualNumberExample :: Effect Unit
dualNumberExample = do
  log "=== Dual Number Automatic Differentiation ==="
  log ""

  -- f(x) = x^2 + 2x + 1 at x = 3
  -- f'(x) = 2x + 2, so f'(3) = 8
  x <- Amari.dualVariable 3.0
  two <- Amari.dualFromReal 2.0
  one <- Amari.dualFromReal 1.0

  -- x^2
  x2 <- Amari.dualMul x x

  -- 2x
  twoX <- Amari.dualMul two x

  -- x^2 + 2x
  sum1 <- Amari.dualAdd x2 twoX

  -- x^2 + 2x + 1
  result <- Amari.dualAdd sum1 one

  realPart <- Amari.dualReal result
  dualPart <- Amari.dualDual result

  log "f(x) = x² + 2x + 1"
  log $ "f(3) = " <> show realPart <> " (expected: 16)"
  log $ "f'(3) = " <> show dualPart <> " (expected: 8)"
  log ""

  -- Trigonometric functions
  log "Trigonometric derivatives:"
  y <- Amari.dualVariable (3.14159265 / 4.0)  -- π/4
  sinY <- Amari.dualSin y
  cosY <- Amari.dualCos y

  sinReal <- Amari.dualReal sinY
  sinDual <- Amari.dualDual sinY
  cosReal <- Amari.dualReal cosY
  cosDual <- Amari.dualDual cosY

  log $ "sin(π/4) = " <> show sinReal <> ", d/dx sin(x)|_{π/4} = " <> show sinDual
  log $ "cos(π/4) = " <> show cosReal <> ", d/dx cos(x)|_{π/4} = " <> show cosDual
  log ""

  -- Exponential and logarithm
  log "Exponential and logarithm derivatives:"
  z <- Amari.dualVariable 1.0
  expZ <- Amari.dualExp z
  logZ <- Amari.dualLog z

  expReal <- Amari.dualReal expZ
  expDual <- Amari.dualDual expZ
  logReal <- Amari.dualReal logZ
  logDual <- Amari.dualDual logZ

  log $ "exp(1) = " <> show expReal <> ", d/dx exp(x)|_1 = " <> show expDual
  log $ "log(1) = " <> show logReal <> ", d/dx log(x)|_1 = " <> show logDual
  log ""

-- | Demonstrate rotor-based rotations
rotorExample :: Effect Unit
rotorExample = do
  log "=== Rotor Rotation Example ==="
  log ""

  -- Create a unit bivector (rotation plane)
  -- For rotation in xy-plane, use e12
  log "Creating rotor for 90° rotation in xy-plane..."

  -- Rotor: R = cos(θ/2) - sin(θ/2) * B
  -- where B is the unit bivector in the rotation plane
  -- For θ = π/2: cos(π/4) ≈ 0.707, sin(π/4) ≈ 0.707
  let cosHalfAngle = 0.7071067811865476
  let sinHalfAngle = 0.7071067811865476

  -- Create scalar part
  scalar <- Amari.multivectorFromScalar cosHalfAngle

  -- Create bivector part (e12)
  bivector <- Amari.multivectorFromBivector [sinHalfAngle, 0.0, 0.0]

  -- Negate bivector (rotor is cos - sin*B)
  negBivector <- Amari.multivectorScale bivector (-1.0)

  -- Combine: R = cos(θ/2) - sin(θ/2)*e12
  rotor <- Amari.multivectorAdd scalar negBivector

  -- Reverse of rotor
  rotorRev <- Amari.multivectorReverse rotor

  -- Vector to rotate: v = e1
  v <- Amari.multivectorFromVector [1.0, 0.0, 0.0]

  -- Rotation: v' = R * v * R†
  temp <- Amari.multivectorMul rotor v
  rotated <- Amari.multivectorMul temp rotorRev

  -- Extract rotated vector components
  rx <- Amari.multivectorGet rotated 1
  ry <- Amari.multivectorGet rotated 2
  rz <- Amari.multivectorGet rotated 3

  log "v = (1, 0, 0)"
  log $ "Rotated v' = (" <> show rx <> ", " <> show ry <> ", " <> show rz <> ")"
  log "Expected: (0, 1, 0) for 90° rotation"
  log ""

-- | Demonstrate bivector operations
bivectorExample :: Effect Unit
bivectorExample = do
  log "=== Bivector Example ==="
  log ""

  -- Create two vectors
  a <- Amari.multivectorFromVector [1.0, 0.0, 0.0]
  b <- Amari.multivectorFromVector [1.0, 1.0, 0.0]

  -- Outer product gives bivector (parallelogram)
  bivector <- Amari.multivectorOuterProduct a b

  -- The magnitude is the area
  norm <- Amari.multivectorNorm bivector

  log "a = (1, 0, 0), b = (1, 1, 0)"
  log "a ∧ b (bivector representing parallelogram):"
  log $ "  Area = ‖a ∧ b‖ = " <> show norm

  -- Get bivector components
  e12 <- Amari.multivectorGet bivector 3
  e13 <- Amari.multivectorGet bivector 4
  e23 <- Amari.multivectorGet bivector 5

  log $ "  e12 = " <> show e12 <> ", e13 = " <> show e13 <> ", e23 = " <> show e23
  log ""

  log "=== Examples Complete ==="
