-- | PureScript FFI bindings for amari-wasm
-- |
-- | This module provides type-safe access to the Amari WASM library
-- | for geometric algebra and multivector operations.
module Amari.Wasm
  ( -- * Types
    Multivector
  , MultivectorClass
  , Dual
  , DualClass
  , Result

  -- * Initialization
  , initAmari

  -- * Multivector Construction
  , multivectorNew
  , multivectorFromScalar
  , multivectorFromVector
  , multivectorFromBivector
  , multivectorZero
  , multivectorOne

  -- * Multivector Operations
  , multivectorAdd
  , multivectorSub
  , multivectorMul
  , multivectorScale
  , multivectorGeometricProduct
  , multivectorOuterProduct
  , multivectorInnerProduct
  , multivectorReverse
  , multivectorConjugate
  , multivectorNorm
  , multivectorNormalize
  , multivectorInverse

  -- * Component Access
  , multivectorGet
  , multivectorSet
  , multivectorGrade
  , multivectorDimension

  -- * Dual Numbers
  , dualNew
  , dualFromReal
  , dualVariable
  , dualAdd
  , dualMul
  , dualSin
  , dualCos
  , dualExp
  , dualLog
  , dualPow
  , dualReal
  , dualDual

  -- * Utilities
  , isOk
  , unwrap
  , fromResult
  ) where

import Prelude

import Data.Either (Either(..))
import Data.Function.Uncurried (Fn2, Fn3, runFn2, runFn3)
import Effect (Effect)
import Effect.Aff (Aff)
import Effect.Aff.Compat (EffectFnAff, fromEffectFnAff)
import Foreign (Foreign)

-- | Opaque type for WASM Multivector
foreign import data Multivector :: Type

-- | Class constraint for Multivector operations
class MultivectorClass a where
  toMultivector :: a -> Multivector
  fromMultivector :: Multivector -> a

instance multivectorClassMultivector :: MultivectorClass Multivector where
  toMultivector = identity
  fromMultivector = identity

-- | Opaque type for WASM Dual numbers
foreign import data Dual :: Type

-- | Class constraint for Dual number operations
class DualClass a where
  toDual :: a -> Dual
  fromDual :: Dual -> a

instance dualClassDual :: DualClass Dual where
  toDual = identity
  fromDual = identity

-- | Result type from WASM operations
foreign import data Result :: Type -> Type

-- | Initialize the WASM module
foreign import initAmariImpl :: EffectFnAff Unit

initAmari :: Aff Unit
initAmari = fromEffectFnAff initAmariImpl

-- | Check if result is Ok
foreign import isOkImpl :: forall a. Result a -> Boolean

isOk :: forall a. Result a -> Boolean
isOk = isOkImpl

-- | Unwrap a successful result (unsafe)
foreign import unwrapImpl :: forall a. Result a -> a

unwrap :: forall a. Result a -> a
unwrap = unwrapImpl

-- | Convert Result to Either
fromResult :: forall a. Result a -> Either String a
fromResult r =
  if isOk r
    then Right (unwrap r)
    else Left "WASM operation failed"

--------------------------------------------------------------------------------
-- Multivector Construction
--------------------------------------------------------------------------------

-- | Create a new multivector with specified signature
foreign import multivectorNewImpl :: Fn3 Int Int Int (Effect Multivector)

multivectorNew :: Int -> Int -> Int -> Effect Multivector
multivectorNew p q r = runFn3 multivectorNewImpl p q r

-- | Create a multivector from a scalar value
foreign import multivectorFromScalarImpl :: Number -> Effect Multivector

multivectorFromScalar :: Number -> Effect Multivector
multivectorFromScalar = multivectorFromScalarImpl

-- | Create a multivector from vector components
foreign import multivectorFromVectorImpl :: Array Number -> Effect Multivector

multivectorFromVector :: Array Number -> Effect Multivector
multivectorFromVector = multivectorFromVectorImpl

-- | Create a multivector from bivector components
foreign import multivectorFromBivectorImpl :: Array Number -> Effect Multivector

multivectorFromBivector :: Array Number -> Effect Multivector
multivectorFromBivector = multivectorFromBivectorImpl

-- | Zero multivector
foreign import multivectorZeroImpl :: Fn3 Int Int Int (Effect Multivector)

multivectorZero :: Int -> Int -> Int -> Effect Multivector
multivectorZero p q r = runFn3 multivectorZeroImpl p q r

-- | Identity multivector (scalar 1)
foreign import multivectorOneImpl :: Fn3 Int Int Int (Effect Multivector)

multivectorOne :: Int -> Int -> Int -> Effect Multivector
multivectorOne p q r = runFn3 multivectorOneImpl p q r

--------------------------------------------------------------------------------
-- Multivector Operations
--------------------------------------------------------------------------------

-- | Add two multivectors
foreign import multivectorAddImpl :: Fn2 Multivector Multivector (Effect Multivector)

multivectorAdd :: Multivector -> Multivector -> Effect Multivector
multivectorAdd a b = runFn2 multivectorAddImpl a b

-- | Subtract two multivectors
foreign import multivectorSubImpl :: Fn2 Multivector Multivector (Effect Multivector)

multivectorSub :: Multivector -> Multivector -> Effect Multivector
multivectorSub a b = runFn2 multivectorSubImpl a b

-- | Multiply two multivectors (geometric product)
foreign import multivectorMulImpl :: Fn2 Multivector Multivector (Effect Multivector)

multivectorMul :: Multivector -> Multivector -> Effect Multivector
multivectorMul a b = runFn2 multivectorMulImpl a b

-- | Scale a multivector by a scalar
foreign import multivectorScaleImpl :: Fn2 Multivector Number (Effect Multivector)

multivectorScale :: Multivector -> Number -> Effect Multivector
multivectorScale mv s = runFn2 multivectorScaleImpl mv s

-- | Geometric product
foreign import multivectorGeometricProductImpl :: Fn2 Multivector Multivector (Effect Multivector)

multivectorGeometricProduct :: Multivector -> Multivector -> Effect Multivector
multivectorGeometricProduct a b = runFn2 multivectorGeometricProductImpl a b

-- | Outer (wedge) product
foreign import multivectorOuterProductImpl :: Fn2 Multivector Multivector (Effect Multivector)

multivectorOuterProduct :: Multivector -> Multivector -> Effect Multivector
multivectorOuterProduct a b = runFn2 multivectorOuterProductImpl a b

-- | Inner (dot) product
foreign import multivectorInnerProductImpl :: Fn2 Multivector Multivector (Effect Multivector)

multivectorInnerProduct :: Multivector -> Multivector -> Effect Multivector
multivectorInnerProduct a b = runFn2 multivectorInnerProductImpl a b

-- | Reverse of a multivector
foreign import multivectorReverseImpl :: Multivector -> Effect Multivector

multivectorReverse :: Multivector -> Effect Multivector
multivectorReverse = multivectorReverseImpl

-- | Clifford conjugate
foreign import multivectorConjugateImpl :: Multivector -> Effect Multivector

multivectorConjugate :: Multivector -> Effect Multivector
multivectorConjugate = multivectorConjugateImpl

-- | Magnitude of a multivector
foreign import multivectorNormImpl :: Multivector -> Effect Number

multivectorNorm :: Multivector -> Effect Number
multivectorNorm = multivectorNormImpl

-- | Normalize a multivector
foreign import multivectorNormalizeImpl :: Multivector -> Effect Multivector

multivectorNormalize :: Multivector -> Effect Multivector
multivectorNormalize = multivectorNormalizeImpl

-- | Inverse of a multivector
foreign import multivectorInverseImpl :: Multivector -> Effect (Result Multivector)

multivectorInverse :: Multivector -> Effect (Result Multivector)
multivectorInverse = multivectorInverseImpl

--------------------------------------------------------------------------------
-- Component Access
--------------------------------------------------------------------------------

-- | Get component at index
foreign import multivectorGetImpl :: Fn2 Multivector Int (Effect Number)

multivectorGet :: Multivector -> Int -> Effect Number
multivectorGet mv i = runFn2 multivectorGetImpl mv i

-- | Set component at index
foreign import multivectorSetImpl :: Fn3 Multivector Int Number (Effect Multivector)

multivectorSet :: Multivector -> Int -> Number -> Effect Multivector
multivectorSet mv i v = runFn3 multivectorSetImpl mv i v

-- | Extract grade-k part
foreign import multivectorGradeImpl :: Fn2 Multivector Int (Effect Multivector)

multivectorGrade :: Multivector -> Int -> Effect Multivector
multivectorGrade mv k = runFn2 multivectorGradeImpl mv k

-- | Get dimension of the algebra
foreign import multivectorDimensionImpl :: Multivector -> Effect Int

multivectorDimension :: Multivector -> Effect Int
multivectorDimension = multivectorDimensionImpl

--------------------------------------------------------------------------------
-- Dual Numbers for Automatic Differentiation
--------------------------------------------------------------------------------

-- | Create a dual number with real and dual parts
foreign import dualNewImpl :: Fn2 Number Number (Effect Dual)

dualNew :: Number -> Number -> Effect Dual
dualNew r d = runFn2 dualNewImpl r d

-- | Create a dual number from a real (constant)
foreign import dualFromRealImpl :: Number -> Effect Dual

dualFromReal :: Number -> Effect Dual
dualFromReal = dualFromRealImpl

-- | Create a dual number representing a variable (for differentiation)
foreign import dualVariableImpl :: Number -> Effect Dual

dualVariable :: Number -> Effect Dual
dualVariable = dualVariableImpl

-- | Add dual numbers
foreign import dualAddImpl :: Fn2 Dual Dual (Effect Dual)

dualAdd :: Dual -> Dual -> Effect Dual
dualAdd a b = runFn2 dualAddImpl a b

-- | Multiply dual numbers
foreign import dualMulImpl :: Fn2 Dual Dual (Effect Dual)

dualMul :: Dual -> Dual -> Effect Dual
dualMul a b = runFn2 dualMulImpl a b

-- | Sine of dual number
foreign import dualSinImpl :: Dual -> Effect Dual

dualSin :: Dual -> Effect Dual
dualSin = dualSinImpl

-- | Cosine of dual number
foreign import dualCosImpl :: Dual -> Effect Dual

dualCos :: Dual -> Effect Dual
dualCos = dualCosImpl

-- | Exponential of dual number
foreign import dualExpImpl :: Dual -> Effect Dual

dualExp :: Dual -> Effect Dual
dualExp = dualExpImpl

-- | Natural log of dual number
foreign import dualLogImpl :: Dual -> Effect Dual

dualLog :: Dual -> Effect Dual
dualLog = dualLogImpl

-- | Power of dual number
foreign import dualPowImpl :: Fn2 Dual Number (Effect Dual)

dualPow :: Dual -> Number -> Effect Dual
dualPow d n = runFn2 dualPowImpl d n

-- | Get the real part of a dual number
foreign import dualRealImpl :: Dual -> Effect Number

dualReal :: Dual -> Effect Number
dualReal = dualRealImpl

-- | Get the dual (derivative) part
foreign import dualDualImpl :: Dual -> Effect Number

dualDual :: Dual -> Effect Number
dualDual = dualDualImpl
