// FFI bindings for amari-wasm
// This file provides the JavaScript implementation for the PureScript FFI

let amari = null;

// Initialize the WASM module
export const initAmariImpl = function(onError, onSuccess) {
  return function() {
    import('@justinelliottcobb/amari-wasm').then(function(module) {
      amari = module;
      onSuccess();
    }).catch(function(err) {
      onError(err);
    });
  };
};

// Result helpers
export const isOkImpl = function(result) {
  return result && result.ok !== undefined ? result.ok : true;
};

export const unwrapImpl = function(result) {
  return result && result.value !== undefined ? result.value : result;
};

// Multivector Construction
export const multivectorNewImpl = function(p, q, r) {
  return function() {
    return new amari.Multivector(p, q, r);
  };
};

export const multivectorFromScalarImpl = function(s) {
  return function() {
    return amari.Multivector.from_scalar(s);
  };
};

export const multivectorFromVectorImpl = function(components) {
  return function() {
    return amari.Multivector.from_vector(components);
  };
};

export const multivectorFromBivectorImpl = function(components) {
  return function() {
    return amari.Multivector.from_bivector(components);
  };
};

export const multivectorZeroImpl = function(p, q, r) {
  return function() {
    return amari.Multivector.zero(p, q, r);
  };
};

export const multivectorOneImpl = function(p, q, r) {
  return function() {
    return amari.Multivector.one(p, q, r);
  };
};

// Multivector Operations
export const multivectorAddImpl = function(a, b) {
  return function() {
    return a.add(b);
  };
};

export const multivectorSubImpl = function(a, b) {
  return function() {
    return a.sub(b);
  };
};

export const multivectorMulImpl = function(a, b) {
  return function() {
    return a.mul(b);
  };
};

export const multivectorScaleImpl = function(mv, s) {
  return function() {
    return mv.scale(s);
  };
};

export const multivectorGeometricProductImpl = function(a, b) {
  return function() {
    return a.geometric_product(b);
  };
};

export const multivectorOuterProductImpl = function(a, b) {
  return function() {
    return a.outer_product(b);
  };
};

export const multivectorInnerProductImpl = function(a, b) {
  return function() {
    return a.inner_product(b);
  };
};

export const multivectorReverseImpl = function(mv) {
  return function() {
    return mv.reverse();
  };
};

export const multivectorConjugateImpl = function(mv) {
  return function() {
    return mv.conjugate();
  };
};

export const multivectorNormImpl = function(mv) {
  return function() {
    return mv.norm();
  };
};

export const multivectorNormalizeImpl = function(mv) {
  return function() {
    return mv.normalize();
  };
};

export const multivectorInverseImpl = function(mv) {
  return function() {
    try {
      return { ok: true, value: mv.inverse() };
    } catch (e) {
      return { ok: false, error: e.message };
    }
  };
};

// Component Access
export const multivectorGetImpl = function(mv, i) {
  return function() {
    return mv.get(i);
  };
};

export const multivectorSetImpl = function(mv, i, v) {
  return function() {
    return mv.set(i, v);
  };
};

export const multivectorGradeImpl = function(mv, k) {
  return function() {
    return mv.grade(k);
  };
};

export const multivectorDimensionImpl = function(mv) {
  return function() {
    return mv.dimension();
  };
};

// Dual Numbers
export const dualNewImpl = function(r, d) {
  return function() {
    return new amari.Dual(r, d);
  };
};

export const dualFromRealImpl = function(r) {
  return function() {
    return amari.Dual.from_real(r);
  };
};

export const dualVariableImpl = function(r) {
  return function() {
    return amari.Dual.variable(r);
  };
};

export const dualAddImpl = function(a, b) {
  return function() {
    return a.add(b);
  };
};

export const dualMulImpl = function(a, b) {
  return function() {
    return a.mul(b);
  };
};

export const dualSinImpl = function(d) {
  return function() {
    return d.sin();
  };
};

export const dualCosImpl = function(d) {
  return function() {
    return d.cos();
  };
};

export const dualExpImpl = function(d) {
  return function() {
    return d.exp();
  };
};

export const dualLogImpl = function(d) {
  return function() {
    return d.log();
  };
};

export const dualPowImpl = function(d, n) {
  return function() {
    return d.pow(n);
  };
};

export const dualRealImpl = function(d) {
  return function() {
    return d.real();
  };
};

export const dualDualImpl = function(d) {
  return function() {
    return d.dual();
  };
};
