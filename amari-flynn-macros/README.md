# amari-flynn-macros

Procedural macros for amari-flynn probabilistic contracts.

## Overview

`amari-flynn-macros` provides procedural macro attributes for specifying probabilistic contracts on Rust functions. These macros generate documentation and verification infrastructure for statistical testing. All macros support zero-parameter, single-parameter, and multi-parameter functions.

## Installation

This crate is automatically included when you use `amari-flynn`. For direct usage:

```toml
[dependencies]
amari-flynn-macros = "0.19"
```

## Macros

### `#[prob_requires]` - Probabilistic Preconditions

Specifies that a precondition should hold with a given probability:

```rust
use amari_flynn_macros::prob_requires;

// Single parameter
#[prob_requires(x > 0.0, 0.95)]
fn compute_sqrt(x: f64) -> f64 {
    x.sqrt()
}

// Multiple parameters
#[prob_requires(x > 0.0 && y > 0.0, 0.9)]
fn product_positive(x: f64, y: f64) -> f64 {
    x * y
}
```

**Syntax**: `#[prob_requires(condition, probability_bound)]`

- `condition`: Boolean expression over function parameters
- `probability_bound`: Minimum probability the condition holds (0.0 to 1.0)

### `#[prob_ensures]` - Probabilistic Postconditions

Specifies that a postcondition should hold with a given probability:

```rust
use amari_flynn_macros::prob_ensures;

// Single parameter
#[prob_ensures(result >= 0.0, 0.99)]
fn safe_operation(x: f64) -> f64 {
    x.abs()
}

// Multiple parameters
#[prob_ensures(result >= 0.0, 0.99)]
fn sum_abs(x: f64, y: f64) -> f64 {
    x.abs() + y.abs()
}

// Zero parameters
#[prob_ensures(result >= 0.0, 0.99)]
fn constant_value() -> f64 {
    42.0
}
```

**Syntax**: `#[prob_ensures(condition, probability_bound)]`

- `condition`: Boolean expression over `result` (the return value)
- `probability_bound`: Minimum probability the condition holds (0.0 to 1.0)

### `#[ensures_expected]` - Expected Value Constraints

Specifies that the expected value of an expression should be within bounds:

```rust
use amari_flynn_macros::ensures_expected;

// Zero parameters
#[ensures_expected(result, 0.5, 0.15)]
fn biased_coin() -> f64 {
    if rand::random::<bool>() { 1.0 } else { 0.0 }
}

// Multiple parameters
#[ensures_expected(result, 0.0, 0.5)]
fn symmetric_diff(x: f64, y: f64) -> f64 {
    x - y
}
```

**Syntax**: `#[ensures_expected(expression, expected_value, epsilon)]`

- `expression`: Expression to evaluate (typically `result`)
- `expected_value`: Expected mean value
- `epsilon`: Maximum deviation from expected value

## Generated Code

Each macro generates:

1. **Documentation**: Doc comments describing the probabilistic contract
2. **Verification Helper**: A function for statistical verification

### Single-Parameter Example

```rust
#[prob_requires(x > 0.0, 0.95)]
fn compute(x: f64) -> f64 { x.sqrt() }

// Generates:
fn verify_compute_precondition<F>(
    input_generator: F,
    samples: usize,
) -> amari_flynn::contracts::VerificationResult
where
    F: Fn() -> f64,
{
    let verifier = amari_flynn::backend::monte_carlo::MonteCarloVerifier::new(samples);
    verifier.verify_probability_bound(
        || { let x = input_generator(); x > 0.0 },
        0.95,
    )
}
```

### Multi-Parameter Example

```rust
#[prob_requires(x > 0.0 && y > 0.0, 0.9)]
fn product(x: f64, y: f64) -> f64 { x * y }

// Generates:
fn verify_product_precondition<F>(
    input_generator: F,
    samples: usize,
) -> amari_flynn::contracts::VerificationResult
where
    F: Fn() -> (f64, f64),  // Tuple of parameter types
{
    let verifier = amari_flynn::backend::monte_carlo::MonteCarloVerifier::new(samples);
    verifier.verify_probability_bound(
        || { let (x, y) = input_generator(); x > 0.0 && y > 0.0 },
        0.9,
    )
}
```

## Usage in Tests

```rust
use amari_flynn::prelude::*;
use amari_flynn::{prob_requires, prob_ensures};

#[prob_requires(x > 0.0, 0.95)]
fn sqrt_positive(x: f64) -> f64 {
    if x > 0.0 { x.sqrt() } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sqrt_precondition() {
        let result = verify_sqrt_positive_precondition(
            || rand::random::<f64>().abs() + 0.001,
            1000,
        );
        assert!(matches!(
            result,
            VerificationResult::Verified | VerificationResult::Inconclusive
        ));
    }
}
```

## Combining Multiple Contracts

Contracts can be stacked:

```rust
use amari_flynn::{prob_requires, prob_ensures, ensures_expected};

#[prob_requires(x > 0.0, 0.95)]
#[prob_ensures(result > 0.0, 0.99)]
#[ensures_expected(result, 1.0, 0.5)]
fn probabilistic_function(x: f64) -> f64 {
    x.sqrt() * (0.5 + rand::random::<f64>())
}
```

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
