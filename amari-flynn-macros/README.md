# amari-flynn-macros

Procedural macros for amari-flynn probabilistic contracts.

## Overview

`amari-flynn-macros` provides procedural macro attributes for specifying probabilistic contracts on Rust functions. These macros generate documentation and verification infrastructure for statistical testing.

## Installation

This crate is automatically included when you use `amari-flynn`. For direct usage:

```toml
[dependencies]
amari-flynn-macros = "0.12"
```

## Macros

### `#[prob_requires]` - Probabilistic Preconditions

Specifies that a precondition should hold with a given probability:

```rust
use amari_flynn_macros::prob_requires;

/// Function that expects positive input with 95% probability
#[prob_requires(x > 0.0, 0.95)]
fn compute_sqrt(x: f64) -> f64 {
    x.sqrt()
}
```

**Syntax**: `#[prob_requires(condition, probability_bound)]`

- `condition`: Boolean expression over function parameters
- `probability_bound`: Minimum probability the condition holds (0.0 to 1.0)

### `#[prob_ensures]` - Probabilistic Postconditions

Specifies that a postcondition should hold with a given probability:

```rust
use amari_flynn_macros::prob_ensures;

/// Result is non-negative with 99% probability
#[prob_ensures(result >= 0.0, 0.99)]
fn safe_operation(x: f64) -> f64 {
    x.abs()
}
```

**Syntax**: `#[prob_ensures(condition, probability_bound)]`

- `condition`: Boolean expression over `result` (the return value)
- `probability_bound`: Minimum probability the condition holds (0.0 to 1.0)

### `#[ensures_expected]` - Expected Value Constraints

Specifies that the expected value of an expression should be within bounds:

```rust
use amari_flynn_macros::ensures_expected;

/// Result should have expected value 5.0 ± 0.1
#[ensures_expected(result, 5.0, 0.1)]
fn random_around_five() -> f64 {
    5.0 + (rand::random::<f64>() - 0.5) * 0.2
}
```

**Syntax**: `#[ensures_expected(expression, expected_value, epsilon)]`

- `expression`: Expression to evaluate (typically `result`)
- `expected_value`: Expected mean value
- `epsilon`: Maximum deviation from expected value

## Generated Code

Each macro generates:

1. **Documentation**: Doc comments describing the probabilistic contract
2. **Verification Helper**: A test function for statistical verification (under `#[cfg(test)]`)

Example generated verification helper:

```rust
#[cfg(test)]
fn verify_compute_sqrt_precondition<F>(
    input_generator: F,
    samples: usize,
) -> amari_flynn::contracts::VerificationResult
where
    F: Fn() -> f64,
{
    let verifier = amari_flynn::backend::monte_carlo::MonteCarloVerifier::new(samples);
    verifier.verify_probability_bound(
        || {
            let x = input_generator();
            x > 0.0
        },
        0.95,
    )
}
```

## Usage with amari-flynn

The macros integrate with the Flynn verification framework:

```rust
use amari_flynn::prelude::*;
use amari_flynn_macros::{prob_requires, prob_ensures};

#[prob_requires(input.len() > 0, 0.99)]
#[prob_ensures(result.is_some(), 0.95)]
fn find_element(input: &[i32], target: i32) -> Option<usize> {
    input.iter().position(|&x| x == target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_element_contracts() {
        // Use generated verification helpers
        let result = verify_find_element_precondition(
            || vec![1, 2, 3, 4, 5],
            10_000,
        );
        assert!(matches!(result, VerificationResult::Verified { .. }));
    }
}
```

## Combining Multiple Contracts

Contracts can be stacked:

```rust
use amari_flynn_macros::{prob_requires, prob_ensures, ensures_expected};

#[prob_requires(x > 0.0, 0.95)]
#[prob_ensures(result > 0.0, 0.99)]
#[ensures_expected(result, 1.0, 0.5)]
fn probabilistic_function(x: f64) -> f64 {
    x.sqrt() * (0.5 + rand::random::<f64>())
}
```

## Limitations

- Single-parameter functions have full verification helper support
- Multi-parameter functions generate documentation but simplified verification
- Zero-parameter functions are fully supported

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
