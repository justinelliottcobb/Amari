# amari-flynn

Probabilistic contracts and verification - named after Kevin Flynn's acceptance of spontaneous perfection.

## Overview

`amari-flynn` implements probabilistic verification, distinguishing between impossible, rare, and emergent events. Named after Kevin Flynn from Tron: Legacy, who discovered that spontaneous, imperfect emergence (the ISOs) represented a form of perfection beyond rigid determinism.

This library embodies that philosophy: formal verification should prove what's impossible while allowing rare, emergent possibilities.

## Features

- **Probabilistic Contracts**: Verify statistical properties of code
- **Event Classification**: Distinguish impossible (P=0), rare (0<P<<1), and emergent (P>0) events
- **Monte Carlo Verification**: Statistical verification of probability bounds
- **Distribution Types**: Uniform, Normal, Bernoulli, Exponential distributions
- **Procedural Macros**: `#[prob_requires]`, `#[prob_ensures]`, `#[ensures_expected]`
- **Rare Event Tracking**: Monitor and bound low-probability events

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
amari-flynn = "0.12"
```

### Feature Flags

```toml
[dependencies]
# Default features
amari-flynn = "0.12"

# Minimal, no-std compatible
amari-flynn = { version = "0.12", default-features = false }
```

## Quick Start

```rust
use amari_flynn::prelude::*;

// Create probabilistic values
let coin_flip = Prob::with_probability(0.5, true);

// Sample from distributions
let die_roll = Uniform::new(1, 6).sample();

// Track rare events
let miracle_shot = RareEvent::<()>::new(0.001, "critical_hit");
```

## The Flynn Philosophy

### Three Types of Events

Flynn teaches us to distinguish between three categories:

1. **Impossible** (P=0): Formally proven to never occur
   - Violates mathematical axioms
   - Proved unreachable via formal verification
   - System invariants guarantee exclusion

2. **Rare** (0 < P << 1): Bounded probability, tracked and verified
   - Low but non-zero probability
   - Statistical bounds verified via Monte Carlo
   - Tracked as legitimate possibilities

3. **Emergent** (P > 0): Possible but not prescribed
   - Not predicted or designed
   - Arise spontaneously from system rules
   - The "ISOs" of your system

## Probabilistic Contracts

### Using Macros

```rust
use amari_flynn::prelude::*;
use amari_flynn_macros::{prob_requires, prob_ensures};

// Precondition: x > 0 holds with P ≥ 0.95
#[prob_requires(x > 0.0, 0.95)]
fn compute(x: f64) -> f64 {
    x.sqrt()
}

// Postcondition: result is non-negative with P ≥ 0.99
#[prob_ensures(result >= 0.0, 0.99)]
fn safe_compute(x: f64) -> f64 {
    x.abs()
}
```

### Monte Carlo Verification

```rust
use amari_flynn::prelude::*;

// Create verifier with 10,000 samples
let verifier = MonteCarloVerifier::new(10_000);

// Verify a probability bound
let result = verifier.verify_probability_bound(
    || rand::random::<f64>() > 0.9, // Event: random > 0.9
    0.15, // Should occur with P ≤ 0.15
);

match result {
    VerificationResult::Verified { .. } => println!("Bound verified!"),
    VerificationResult::Failed { .. } => println!("Bound violated!"),
    _ => {}
}
```

## Key Types

### Prob<T>

Probabilistic values with associated probability:

```rust
use amari_flynn::prelude::*;

// Value with 70% probability
let likely = Prob::with_probability(0.7, "success");

// Sample the value
if likely.sample() {
    println!("Got: {}", likely.value());
}
```

### Distributions

```rust
use amari_flynn::prelude::*;

// Uniform distribution over [1, 6]
let die = Uniform::new(1, 6);

// Normal distribution N(0, 1)
let standard_normal = Normal::new(0.0, 1.0);

// Bernoulli with P(true) = 0.3
let coin = Bernoulli::new(0.3);

// Exponential with rate λ = 2.0
let waiting_time = Exponential::new(2.0);
```

### RareEvent<T>

Track events that should occur rarely:

```rust
use amari_flynn::prelude::*;

// Critical hit with 1% chance
let crit = RareEvent::<()>::new(0.01, "critical_hit");

// Network timeout with 0.1% chance
let timeout = RareEvent::<String>::new(0.001, "network_timeout");
```

## Modules

| Module | Description |
|--------|-------------|
| `prob` | Core Prob<T> type and probability operations |
| `distributions` | Statistical distributions |
| `contracts` | ProbabilisticContract, RareEvent, VerificationResult |
| `backend` | Verification backends (Monte Carlo) |
| `statistical` | Statistical analysis utilities |

## Use Cases

- **Game Mechanics**: Verify critical hit rates, loot drops stay within bounds
- **Reliability Engineering**: Bound failure probabilities in distributed systems
- **Financial Modeling**: Verify risk bounds in Monte Carlo simulations
- **Quality Assurance**: Statistical testing of randomized algorithms

## Example: Game Mechanics

```rust
use amari_flynn::prelude::*;

// Verify critical hit rate is bounded
let crit_rate = 0.15;
let crit_prob = Prob::with_probability(crit_rate, ());

// Statistical verification that P(crit) ≤ 0.20
// (leaving room for buffs/modifiers while maintaining balance)

let verifier = MonteCarloVerifier::new(50_000);
let result = verifier.verify_probability_bound(
    || rand::random::<f64>() < crit_rate,
    0.20, // upper bound
);
```

## The ISO Philosophy

> "The ISOs, they were a miracle. They weren't meant to be - they just were."
>
> - Kevin Flynn

Like the ISOs in Tron: Legacy, the most valuable behaviors in a system are often those that emerge spontaneously, unpredicted by design. Flynn enables you to prove safety boundaries while preserving space for emergence.

## Roadmap

- **Why3 Integration**: Formal verification of probability bounds
- **Creusot Support**: Rust-native formal verification
- **SMT Backend**: Automated theorem proving for event impossibility
- **Geometric Integration**: Probabilistic contracts over geometric algebra types

## License

Licensed under either of Apache License, Version 2.0 or MIT License at your option.

## Part of Amari

This crate is part of the [Amari](https://github.com/justinelliottcobb/Amari) mathematical computing library.
