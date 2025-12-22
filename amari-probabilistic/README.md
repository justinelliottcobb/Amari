# amari-probabilistic

Probability theory on geometric algebra spaces for the Amari mathematical computing library.

## Overview

`amari-probabilistic` provides probability distributions, stochastic processes, MCMC sampling, and Bayesian inference for multivector-valued random variables in Clifford algebras Cl(P,Q,R).

This crate builds on:
- `amari-measure` for measure-theoretic foundations
- `amari-info-geom` for Fisher-Riemannian geometry integration
- `amari-core` for the underlying Multivector type

## Features

### Distributions

Probability distributions over multivector spaces:

```rust
use amari_probabilistic::distribution::{Distribution, GaussianMultivector};
use amari_core::Multivector;

// Standard Gaussian on Cl(2,0,0)
let gaussian = GaussianMultivector::<2, 0, 0>::standard();

// Sample from the distribution
let mut rng = rand::thread_rng();
let sample: Multivector<2, 0, 0> = gaussian.sample(&mut rng);

// Evaluate log-probability
let log_p = gaussian.log_prob(&sample).unwrap();
```

### MCMC Sampling

Metropolis-Hastings and Hamiltonian Monte Carlo:

```rust
use amari_probabilistic::sampling::{MetropolisHastings, Sampler};

// Create sampler with proposal standard deviation
let mut sampler = MetropolisHastings::new(&target_distribution, 0.1);

// Run MCMC with burnin
let samples = sampler.run(&mut rng, 1000, 100)?;

// Check diagnostics
let diag = sampler.diagnostics();
println!("Acceptance rate: {}", diag.acceptance_rate);
```

### Stochastic Differential Equations

SDE solvers for multivector-valued stochastic processes:

```rust
use amari_probabilistic::stochastic::{
    GeometricBrownianMotion, EulerMaruyama, SDESolver
};

// Geometric Brownian motion: dX = μX dt + σX dW
let gbm = GeometricBrownianMotion::<2, 0, 0>::new(0.05, 0.2)?;

// Solve with Euler-Maruyama
let solver = EulerMaruyama::new();
let initial = Multivector::scalar(1.0);
let path = solver.solve(&gbm, initial, 0.0, 1.0, 100, &mut rng)?;
```

### Bayesian Inference

Bayesian models on geometric algebra spaces:

```rust
use amari_probabilistic::bayesian::{BayesianGA, GaussianPrior};

// Define prior
let prior = GaussianPrior::<2, 0, 0>::diffuse();

// Define likelihood function
let likelihood = |theta: &Multivector<2, 0, 0>, data: &[Multivector<2, 0, 0>]| {
    // Compute log-likelihood
    Ok(log_lik)
};

// Create Bayesian model
let mut model = BayesianGA::new(prior, likelihood);
model.observe(observations);

// Sample from posterior
let posterior_samples = model.sample_posterior(&mut rng, 1000, 100, 0.1)?;
```

### Uncertainty Propagation

Propagate uncertainty through geometric operations:

```rust
use amari_probabilistic::uncertainty::{UncertainMultivector, LinearPropagation};

// Create uncertain multivector with mean and covariance
let um = UncertainMultivector::diagonal(mean, &variances)?;

// Propagate through operations
let prop = LinearPropagation::new();
let result = prop.scalar_multiply(&um, 2.0);
```

## Module Structure

- **`distribution`**: Core `Distribution` trait and implementations
  - `GaussianMultivector`: Gaussian distribution on multivector space
  - `UniformMultivector`: Uniform distribution on hypercube
  - `GradeProjectedDistribution`: Marginal distribution on specific grades

- **`random`**: Random variable traits and moment computation
  - `GeometricRandomVariable`: Trait for RVs with computable moments
  - `CovarianceMatrix`: Covariance representation
  - `MomentComputer`: Monte Carlo moment estimation

- **`sampling`**: MCMC and sampling algorithms
  - `MetropolisHastings`: Random-walk Metropolis
  - `HamiltonianMonteCarlo`: HMC with leapfrog integration
  - `RejectionSampling`: Accept-reject sampling

- **`stochastic`**: Stochastic processes and SDE solvers
  - `StochasticProcess`: Trait for SDEs
  - `EulerMaruyama`: First-order SDE solver
  - `Milstein`: Higher-order SDE solver
  - `WienerProcess`, `GeometricBrownianMotion`, `OrnsteinUhlenbeck`

- **`bayesian`**: Bayesian inference
  - `BayesianGA`: Bayesian model with prior and likelihood
  - `JeffreysPrior`: Non-informative prior
  - `GaussianPrior`: Conjugate Gaussian prior
  - `PosteriorSampler`: Posterior analysis utilities

- **`uncertainty`**: Uncertainty propagation
  - `LinearPropagation`: Jacobian-based propagation
  - `UnscentedTransform`: Sigma-point propagation

- **`monte_carlo`**: Monte Carlo integration extensions

- **`phantom`**: Compile-time distribution property markers
  - `Bounded`, `LightTailed`, `RotorValued`, etc.

## Error Handling

The crate uses `ProbabilisticError` for error handling:

```rust
pub enum ProbabilisticError {
    NotNormalized { total: f64 },
    OutOfSupport { sample: String },
    SamplerNotConverged { iterations: usize, reason: String },
    SDEInstability { time: f64, details: String },
    InvalidParameters { description: String },
    // ...
}
```

## Feature Flags

- `std` (default): Standard library support
- `parallel`: Parallel sampling with Rayon
- `gpu`: GPU-accelerated sampling via `amari-gpu`

## Mathematical Background

### Distributions on Cl(P,Q,R)

A probability distribution on the Clifford algebra Cl(P,Q,R) assigns probabilities to multivector-valued events. The dimension of the space is 2^(P+Q+R).

### Stochastic Differential Equations

SDEs on multivector spaces take the form:
```
dX = μ(X,t) dt + σ(X,t) dW
```
where μ is the drift (a multivector), σ is the diffusion coefficient, and W is a Wiener process.

### Bayesian Inference

For a parameter θ (a multivector), data D, prior p(θ), and likelihood p(D|θ):
```
p(θ|D) ∝ p(D|θ) p(θ)
```

## License

MIT OR Apache-2.0
