# Probability on Geometric Algebra (amari-probabilistic)

*Added in v0.13.0*

Distributions on multivector spaces, MCMC sampling, and Monte Carlo estimation.

## Quick Start

```typescript
import init, {
  WasmGaussianMultivector,
  WasmUniformMultivector,
  WasmMonteCarloEstimator
} from '@justinelliottcobb/amari-wasm';

async function probabilisticDemo() {
  await init();

  // Create a standard Gaussian distribution on Cl(3,0,0)
  const gaussian = WasmGaussianMultivector.standard();

  // Draw samples
  const samples = [];
  for (let i = 0; i < 1000; i++) {
    samples.push(gaussian.sample());
  }
  console.log(`Drew ${samples.length} Gaussian samples`);

  // Compute log probability
  const sample = gaussian.sample();
  const logProb = gaussian.logProb(sample);
  console.log(`Log probability: ${logProb}`);

  // Grade-concentrated distribution (e.g., only on bivectors)
  const bivectorDist = WasmGaussianMultivector.gradeConcentrated(2, 1.0);

  // Uniform distribution on unit multivectors
  const uniform = WasmUniformMultivector.unitSphere();

  // Monte Carlo estimation
  const estimator = new WasmMonteCarloEstimator();
  const estimate = estimator.estimate((mv) => mv.norm(), gaussian, 10000);
  console.log(`Expected norm: ${estimate.mean} +/- ${estimate.stdError}`);

  // Clean up WASM memory
  gaussian.free(); bivectorDist.free(); uniform.free();
  sample.free(); estimator.free();
  samples.forEach(s => s.free());
}

probabilisticDemo();
```

## API Reference

### WasmGaussianMultivector

- `standard()`: Create standard Gaussian on full multivector space
- `new(mean, covariance)`: Create with specified mean and covariance
- `gradeConcentrated(grade, scale)`: Gaussian concentrated on specific grade
- `sample()`: Draw a random sample
- `logProb(sample)`: Compute log probability density

### WasmUniformMultivector

- `unitSphere()`: Uniform distribution on unit multivectors
- `gradeSimplex(grade)`: Uniform on grade components summing to 1
- `sample()`: Draw a random sample

### WasmMonteCarloEstimator

- `estimate(fn, distribution, nSamples)`: Estimate expectation
- `estimateVariance(fn, distribution, nSamples)`: Estimate variance

## Use Cases

- **Bayesian Inference**: Probabilistic modeling on geometric algebra spaces
- **Uncertainty Quantification**: Monte Carlo methods for error propagation
- **Stochastic Processes**: MCMC sampling on multivector spaces
