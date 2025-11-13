//! Common probability distributions

use crate::prob::Prob;
use rand::Rng;
use rand_distr::{Bernoulli as RandBernoulli, Distribution, Exp, Normal as RandNormal};

/// Uniform distribution over a range
#[derive(Clone, Copy, Debug)]
pub struct Uniform<T> {
    min: T,
    max: T,
}

impl Uniform<i32> {
    /// Create uniform distribution over [min, max)
    pub fn new(min: i32, max: i32) -> Self {
        Self { min, max }
    }

    /// Sample from the distribution
    pub fn sample(&self) -> Prob<i32> {
        let mut rng = rand::thread_rng();
        let value = rng.gen_range(self.min..self.max);
        Prob::new(value)
    }
}

/// Bernoulli distribution (binary outcome)
#[derive(Clone, Copy, Debug)]
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    /// Create Bernoulli distribution with probability p
    pub fn new(p: f64) -> Self {
        assert!((0.0..=1.0).contains(&p), "Probability must be in [0, 1]");
        Self { p }
    }

    /// Sample from the distribution
    pub fn sample(&self) -> Prob<bool> {
        let mut rng = rand::thread_rng();
        let dist = RandBernoulli::new(self.p).unwrap();
        let value = dist.sample(&mut rng);
        Prob::with_probability(self.p, value)
    }
}

/// Normal (Gaussian) distribution
#[derive(Clone, Copy, Debug)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    /// Create normal distribution with given mean and standard deviation
    pub fn new(mean: f64, std_dev: f64) -> Self {
        assert!(std_dev > 0.0, "Standard deviation must be positive");
        Self { mean, std_dev }
    }

    /// Sample from the distribution
    pub fn sample(&self) -> Prob<f64> {
        let mut rng = rand::thread_rng();
        let dist = RandNormal::new(self.mean, self.std_dev).unwrap();
        let value = dist.sample(&mut rng);
        Prob::new(value)
    }
}

/// Exponential distribution
#[derive(Clone, Copy, Debug)]
pub struct Exponential {
    lambda: f64,
}

impl Exponential {
    /// Create exponential distribution with rate parameter lambda
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "Lambda must be positive");
        Self { lambda }
    }

    /// Sample from the distribution
    pub fn sample(&self) -> Prob<f64> {
        let mut rng = rand::thread_rng();
        let dist = Exp::new(self.lambda).unwrap();
        let value = dist.sample(&mut rng);
        Prob::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_range() {
        let dist = Uniform::new(1, 7);
        for _ in 0..100 {
            let sample = dist.sample().into_inner();
            assert!((1..7).contains(&sample));
        }
    }

    #[test]
    fn test_bernoulli() {
        let dist = Bernoulli::new(0.5);
        let sample = dist.sample();
        assert_eq!(sample.probability(), 0.5);
    }

    #[test]
    fn test_normal() {
        let dist = Normal::new(0.0, 1.0);
        let sample = dist.sample();
        assert_eq!(sample.probability(), 1.0); // Sampling is deterministic
    }
}
