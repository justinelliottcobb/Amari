//! Common functions with automatic differentiation support

use crate::DualNumber;
use alloc::vec::Vec;
use num_traits::Float;

/// Softmax function with automatic differentiation
pub fn softmax<T: Float>(inputs: &[DualNumber<T>]) -> Vec<DualNumber<T>> {
    if inputs.is_empty() {
        return Vec::new();
    }
    
    // Find max for numerical stability
    let max_val = inputs.iter().map(|x| x.real).fold(inputs[0].real, |a, b| a.max(b));
    
    // Compute exp(x - max) for each input
    let exp_vals: Vec<DualNumber<T>> = inputs.iter()
        .map(|&x| (x - DualNumber::constant(max_val)).exp())
        .collect();
    
    // Compute sum of exponentials
    let sum = exp_vals.iter().fold(DualNumber::constant(T::zero()), |acc, &x| acc + x);
    
    // Divide each by the sum
    exp_vals.into_iter().map(|exp_val| exp_val / sum).collect()
}

/// Log-sum-exp function with automatic differentiation
pub fn log_sum_exp<T: Float>(inputs: &[DualNumber<T>]) -> DualNumber<T> {
    if inputs.is_empty() {
        return DualNumber::constant(T::neg_infinity());
    }
    
    let max_val = inputs.iter().map(|x| x.real).fold(inputs[0].real, |a, b| a.max(b));
    let max_dual = DualNumber::constant(max_val);
    
    let sum: DualNumber<T> = inputs.iter()
        .map(|&x| (x - max_dual).exp())
        .fold(DualNumber::constant(T::zero()), |acc, x| acc + x);
    
    max_dual + sum.ln()
}

/// Cross-entropy loss with automatic differentiation
pub fn cross_entropy_loss<T: Float>(
    predictions: &[DualNumber<T>],
    targets: &[T],
) -> DualNumber<T> {
    assert_eq!(predictions.len(), targets.len());
    
    let log_probs = softmax(predictions).into_iter().map(|x| x.ln()).collect::<Vec<_>>();
    
    let mut loss = DualNumber::constant(T::zero());
    for (i, &target) in targets.iter().enumerate() {
        loss = loss - DualNumber::constant(target) * log_probs[i];
    }
    
    loss
}

/// KL divergence with automatic differentiation
pub fn kl_divergence<T: Float>(
    p_logits: &[DualNumber<T>],
    q_logits: &[DualNumber<T>],
) -> DualNumber<T> {
    let p_probs = softmax(p_logits);
    let q_probs = softmax(q_logits);
    
    let mut kl = DualNumber::constant(T::zero());
    
    for (p, q) in p_probs.iter().zip(q_probs.iter()) {
        if p.real > T::from(1e-12).unwrap() {
            kl = kl + *p * (p.ln() - q.ln());
        }
    }
    
    kl
}

/// Jensen-Shannon divergence with automatic differentiation
pub fn js_divergence<T: Float>(
    p_logits: &[DualNumber<T>],
    q_logits: &[DualNumber<T>],
) -> DualNumber<T> {
    let p_probs = softmax(p_logits);
    let q_probs = softmax(q_logits);
    
    // Compute M = (P + Q) / 2
    let m_probs: Vec<DualNumber<T>> = p_probs.iter()
        .zip(q_probs.iter())
        .map(|(&p, &q)| (p + q) * DualNumber::constant(T::from(0.5).unwrap()))
        .collect();
    
    // JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    let kl_pm = kl_divergence_probs(&p_probs, &m_probs);
    let kl_qm = kl_divergence_probs(&q_probs, &m_probs);
    
    (kl_pm + kl_qm) * DualNumber::constant(T::from(0.5).unwrap())
}

/// KL divergence between probability vectors
fn kl_divergence_probs<T: Float>(
    p_probs: &[DualNumber<T>],
    q_probs: &[DualNumber<T>],
) -> DualNumber<T> {
    let mut kl = DualNumber::constant(T::zero());
    
    for (p, q) in p_probs.iter().zip(q_probs.iter()) {
        if p.real > T::from(1e-12).unwrap() && q.real > T::from(1e-12).unwrap() {
            kl = kl + *p * (p.ln() - q.ln());
        }
    }
    
    kl
}

/// Attention mechanism with automatic differentiation
pub fn attention<T: Float>(
    queries: &[DualNumber<T>],
    keys: &[DualNumber<T>],
    values: &[DualNumber<T>],
    temperature: DualNumber<T>,
) -> Vec<DualNumber<T>> {
    let d_k = T::from(keys.len()).unwrap().sqrt();
    let scale = temperature / DualNumber::constant(d_k);
    
    // Compute attention scores: Q * K^T / sqrt(d_k)
    let mut scores = Vec::new();
    for query in queries {
        let mut score = DualNumber::constant(T::zero());
        for (i, key) in keys.iter().enumerate() {
            score = score + *query * *key;
        }
        scores.push(score * scale);
    }
    
    // Apply softmax to get attention weights
    let weights = softmax(&scores);
    
    // Compute weighted sum of values
    let mut output = Vec::new();
    for weight in weights {
        let mut weighted_value = DualNumber::constant(T::zero());
        for (i, value) in values.iter().enumerate() {
            weighted_value = weighted_value + weight * *value;
        }
        output.push(weighted_value);
    }
    
    output
}

/// GELU activation function with automatic differentiation
pub fn gelu<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    let sqrt_2_pi = T::from(2.0).unwrap().sqrt() / T::from(3.141592653589793).unwrap().sqrt();
    let coeff = DualNumber::constant(T::from(0.5).unwrap());
    let tanh_input = DualNumber::constant(sqrt_2_pi) * (x + DualNumber::constant(T::from(0.044715).unwrap()) * x.powf(T::from(3.0).unwrap()));
    
    x * coeff * (DualNumber::constant(T::one()) + tanh_input.tanh())
}

/// Swish/SiLU activation function with automatic differentiation
pub fn swish<T: Float>(x: DualNumber<T>) -> DualNumber<T> {
    x * x.sigmoid()
}

/// Layer normalization with automatic differentiation
pub fn layer_norm<T: Float>(
    inputs: &[DualNumber<T>],
    gamma: &[DualNumber<T>],
    beta: &[DualNumber<T>],
    epsilon: T,
) -> Vec<DualNumber<T>> {
    let n = T::from(inputs.len()).unwrap();
    
    // Compute mean
    let mean = inputs.iter().fold(DualNumber::constant(T::zero()), |acc, &x| acc + x) 
             / DualNumber::constant(n);
    
    // Compute variance
    let variance = inputs.iter()
        .map(|&x| (x - mean).powf(T::from(2.0).unwrap()))
        .fold(DualNumber::constant(T::zero()), |acc, x| acc + x)
        / DualNumber::constant(n);
    
    let std_dev = (variance + DualNumber::constant(epsilon)).sqrt();
    
    // Normalize and scale
    inputs.iter().enumerate().map(|(i, &x)| {
        let normalized = (x - mean) / std_dev;
        gamma[i] * normalized + beta[i]
    }).collect()
}

/// Batch normalization with automatic differentiation
pub fn batch_norm<T: Float>(
    inputs: &[DualNumber<T>],
    running_mean: T,
    running_var: T,
    gamma: DualNumber<T>,
    beta: DualNumber<T>,
    epsilon: T,
) -> Vec<DualNumber<T>> {
    let std_dev = DualNumber::constant((running_var + epsilon).sqrt());
    let mean = DualNumber::constant(running_mean);
    
    inputs.iter().map(|&x| {
        let normalized = (x - mean) / std_dev;
        gamma * normalized + beta
    }).collect()
}

/// Dropout function (for completeness, though deterministic here)
pub fn dropout<T: Float>(
    inputs: &[DualNumber<T>],
    keep_prob: T,
    training: bool,
) -> Vec<DualNumber<T>> {
    if !training {
        return inputs.to_vec();
    }
    
    let scale = DualNumber::constant(T::one() / keep_prob);
    
    // In a real implementation, this would use random sampling
    // Here we just apply the scaling factor
    inputs.iter().map(|&x| x * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_softmax() {
        let inputs = vec![
            DualNumber::variable(1.0),
            DualNumber::variable(2.0),
            DualNumber::variable(3.0),
        ];
        
        let result = softmax(&inputs);
        
        // Check that probabilities sum to 1
        let sum = result.iter().fold(DualNumber::constant(0.0), |acc, &x| acc + x);
        assert_relative_eq!(sum.real, 1.0, epsilon = 1e-10);
        
        // Check that derivatives exist
        for prob in &result {
            assert!(prob.dual.abs() > 0.0);
        }
    }
    
    #[test]
    fn test_log_sum_exp() {
        let inputs = vec![
            DualNumber::variable(1.0),
            DualNumber::variable(2.0),
            DualNumber::variable(3.0),
        ];
        
        let result = log_sum_exp(&inputs);
        
        // Should be close to max + ln(sum of exp(x - max))
        let expected = 3.0 + ((-2.0f64).exp() + (-1.0f64).exp() + 0.0f64.exp()).ln();
        assert_relative_eq!(result.real, expected, epsilon = 1e-10);
        
        // Should have non-zero derivative
        assert!(result.dual.abs() > 0.0);
    }
    
    #[test]
    fn test_cross_entropy_loss() {
        let predictions = vec![
            DualNumber::variable(1.0),
            DualNumber::variable(2.0),
            DualNumber::variable(3.0),
        ];
        
        let targets = vec![0.0, 0.0, 1.0]; // One-hot encoded
        
        let loss = cross_entropy_loss(&predictions, &targets);
        
        // Loss should be positive
        assert!(loss.real > 0.0);
        
        // Should have gradient
        assert!(loss.dual.abs() > 0.0);
    }
    
    #[test]
    fn test_kl_divergence() {
        let p_logits = vec![
            DualNumber::variable(1.0),
            DualNumber::variable(1.0),
        ];
        
        let q_logits = vec![
            DualNumber::constant(2.0),
            DualNumber::constant(0.5),
        ];
        
        let kl = kl_divergence(&p_logits, &q_logits);
        
        // KL divergence should be non-negative
        assert!(kl.real >= 0.0);
        
        // Should have gradient with respect to p
        assert!(kl.dual.abs() > 0.0);
    }
    
    #[test]
    fn test_gelu_activation() {
        let x = DualNumber::variable(1.0);
        let result = gelu(x);
        
        // GELU(1) ≈ 0.841
        assert_relative_eq!(result.real, 0.841, epsilon = 0.01);
        
        // Should have derivative
        assert!(result.dual > 0.0);
    }
    
    #[test]
    fn test_layer_norm() {
        let inputs = vec![
            DualNumber::variable(1.0),
            DualNumber::variable(2.0),
            DualNumber::variable(3.0),
        ];
        
        let gamma = vec![
            DualNumber::constant(1.0),
            DualNumber::constant(1.0),
            DualNumber::constant(1.0),
        ];
        
        let beta = vec![
            DualNumber::constant(0.0),
            DualNumber::constant(0.0),
            DualNumber::constant(0.0),
        ];
        
        let result = layer_norm(&inputs, &gamma, &beta, 1e-5);
        
        // Check that normalized values have mean ≈ 0
        let mean = result.iter().fold(DualNumber::constant(0.0), |acc, &x| acc + x) 
                 / DualNumber::constant(3.0);
        assert_relative_eq!(mean.real, 0.0, epsilon = 1e-10);
        
        // Check that all outputs have derivatives
        for output in &result {
            assert!(output.dual.abs() > 0.0);
        }
    }
}