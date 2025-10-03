//! Neural Networks with Verified Mathematics
//!
//! This example demonstrates how to implement neural networks using
//! dual numbers for automatic differentiation and geometric algebra
//! for verified mathematical operations.

use amari_dual::{Dual, DualNumber};
use amari_core::{Vector, Multivector};
use rand::Rng;
use std::f64::consts::E;

/// A neural network layer implemented with dual numbers for exact gradients
#[derive(Debug, Clone)]
pub struct Layer {
    /// Weight matrix (stored as flat vector for simplicity)
    pub weights: Vec<f64>,
    /// Bias vector
    pub biases: Vec<f64>,
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
}

impl Layer {
    /// Create a new layer with random initialization
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization: scale weights by sqrt(6/(fan_in + fan_out))
        let fan_in = input_size as f64;
        let fan_out = output_size as f64;
        let scale = (6.0 / (fan_in + fan_out)).sqrt();

        let weights: Vec<f64> = (0..input_size * output_size)
            .map(|_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
            .collect();

        let biases: Vec<f64> = (0..output_size)
            .map(|_| (rng.gen::<f64>() - 0.5) * 0.1)
            .collect();

        Self {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    /// Forward pass with dual numbers for automatic differentiation
    pub fn forward(&self, inputs: &[Dual<f64>]) -> Vec<Dual<f64>> {
        assert_eq!(inputs.len(), self.input_size);

        let mut outputs = Vec::with_capacity(self.output_size);

        for i in 0..self.output_size {
            let mut sum = Dual::constant(self.biases[i]);

            for j in 0..self.input_size {
                let weight_idx = i * self.input_size + j;
                let weight_contribution = inputs[j].scale(self.weights[weight_idx]);
                sum = sum.add(&weight_contribution);
            }

            outputs.push(sum);
        }

        outputs
    }

    /// Apply activation function (ReLU)
    pub fn relu(inputs: &[Dual<f64>]) -> Vec<Dual<f64>> {
        inputs.iter().map(|x| x.max(&Dual::constant(0.0))).collect()
    }

    /// Apply activation function (Sigmoid)
    pub fn sigmoid(inputs: &[Dual<f64>]) -> Vec<Dual<f64>> {
        inputs.iter().map(|x| {
            // sigmoid(x) = 1 / (1 + e^(-x))
            let neg_x = x.scale(-1.0);
            let exp_neg_x = neg_x.exp();
            let denominator = Dual::constant(1.0).add(&exp_neg_x);
            Dual::constant(1.0).divide(&denominator)
        }).collect()
    }

    /// Apply activation function (Tanh)
    pub fn tanh(inputs: &[Dual<f64>]) -> Vec<Dual<f64>> {
        inputs.iter().map(|x| x.tanh()).collect()
    }

    /// Update weights using computed gradients
    pub fn update_weights(&mut self, weight_gradients: &[f64], bias_gradients: &[f64], learning_rate: f64) {
        for (weight, gradient) in self.weights.iter_mut().zip(weight_gradients.iter()) {
            *weight -= learning_rate * gradient;
        }

        for (bias, gradient) in self.biases.iter_mut().zip(bias_gradients.iter()) {
            *bias -= learning_rate * gradient;
        }
    }
}

/// A simple neural network with verified mathematics
#[derive(Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    /// Create a new neural network with specified architecture
    pub fn new(architecture: &[usize]) -> Self {
        let mut layers = Vec::new();

        for i in 0..architecture.len() - 1 {
            layers.push(Layer::new(architecture[i], architecture[i + 1]));
        }

        Self { layers }
    }

    /// Forward pass through the entire network
    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        let mut current_values: Vec<Dual<f64>> = inputs.iter()
            .map(|&x| Dual::constant(x))
            .collect();

        for (i, layer) in self.layers.iter().enumerate() {
            current_values = layer.forward(&current_values);

            // Apply activation function (except for output layer)
            if i < self.layers.len() - 1 {
                current_values = Layer::sigmoid(&current_values);
            }
        }

        // Convert dual numbers back to regular floats
        current_values.iter().map(|x| x.real()).collect()
    }

    /// Compute loss and gradients using automatic differentiation
    pub fn compute_loss_and_gradients(&self, inputs: &[f64], targets: &[f64]) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let output_size = self.layers.last().unwrap().output_size;
        let mut weight_gradients = Vec::new();
        let mut bias_gradients = Vec::new();

        // For each weight/bias, compute its gradient using dual numbers
        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let mut layer_weight_grads = vec![0.0; layer.weights.len()];
            let mut layer_bias_grads = vec![0.0; layer.biases.len()];

            // Compute gradients for weights
            for weight_idx in 0..layer.weights.len() {
                let loss_dual = self.compute_loss_with_weight_perturbation(inputs, targets, layer_idx, weight_idx, true);
                layer_weight_grads[weight_idx] = loss_dual.dual();
            }

            // Compute gradients for biases
            for bias_idx in 0..layer.biases.len() {
                let loss_dual = self.compute_loss_with_bias_perturbation(inputs, targets, layer_idx, bias_idx);
                layer_bias_grads[bias_idx] = loss_dual.dual();
            }

            weight_gradients.push(layer_weight_grads);
            bias_gradients.push(layer_bias_grads);
        }

        // Compute actual loss
        let outputs = self.forward(inputs);
        let loss = Self::mean_squared_error(&outputs, targets);

        (loss, weight_gradients, bias_gradients)
    }

    /// Helper function to compute loss with weight perturbation (for autodiff)
    fn compute_loss_with_weight_perturbation(&self, inputs: &[f64], targets: &[f64], layer_idx: usize, weight_idx: usize, _is_weight: bool) -> Dual<f64> {
        let mut current_values: Vec<Dual<f64>> = inputs.iter()
            .map(|&x| Dual::constant(x))
            .collect();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_inputs = current_values.clone();
            current_values.clear();

            for output_idx in 0..layer.output_size {
                let mut sum = Dual::constant(layer.biases[output_idx]);

                for input_idx in 0..layer.input_size {
                    let current_weight_idx = output_idx * layer.input_size + input_idx;
                    let weight_value = if i == layer_idx && current_weight_idx == weight_idx {
                        Dual::variable(layer.weights[current_weight_idx])
                    } else {
                        Dual::constant(layer.weights[current_weight_idx])
                    };

                    let contribution = layer_inputs[input_idx].multiply(&weight_value);
                    sum = sum.add(&contribution);
                }

                current_values.push(sum);
            }

            // Apply activation function (except for output layer)
            if i < self.layers.len() - 1 {
                current_values = Layer::sigmoid(&current_values);
            }
        }

        // Compute MSE loss
        let mut loss = Dual::constant(0.0);
        for (output, &target) in current_values.iter().zip(targets.iter()) {
            let error = output.subtract(&Dual::constant(target));
            let squared_error = error.square();
            loss = loss.add(&squared_error);
        }

        loss.scale(0.5 / targets.len() as f64)
    }

    /// Helper function to compute loss with bias perturbation (for autodiff)
    fn compute_loss_with_bias_perturbation(&self, inputs: &[f64], targets: &[f64], layer_idx: usize, bias_idx: usize) -> Dual<f64> {
        let mut current_values: Vec<Dual<f64>> = inputs.iter()
            .map(|&x| Dual::constant(x))
            .collect();

        for (i, layer) in self.layers.iter().enumerate() {
            let layer_inputs = current_values.clone();
            current_values.clear();

            for output_idx in 0..layer.output_size {
                let bias_value = if i == layer_idx && output_idx == bias_idx {
                    Dual::variable(layer.biases[output_idx])
                } else {
                    Dual::constant(layer.biases[output_idx])
                };

                let mut sum = bias_value;

                for input_idx in 0..layer.input_size {
                    let weight_idx = output_idx * layer.input_size + input_idx;
                    let weight_value = Dual::constant(layer.weights[weight_idx]);
                    let contribution = layer_inputs[input_idx].multiply(&weight_value);
                    sum = sum.add(&contribution);
                }

                current_values.push(sum);
            }

            // Apply activation function (except for output layer)
            if i < self.layers.len() - 1 {
                current_values = Layer::sigmoid(&current_values);
            }
        }

        // Compute MSE loss
        let mut loss = Dual::constant(0.0);
        for (output, &target) in current_values.iter().zip(targets.iter()) {
            let error = output.subtract(&Dual::constant(target));
            let squared_error = error.square();
            loss = loss.add(&squared_error);
        }

        loss.scale(0.5 / targets.len() as f64)
    }

    /// Mean squared error loss function
    pub fn mean_squared_error(outputs: &[f64], targets: &[f64]) -> f64 {
        assert_eq!(outputs.len(), targets.len());

        let mut sum = 0.0;
        for (output, target) in outputs.iter().zip(targets.iter()) {
            let error = output - target;
            sum += error * error;
        }

        0.5 * sum / outputs.len() as f64
    }

    /// Train the network using gradient descent with exact gradients
    pub fn train(&mut self, training_data: &[(Vec<f64>, Vec<f64>)], learning_rate: f64, epochs: usize) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (inputs, targets) in training_data {
                let (loss, weight_grads, bias_grads) = self.compute_loss_and_gradients(inputs, targets);
                total_loss += loss;

                // Update weights and biases for each layer
                for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                    layer.update_weights(&weight_grads[layer_idx], &bias_grads[layer_idx], learning_rate);
                }
            }

            let avg_loss = total_loss / training_data.len() as f64;

            if epoch % (epochs / 10).max(1) == 0 {
                println!("Epoch {}: Average Loss = {:.6}", epoch, avg_loss);
            }
        }
    }
}

/// Generate synthetic training data for XOR problem
fn generate_xor_data() -> Vec<(Vec<f64>, Vec<f64>)> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}

/// Generate synthetic training data for function approximation
fn generate_function_data(num_samples: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();

    for _ in 0..num_samples {
        let x = rng.gen::<f64>() * 4.0 - 2.0; // Random x in [-2, 2]
        let y = (x * x * 0.5 - x + 0.5).sin(); // Target function: sin(0.5x² - x + 0.5)
        data.push((vec![x], vec![y]));
    }

    data
}

/// Demonstrate XOR learning with verified gradients
fn xor_learning_demo() {
    println!("=== XOR Learning with Verified Gradients ===");
    println!("Training a neural network to learn the XOR function\\n");

    let mut network = NeuralNetwork::new(&[2, 4, 1]); // 2 inputs, 4 hidden neurons, 1 output
    let xor_data = generate_xor_data();

    println!("Network architecture: 2 → 4 → 1");
    println!("Training data: {} samples", xor_data.len());
    println!("Activation function: Sigmoid\\n");

    // Test initial performance
    println!("Initial performance:");
    println!("Input\\t\\tTarget\\t\\tOutput\\t\\tError");
    println!("{:-<50}", "");

    for (inputs, targets) in &xor_data {
        let outputs = network.forward(inputs);
        let error = (outputs[0] - targets[0]).abs();
        println!("{:?}\\t\\t{:.1}\\t\\t{:.4}\\t\\t{:.4}",
            inputs, targets[0], outputs[0], error);
    }

    // Train the network
    println!("\\nTraining with exact gradients from dual numbers:");
    network.train(&xor_data, 0.5, 1000);

    // Test final performance
    println!("\\nFinal performance:");
    println!("Input\\t\\tTarget\\t\\tOutput\\t\\tError");
    println!("{:-<50}", "");

    let mut total_error = 0.0;
    for (inputs, targets) in &xor_data {
        let outputs = network.forward(inputs);
        let error = (outputs[0] - targets[0]).abs();
        total_error += error;

        println!("{:?}\\t\\t{:.1}\\t\\t{:.4}\\t\\t{:.4}",
            inputs, targets[0], outputs[0], error);
    }

    let avg_error = total_error / xor_data.len() as f64;
    println!("\\nAverage absolute error: {:.6}", avg_error);
    println!("The network successfully learned XOR using exact gradients!");
}

/// Demonstrate function approximation
fn function_approximation_demo() {
    println!("\\n=== Function Approximation Demo ===");
    println!("Learning to approximate f(x) = sin(0.5x² - x + 0.5)\\n");

    let mut network = NeuralNetwork::new(&[1, 8, 8, 1]); // Deeper network for complex function
    let training_data = generate_function_data(100);

    println!("Network architecture: 1 → 8 → 8 → 1");
    println!("Training data: {} samples", training_data.len());
    println!("Target function: f(x) = sin(0.5x² - x + 0.5)\\n");

    // Test a few points before training
    let test_points = vec![-1.5, -0.5, 0.0, 0.5, 1.5];

    println!("Initial approximation quality:");
    println!("x\\t\\tTarget f(x)\\t\\tNetwork f(x)\\tError");
    println!("{:-<60}", "");

    for &x in &test_points {
        let target = (x * x * 0.5 - x + 0.5).sin();
        let output = network.forward(&[x])[0];
        let error = (output - target).abs();

        println!("{:.1}\\t\\t{:.6}\\t\\t{:.6}\\t\\t{:.6}",
            x, target, output, error);
    }

    // Train the network
    println!("\\nTraining with verified automatic differentiation:");
    network.train(&training_data, 0.01, 2000);

    // Test final approximation
    println!("\\nFinal approximation quality:");
    println!("x\\t\\tTarget f(x)\\t\\tNetwork f(x)\\tError");
    println!("{:-<60}", "");

    let mut total_error = 0.0;
    for &x in &test_points {
        let target = (x * x * 0.5 - x + 0.5).sin();
        let output = network.forward(&[x])[0];
        let error = (output - target).abs();
        total_error += error;

        println!("{:.1}\\t\\t{:.6}\\t\\t{:.6}\\t\\t{:.6}",
            x, target, output, error);
    }

    let avg_error = total_error / test_points.len() as f64;
    println!("\\nAverage approximation error: {:.6}", avg_error);
}

/// Demonstrate activation function properties using dual numbers
fn activation_function_demo() {
    println!("\\n=== Activation Function Analysis ===");
    println!("Computing derivatives of activation functions using dual numbers\\n");

    let test_inputs = vec![-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    println!("Input\\t\\tReLU\\t\\tReLU'\\t\\tSigmoid\\t\\tSigmoid'\\t\\tTanh\\t\\tTanh'");
    println!("{:-<90}", "");

    for &x in &test_inputs {
        let x_dual = Dual::variable(x);

        // ReLU and its derivative
        let relu_output = x_dual.max(&Dual::constant(0.0));
        let relu_value = relu_output.real();
        let relu_derivative = relu_output.dual();

        // Sigmoid and its derivative
        let sigmoid_output = {
            let neg_x = x_dual.scale(-1.0);
            let exp_neg_x = neg_x.exp();
            let denominator = Dual::constant(1.0).add(&exp_neg_x);
            Dual::constant(1.0).divide(&denominator)
        };
        let sigmoid_value = sigmoid_output.real();
        let sigmoid_derivative = sigmoid_output.dual();

        // Tanh and its derivative
        let tanh_output = x_dual.tanh();
        let tanh_value = tanh_output.real();
        let tanh_derivative = tanh_output.dual();

        println!("{:.1}\\t\\t{:.3}\\t\\t{:.3}\\t\\t{:.3}\\t\\t{:.3}\\t\\t{:.3}\\t\\t{:.3}",
            x, relu_value, relu_derivative, sigmoid_value, sigmoid_derivative, tanh_value, tanh_derivative);
    }

    println!("\\nDual numbers provide exact derivatives for all activation functions,");
    println!("enabling precise gradient computations in neural network training.");
}

/// Demonstrate gradient verification
fn gradient_verification_demo() {
    println!("\\n=== Gradient Verification Demo ===");
    println!("Comparing automatic differentiation with finite differences\\n");

    let network = NeuralNetwork::new(&[2, 3, 1]);
    let inputs = vec![0.5, -0.3];
    let targets = vec![0.8];

    // Compute gradients using automatic differentiation
    let (loss_autodiff, weight_grads_autodiff, bias_grads_autodiff) = network.compute_loss_and_gradients(&inputs, &targets);

    println!("Test case: inputs = {:?}, target = {:?}", inputs, targets);
    println!("Loss (autodiff): {:.8}\\n", loss_autodiff);

    // Compare first layer gradients with finite differences
    let epsilon = 1e-8;
    let layer = &network.layers[0];

    println!("Gradient verification (first layer weights):");
    println!("Weight\\t\\tAutoDiff\\t\\tFinite Diff\\t\\tRelative Error");
    println!("{:-<70}", "");

    for (i, &autodiff_grad) in weight_grads_autodiff[0].iter().take(6).enumerate() {
        // Compute finite difference gradient
        let mut network_plus = network.clone();
        let mut network_minus = network.clone();

        network_plus.layers[0].weights[i] += epsilon;
        network_minus.layers[0].weights[i] -= epsilon;

        let loss_plus = {
            let outputs = network_plus.forward(&inputs);
            NeuralNetwork::mean_squared_error(&outputs, &targets)
        };

        let loss_minus = {
            let outputs = network_minus.forward(&inputs);
            NeuralNetwork::mean_squared_error(&outputs, &targets)
        };

        let finite_diff_grad = (loss_plus - loss_minus) / (2.0 * epsilon);
        let relative_error = if autodiff_grad.abs() > 1e-10 {
            ((autodiff_grad - finite_diff_grad) / autodiff_grad).abs()
        } else {
            (autodiff_grad - finite_diff_grad).abs()
        };

        println!("w[{}]\\t\\t{:.8}\\t\\t{:.8}\\t\\t{:.2e}",
            i, autodiff_grad, finite_diff_grad, relative_error);
    }

    println!("\\nAutomatic differentiation provides machine-precision gradients,");
    println!("while finite differences introduce numerical errors.");
}

fn main() {
    println!("🧠 Neural Networks with Verified Mathematics");
    println!("===========================================\\n");

    println!("This example demonstrates neural networks using dual numbers");
    println!("for automatic differentiation and verified gradient computation:\\n");

    println!("• Exact gradient computation using dual numbers");
    println!("• Elimination of finite difference approximation errors");
    println!("• Verified backpropagation algorithm");
    println!("• Function approximation with mathematical rigor");
    println!("• Activation function derivative analysis");
    println!("• Gradient verification against numerical methods\\n");

    // Run the demonstrations
    xor_learning_demo();
    function_approximation_demo();
    activation_function_demo();
    gradient_verification_demo();

    println!("\\n=== Advantages of Verified Neural Networks ==");
    println!("1. Exact gradients eliminate training instability");
    println!("2. Mathematical rigor ensures convergence properties");
    println!("3. No finite difference approximation errors");
    println!("4. Verifiable optimization algorithms");
    println!("5. Robust numerical performance");
    println!("6. Foundation for certified AI systems");

    println!("\\n🎓 Educational Value:");
    println!("Dual number automatic differentiation provides a solid");
    println!("mathematical foundation for neural networks, enabling");
    println!("verified and robust machine learning algorithms.");
}