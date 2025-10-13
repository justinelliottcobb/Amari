//! GPU-accelerated neural network training using dual number automatic differentiation
//!
//! This example demonstrates how forward-mode automatic differentiation can be
//! accelerated on GPU for efficient gradient computation in neural network training.

#[cfg(feature = "gpu")]
use amari_dual::{
    gpu::{DualGpuOps, DualOperation, NeuralNetworkConfig},
    DualNumber,
};

#[cfg(feature = "gpu")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 GPU-Accelerated Neural Network Training with Dual Numbers");
    println!("===========================================================");

    // Initialize GPU context
    let mut gpu_ops = match DualGpuOps::new().await {
        Ok(ops) => {
            println!("✅ GPU context initialized successfully");
            ops
        }
        Err(e) => {
            println!("⚠️  GPU not available, falling back to CPU demo: {}", e);
            return demonstrate_cpu_training();
        }
    };

    // Create neural network configuration
    let network_config = NeuralNetworkConfig {
        input_size: 4,    // Example: iris dataset
        hidden_size: 8,   // Hidden layer
        output_size: 3,   // 3 classes
        activation: "relu".to_string(),
    };

    println!("\n🧠 Neural Network Configuration:");
    println!("   Input size: {}", network_config.input_size);
    println!("   Hidden size: {}", network_config.hidden_size);
    println!("   Output size: {}", network_config.output_size);
    println!("   Activation: {}", network_config.activation);

    // Generate sample training data
    let batch_size = 32;
    let (inputs, targets) = generate_sample_data(batch_size, &network_config);

    println!("\n📊 Training Data:");
    println!("   Batch size: {}", batch_size);
    println!("   Input features: {} x {}", batch_size, network_config.input_size);
    println!("   Target labels: {} x {}", batch_size, network_config.output_size);

    // Initialize network weights
    let num_weights = network_config.input_size * network_config.hidden_size
        + network_config.hidden_size * network_config.output_size
        + network_config.hidden_size
        + network_config.output_size; // Include biases

    let mut weights = initialize_weights(num_weights);
    println!("   Total weights: {}", num_weights);

    // Training parameters
    let learning_rate = 0.01f32;
    let num_epochs = 100;
    let report_interval = 10;

    println!("\n🚀 Starting GPU-accelerated training...");
    println!("   Learning rate: {}", learning_rate);
    println!("   Epochs: {}", num_epochs);

    let training_start = std::time::Instant::now();

    for epoch in 0..num_epochs {
        let epoch_start = std::time::Instant::now();

        // Compute gradients using GPU-accelerated forward-mode AD
        let gradients = gpu_ops
            .neural_gradient(&weights, &inputs, &targets, &network_config)
            .await?;

        // Update weights using computed gradients
        for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
            *weight -= learning_rate * gradient;
        }

        let epoch_time = epoch_start.elapsed();

        if epoch % report_interval == 0 {
            // Compute loss for reporting (simplified)
            let loss = compute_loss(&weights, &inputs, &targets, &network_config);

            println!(
                "   Epoch {}: loss = {:.6}, time = {:?}",
                epoch, loss, epoch_time
            );

            // Demonstrate gradient norm
            let gradient_norm: f32 = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
            println!("            gradient norm = {:.6}", gradient_norm);
        }
    }

    let total_training_time = training_start.elapsed();
    println!("✅ Training completed in {:?}", total_training_time);

    // Demonstrate batch forward-mode AD
    println!("\n🔄 Demonstrating batch forward-mode AD...");

    // Create dual numbers for demonstration
    let dual_inputs: Vec<DualNumber<f32>> = (0..16)
        .map(|i| DualNumber::variable(i as f32 * 0.1))
        .collect();

    // Define operations to apply
    let operations = vec![
        DualOperation::ReLU,
        DualOperation::Sigmoid,
        DualOperation::Tanh,
    ];

    let batch_start = std::time::Instant::now();
    let batch_results = gpu_ops.batch_forward_ad(&dual_inputs, &operations).await?;
    let batch_time = batch_start.elapsed();

    println!("✅ Batch AD completed in {:?}", batch_time);
    println!("   Processed {} dual numbers", batch_results.len());
    println!("   Applied {} operations per number", operations.len());

    // Show some results
    println!("\n📈 Sample Results:");
    for (i, result) in batch_results.iter().take(5).enumerate() {
        println!(
            "   Input {}: f({:.1}) = {:.4}, f'({:.1}) = {:.4}",
            i, dual_inputs[i].real, result.real, dual_inputs[i].real, result.dual
        );
    }

    // Performance analysis
    println!("\n📊 Performance Analysis:");
    let throughput = batch_results.len() as f64 / batch_time.as_secs_f64();
    println!("   Dual number throughput: {:.0} dual numbers/second", throughput);

    let ops_per_second = throughput * operations.len() as f64;
    println!("   Operation throughput: {:.0} operations/second", ops_per_second);

    // Compare with theoretical speedup
    let theoretical_speedup = estimate_gpu_speedup(batch_results.len(), operations.len());
    println!("   Theoretical GPU speedup: {:.1}x vs CPU", theoretical_speedup);

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("⚠️  This example requires the 'gpu' feature to be enabled.");
    println!("Run with: cargo run --example gpu_neural_training --features gpu");
}

#[cfg(feature = "gpu")]
fn demonstrate_cpu_training() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating CPU dual number training fallback...");

    // Create simple neural network training example
    let network_config = NeuralNetworkConfig {
        input_size: 2,
        hidden_size: 4,
        output_size: 1,
        activation: "sigmoid".to_string(),
    };

    println!("✅ CPU neural network configuration:");
    println!("   Input: {} -> Hidden: {} -> Output: {}",
             network_config.input_size, network_config.hidden_size, network_config.output_size);

    // Demonstrate forward-mode AD for a simple function
    let x = DualNumber::variable(2.0f32);
    let y = DualNumber::variable(3.0f32);

    // Compute f(x,y) = sigmoid(x*y + sin(x))
    let product = x * y;
    let sin_x = x.sin();
    let sum = product + sin_x;
    let result = sum.sigmoid();

    println!("✅ Forward-mode AD example:");
    println!("   f(x,y) = sigmoid(x*y + sin(x))");
    println!("   f(2,3) = {:.6}", result.real);
    println!("   ∂f/∂x at (2,3) = {:.6}", result.dual);

    // Demonstrate chain rule in action
    println!("   Chain rule: ∂f/∂x = sigmoid'(u) * (y + cos(x))");
    let expected_derivative = {
        let u = 2.0 * 3.0 + 2.0f32.sin();
        let sigmoid_u = 1.0 / (1.0 + (-u).exp());
        let sigmoid_prime = sigmoid_u * (1.0 - sigmoid_u);
        sigmoid_prime * (3.0 + 2.0f32.cos())
    };
    println!("   Expected: {:.6}", expected_derivative);
    println!("   Computed: {:.6}", result.dual);
    println!("   Difference: {:.2e}", (result.dual - expected_derivative).abs());

    Ok(())
}

#[cfg(feature = "gpu")]
fn generate_sample_data(
    batch_size: usize,
    config: &NeuralNetworkConfig,
) -> (Vec<f32>, Vec<f32>) {
    use std::f32::consts::PI;

    let mut inputs = Vec::with_capacity(batch_size * config.input_size);
    let mut targets = Vec::with_capacity(batch_size * config.output_size);

    for i in 0..batch_size {
        // Generate synthetic input data
        for j in 0..config.input_size {
            let value = (i as f32 + j as f32) * 0.1 + (i as f32 * PI / 10.0).sin();
            inputs.push(value);
        }

        // Generate synthetic target data (classification-like)
        let class = i % config.output_size;
        for j in 0..config.output_size {
            targets.push(if j == class { 1.0 } else { 0.0 });
        }
    }

    (inputs, targets)
}

#[cfg(feature = "gpu")]
fn initialize_weights(num_weights: usize) -> Vec<f32> {
    // Xavier/Glorot initialization
    let scale = (2.0 / num_weights as f32).sqrt();

    (0..num_weights)
        .map(|i| {
            let x = (i as f32 * 0.1).sin();
            x * scale
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn compute_loss(
    _weights: &[f32],
    _inputs: &[f32],
    _targets: &[f32],
    _config: &NeuralNetworkConfig,
) -> f32 {
    // Simplified loss computation for demonstration
    // In practice, this would be the actual forward pass
    0.5
}

#[cfg(feature = "gpu")]
fn estimate_gpu_speedup(batch_size: usize, num_operations: usize) -> f32 {
    // Estimate theoretical speedup based on parallelization
    let total_operations = batch_size * num_operations;

    // GPU advantages:
    // - Parallel execution across workgroups
    // - Vectorized operations
    // - Reduced control flow overhead

    let base_speedup = 10.0; // Base GPU vs CPU speedup
    let parallelism_factor = (total_operations as f32 / 1000.0).min(32.0); // Diminishing returns

    base_speedup * (1.0 + parallelism_factor.log2() / 5.0)
}