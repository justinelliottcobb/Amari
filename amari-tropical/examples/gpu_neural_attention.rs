//! GPU-accelerated neural attention using tropical algebra
//!
//! This example demonstrates how tropical algebra can be used to accelerate
//! neural network attention mechanisms by replacing expensive softmax operations
//! with simple max operations on GPU.

#[cfg(feature = "gpu")]
use amari_tropical::{gpu::TropicalGpuOps, TropicalMatrix};

#[cfg(feature = "gpu")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌴 GPU Tropical Algebra Neural Attention Example");
    println!("================================================");

    // Initialize GPU context
    let mut gpu_ops = match TropicalGpuOps::new().await {
        Ok(ops) => {
            println!("✅ GPU context initialized successfully");
            ops
        }
        Err(e) => {
            println!("⚠️  GPU not available, falling back to CPU: {}", e);
            return demonstrate_cpu_attention();
        }
    };

    // Create sample attention matrices (Query, Key, Value)
    let seq_len = 128;
    let d_model = 64;

    println!("\nCreating attention matrices ({}x{})...", seq_len, d_model);

    // Query matrix - log probabilities for attention queries
    let query_logits = create_sample_logits(seq_len, d_model, 0.0);
    let query = TropicalMatrix::from_log_probs(&query_logits);

    // Key matrix - log probabilities for attention keys
    let key_logits = create_sample_logits(d_model, seq_len, 1.0);
    let key = TropicalMatrix::from_log_probs(&key_logits);

    // Value matrix - log probabilities for attention values
    let value_logits = create_sample_logits(seq_len, d_model, 2.0);
    let value = TropicalMatrix::from_log_probs(&value_logits);

    println!(
        "✅ Created Q({},{}), K({},{}), V({},{}) matrices",
        query.rows(),
        query.cols(),
        key.rows(),
        key.cols(),
        value.rows(),
        value.cols()
    );

    // Perform GPU-accelerated tropical attention
    println!("\n🚀 Computing GPU tropical attention...");
    let start = std::time::Instant::now();

    let attention_output = gpu_ops.neural_attention(&query, &key, &value).await?;

    let gpu_time = start.elapsed();
    println!("✅ GPU tropical attention completed in {:?}", gpu_time);

    // Verify output structure
    println!(
        "📊 Output matrix: {}x{}",
        attention_output.rows(),
        attention_output.cols()
    );

    // Convert back to attention scores for analysis
    let attention_scores = attention_output.to_attention_scores();
    println!(
        "📈 Attention pattern extracted ({} heads)",
        attention_scores.len()
    );

    // Analyze attention patterns
    analyze_attention_patterns(&attention_scores);

    // Demonstrate batch processing
    println!("\n🔄 Demonstrating batch attention processing...");
    let batch_size = 4;
    let queries = vec![query.clone(); batch_size];
    let keys = vec![key.clone(); batch_size];
    let values = vec![value.clone(); batch_size];

    let batch_start = std::time::Instant::now();
    let mut batch_outputs = Vec::new();

    for i in 0..batch_size {
        let output = gpu_ops
            .neural_attention(&queries[i], &keys[i], &values[i])
            .await?;
        batch_outputs.push(output);
    }

    let batch_time = batch_start.elapsed();
    println!(
        "✅ Batch processing ({} items) completed in {:?}",
        batch_size, batch_time
    );
    println!(
        "📊 Average per-item time: {:?}",
        batch_time / batch_size as u32
    );

    // Compare with theoretical speedup
    let theoretical_speedup = calculate_theoretical_speedup(seq_len, d_model);
    println!("\n📈 Performance Analysis:");
    println!(
        "   Theoretical speedup vs softmax: {:.2}x",
        theoretical_speedup
    );
    println!("   GPU utilization: Parallel tropical operations");
    println!("   Memory efficiency: Reduced precision requirements");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("⚠️  This example requires the 'gpu' feature to be enabled.");
    println!("Run with: cargo run --example gpu_neural_attention --features gpu");
}

#[cfg(feature = "gpu")]
fn demonstrate_cpu_attention() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating CPU tropical attention fallback...");

    let seq_len = 32; // Smaller for CPU demo
    let d_model = 16;

    // Create smaller matrices for CPU processing
    let query_logits = create_sample_logits(seq_len, d_model, 0.0);
    let query = TropicalMatrix::from_log_probs(&query_logits);

    let key_logits = create_sample_logits(d_model, seq_len, 1.0);
    let key = TropicalMatrix::from_log_probs(&key_logits);

    println!("✅ CPU tropical attention simulation completed");
    println!(
        "📊 Q({},{}), K({},{}) - using tropical max-plus operations",
        query.rows(),
        query.cols(),
        key.rows(),
        key.cols()
    );

    // Demonstrate tropical matrix multiplication (attention scores)
    let attention_scores = query.mul(&key);
    println!("📈 Attention scores computed using tropical multiplication");
    println!("   Traditional: softmax(QK^T/√d)");
    println!("   Tropical: max(Q ⊗ K^T) where ⊗ is tropical multiplication");

    let scores = attention_scores.to_attention_scores();
    analyze_attention_patterns(&scores);

    Ok(())
}

#[cfg(feature = "gpu")]
fn create_sample_logits(rows: usize, cols: usize, offset: f32) -> Vec<Vec<f32>> {
    use std::f32::consts::PI;

    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| {
                    // Create sinusoidal patterns to simulate realistic attention logits
                    let x = i as f32 / rows as f32 * 2.0 * PI;
                    let y = j as f32 / cols as f32 * 2.0 * PI;

                    -2.0 + offset
                        + 0.5 * (x + y).sin()
                        + 0.3 * (2.0 * x).cos()
                        + 0.2 * (3.0 * y).sin()
                })
                .collect()
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn analyze_attention_patterns(attention_scores: &[Vec<f32>]) {
    println!("\n🔍 Attention Pattern Analysis:");

    if attention_scores.is_empty() || attention_scores[0].is_empty() {
        println!("   No attention patterns to analyze");
        return;
    }

    let rows = attention_scores.len();
    let cols = attention_scores[0].len();

    // Find sparsity pattern
    let mut non_zero_count = 0;
    let mut max_attention = 0.0f32;
    let mut total_attention = 0.0f32;

    for row in attention_scores {
        for &score in row {
            if score > 0.001 {
                // Consider very small values as zero
                non_zero_count += 1;
            }
            max_attention = max_attention.max(score);
            total_attention += score;
        }
    }

    let sparsity = 1.0 - (non_zero_count as f32) / (rows * cols) as f32;
    let avg_attention = total_attention / (rows * cols) as f32;

    println!("   Matrix size: {}x{}", rows, cols);
    println!(
        "   Sparsity: {:.1}% (tropical algebra promotes sparsity)",
        sparsity * 100.0
    );
    println!("   Max attention: {:.4}", max_attention);
    println!("   Avg attention: {:.4}", avg_attention);

    // Find attention focus patterns
    let mut row_maxes = Vec::new();
    for row in attention_scores {
        let row_max = row.iter().fold(0.0f32, |acc, &x| acc.max(x));
        row_maxes.push(row_max);
    }

    let focus_variance = row_maxes
        .iter()
        .map(|&x| (x - avg_attention).powi(2))
        .sum::<f32>()
        / rows as f32;

    println!(
        "   Focus variance: {:.4} (higher = more focused attention)",
        focus_variance
    );

    if focus_variance > 0.1 {
        println!("   Pattern: FOCUSED - Clear attention peaks detected");
    } else {
        println!("   Pattern: DIFFUSE - Attention spread across tokens");
    }
}

#[cfg(feature = "gpu")]
fn calculate_theoretical_speedup(seq_len: usize, d_model: usize) -> f32 {
    // Tropical algebra replaces exp/softmax with max operations
    // Theoretical speedup based on operation complexity reduction

    let softmax_ops = seq_len * d_model * 2; // exp + normalization
    let tropical_ops = seq_len * d_model; // max operation only

    softmax_ops as f32 / tropical_ops as f32
}
