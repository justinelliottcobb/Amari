//! GPU-accelerated LLM evaluation using Tropical-Dual-Clifford fusion systems
//!
//! This example demonstrates how the fusion of tropical algebra, dual numbers,
//! and Clifford algebra can accelerate LLM evaluation with geometric awareness.

#[cfg(feature = "gpu")]
use amari_dual::gpu::GpuDualNumber;
#[cfg(feature = "gpu")]
use amari_fusion::{
    gpu::{
        FusionGpuOps, FusionObjective, FusionOptimizationConfig, GeometricAttentionConfig,
        GpuTropicalDualClifford, LlmEvaluationConfig,
    },
    TropicalDualClifford,
};
#[cfg(feature = "gpu")]
use amari_tropical::gpu::GpuTropicalNumber;

#[cfg(feature = "gpu")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 GPU-Accelerated LLM Evaluation with Fusion Systems");
    println!("====================================================");

    // Initialize GPU context
    let mut fusion_ops = match FusionGpuOps::new().await {
        Ok(ops) => {
            println!("✅ Fusion GPU context initialized successfully");
            ops
        }
        Err(e) => {
            println!("⚠️  GPU not available, falling back to CPU demo: {}", e);
            return demonstrate_cpu_fusion();
        }
    };

    // Configuration for LLM evaluation
    let eval_config = LlmEvaluationConfig {
        tropical_weight: 0.4,   // Emphasis on efficient path finding
        dual_weight: 0.35,      // Automatic differentiation importance
        geometric_weight: 0.25, // Geometric relationship weighting
    };

    println!("\n🧠 LLM Evaluation Configuration:");
    println!("   Tropical weight: {:.2}", eval_config.tropical_weight);
    println!("   Dual weight: {:.2}", eval_config.dual_weight);
    println!("   Geometric weight: {:.2}", eval_config.geometric_weight);

    // Create sample LLM embeddings for evaluation
    let vocabulary_size = 256;
    let sequence_length = 128;

    println!("\n📊 Generating LLM Embeddings:");
    println!("   Vocabulary size: {}", vocabulary_size);
    println!("   Sequence length: {}", sequence_length);

    let input_embeddings = generate_sample_embeddings(sequence_length, "input_sequence");
    let reference_embeddings = generate_reference_embeddings(vocabulary_size, "reference_vocab");

    println!("   Input embeddings: {} items", input_embeddings.len());
    println!(
        "   Reference embeddings: {} items",
        reference_embeddings.len()
    );

    // Perform GPU-accelerated LLM evaluation
    println!("\n🚀 Executing GPU LLM Evaluation...");
    let eval_start = std::time::Instant::now();

    let evaluation_result = fusion_ops
        .llm_evaluation(&input_embeddings, &reference_embeddings, &eval_config)
        .await?;

    let eval_time = eval_start.elapsed();
    println!("✅ LLM evaluation completed in {:?}", eval_time);

    // Analyze results
    println!("\n📈 Evaluation Results:");
    println!(
        "   Average tropical score: {:.6}",
        evaluation_result.average_tropical_score
    );
    println!(
        "   Average dual sensitivity: {:.6}",
        evaluation_result.average_dual_sensitivity
    );
    println!(
        "   Average geometric alignment: {:.6}",
        evaluation_result.average_geometric_alignment
    );
    println!(
        "   Best match index: {}",
        evaluation_result.best_match_index
    );
    println!(
        "   Best combined score: {:.6}",
        evaluation_result.best_combined_score
    );

    // Demonstrate geometric attention
    println!("\n🔍 Demonstrating Geometric Attention...");

    let attention_config = GeometricAttentionConfig {
        tropical_weight: 0.33,
        dual_weight: 0.33,
        geometric_weight: 0.34,
        temperature: 1.0,
    };

    // Use a smaller subset for attention computation
    let attention_sequence = input_embeddings[..16.min(input_embeddings.len())].to_vec();

    let attention_start = std::time::Instant::now();
    let attention_output = fusion_ops
        .geometric_attention(
            &attention_sequence,
            &attention_sequence,
            &attention_sequence,
            &attention_config,
        )
        .await?;

    let attention_time = attention_start.elapsed();
    println!("✅ Geometric attention completed in {:?}", attention_time);
    println!("   Output sequence length: {}", attention_output.len());

    // Analyze attention patterns
    analyze_attention_patterns(&attention_sequence, &attention_output);

    // Demonstrate fusion optimization
    println!("\n🎯 Demonstrating Fusion Optimization...");

    let optimization_config = FusionOptimizationConfig {
        learning_rate: 0.01,
        max_iterations: 50, // Reduced for demo
        convergence_threshold: 1e-4,
    };

    // Create optimization objectives
    let objectives = create_optimization_objectives(4);

    let optimization_start = std::time::Instant::now();
    let optimized_params = fusion_ops
        .batch_fusion_optimization(&attention_sequence[..4], &objectives, &optimization_config)
        .await?;

    let optimization_time = optimization_start.elapsed();
    println!(
        "✅ Fusion optimization completed in {:?}",
        optimization_time
    );
    println!("   Optimized {} parameters", optimized_params.len());

    // Performance analysis
    println!("\n📊 Performance Analysis:");
    let total_operations = input_embeddings.len() * reference_embeddings.len();
    let throughput = total_operations as f64 / eval_time.as_secs_f64();
    println!("   Total comparisons: {}", total_operations);
    println!(
        "   Evaluation throughput: {:.0} comparisons/second",
        throughput
    );

    let theoretical_speedup = estimate_fusion_speedup(sequence_length, vocabulary_size);
    println!("   Theoretical speedup vs CPU: {:.1}x", theoretical_speedup);

    // Demonstrate fusion algebra properties
    println!("\n🔬 Fusion Algebra Properties:");
    demonstrate_fusion_properties(&input_embeddings[..3]);

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("⚠️  This example requires the 'gpu' feature to be enabled.");
    println!("Run with: cargo run --example gpu_llm_evaluation --features gpu");
}

#[cfg(feature = "gpu")]
fn demonstrate_cpu_fusion() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔄 Demonstrating CPU fusion fallback...");

    // Create sample TDC objects for demonstration
    let logits1 = vec![2.0f32, 1.5, 3.0, 0.8, 1.2];
    let logits2 = vec![1.8f32, 2.1, 2.5, 1.0, 1.5];

    let tdc1 = TropicalDualClifford::<f32, 8>::from_logits(&logits1);
    let tdc2 = TropicalDualClifford::<f32, 8>::from_logits(&logits2);

    println!("✅ Created TropicalDualClifford objects:");
    println!(
        "   TDC1 tropical max: {:.3}",
        tdc1.tropical().max_element().value()
    );
    println!(
        "   TDC2 tropical max: {:.3}",
        tdc2.tropical().max_element().value()
    );

    // Demonstrate evaluation
    let evaluation = tdc1.evaluate(&tdc2);
    println!("✅ Fusion evaluation completed:");
    println!(
        "   Best path score: {:.6}",
        evaluation.best_path_score.value()
    );
    println!("   Gradient norm: {:.6}", evaluation.gradient_norm);
    println!(
        "   Geometric distance: {:.6}",
        evaluation.geometric_distance
    );
    println!("   Combined score: {:.6}", evaluation.combined_score);

    // Demonstrate distance computation
    let distance = tdc1.distance(&tdc2);
    println!("✅ Fusion distance: {:.6}", distance);

    // Demonstrate sensitivity analysis
    let sensitivity = tdc1.sensitivity_analysis();
    println!("✅ Sensitivity analysis:");
    println!(
        "   Total sensitivity: {:.6}",
        sensitivity.total_sensitivity()
    );
    println!(
        "   Most sensitive components: {:?}",
        sensitivity.most_sensitive(3)
    );

    Ok(())
}

#[cfg(feature = "gpu")]
fn generate_sample_embeddings(count: usize, seed: &str) -> Vec<GpuTropicalDualClifford> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Create deterministic seed from string
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let seed_value = hasher.finish();

    // Use seed to create reproducible "random" embeddings
    (0..count)
        .map(|i| {
            let base_seed = seed_value.wrapping_add(i as u64);

            // Generate tropical component
            let tropical_value = ((base_seed % 1000) as f32 / 1000.0) * 4.0 - 2.0;

            // Generate dual component
            let dual_real = ((base_seed.wrapping_mul(17) % 1000) as f32 / 1000.0) * 2.0 - 1.0;
            let dual_dual = ((base_seed.wrapping_mul(31) % 1000) as f32 / 1000.0) * 0.5;

            // Generate Clifford components
            let mut clifford = [0.0f32; 8];
            for (j, value) in clifford.iter_mut().enumerate() {
                *value = ((base_seed.wrapping_mul((j + 1) as u64 * 13) % 1000) as f32 / 1000.0)
                    * 1.0
                    - 0.5;
            }

            GpuTropicalDualClifford {
                tropical: GpuTropicalNumber {
                    value: tropical_value,
                },
                dual: GpuDualNumber {
                    real: dual_real,
                    dual: dual_dual,
                },
                clifford,
            }
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn generate_reference_embeddings(count: usize, seed: &str) -> Vec<GpuTropicalDualClifford> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let seed_value = hasher.finish();

    (0..count)
        .map(|i| {
            let base_seed = seed_value.wrapping_add((i as u64).wrapping_mul(47));

            // Generate more structured reference embeddings
            let tropical_value = ((base_seed % 100) as f32 / 100.0) * 3.0;
            let dual_real = ((base_seed.wrapping_mul(23) % 100) as f32 / 100.0) * 1.5;
            let dual_dual = ((base_seed.wrapping_mul(41) % 100) as f32 / 100.0) * 0.3;

            let mut clifford = [0.0f32; 8];
            for (j, value) in clifford.iter_mut().enumerate() {
                *value =
                    ((base_seed.wrapping_mul((j + 1) as u64 * 7) % 100) as f32 / 100.0) * 0.8 - 0.4;
            }

            GpuTropicalDualClifford {
                tropical: GpuTropicalNumber {
                    value: tropical_value,
                },
                dual: GpuDualNumber {
                    real: dual_real,
                    dual: dual_dual,
                },
                clifford,
            }
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn analyze_attention_patterns(
    input: &[GpuTropicalDualClifford],
    output: &[GpuTropicalDualClifford],
) {
    println!("\n🔍 Attention Pattern Analysis:");

    if input.is_empty() || output.is_empty() {
        println!("   No attention patterns to analyze");
        return;
    }

    // Compute attention focus metrics
    let mut tropical_focus = 0.0f32;
    let mut dual_focus = 0.0f32;
    let mut geometric_focus = 0.0f32;

    for (inp, out) in input.iter().zip(output.iter()) {
        // Tropical attention change
        tropical_focus += (out.tropical.value - inp.tropical.value).abs();

        // Dual attention change
        dual_focus += (out.dual.real - inp.dual.real).abs();
        dual_focus += (out.dual.dual - inp.dual.dual).abs();

        // Geometric attention change
        for i in 0..8 {
            geometric_focus += (out.clifford[i] - inp.clifford[i]).abs();
        }
    }

    let seq_len = input.len() as f32;
    println!("   Tropical focus change: {:.4}", tropical_focus / seq_len);
    println!("   Dual focus change: {:.4}", dual_focus / seq_len);
    println!(
        "   Geometric focus change: {:.4}",
        geometric_focus / seq_len
    );

    // Compute attention concentration
    let output_variance = compute_sequence_variance(output);
    println!(
        "   Attention concentration: {:.4}",
        1.0 / (1.0 + output_variance)
    );
}

#[cfg(feature = "gpu")]
fn compute_sequence_variance(sequence: &[GpuTropicalDualClifford]) -> f32 {
    if sequence.is_empty() {
        return 0.0;
    }

    let mean_tropical =
        sequence.iter().map(|x| x.tropical.value).sum::<f32>() / sequence.len() as f32;
    let variance = sequence
        .iter()
        .map(|x| (x.tropical.value - mean_tropical).powi(2))
        .sum::<f32>()
        / sequence.len() as f32;

    variance
}

#[cfg(feature = "gpu")]
fn create_optimization_objectives(count: usize) -> Vec<FusionObjective> {
    (0..count)
        .map(|i| FusionObjective {
            target_tropical: (i as f32 * 0.5) + 1.0,
            target_dual_real: (i as f32 * 0.3) + 0.5,
            target_dual_dual: (i as f32 * 0.1) + 0.1,
            target_clifford: [i as f32 * 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            weight: 1.0 / (i + 1) as f32,
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn estimate_fusion_speedup(sequence_length: usize, vocab_size: usize) -> f32 {
    // Theoretical speedup based on:
    // - Tropical algebra: O(1) max vs O(n) softmax
    // - Dual numbers: automatic gradients vs backpropagation
    // - Clifford algebra: geometric operations vs matrix multiplications

    let comparison_operations = (sequence_length * vocab_size) as f32;

    // Base GPU parallelism speedup
    let gpu_parallelism = 32.0; // Typical GPU warp size

    // Algorithmic improvements
    let tropical_speedup = 2.0; // Max vs softmax
    let dual_speedup = 1.5; // Forward vs backward AD
    let clifford_speedup = 1.3; // Geometric vs matrix ops

    let combined_algorithmic = tropical_speedup * dual_speedup * clifford_speedup;

    // Diminishing returns for very large problems
    let scale_factor = (comparison_operations.log2() / 20.0).min(2.0);

    gpu_parallelism * combined_algorithmic * scale_factor
}

#[cfg(feature = "gpu")]
fn demonstrate_fusion_properties(embeddings: &[GpuTropicalDualClifford]) {
    if embeddings.len() < 3 {
        println!("   Insufficient embeddings for property demonstration");
        return;
    }

    let e1 = &embeddings[0];
    let e2 = &embeddings[1];
    let _e3 = &embeddings[2];

    // Demonstrate tropical properties (max-plus algebra)
    println!("   Tropical properties:");
    let tropical_max = e1.tropical.value.max(e2.tropical.value);
    println!(
        "     max({:.3}, {:.3}) = {:.3}",
        e1.tropical.value, e2.tropical.value, tropical_max
    );

    // Demonstrate dual number properties (automatic differentiation)
    println!("   Dual number properties:");
    let dual_product_real = e1.dual.real * e2.dual.real - e1.dual.dual * e2.dual.dual;
    let dual_product_dual = e1.dual.real * e2.dual.dual + e1.dual.dual * e2.dual.real;
    println!(
        "     ({:.3}+{:.3}ε) * ({:.3}+{:.3}ε) = {:.3}+{:.3}ε",
        e1.dual.real,
        e1.dual.dual,
        e2.dual.real,
        e2.dual.dual,
        dual_product_real,
        dual_product_dual
    );

    // Demonstrate Clifford algebra properties (geometric product)
    println!("   Clifford algebra properties:");
    let mut geometric_product = 0.0f32;
    for i in 0..8 {
        geometric_product += e1.clifford[i] * e2.clifford[i];
    }
    println!("     Geometric product magnitude: {:.6}", geometric_product);

    // Demonstrate fusion coherence
    let fusion_coherence = (tropical_max + dual_product_real + geometric_product) / 3.0;
    println!("   Fusion coherence: {:.6}", fusion_coherence);
}
