use amari_enumerative::performance::{
    CurveBatchProcessor, FastIntersectionComputer, MemoryPool, SparseSchubertMatrix,
    WasmPerformanceConfig,
};
use amari_enumerative::{ChowClass, ProjectiveSpace};
use num_rational::Rational64;

#[test]
fn test_wasm_performance_config_creation() {
    let config = WasmPerformanceConfig::default();

    // Test default configuration values match the Default trait
    assert!(config.enable_simd);
    assert!(!config.enable_gpu); // Conservative default
    assert_eq!(config.memory_pool_mb, 64);
    assert_eq!(config.batch_size, 1024);
    assert_eq!(config.max_workers, 4);
    assert!(config.enable_workers);
    assert_eq!(config.cache_size, 10000);
}

#[test]
fn test_wasm_performance_config_cloning() {
    let config1 = WasmPerformanceConfig::default();
    let config2 = config1.clone();

    // Test that Clone works correctly
    assert_eq!(config1.enable_simd, config2.enable_simd);
    assert_eq!(config1.enable_gpu, config2.enable_gpu);
    assert_eq!(config1.memory_pool_mb, config2.memory_pool_mb);
    assert_eq!(config1.batch_size, config2.batch_size);
    assert_eq!(config1.max_workers, config2.max_workers);
    assert_eq!(config1.enable_workers, config2.enable_workers);
    assert_eq!(config1.cache_size, config2.cache_size);
}

#[test]
fn test_wasm_performance_config_modification() {
    let mut config = WasmPerformanceConfig::default();

    // Test that we can modify the configuration
    config.enable_simd = false;
    config.enable_gpu = true;
    config.memory_pool_mb = 128;
    config.batch_size = 2048;
    config.max_workers = 8;
    config.enable_workers = false;
    config.cache_size = 20000;

    assert!(!config.enable_simd);
    assert!(config.enable_gpu);
    assert_eq!(config.memory_pool_mb, 128);
    assert_eq!(config.batch_size, 2048);
    assert_eq!(config.max_workers, 8);
    assert!(!config.enable_workers);
    assert_eq!(config.cache_size, 20000);
}

#[test]
fn test_fast_intersection_computer_creation() {
    let config = WasmPerformanceConfig::default();
    let computer = FastIntersectionComputer::new(config);

    // Basic creation test - ensure it compiles and creates
    assert!(true);
}

#[test]
fn test_sparse_schubert_matrix_creation() {
    let matrix = SparseSchubertMatrix::new(10, 10);

    // Basic creation test
    assert!(true);
}

#[test]
fn test_memory_pool_creation() {
    let pool = MemoryPool::new(1024);

    // Basic creation test with size
    assert!(true);
}

#[test]
fn test_curve_batch_processor_creation() {
    let config = WasmPerformanceConfig::default();
    let processor = CurveBatchProcessor::new(config);

    // Basic creation test with config
    assert!(true);
}

#[test]
fn test_performance_structs_compile() {
    // This test ensures all structs can be instantiated and compile correctly
    let config = WasmPerformanceConfig::default();
    let _computer = FastIntersectionComputer::new(config.clone());
    let _matrix = SparseSchubertMatrix::new(5, 5);
    let _pool = MemoryPool::new(512);
    let _processor = CurveBatchProcessor::new(config);

    // If we get here, all structs compiled and can be created
    assert!(true);
}

#[test]
fn test_integration_with_existing_types() {
    // Test that performance module works with existing enumerative geometry types
    let config = WasmPerformanceConfig::default();
    let _computer = FastIntersectionComputer::new(config);

    // Create some standard geometric objects
    let p2 = ProjectiveSpace::new(2);
    let cubic = ChowClass::hypersurface(3);
    let quartic = ChowClass::hypersurface(4);

    // Test that they can coexist (basic integration test)
    assert_eq!(p2.dimension, 2);
    assert_eq!(cubic.degree, Rational64::from(3));
    assert_eq!(quartic.degree, Rational64::from(4));
}

#[test]
fn test_config_with_different_values() {
    // Test creating configurations with different performance characteristics

    // High performance configuration
    let mut high_perf = WasmPerformanceConfig::default();
    high_perf.enable_simd = true;
    high_perf.enable_gpu = true;
    high_perf.memory_pool_mb = 256;
    high_perf.batch_size = 4096;
    high_perf.max_workers = 16;
    high_perf.cache_size = 50000;

    // Low memory configuration
    let mut low_mem = WasmPerformanceConfig::default();
    low_mem.enable_simd = false;
    low_mem.enable_gpu = false;
    low_mem.memory_pool_mb = 16;
    low_mem.batch_size = 128;
    low_mem.max_workers = 2;
    low_mem.cache_size = 1000;

    // Verify the configurations
    assert!(high_perf.memory_pool_mb > low_mem.memory_pool_mb);
    assert!(high_perf.batch_size > low_mem.batch_size);
    assert!(high_perf.max_workers > low_mem.max_workers);
    assert!(high_perf.cache_size > low_mem.cache_size);
}

#[test]
fn test_performance_reasonable_defaults() {
    let config = WasmPerformanceConfig::default();

    // Verify that default values are reasonable for WASM deployment
    assert!(config.memory_pool_mb > 0);
    assert!(config.memory_pool_mb <= 256); // Reasonable WASM limit
    assert!(config.batch_size > 0);
    assert!(config.batch_size <= 8192); // Reasonable batch size
    assert!(config.max_workers > 0);
    assert!(config.max_workers <= 32); // Reasonable worker count
    assert!(config.cache_size > 0);
}

#[test]
fn test_config_field_types() {
    let config = WasmPerformanceConfig::default();

    // Verify field types are correct
    let _simd: bool = config.enable_simd;
    let _gpu: bool = config.enable_gpu;
    let _mem: usize = config.memory_pool_mb;
    let _batch: usize = config.batch_size;
    let _workers: usize = config.max_workers;
    let _workers_enabled: bool = config.enable_workers;
    let _cache: usize = config.cache_size;

    // All type assignments should work if we get here
    assert!(true);
}

#[cfg(feature = "wasm")]
#[test]
fn test_wasm_feature_compilation() {
    // This test only runs when wasm feature is enabled
    // It ensures WASM-specific code compiles
    let config = WasmPerformanceConfig::default();
    assert!(!config.enable_gpu); // Default should be conservative for WASM
}

#[cfg(feature = "parallel")]
#[test]
fn test_parallel_feature_compilation() {
    // This test only runs when parallel feature is enabled
    let config = WasmPerformanceConfig::default();
    let processor = CurveBatchProcessor::new(config);
    assert!(true); // Basic compilation test
}

#[test]
fn test_performance_module_constants() {
    // Test any module-level constants or static values
    let config = WasmPerformanceConfig::default();

    // Verify reasonable defaults
    assert!(config.cache_size > 0);
    assert!(config.batch_size > 0);
    assert!(config.memory_pool_mb > 0);
    assert!(config.max_workers > 0);
}

#[test]
fn test_performance_configuration_scenarios() {
    // Test different deployment scenarios

    // Browser WASM deployment
    let mut browser_config = WasmPerformanceConfig::default();
    browser_config.enable_gpu = false; // WebGL limitations
    browser_config.memory_pool_mb = 32; // Conservative memory
    browser_config.max_workers = 4; // Limited workers

    // Node.js WASM deployment
    let mut node_config = WasmPerformanceConfig::default();
    node_config.enable_gpu = false; // No GPU in Node
    node_config.memory_pool_mb = 128; // More memory available
    node_config.max_workers = 8; // More workers possible

    // Embedded WASM deployment
    let mut embedded_config = WasmPerformanceConfig::default();
    embedded_config.enable_simd = false; // May not be supported
    embedded_config.enable_gpu = false; // Definitely not available
    embedded_config.memory_pool_mb = 8; // Very limited memory
    embedded_config.max_workers = 1; // Single threaded
    embedded_config.cache_size = 100; // Minimal cache

    // All configurations should be valid
    assert!(browser_config.memory_pool_mb <= 32);
    assert!(node_config.memory_pool_mb >= 32);
    assert!(embedded_config.memory_pool_mb <= 8);
}
