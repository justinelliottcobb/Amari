//! WebAssembly Edge Computing Tests - TDD Phase 2
//!
//! Tests for zero-copy TypedArray integration and JavaScript interop
//! for high-performance edge computing applications.

use wasm_bindgen_test::*;
use amari_wasm::{WasmGpuInfoGeometry, WasmMultivector, AmariEdgeCompute};
use js_sys::{Float64Array, Uint32Array};

wasm_bindgen_test_configure!(run_in_browser);

/// Test WebAssembly zero-copy TypedArray integration
#[wasm_bindgen_test]
async fn test_wasm_zero_copy_tensor_computation() {
    let edge_compute = AmariEdgeCompute::new().await
        .expect("Should initialize edge compute");

    // Create TypedArrays from JavaScript side
    let vector_data = Float64Array::new_with_length(900); // 100 triplets × 9 components

    // Fill with test data
    for i in 0..900 {
        vector_data.set_index(i, (i as f64) * 0.1);
    }

    // Compute tensor batch using zero-copy operations
    let results = edge_compute
        .amari_chentsov_tensor_batch_typed_array(&vector_data, 100)
        .await
        .expect("Batch computation should succeed");

    assert_eq!(results.length(), 100);

    // Verify some computations
    for i in 0..10 {
        let result = results.get_index(i);
        assert!(result.is_finite(), "Result should be finite");
    }
}

/// Test WebGPU device detection in WebAssembly
#[wasm_bindgen_test]
async fn test_wasm_webgpu_device_detection() {
    // Test automatic device detection
    match WasmGpuInfoGeometry::new_auto().await {
        Ok(gpu_info_geom) => {
            let device_info = gpu_info_geom.get_device_info();

            // Should detect WebGPU capability
            assert!(device_info.supports_webgpu(), "Should detect WebGPU support");

            // Test a simple computation
            let x = WasmMultivector::new_e1();
            let y = WasmMultivector::new_e2();
            let z = WasmMultivector::new_e3();

            let result = gpu_info_geom.amari_chentsov_tensor(&x, &y, &z).await
                .expect("Tensor computation should work");

            assert!((result - 1.0).abs() < 1e-10);
        }
        Err(_) => {
            // Fallback to CPU if WebGPU not available
            let cpu_compute = AmariEdgeCompute::new_cpu_fallback()
                .expect("CPU fallback should always work");

            let device_info = cpu_compute.get_device_info();
            assert!(!device_info.supports_webgpu(), "Should indicate CPU mode");
        }
    }
}

/// Test worker thread integration for background computation
#[wasm_bindgen_test]
async fn test_worker_thread_integration() {
    let edge_compute = AmariEdgeCompute::new().await
        .expect("Should initialize edge compute");

    // Test data that would normally block the main thread
    let large_vector_data = Float64Array::new_with_length(90000); // 10k triplets

    // Fill with test data
    for i in 0..90000 {
        large_vector_data.set_index(i, (i as f64) * 0.001);
    }

    // Compute in worker thread (non-blocking)
    let worker_promise = edge_compute
        .amari_chentsov_tensor_batch_worker(&large_vector_data, 10000);

    // This should not block the main thread
    let start_time = js_sys::Date::now();

    let results = worker_promise.await
        .expect("Worker computation should complete");

    let duration = js_sys::Date::now() - start_time;

    assert_eq!(results.length(), 10000);

    // Should complete in reasonable time for worker thread
    assert!(duration < 5000.0, "Worker computation should complete in <5s");
}

/// Test progressive enhancement: CPU → WebGPU → Edge device
#[wasm_bindgen_test]
async fn test_progressive_enhancement() {
    // Try devices in order of preference
    let device_preferences = ["discrete-gpu", "integrated-gpu", "cpu"];

    let mut best_compute = None;

    for device_type in device_preferences {
        if let Ok(compute) = AmariEdgeCompute::new_with_device(device_type).await {
            best_compute = Some(compute);
            break;
        }
    }

    let edge_compute = best_compute
        .expect("Should find at least CPU fallback");

    // Test computation works regardless of device
    let test_data = Float64Array::new_with_length(27); // 3 triplets

    // Fill with identity basis vectors
    test_data.set_index(0, 1.0); // x = e1
    test_data.set_index(10, 1.0); // y = e2
    test_data.set_index(20, 1.0); // z = e3

    let results = edge_compute
        .amari_chentsov_tensor_batch_typed_array(&test_data, 3)
        .await
        .expect("Computation should work on any device");

    // First result should be 1.0 (e1 × e2 × e3)
    assert!((results.get_index(0) - 1.0).abs() < 1e-10);
}

/// Test memory management with large datasets
#[wasm_bindgen_test]
async fn test_memory_management_large_datasets() {
    let edge_compute = AmariEdgeCompute::new().await
        .expect("Should initialize edge compute");

    // Test progressively larger datasets
    let dataset_sizes = [1000, 5000, 10000];

    for &size in &dataset_sizes {
        let vector_data = Float64Array::new_with_length(size * 9);

        // Fill with random-ish data
        for i in 0..(size * 9) {
            vector_data.set_index(i, (i as f64 * 0.001).sin());
        }

        let memory_before = edge_compute.get_memory_usage();

        let results = edge_compute
            .amari_chentsov_tensor_batch_typed_array(&vector_data, size)
            .await
            .expect("Large dataset computation should work");

        let memory_after = edge_compute.get_memory_usage();

        assert_eq!(results.length() as usize, size);

        // Memory should be efficiently managed
        let memory_increase = memory_after - memory_before;
        assert!(memory_increase < (size * 64) as f64, // Reasonable memory per computation
               "Memory usage should be efficient for dataset size {}", size);

        // Force garbage collection to test cleanup
        edge_compute.cleanup_memory();
    }
}

/// Test JavaScript API ergonomics and performance
#[wasm_bindgen_test]
async fn test_javascript_api_ergonomics() {
    let edge_compute = AmariEdgeCompute::new().await
        .expect("Should initialize edge compute");

    // Test fluent API design
    let result = edge_compute
        .create_computation_pipeline()
        .add_tensor_operation("amari-chentsov")
        .add_batch_size(100)
        .add_device_preference("auto")
        .execute_pipeline()
        .await
        .expect("Pipeline should execute");

    assert!(result.is_success(), "Pipeline execution should succeed");

    // Test callback-based API for streaming results
    let mut received_results = 0;

    let callback = wasm_bindgen::closure::Closure::wrap(Box::new(move |result: f64| {
        received_results += 1;
        assert!(result.is_finite(), "Streamed result should be finite");
    }) as Box<dyn FnMut(f64)>);

    // Create test data
    let streaming_data = Float64Array::new_with_length(2700); // 300 triplets
    for i in 0..2700 {
        streaming_data.set_index(i, (i as f64) * 0.01);
    }

    edge_compute
        .amari_chentsov_tensor_batch_streaming(
            &streaming_data,
            300,
            callback.as_ref().unchecked_ref()
        )
        .await
        .expect("Streaming computation should work");

    // Clean up
    callback.forget();
}

/// Test WebAssembly module optimization and size
#[wasm_bindgen_test]
fn test_wasm_module_optimization() {
    // Test that WASM module is appropriately sized
    let module_size = amari_wasm::get_module_size();

    // Should be optimized for edge deployment
    assert!(module_size < 2_000_000, // <2MB
           "WASM module should be optimized for edge deployment, got {} bytes", module_size);

    // Test that critical functions are exported
    let exports = amari_wasm::get_exported_functions();

    let required_exports = [
        "amari_chentsov_tensor_batch_typed_array",
        "fisher_information_matrix_batch",
        "bregman_divergence_batch",
        "create_computation_pipeline",
    ];

    for export in required_exports {
        assert!(exports.contains(&export.to_string()),
               "Required function {} should be exported", export);
    }
}

/// Test cross-origin resource sharing for edge deployment
#[wasm_bindgen_test]
async fn test_cors_edge_deployment() {
    // Test that WASM module can be loaded from different origins
    let edge_compute = AmariEdgeCompute::new_from_url(
        "https://cdn.example.com/amari-wasm/amari_edge_compute.wasm"
    ).await;

    match edge_compute {
        Ok(compute) => {
            // If CDN is available, test basic functionality
            let device_info = compute.get_device_info();
            assert!(device_info.is_initialized(), "Remote module should initialize");
        }
        Err(_) => {
            // Expected in test environment - just verify local module works
            let local_compute = AmariEdgeCompute::new().await
                .expect("Local module should work");

            assert!(local_compute.get_device_info().is_initialized());
        }
    }
}

/// Test TypeScript type definitions accuracy
#[wasm_bindgen_test]
fn test_typescript_type_definitions() {
    // Verify that TypeScript definitions match actual WASM exports
    let type_definitions = amari_wasm::get_typescript_definitions();

    // Should include proper type definitions for all major functions
    assert!(type_definitions.contains("amariChentsovTensorBatch"),
           "TypeScript definitions should include tensor operations");
    assert!(type_definitions.contains("Float64Array"),
           "TypeScript definitions should include TypedArray types");
    assert!(type_definitions.contains("Promise<Float64Array>"),
           "TypeScript definitions should include async return types");

    // Verify no `any` types in critical functions (type safety)
    assert!(!type_definitions.contains("): any"),
           "TypeScript definitions should avoid 'any' types");
}