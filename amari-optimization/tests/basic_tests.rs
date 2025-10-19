use amari_optimization::{OptimizationError, OptimizationResult};

#[test]
fn test_optimization_error_creation() {
    let error = OptimizationError::ConvergenceFailure { iterations: 100 };
    assert!(error.to_string().contains("100"));

    let error = OptimizationError::InvalidProblem {
        message: "Test problem".to_string(),
    };
    assert!(error.to_string().contains("Test problem"));

    let error = OptimizationError::NumericalError {
        message: "Numerical instability".to_string(),
    };
    assert!(error.to_string().contains("Numerical instability"));
}

#[test]
fn test_optimization_result() {
    let success: OptimizationResult<f64> = Ok(42.0);
    assert_eq!(success.unwrap(), 42.0);

    let failure: OptimizationResult<f64> = Err(OptimizationError::ConvergenceFailure {
        iterations: 1000,
    });
    assert!(failure.is_err());
}