//! Error types for automata operations

use alloc::string::String;
use core::fmt;

/// Errors that can occur in automata operations
#[derive(Debug, Clone, PartialEq)]
pub enum AutomataError {
    /// Invalid automaton configuration
    InvalidConfiguration(String),

    /// Invalid rule specification
    InvalidRule(String),

    /// Invalid grid dimensions or index
    InvalidDimensions(String),

    /// Invalid coordinates provided
    InvalidCoordinates(usize, usize),

    /// Configuration not found during inverse design
    ConfigurationNotFound,

    /// Assembly failed due to constraints
    AssemblyFailed(String),

    /// Connection error in self-assembly
    ConnectionError(String),

    /// Invalid component specification
    InvalidComponent(String),

    /// Evolution step failed
    EvolutionFailed(String),

    /// Invalid target pattern
    InvalidTarget(String),

    /// Optimization failed to converge
    OptimizationFailed(String),

    /// Generic error with custom message
    Other(String),
}

impl fmt::Display for AutomataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutomataError::InvalidConfiguration(msg) => {
                write!(f, "Invalid automaton configuration: {}", msg)
            }
            AutomataError::InvalidRule(msg) => write!(f, "Invalid rule: {}", msg),
            AutomataError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            AutomataError::InvalidCoordinates(x, y) => {
                write!(f, "Invalid coordinates: ({}, {})", x, y)
            }
            AutomataError::ConfigurationNotFound => {
                write!(f, "Configuration not found during inverse design")
            }
            AutomataError::AssemblyFailed(msg) => write!(f, "Assembly failed: {}", msg),
            AutomataError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            AutomataError::InvalidComponent(msg) => write!(f, "Invalid component: {}", msg),
            AutomataError::EvolutionFailed(msg) => write!(f, "Evolution failed: {}", msg),
            AutomataError::InvalidTarget(msg) => write!(f, "Invalid target: {}", msg),
            AutomataError::OptimizationFailed(msg) => write!(f, "Optimization failed: {}", msg),
            AutomataError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl core::error::Error for AutomataError {}

/// Result type for automata operations
pub type AutomataResult<T> = Result<T, AutomataError>;
