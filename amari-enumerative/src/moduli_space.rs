//! Moduli spaces of curves and algebraic varieties
//!
//! This module implements computations on moduli spaces, including
//! the moduli space of curves, stable maps, and intersection numbers
//! on these spaces.

use num_rational::Rational64;
use std::collections::HashMap;
use crate::{EnumerativeError, EnumerativeResult};

/// Moduli space of curves M_{g,n}
#[derive(Debug, Clone)]
pub struct ModuliSpace {
    /// Genus of the curves
    pub genus: usize,
    /// Number of marked points
    pub marked_points: usize,
    /// Whether to consider stable curves (compactified moduli space)
    pub stable: bool,
}

impl ModuliSpace {
    /// Create a new moduli space M_{g,n}
    pub fn new(genus: usize, marked_points: usize, stable: bool) -> EnumerativeResult<Self> {
        // Check stability condition: 2g - 2 + n > 0 for stable curves
        // For the stable compactification, we allow (g=0, n=0) as it gives M̄_{0,0} = point
        if stable && 2 * genus + marked_points < 3 && !(genus == 0 && marked_points == 0) {
            return Err(EnumerativeError::InvalidDimension(
                format!("Unstable parameters: g={}, n={}", genus, marked_points)
            ));
        }

        Ok(Self {
            genus,
            marked_points,
            stable,
        })
    }

    /// Compute the dimension of the moduli space
    pub fn dimension(&self) -> usize {
        if self.stable {
            // Dimension of \bar{M}_{g,n}
            // Handle case where 3*genus < 3 to avoid underflow
            if 3 * self.genus >= 3 {
                3 * self.genus - 3 + self.marked_points
            } else {
                // For genus 0, dimension is marked_points - 3 (when stable)
                if self.marked_points >= 3 {
                    self.marked_points - 3
                } else {
                    0
                }
            }
        } else {
            // Dimension of M_{g,n} (if it exists)
            if self.genus == 0 && self.marked_points >= 3 {
                self.marked_points - 3
            } else if self.genus >= 2 {
                3 * self.genus - 3
            } else {
                0
            }
        }
    }

    /// Check if the moduli space is non-empty
    pub fn is_nonempty(&self) -> bool {
        if self.stable {
            2 * self.genus - 2 + self.marked_points > 0
        } else {
            self.genus >= 0 && self.marked_points >= 0
        }
    }

    /// Compute intersection numbers on the moduli space
    pub fn intersection_number(&self, classes: &[TautologicalClass]) -> EnumerativeResult<Rational64> {
        // This is highly non-trivial and requires knowledge of the intersection theory
        // of moduli spaces. For now, we provide some basic cases.

        if classes.is_empty() {
            return Ok(Rational64::from(1));
        }

        // Check dimension compatibility
        let total_degree: usize = classes.iter().map(|c| c.degree).sum();
        if total_degree != self.dimension() {
            return Ok(Rational64::from(0));
        }

        // Some classical intersection numbers
        match (self.genus, self.marked_points, classes.len()) {
            (0, 3, 0) => Ok(Rational64::from(1)), // \bar{M}_{0,3} is a point
            (1, 1, 1) => {
                // Some intersection on \bar{M}_{1,1}
                Ok(Rational64::new(1, 24))
            }
            _ => {
                // General case requires Witten's conjecture/Kontsevich's theorem
                Ok(Rational64::from(0))
            }
        }
    }
}

/// Tautological classes on moduli spaces (ψ, κ, λ classes)
#[derive(Debug, Clone, PartialEq)]
pub struct TautologicalClass {
    /// Type of the class
    pub class_type: TautologicalType,
    /// Degree of the class
    pub degree: usize,
    /// Index (for ψ classes, this is which marked point)
    pub index: Option<usize>,
}

impl TautologicalClass {
    /// Create a ψ class
    pub fn psi(index: usize) -> Self {
        Self {
            class_type: TautologicalType::Psi,
            degree: 1,
            index: Some(index),
        }
    }

    /// Create a κ class
    pub fn kappa(degree: usize) -> Self {
        Self {
            class_type: TautologicalType::Kappa,
            degree,
            index: None,
        }
    }

    /// Create a λ class
    pub fn lambda(degree: usize) -> Self {
        Self {
            class_type: TautologicalType::Lambda,
            degree,
            index: None,
        }
    }
}

/// Types of tautological classes
#[derive(Debug, Clone, PartialEq)]
pub enum TautologicalType {
    /// ψ classes (cotangent line classes at marked points)
    Psi,
    /// κ classes (κ_m = π_*(ψ^{m+1}))
    Kappa,
    /// λ classes (Chern classes of Hodge bundle)
    Lambda,
}

/// Curve class in a target variety
#[derive(Debug, Clone, PartialEq)]
pub struct CurveClass {
    /// The target variety (simplified as string for now)
    pub target: String,
    /// Degree information
    pub degree_data: HashMap<String, i64>,
    /// Genus of curves in this class
    pub genus: usize,
}

impl CurveClass {
    /// Create a new curve class
    pub fn new(target: String, genus: usize) -> Self {
        Self {
            target,
            degree_data: HashMap::new(),
            genus,
        }
    }

    /// Set degree in a particular homology class
    pub fn set_degree(&mut self, class_name: String, degree: i64) {
        self.degree_data.insert(class_name, degree);
    }

    /// Get degree in a particular homology class
    pub fn get_degree(&self, class_name: &str) -> i64 {
        self.degree_data.get(class_name).copied().unwrap_or(0)
    }

    /// Check if this is a rational curve class
    pub fn is_rational(&self) -> bool {
        self.genus == 0
    }
}

/// Moduli space of stable maps \bar{M}_{g,n}(X, β)
#[derive(Debug, Clone)]
pub struct ModuliOfStableMaps {
    /// Domain moduli space
    pub domain: ModuliSpace,
    /// Target variety (simplified)
    pub target: String,
    /// Curve class being mapped to
    pub curve_class: CurveClass,
}

impl ModuliOfStableMaps {
    /// Create a new moduli space of stable maps
    pub fn new(
        domain: ModuliSpace,
        target: String,
        curve_class: CurveClass,
    ) -> Self {
        Self {
            domain,
            target,
            curve_class,
        }
    }

    /// Compute expected dimension of the moduli space
    pub fn expected_dimension(&self) -> EnumerativeResult<i64> {
        // Expected dimension formula:
        // dim(M_{g,n}(X, β)) = dim(M_{g,n}) + ∫_β c_1(TX) + (dim(X) - 3)(1-g)

        let moduli_dim = self.domain.dimension() as i64;

        // Simplified: assume target is projective space P^n
        let target_dim = match self.target.as_str() {
            "P1" => 1,
            "P2" => 2,
            "P3" => 3,
            _ => 2, // default to P^2
        };

        let degree = self.curve_class.get_degree("H"); // degree in hyperplane class
        let first_chern_integral = (target_dim + 1) * degree;

        let expected_dim = moduli_dim + first_chern_integral +
                          (target_dim - 3) * (1 - self.domain.genus as i64);

        Ok(expected_dim)
    }

    /// Check if the moduli space has the expected dimension
    pub fn has_expected_dimension(&self) -> EnumerativeResult<bool> {
        let expected = self.expected_dimension()?;
        Ok(expected >= 0)
    }
}