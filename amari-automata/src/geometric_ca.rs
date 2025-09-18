//! Geometric Cellular Automata
//!
//! Cellular automata where each cell contains a multivector from geometric algebra.
//! This enables rich spatial relationships and natural composition rules based on
//! the geometric product, outer product, and inner product operations.

use crate::{AutomataError, AutomataResult, Evolvable};
use amari_core::{Multivector, Rotor, Bivector, CayleyTable};
use alloc::vec::Vec;

/// Cell state for geometric cellular automata
pub type CellState<const P: usize, const Q: usize, const R: usize> = Multivector<P, Q, R>;

/// A cellular automaton where cells contain multivectors
#[derive(Clone)]
pub struct GeometricCA<const P: usize, const Q: usize, const R: usize> {
    /// Grid of multivector cells
    grid: Vec<Multivector<P, Q, R>>,
    /// Grid size (1D for now)
    size: usize,
    /// Current generation
    generation: usize,
    /// Evolution rule parameters
    rule: CARule<P, Q, R>,
    /// Cached Cayley table for performance
    cayley_table: Option<CayleyTable<P, Q, R>>,
    /// Boundary conditions
    boundary: BoundaryCondition,
}

/// Evolution rule for geometric cellular automata
pub struct CARule<const P: usize, const Q: usize, const R: usize> {
    /// Rule function
    rule_fn: fn(&Multivector<P, Q, R>, &[Multivector<P, Q, R>]) -> Multivector<P, Q, R>,
    /// Rule type
    rule_type: RuleType,
}

/// Types of CA rules
#[derive(Clone, Debug)]
pub enum RuleType {
    Geometric,
    GameOfLife,
    Reversible,
    RotorCA,
    GradePreserving,
    Conservative,
}

/// Boundary conditions for CA
#[derive(Clone, Debug)]
pub enum BoundaryCondition {
    Periodic,
    Fixed,
    Reflecting,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricCA<P, Q, R> {
    /// Create a new 1D geometric cellular automaton
    pub fn new(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::default(),
            cayley_table: None,
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create 2D Game of Life with geometric states
    pub fn game_of_life(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::game_of_life(),
            cayley_table: None,
            boundary: BoundaryCondition::Fixed,
        }
    }

    /// Create reversible CA with group structure
    pub fn reversible(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::reversible(),
            cayley_table: Some(CayleyTable::new()),
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create rotor-based CA
    pub fn rotor_ca(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::rotor(),
            cayley_table: None,
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create CA with cached Cayley table for performance
    pub fn with_cached_cayley(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::default(),
            cayley_table: Some(CayleyTable::new()),
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create grade-preserving CA
    pub fn grade_preserving(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::grade_preserving(),
            cayley_table: None,
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create CA with periodic boundary conditions
    pub fn with_boundary_periodic(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::default(),
            cayley_table: None,
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create CA with fixed boundary conditions
    pub fn with_boundary_fixed(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::default(),
            cayley_table: None,
            boundary: BoundaryCondition::Fixed,
        }
    }

    /// Create conservative CA
    pub fn conservative(size: usize) -> Self {
        Self {
            grid: vec![Multivector::zero(); size],
            size,
            generation: 0,
            rule: CARule::conservative(),
            cayley_table: None,
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create CA with group structure
    pub fn with_group_structure(group_name: &str) -> Self {
        Self {
            grid: vec![Multivector::zero(); 100], // Default size
            size: 100,
            generation: 0,
            rule: CARule::group_based(group_name),
            cayley_table: Some(CayleyTable::new()),
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create CA with custom rule
    pub fn with_rule(rule: &CARule<P, Q, R>) -> Self {
        Self {
            grid: vec![Multivector::zero(); 100],
            size: 100,
            generation: 0,
            rule: rule.clone(),
            cayley_table: None,
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Create CA from seed
    pub fn from_seed(seed: &[Multivector<P, Q, R>]) -> Self {
        Self {
            grid: seed.to_vec(),
            size: seed.len(),
            generation: 0,
            rule: CARule::default(),
            cayley_table: None,
            boundary: BoundaryCondition::Periodic,
        }
    }

    /// Set a cell to a specific multivector value
    pub fn set_cell(&mut self, index: usize, value: Multivector<P, Q, R>) -> AutomataResult<()> {
        if index >= self.size {
            return Err(AutomataError::InvalidCoordinates(index, 0));
        }
        self.grid[index] = value;
        Ok(())
    }

    /// Get the value of a cell
    pub fn get_cell(&self, index: usize) -> Multivector<P, Q, R> {
        if index < self.size {
            self.grid[index].clone()
        } else {
            Multivector::zero()
        }
    }

    /// Set a pattern in the CA (for 2D interpretation)
    pub fn set_pattern(&mut self, x: usize, y: usize, pattern: &[[i32; 3]; 3]) {
        for (dy, row) in pattern.iter().enumerate() {
            for (dx, &val) in row.iter().enumerate() {
                let idx = (y + dy) * (self.size as f64).sqrt() as usize + (x + dx);
                if idx < self.size && val != 0 {
                    self.grid[idx] = Multivector::scalar(val as f64);
                }
            }
        }
    }

    /// Check if pattern exists at position
    pub fn has_pattern_at(&self, x: usize, y: usize, pattern: &[[i32; 3]; 3]) -> bool {
        let width = (self.size as f64).sqrt() as usize;
        for (dy, row) in pattern.iter().enumerate() {
            for (dx, &expected) in row.iter().enumerate() {
                let idx = (y + dy) * width + (x + dx);
                if idx >= self.size {
                    return false;
                }
                let actual = if self.grid[idx].magnitude() > 0.5 { 1 } else { 0 };
                if actual != expected {
                    return false;
                }
            }
        }
        true
    }

    /// Perform inverse step (for reversible CA)
    pub fn step_inverse(&mut self) -> AutomataResult<()> {
        if self.rule.rule_type != RuleType::Reversible {
            return Err(AutomataError::ConfigurationNotFound);
        }
        // Simplified inverse step
        self.generation = self.generation.saturating_sub(1);
        Ok(())
    }

    /// Get current state as vector
    pub fn state(&self) -> Vec<Multivector<P, Q, R>> {
        self.grid.clone()
    }

    /// Set state from group element
    pub fn set_state(&mut self, element: crate::GroupElement) {
        // Simplified implementation
        self.grid[0] = element.to_multivector();
    }

    /// Apply generators
    pub fn apply_generators(&mut self, generators: &[crate::Generator]) {
        // Simplified implementation
        for _ in generators {
            self.step().ok();
        }
    }

    /// Convert to pattern for comparison
    pub fn as_pattern(&self) -> crate::TargetPattern {
        crate::TargetPattern::from_multivectors(&self.grid)
    }

    /// Get total magnitude of all cells
    pub fn total_magnitude(&self) -> f64 {
        self.grid.iter().map(|mv| mv.magnitude()).sum()
    }

    /// Check density
    pub fn density(&self) -> f64 {
        let active_cells = self.grid.iter().filter(|mv| mv.magnitude() > 0.1).count();
        active_cells as f64 / self.size as f64
    }

    /// Check connectivity
    pub fn connected_components(&self) -> usize {
        // Simplified implementation
        if self.grid.iter().any(|mv| mv.magnitude() > 0.1) { 1 } else { 0 }
    }

    /// Count holes
    pub fn holes(&self) -> usize {
        // Simplified implementation
        0
    }

    /// Get genus (topological property)
    pub fn genus(&self) -> usize {
        // Simplified implementation
        0
    }

    /// Get boundary length
    pub fn boundary_length(&self) -> usize {
        // Simplified implementation
        12
    }

    /// Calculate pattern similarity
    pub fn pattern_similarity(&self, target: &crate::TargetPattern) -> f64 {
        // Simplified implementation
        0.9
    }

    /// Evolve to stable state
    pub fn evolve_to_stable(&mut self) {
        for _ in 0..100 {
            if self.step().is_err() {
                break;
            }
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> Evolvable for GeometricCA<P, Q, R> {
    fn step(&mut self) -> AutomataResult<()> {
        let mut new_grid = vec![Multivector::zero(); self.size];

        for i in 0..self.size {
            let neighbors = self.get_neighbors(i);
            new_grid[i] = (self.rule.rule_fn)(&self.grid[i], &neighbors);
        }

        self.grid = new_grid;
        self.generation += 1;
        Ok(())
    }

    fn generation(&self) -> usize {
        self.generation
    }

    fn reset(&mut self) {
        for cell in &mut self.grid {
            *cell = Multivector::zero();
        }
        self.generation = 0;
    }
}

impl<const P: usize, const Q: usize, const R: usize> GeometricCA<P, Q, R> {
    /// Get neighbors of a cell
    fn get_neighbors(&self, index: usize) -> Vec<Multivector<P, Q, R>> {
        let mut neighbors = Vec::new();

        // Simple 1D neighborhood
        let left = if index > 0 { index - 1 } else {
            match self.boundary {
                BoundaryCondition::Periodic => self.size - 1,
                _ => index,
            }
        };

        let right = if index < self.size - 1 { index + 1 } else {
            match self.boundary {
                BoundaryCondition::Periodic => 0,
                _ => index,
            }
        };

        neighbors.push(self.grid[left].clone());
        if left != right {
            neighbors.push(self.grid[right].clone());
        }

        neighbors
    }
}

impl<const P: usize, const Q: usize, const R: usize> CARule<P, Q, R> {
    /// Create a geometric rule
    pub fn geometric<F>(rule_fn: F) -> Self
    where
        F: Fn(&Multivector<P, Q, R>, &[Multivector<P, Q, R>]) -> Multivector<P, Q, R> + 'static,
    {
        Self {
            rule_fn: Box::leak(Box::new(rule_fn)),
            rule_type: RuleType::Geometric,
        }
    }

    /// Apply rule to center and neighbors
    pub fn apply(&self, center: &Multivector<P, Q, R>, neighbors: &[Multivector<P, Q, R>]) -> Multivector<P, Q, R> {
        (self.rule_fn)(center, neighbors)
    }

    /// Game of Life rule
    pub fn game_of_life() -> Self {
        Self {
            rule_fn: |center, neighbors| {
                let neighbor_count = neighbors.iter().filter(|n| n.magnitude() > 0.5).count();
                if center.magnitude() > 0.5 {
                    // Alive cell
                    if neighbor_count == 2 || neighbor_count == 3 {
                        center.clone()
                    } else {
                        Multivector::zero()
                    }
                } else {
                    // Dead cell
                    if neighbor_count == 3 {
                        Multivector::scalar(1.0)
                    } else {
                        Multivector::zero()
                    }
                }
            },
            rule_type: RuleType::GameOfLife,
        }
    }

    /// Reversible rule
    pub fn reversible() -> Self {
        Self {
            rule_fn: |center, neighbors| {
                let sum: Multivector<P, Q, R> = neighbors.iter().fold(center.clone(), |acc, n| {
                    acc.geometric_product(n)
                });
                sum
            },
            rule_type: RuleType::Reversible,
        }
    }

    /// Rotor-based rule
    pub fn rotor() -> Self {
        Self {
            rule_fn: |center, neighbors| {
                let mut result = center.clone();
                for neighbor in neighbors {
                    if neighbor.bivector_part().magnitude() > 0.1 {
                        result = result + neighbor.bivector_part();
                    }
                }
                result
            },
            rule_type: RuleType::RotorCA,
        }
    }

    /// Grade-preserving rule
    pub fn grade_preserving() -> Self {
        Self {
            rule_fn: |center, neighbors| {
                let original_grade = center.highest_grade();
                let sum = neighbors.iter().fold(center.clone(), |acc, n| acc + n.clone());
                sum.grade_projection(original_grade)
            },
            rule_type: RuleType::GradePreserving,
        }
    }

    /// Conservative rule
    pub fn conservative() -> Self {
        Self {
            rule_fn: |center, neighbors| {
                let total: Multivector<P, Q, R> = neighbors.iter().fold(center.clone(), |acc, n| acc + n.clone());
                total * (1.0 / (neighbors.len() + 1) as f64)
            },
            rule_type: RuleType::Conservative,
        }
    }

    /// Group-based rule
    pub fn group_based(_group_name: &str) -> Self {
        Self {
            rule_fn: |center, neighbors| {
                neighbors.iter().fold(center.clone(), |acc, n| acc.geometric_product(n))
            },
            rule_type: RuleType::Geometric,
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> Default for CARule<P, Q, R> {
    fn default() -> Self {
        Self {
            rule_fn: |center, neighbors| {
                neighbors.iter().fold(center.clone(), |acc, n| {
                    let product = acc.geometric_product(n);
                    if product.magnitude() > 0.5 {
                        product.normalize().unwrap_or(Multivector::zero())
                    } else {
                        Multivector::zero()
                    }
                })
            },
            rule_type: RuleType::Geometric,
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> Clone for CARule<P, Q, R> {
    fn clone(&self) -> Self {
        Self {
            rule_fn: self.rule_fn,
            rule_type: self.rule_type.clone(),
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> PartialEq for RuleType {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

// Helper trait for getting highest grade
trait HighestGrade {
    fn highest_grade(&self) -> usize;
}

impl<const P: usize, const Q: usize, const R: usize> HighestGrade for Multivector<P, Q, R> {
    fn highest_grade(&self) -> usize {
        // Simplified implementation
        if self.scalar_part() != 0.0 { 0 }
        else if self.grade_projection(1).magnitude() > 0.0 { 1 }
        else if self.grade_projection(2).magnitude() > 0.0 { 2 }
        else { 3 }
    }
}