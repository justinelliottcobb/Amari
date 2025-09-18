//! Geometric Cellular Automata
//!
//! Cellular automata where each cell contains a multivector from geometric algebra.
//! This enables rich spatial relationships and natural composition rules based on
//! the geometric product, outer product, and inner product operations.

use crate::{AutomataError, AutomataResult, Evolvable};
use amari_core::Multivector;
use alloc::vec::Vec;

/// A cellular automaton where cells contain multivectors
pub struct GeometricCA<const P: usize, const Q: usize, const R: usize> {
    /// Grid of multivector cells
    grid: Vec<Vec<Multivector<P, Q, R>>>,
    /// Grid dimensions
    width: usize,
    height: usize,
    /// Current generation
    generation: usize,
    /// Evolution rule parameters
    rule_params: CARule<P, Q, R>,
}

/// Evolution rule for geometric cellular automata
pub struct CARule<const P: usize, const Q: usize, const R: usize> {
    /// Threshold for activation
    threshold: f64,
    /// Geometric product weight
    geo_weight: f64,
    /// Outer product weight
    outer_weight: f64,
    /// Inner product weight
    inner_weight: f64,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricCA<P, Q, R> {
    /// Create a new geometric cellular automaton
    pub fn new(width: usize, height: usize) -> Self {
        let grid = vec![vec![Multivector::zero(); width]; height];
        Self {
            grid,
            width,
            height,
            generation: 0,
            rule_params: CARule::default(),
        }
    }

    /// Set the value of a cell
    pub fn set_cell(&mut self, x: usize, y: usize, value: Multivector<P, Q, R>) -> AutomataResult<()> {
        if x >= self.width || y >= self.height {
            return Err(AutomataError::InvalidCoordinates(x, y));
        }
        self.grid[y][x] = value;
        Ok(())
    }

    /// Get the value of a cell
    pub fn get_cell(&self, x: usize, y: usize) -> AutomataResult<&Multivector<P, Q, R>> {
        if x >= self.width || y >= self.height {
            return Err(AutomataError::InvalidCoordinates(x, y));
        }
        Ok(&self.grid[y][x])
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Set evolution rule parameters
    pub fn set_rule(&mut self, rule: CARule<P, Q, R>) {
        self.rule_params = rule;
    }

    /// Get neighbors of a cell (8-neighborhood)
    fn get_neighbors(&self, x: usize, y: usize) -> Vec<&Multivector<P, Q, R>> {
        let mut neighbors = Vec::new();

        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 { continue; }

                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && ny >= 0 && (nx as usize) < self.width && (ny as usize) < self.height {
                    neighbors.push(&self.grid[ny as usize][nx as usize]);
                }
            }
        }

        neighbors
    }

    /// Apply evolution rule to compute next state
    fn evolve_cell(&self, x: usize, y: usize) -> Multivector<P, Q, R> {
        let current = &self.grid[y][x];
        let neighbors = self.get_neighbors(x, y);

        // Combine neighbor contributions using geometric algebra operations
        let mut sum = Multivector::zero();

        for neighbor in neighbors {
            // Geometric product contribution
            let geo = current.geometric_product(neighbor);
            sum = sum + geo * self.rule_params.geo_weight;

            // Outer product contribution
            let outer = current.outer_product(neighbor);
            sum = sum + outer * self.rule_params.outer_weight;

            // Inner product contribution
            let inner_scalar = current.inner_product(neighbor);
            let inner_mv = Multivector::scalar(inner_scalar);
            sum = sum + inner_mv * self.rule_params.inner_weight;
        }

        // Apply threshold and normalization
        let magnitude = sum.magnitude();
        if magnitude > self.rule_params.threshold {
            sum.normalize().unwrap_or(Multivector::zero())
        } else {
            Multivector::zero()
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> Evolvable for GeometricCA<P, Q, R> {
    fn step(&mut self) -> AutomataResult<()> {
        let mut new_grid = vec![vec![Multivector::zero(); self.width]; self.height];

        for y in 0..self.height {
            for x in 0..self.width {
                new_grid[y][x] = self.evolve_cell(x, y);
            }
        }

        self.grid = new_grid;
        self.generation += 1;
        Ok(())
    }

    fn generation(&self) -> usize {
        self.generation
    }

    fn reset(&mut self) {
        for row in &mut self.grid {
            for cell in row {
                *cell = Multivector::zero();
            }
        }
        self.generation = 0;
    }
}

impl<const P: usize, const Q: usize, const R: usize> Default for CARule<P, Q, R> {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            geo_weight: 1.0,
            outer_weight: 0.5,
            inner_weight: 0.5,
        }
    }
}

impl<const P: usize, const Q: usize, const R: usize> CARule<P, Q, R> {
    /// Create a new rule with custom parameters
    pub fn new(threshold: f64, geo_weight: f64, outer_weight: f64, inner_weight: f64) -> Self {
        Self {
            threshold,
            geo_weight,
            outer_weight,
            inner_weight,
        }
    }
}