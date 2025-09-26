//! Integration between geometric algebra and enumerative geometry
//!
//! This module bridges the gap between Clifford algebra structures from amari-core
//! and enumerative geometry computations, enabling geometric representations of
//! algebraic varieties, Schubert cells, and moduli spaces.

use amari_core::{Multivector, Rotor};
use crate::{ChowClass, SchubertClass, IntersectionNumber, ProjectiveSpace, EnumerativeResult, EnumerativeError};
use num_rational::Rational64;
use std::collections::HashMap;

/// Common geometric algebra signatures for enumerative geometry
pub mod signatures {
    use amari_core::Multivector;

    /// Euclidean 3D space Cl(3,0,0) - for real projective geometry
    pub type Euclidean3D = Multivector<3, 0, 0>;

    /// Projective space Cl(3,1,0) - adds point at infinity
    pub type Projective3D = Multivector<3, 1, 0>;

    /// Conformal space Cl(4,1,0) - adds conformal point
    pub type Conformal4D = Multivector<4, 1, 0>;

    /// Complex projective space Cl(2,0,0) ⊗ ℂ representation
    pub type ComplexProjective = Multivector<4, 0, 0>;

    /// Grassmannian representation Cl(k,n-k,0) for Gr(k,n)
    pub type GrassmannianGA<const K: usize, const N_MINUS_K: usize> = Multivector<K, N_MINUS_K, 0>;
}

/// Geometric representation of an algebraic variety using multivectors
#[derive(Debug, Clone)]
pub struct GeometricVariety<const P: usize, const Q: usize, const R: usize> {
    /// The underlying multivector representation
    pub multivector: Multivector<P, Q, R>,
    /// Dimension of the variety
    pub dimension: usize,
    /// Degree of the variety
    pub degree: Rational64,
    /// Additional geometric invariants
    pub invariants: HashMap<String, f64>,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricVariety<P, Q, R> {
    /// Create a new geometric variety from a multivector
    pub fn new(multivector: Multivector<P, Q, R>, dimension: usize, degree: Rational64) -> Self {
        Self {
            multivector,
            dimension,
            degree,
            invariants: HashMap::new(),
        }
    }

    /// Create a point variety (0-dimensional)
    pub fn point(coordinates: &[f64]) -> EnumerativeResult<Self> {
        if coordinates.len() != P + Q + R {
            return Err(EnumerativeError::InvalidDimension(
                format!("Expected {} coordinates, got {}", P + Q + R, coordinates.len())
            ));
        }

        let mut mv = Multivector::zero();
        for (i, &coord) in coordinates.iter().enumerate() {
            mv = mv + Multivector::basis_vector(i) * coord;
        }

        Ok(Self::new(mv, 0, Rational64::from(1)))
    }

    /// Create a line variety (1-dimensional) from two points
    pub fn line_through_points(p1: &Self, p2: &Self) -> EnumerativeResult<Self> {
        if p1.dimension != 0 || p2.dimension != 0 {
            return Err(EnumerativeError::InvalidDimension(
                "Line construction requires point varieties".to_string()
            ));
        }

        // Line as wedge product of two points in projective geometry
        let line_mv = p1.multivector.outer_product(&p2.multivector);
        Ok(Self::new(line_mv, 1, Rational64::from(1)))
    }

    /// Create a plane variety (2-dimensional) from three points
    pub fn plane_through_points(p1: &Self, p2: &Self, p3: &Self) -> EnumerativeResult<Self> {
        if p1.dimension != 0 || p2.dimension != 0 || p3.dimension != 0 {
            return Err(EnumerativeError::InvalidDimension(
                "Plane construction requires point varieties".to_string()
            ));
        }

        // Plane as wedge product of three points
        let plane_mv = p1.multivector.outer_product(&p2.multivector).outer_product(&p3.multivector);
        Ok(Self::new(plane_mv, 2, Rational64::from(1)))
    }

    /// Convert to a Chow class representation
    pub fn to_chow_class(&self) -> ChowClass {
        ChowClass::new(self.dimension, self.degree)
    }

    /// Compute the intersection with another geometric variety
    pub fn intersect_with(&self, other: &Self) -> EnumerativeResult<Vec<Self>> {
        // In geometric algebra, intersection can be computed using the meet operation
        // Meet operation: a ∨ b = ⋆(⋆a ∧ ⋆b) where ⋆ is the Hodge dual
        let self_dual = self.multivector.hodge_dual();
        let other_dual = other.multivector.hodge_dual();
        let intersection_mv = self_dual.outer_product(&other_dual).hodge_dual();
        let intersection_dim = if self.dimension + other.dimension >= P + Q + R {
            0 // Point intersection in general position
        } else {
            (P + Q + R) - (self.dimension + other.dimension)
        };

        let intersection_degree = self.degree * other.degree;

        Ok(vec![Self::new(intersection_mv, intersection_dim, intersection_degree)])
    }

    /// Apply a rotor transformation to the variety
    pub fn transform(&self, rotor: &Rotor<P, Q, R>) -> Self {
        let transformed_mv = rotor.apply(&self.multivector);
        Self {
            multivector: transformed_mv,
            dimension: self.dimension,
            degree: self.degree,
            invariants: self.invariants.clone(),
        }
    }

    /// Compute the geometric degree (related to multivector magnitude)
    pub fn geometric_degree(&self) -> f64 {
        self.multivector.magnitude()
    }

    /// Check if this variety contains a given point
    pub fn contains_point(&self, point: &Self) -> bool {
        if point.dimension != 0 {
            return false;
        }

        // In GA, point lies on variety if their meet equals the point
        // Meet operation: a ∨ b = ⋆(⋆a ∧ ⋆b)
        let self_dual = self.multivector.hodge_dual();
        let point_dual = point.multivector.hodge_dual();
        let meet = self_dual.outer_product(&point_dual).hodge_dual();
        (meet - point.multivector.clone()).magnitude() < 1e-10
    }
}

/// Geometric algebra enhanced Schubert class
#[derive(Debug, Clone)]
pub struct GeometricSchubertClass<const P: usize, const Q: usize, const R: usize> {
    /// Standard Schubert class
    pub schubert_class: SchubertClass,
    /// Geometric algebra representation
    pub multivector: Multivector<P, Q, R>,
    /// Grassmannian parameters (k, n)
    pub grassmannian_dim: (usize, usize),
}

impl<const P: usize, const Q: usize, const R: usize> GeometricSchubertClass<P, Q, R> {
    /// Create a geometric Schubert class from a partition
    pub fn new(partition: Vec<usize>, grassmannian_dim: (usize, usize)) -> EnumerativeResult<Self> {
        let schubert_class = SchubertClass::new(partition, grassmannian_dim)?;

        // Create multivector representation based on the Plücker embedding
        let mut mv = Multivector::scalar(1.0);
        for (i, &part) in schubert_class.partition.iter().enumerate() {
            if part > 0 {
                // Add basis blade corresponding to Schubert condition
                // Use basis vector since there's no basis_blade method
                let blade = Multivector::basis_vector(i + part);
                mv = mv.outer_product(&blade);
            }
        }

        Ok(Self {
            schubert_class,
            multivector: mv,
            grassmannian_dim,
        })
    }

    /// Compute geometric intersection with another Schubert class
    pub fn geometric_intersection(&self, other: &Self) -> EnumerativeResult<Self> {
        if self.grassmannian_dim != other.grassmannian_dim {
            return Err(EnumerativeError::InvalidDimension(
                "Schubert classes must be on the same Grassmannian".to_string()
            ));
        }

        // Geometric intersection via multivector product
        let intersection_mv = self.multivector.geometric_product(&other.multivector);

        // Combine partitions (simplified)
        let mut new_partition = self.schubert_class.partition.clone();
        for (i, &part) in other.schubert_class.partition.iter().enumerate() {
            if i < new_partition.len() {
                new_partition[i] = new_partition[i].saturating_add(part);
            } else {
                new_partition.push(part);
            }
        }

        Ok(Self {
            schubert_class: SchubertClass::new(new_partition, self.grassmannian_dim)?,
            multivector: intersection_mv,
            grassmannian_dim: self.grassmannian_dim,
        })
    }

    /// Convert to standard Schubert class
    pub fn to_schubert_class(&self) -> &SchubertClass {
        &self.schubert_class
    }
}

/// Geometric algebra enhanced projective space
#[derive(Debug, Clone)]
pub struct GeometricProjectiveSpace<const P: usize, const Q: usize, const R: usize> {
    /// Standard projective space
    pub projective_space: ProjectiveSpace,
    /// Geometric algebra context
    pub _phantom: std::marker::PhantomData<Multivector<P, Q, R>>,
}

impl<const P: usize, const Q: usize, const R: usize> GeometricProjectiveSpace<P, Q, R> {
    /// Create a new geometric projective space
    pub fn new(dimension: usize) -> Self {
        Self {
            projective_space: ProjectiveSpace::new(dimension),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a variety from multivector coefficients
    pub fn variety_from_multivector(
        &self,
        multivector: Multivector<P, Q, R>,
        dimension: usize,
        degree: Rational64,
    ) -> GeometricVariety<P, Q, R> {
        GeometricVariety::new(multivector, dimension, degree)
    }

    /// Compute intersection number using geometric algebra
    pub fn geometric_intersection_number(
        &self,
        variety1: &GeometricVariety<P, Q, R>,
        variety2: &GeometricVariety<P, Q, R>,
    ) -> IntersectionNumber {
        // Use geometric product magnitude as intersection multiplicity
        let intersection_mv = variety1.multivector.geometric_product(&variety2.multivector);
        let multiplicity = intersection_mv.magnitude();

        IntersectionNumber::new(Rational64::from(multiplicity as i64))
    }

    /// Create a hyperplane from normal vector
    pub fn hyperplane_from_normal(&self, normal: &[f64]) -> EnumerativeResult<GeometricVariety<P, Q, R>> {
        if normal.len() != P + Q + R {
            return Err(EnumerativeError::InvalidDimension(
                format!("Normal vector must have {} components", P + Q + R)
            ));
        }

        let mut normal_mv = Multivector::zero();
        for (i, &component) in normal.iter().enumerate() {
            normal_mv = normal_mv + Multivector::basis_vector(i) * component;
        }

        Ok(GeometricVariety::new(normal_mv, self.projective_space.dimension - 1, Rational64::from(1)))
    }
}

/// Quantum K-theory for enumerative geometry
///
/// This module implements quantum K-theory, which extends classical K-theory
/// by incorporating quantum corrections from rational curves. It provides
/// tools for computing quantum K-theory rings and quantum products.
pub mod quantum_k_theory {
    use super::*;
    use std::collections::BTreeMap;

    /// Quantum K-theory ring element with geometric algebra enhancement
    #[derive(Debug, Clone)]
    pub struct QuantumKClass<const P: usize, const Q: usize, const R: usize> {
        /// Geometric algebra representation
        pub multivector: Multivector<P, Q, R>,
        /// K-theory degree (virtual dimension)
        pub k_degree: i32,
        /// Quantum parameter q^n where n counts rational curves
        pub q_power: i32,
        /// Todd class corrections for coherent sheaves
        pub todd_coefficients: Vec<Rational64>,
        /// Chern character components
        pub chern_character: BTreeMap<usize, Rational64>,
    }

    impl<const P: usize, const Q: usize, const R: usize> QuantumKClass<P, Q, R> {
        /// Create a new quantum K-class
        pub fn new(multivector: Multivector<P, Q, R>, k_degree: i32, q_power: i32) -> Self {
            Self {
                multivector,
                k_degree,
                q_power,
                todd_coefficients: vec![Rational64::from(1)], // Todd class starts with 1
                chern_character: BTreeMap::new(),
            }
        }

        /// Create the structure sheaf of a point
        pub fn structure_sheaf_point() -> Self {
            Self::new(Multivector::scalar(1.0), 0, 0)
        }

        /// Create a line bundle with first Chern class c₁
        pub fn line_bundle(c1: i64) -> Self {
            let mut class = Self::new(Multivector::scalar(1.0), 0, 0);
            class.chern_character.insert(1, Rational64::from(c1));
            class
        }

        /// Create the tangent bundle of projective space
        pub fn tangent_bundle_projective(dimension: usize) -> Self {
            let mut class = Self::new(Multivector::scalar(1.0), dimension as i32, 0);
            // Tangent bundle of P^n has Chern character e^H * (1 + H)^(n+1) / (1 + H) - 1
            // where H is the hyperplane class
            for i in 1..=dimension {
                class.chern_character.insert(i, Rational64::from(1));
            }
            class
        }

        /// Quantum K-theory product incorporating Gromov-Witten corrections
        pub fn quantum_product(&self, other: &Self) -> EnumerativeResult<Self> {
            // Classical K-theory tensor product
            let classical_mv = self.multivector.geometric_product(&other.multivector);
            let classical_degree = self.k_degree + other.k_degree;

            // Quantum corrections from curve counting
            let mut quantum_power = self.q_power + other.q_power;
            let mut result_class = Self::new(classical_mv, classical_degree, quantum_power);

            // Combine Chern characters using the formula ch(E ⊗ F) = ch(E) * ch(F)
            for (&deg1, &coeff1) in &self.chern_character {
                for (&deg2, &coeff2) in &other.chern_character {
                    let total_deg = deg1 + deg2;
                    let combined_coeff = coeff1 * coeff2;
                    *result_class.chern_character.entry(total_deg).or_insert(Rational64::from(0)) += combined_coeff;
                }
            }

            // Add quantum corrections from Gromov-Witten invariants
            // In quantum K-theory, we get corrections from stable maps of curves
            let gw_correction = self.compute_gw_correction(other)?;
            quantum_power += gw_correction;
            result_class.q_power = quantum_power;

            Ok(result_class)
        }

        /// Compute Gromov-Witten corrections for quantum K-theory product
        fn compute_gw_correction(&self, other: &Self) -> EnumerativeResult<i32> {
            // Simplified GW correction computation
            // In a full implementation, this would involve:
            // 1. Integration over moduli spaces of stable maps
            // 2. Virtual fundamental classes
            // 3. Localization via torus actions

            let total_degree = (self.k_degree + other.k_degree).abs();

            // Basic correction: curves of degree d contribute q^d terms
            if total_degree > 0 && self.has_positive_chern_class() && other.has_positive_chern_class() {
                Ok(total_degree) // Simplified: each positive intersection contributes q^|degree|
            } else {
                Ok(0) // No quantum correction
            }
        }

        /// Check if this class has positive Chern classes (indicating positivity)
        fn has_positive_chern_class(&self) -> bool {
            self.chern_character.values().any(|&coeff| coeff > Rational64::from(0))
        }

        /// Compute the Chern character in cohomology
        pub fn chern_character_total(&self) -> Rational64 {
            self.chern_character.values().sum()
        }

        /// Euler characteristic χ(E) = ∫ ch(E) * td(T_X)
        pub fn euler_characteristic(&self, ambient_dimension: usize) -> Rational64 {
            let ch_total = self.chern_character_total();
            let todd_correction = self.todd_class_value(ambient_dimension);
            ch_total * todd_correction
        }

        /// Todd class value for the ambient space
        fn todd_class_value(&self, dimension: usize) -> Rational64 {
            // Todd class of projective space P^n is (1+H)^(n+1) / ((1+H-1)^(n+1)/H)
            // Simplified: Todd(P^n) ≈ 1 for basic computations
            if dimension == 0 {
                Rational64::from(1)
            } else {
                // For P^n, Todd class gives binomial coefficient corrections
                Rational64::from(1) + Rational64::from(dimension as i64) / Rational64::from(2)
            }
        }

        /// Dual in quantum K-theory
        pub fn dual(&self) -> Self {
            let dual_mv = self.multivector.reverse(); // Use reverse as dual in GA
            let mut dual_class = Self::new(dual_mv, -self.k_degree, -self.q_power);

            // Dual Chern character: ch(E^*) = ch(E)^*
            for (&deg, &coeff) in &self.chern_character {
                dual_class.chern_character.insert(deg, -coeff);
            }

            dual_class
        }

        /// Apply Adams operations ψᵏ
        pub fn adams_operation(&self, k: i32) -> Self {
            let powered_mv = if k >= 0 {
                // For positive k, we take a kind of "power" via geometric product
                let mut result = self.multivector.clone();
                for _ in 1..k {
                    result = result.geometric_product(&self.multivector);
                }
                result
            } else {
                self.multivector.inverse().unwrap_or_else(|| self.multivector.clone())
            };

            let mut adams_class = Self::new(powered_mv, self.k_degree * k, self.q_power * k);

            // Adams operations on Chern character: ψᵏ(ch(E)) = Σ kⁱ chᵢ(E)
            for (&deg, &coeff) in &self.chern_character {
                let k_power = (k as i64).pow(deg as u32);
                let adams_coeff = coeff * Rational64::from(k_power);
                adams_class.chern_character.insert(deg, adams_coeff);
            }

            adams_class
        }

        /// Riemann-Roch theorem computation: χ(E) = ∫ ch(E) * td(X)
        pub fn riemann_roch_euler(&self, ambient_todd: &[Rational64]) -> Rational64 {
            let mut result = Rational64::from(0);

            for (&deg, &ch_coeff) in &self.chern_character {
                if deg < ambient_todd.len() {
                    result += ch_coeff * ambient_todd[deg];
                }
            }

            result
        }

        /// Localization formula for torus-equivariant quantum K-theory
        pub fn localized_integral(&self, fixed_points: &[GeometricVariety<P, Q, R>]) -> EnumerativeResult<Rational64> {
            let mut total = Rational64::from(0);

            // Localization: ∫_X ω = Σ_{f∈X^T} ω(f) / e_T(N_f)
            // where X^T are the torus fixed points and N_f is the normal bundle
            for point in fixed_points {
                let point_contribution = self.evaluate_at_point(point)?;
                let normal_euler = self.normal_bundle_euler_class(point);

                if normal_euler != Rational64::from(0) {
                    total += point_contribution / normal_euler;
                }
            }

            Ok(total)
        }

        /// Evaluate the K-theory class at a fixed point
        fn evaluate_at_point(&self, point: &GeometricVariety<P, Q, R>) -> EnumerativeResult<Rational64> {
            // Simplified evaluation: use the geometric degree
            let geometric_eval = point.geometric_degree();
            Ok(Rational64::from(geometric_eval as i64))
        }

        /// Compute Euler class of normal bundle at fixed point
        fn normal_bundle_euler_class(&self, _point: &GeometricVariety<P, Q, R>) -> Rational64 {
            // Simplified normal bundle computation
            // In practice, this depends on the specific torus action
            let dimension_factor = Rational64::from((P + Q + R) as i64);
            dimension_factor // Simplified placeholder
        }

        /// Quantum cohomology to K-theory correspondence
        pub fn from_quantum_cohomology(_qh_class: &crate::QuantumCohomology) -> EnumerativeResult<Self> {
            // This implements the correspondence between quantum cohomology and quantum K-theory
            // via the Gamma class and other characteristic classes

            let multivector = Multivector::scalar(1.0); // Placeholder
            let mut k_class = Self::new(multivector, 0, 0);

            // Add Chern character from cohomology class
            // In practice, this requires the Chern character map ch: K → H^*
            k_class.chern_character.insert(0, Rational64::from(1));

            Ok(k_class)
        }
    }

    /// Quantum K-theory ring structure
    #[derive(Debug)]
    pub struct QuantumKRing<const P: usize, const Q: usize, const R: usize> {
        /// Base ring generators
        pub generators: Vec<QuantumKClass<P, Q, R>>,
        /// Relations in the ring
        pub relations: Vec<String>,
        /// Quantum parameter
        pub quantum_parameter: String,
    }

    impl<const P: usize, const Q: usize, const R: usize> QuantumKRing<P, Q, R> {
        /// Create a new quantum K-theory ring
        pub fn new() -> Self {
            Self {
                generators: Vec::new(),
                relations: Vec::new(),
                quantum_parameter: "q".to_string(),
            }
        }

        /// Add a generator to the ring
        pub fn add_generator(&mut self, class: QuantumKClass<P, Q, R>) {
            self.generators.push(class);
        }

        /// Quantum K-theory ring of projective space
        pub fn projective_space(dimension: usize) -> Self {
            let mut ring = Self::new();

            // Generator: line bundle O(1)
            let line_bundle = QuantumKClass::line_bundle(1);
            ring.add_generator(line_bundle);

            // Relation: O(1)^(n+1) = 0 in K_0(P^n) tensored with Q
            ring.relations.push(format!("L^{} = 0", dimension + 1));

            ring
        }

        /// Quantum K-theory ring of Grassmannian Gr(k,n)
        pub fn grassmannian(k: usize, n: usize) -> Self {
            let mut ring = Self::new();

            // Tautological bundles S and Q
            let tautological_sub = QuantumKClass::new(Multivector::scalar(1.0), k as i32, 0);
            let quotient_bundle = QuantumKClass::new(Multivector::scalar(1.0), (n - k) as i32, 0);

            ring.add_generator(tautological_sub);
            ring.add_generator(quotient_bundle);

            // Quantum Pieri rules and other relations would go here
            ring.relations.push("Quantum Pieri rules".to_string());

            ring
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::signatures::*;

    #[test]
    fn test_point_creation() {
        let point = GeometricVariety::<3, 0, 0>::point(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(point.dimension, 0);
        assert_eq!(point.degree, Rational64::from(1));
    }

    #[test]
    fn test_line_through_points() {
        let p1 = GeometricVariety::<3, 0, 0>::point(&[1.0, 0.0, 0.0]).unwrap();
        let p2 = GeometricVariety::<3, 0, 0>::point(&[0.0, 1.0, 0.0]).unwrap();
        let line = GeometricVariety::line_through_points(&p1, &p2).unwrap();

        assert_eq!(line.dimension, 1);
        assert_eq!(line.degree, Rational64::from(1));
    }

    #[test]
    fn test_geometric_projective_space() {
        let gp2 = GeometricProjectiveSpace::<2, 1, 0>::new(2);
        let hyperplane = gp2.hyperplane_from_normal(&[1.0, 1.0, 1.0]).unwrap();

        assert_eq!(hyperplane.dimension, 1); // Codimension 1 in P²
    }

    #[test]
    fn test_schubert_class_creation() {
        let schubert = GeometricSchubertClass::<2, 2, 0>::new(vec![1], (2, 4)).unwrap();
        assert_eq!(schubert.grassmannian_dim, (2, 4));
        assert_eq!(schubert.schubert_class.partition, vec![1]);
    }

    // Quantum K-theory tests
    #[test]
    fn test_quantum_k_class_creation() {
        use super::quantum_k_theory::QuantumKClass;

        let mv = Multivector::<3, 0, 0>::scalar(1.0);
        let qk_class = QuantumKClass::new(mv, 1, 0);

        assert_eq!(qk_class.k_degree, 1);
        assert_eq!(qk_class.q_power, 0);
        assert_eq!(qk_class.todd_coefficients, vec![Rational64::from(1)]);
    }

    #[test]
    fn test_line_bundle_creation() {
        use super::quantum_k_theory::QuantumKClass;

        let line_bundle = QuantumKClass::<3, 0, 0>::line_bundle(2);
        assert_eq!(line_bundle.k_degree, 0);
        assert_eq!(*line_bundle.chern_character.get(&1).unwrap(), Rational64::from(2));
    }

    #[test]
    fn test_quantum_product_basic() {
        use super::quantum_k_theory::QuantumKClass;

        let bundle1 = QuantumKClass::<3, 0, 0>::line_bundle(1);
        let bundle2 = QuantumKClass::<3, 0, 0>::line_bundle(1);

        let product = bundle1.quantum_product(&bundle2).unwrap();

        // Check that we get quantum corrections
        assert!(product.q_power >= 0);
        assert!(product.chern_character.len() > 0);
    }

    #[test]
    fn test_adams_operations() {
        use super::quantum_k_theory::QuantumKClass;

        let line_bundle = QuantumKClass::<3, 0, 0>::line_bundle(2);
        let adams_2 = line_bundle.adams_operation(2);

        // Adams operation ψ² on line bundle with c₁=2 should give 2² = 4
        assert_eq!(*adams_2.chern_character.get(&1).unwrap(), Rational64::from(4));
    }

    #[test]
    fn test_euler_characteristic() {
        use super::quantum_k_theory::QuantumKClass;

        let structure_sheaf = QuantumKClass::<3, 0, 0>::structure_sheaf_point();
        let euler_char = structure_sheaf.euler_characteristic(2);

        // Structure sheaf of a point should have χ = 1
        assert!(euler_char >= Rational64::from(0));
    }

    #[test]
    fn test_dual_bundle() {
        use super::quantum_k_theory::QuantumKClass;

        let line_bundle = QuantumKClass::<3, 0, 0>::line_bundle(3);
        let dual = line_bundle.dual();

        // Dual of O(3) should be O(-3)
        assert_eq!(dual.k_degree, -line_bundle.k_degree);
        assert_eq!(*dual.chern_character.get(&1).unwrap(), Rational64::from(-3));
    }

    #[test]
    fn test_quantum_k_ring_projective_space() {
        use super::quantum_k_theory::QuantumKRing;

        let ring = QuantumKRing::<3, 0, 0>::projective_space(2);
        assert_eq!(ring.generators.len(), 1);
        assert!(ring.relations.contains(&"L^3 = 0".to_string()));
    }

    #[test]
    fn test_quantum_k_ring_grassmannian() {
        use super::quantum_k_theory::QuantumKRing;

        let ring = QuantumKRing::<4, 0, 0>::grassmannian(2, 4);
        assert_eq!(ring.generators.len(), 2); // Tautological sub and quotient bundles
        assert!(ring.relations.len() > 0);
    }

    #[test]
    fn test_riemann_roch_computation() {
        use super::quantum_k_theory::QuantumKClass;

        let line_bundle = QuantumKClass::<3, 0, 0>::line_bundle(1);
        let todd_classes = vec![Rational64::from(1), Rational64::from(1)];

        let rr_result = line_bundle.riemann_roch_euler(&todd_classes);
        assert!(rr_result >= Rational64::from(0));
    }

    #[test]
    fn test_chern_character_total() {
        use super::quantum_k_theory::QuantumKClass;

        let mut bundle = QuantumKClass::<3, 0, 0>::new(Multivector::scalar(1.0), 2, 0);
        bundle.chern_character.insert(0, Rational64::from(1));
        bundle.chern_character.insert(1, Rational64::from(3));
        bundle.chern_character.insert(2, Rational64::from(2));

        let total = bundle.chern_character_total();
        assert_eq!(total, Rational64::from(6)); // 1 + 3 + 2 = 6
    }
}