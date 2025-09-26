use amari_enumerative::{
    IntersectionRing, ChowClass, IntersectionNumber,
    ProjectiveSpace, Grassmannian, Constraint, AlgebraicVariety
};

#[test]
fn test_bezouts_theorem() {
    // Two curves of degrees d and e in P² intersect in d*e points
    let p2 = ProjectiveSpace::new(2);

    let curve1 = ChowClass::hypersurface(3); // Cubic curve
    let curve2 = ChowClass::hypersurface(4); // Quartic curve

    let intersection = p2.intersect(&curve1, &curve2);

    assert_eq!(intersection.multiplicity(), 12); // 3 * 4 = 12
}

#[test]
fn test_lines_through_points() {
    // Exactly one line passes through 2 points in P²
    let p2 = ProjectiveSpace::new(2);

    let point1 = ChowClass::point();
    let point2 = ChowClass::point();
    let line_class = ChowClass::linear_subspace(1);

    let count = p2.count_objects(
        line_class,
        vec![
            Constraint::PassesThrough(point1),
            Constraint::PassesThrough(point2),
        ]
    );

    assert_eq!(count, 1);
}

#[test]
fn test_degree_genus_formula() {
    // For plane curves: genus = (d-1)(d-2)/2
    let degree = 4;
    let curve = ChowClass::plane_curve(degree);

    let genus = curve.arithmetic_genus();

    assert_eq!(genus, 3); // (4-1)(4-2)/2 = 3
}

#[test]
fn test_chow_ring_product() {
    // In P^n, H^(n+1) = 0 where H is hyperplane class
    let p3 = ProjectiveSpace::new(3);
    let h = p3.hyperplane_class();

    // H^3 should be the point class (dimension 0)
    let h3 = h.power(3);
    assert_eq!(h3.dimension, 0);
    assert_eq!(h3.degree.to_integer(), 1);

    // H^4 should be zero
    let h4 = h.power(4);
    assert!(h4.is_zero());
}

#[test]
fn test_intersection_with_geometric_algebra() {
    // Use Clifford algebra to compute intersections
    use amari_enumerative::intersection::MockMultivector;

    let variety = AlgebraicVariety::from_multivector(
        MockMultivector::from_polynomial("x^2 + y^2 - z^2")
    );

    let line = AlgebraicVariety::from_multivector(
        AlgebraicVariety::line_through_points([0, 0, 0], [1, 1, 1])
    );

    let intersections = variety.intersect_with(&line);

    assert_eq!(intersections.len(), 2); // Line meets quadric in 2 points
}

#[test]
fn test_projective_space_dimension() {
    let p2 = ProjectiveSpace::new(2);
    let p3 = ProjectiveSpace::new(3);

    // Basic dimension checks
    assert_eq!(p2.dimension, 2);
    assert_eq!(p3.dimension, 3);
}

#[test]
fn test_grassmannian_creation() {
    // Test valid Grassmannian creation
    let gr_2_4 = Grassmannian::new(2, 4).unwrap();
    assert_eq!(gr_2_4.k, 2);
    assert_eq!(gr_2_4.n, 4);
    assert_eq!(gr_2_4.dimension(), 4); // 2 * (4 - 2) = 4

    // Test invalid Grassmannian (k > n)
    let invalid = Grassmannian::new(5, 3);
    assert!(invalid.is_err());
}

#[test]
fn test_chow_class_operations() {
    let point = ChowClass::point();
    let line = ChowClass::linear_subspace(1);
    let plane = ChowClass::linear_subspace(2);

    // Test basic properties
    assert_eq!(point.dimension, 0);
    assert_eq!(line.dimension, 1);
    assert_eq!(plane.dimension, 2);

    // Test multiplication
    let product = line.multiply(&line);
    assert_eq!(product.dimension, 2);
    assert_eq!(product.degree.to_integer(), 1);
}

#[test]
fn test_hypersurface_intersections() {
    let p3 = ProjectiveSpace::new(3);

    let quadric = ChowClass::hypersurface(2);
    let cubic = ChowClass::hypersurface(3);

    let intersection = p3.intersect(&quadric, &cubic);

    // Bézout's theorem: 2 * 3 = 6
    assert_eq!(intersection.multiplicity(), 6);
}

#[test]
fn test_empty_intersection() {
    let p2 = ProjectiveSpace::new(2);

    // Two general lines in P^2 should intersect in 1 point
    let line1 = ChowClass::linear_subspace(1);
    let line2 = ChowClass::linear_subspace(1);

    let intersection = p2.intersect(&line1, &line2);
    assert_eq!(intersection.multiplicity(), 1);

    // But if we force dimension too high, we get empty intersection
    let high_dim_class = ChowClass::new(3, num_rational::Rational64::from(1));
    let empty_intersection = p2.intersect(&line1, &high_dim_class);
    assert_eq!(empty_intersection.multiplicity(), 0);
}