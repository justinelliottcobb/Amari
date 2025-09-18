//! Tests for Self-Assembly System

use amari_automata::{
    SelfAssembler, SelfAssembling, Component, Assembly, ComponentType, UIComponentType,
    AssemblyConfig,
};
use amari_core::{Multivector, Vector, Bivector};
use approx::assert_relative_eq;

type TestAssembler = SelfAssembler<3, 0, 0>;
type TestComponent = Component<3, 0, 0>;
type TestAssembly = Assembly<3, 0, 0>;

#[test]
fn test_component_creation() {
    let signature = Multivector::basis_vector(0);
    let position = Vector::e1();
    let component = TestComponent::new(signature.clone(), position.clone(), ComponentType::Basic);

    assert_eq!(component.component_type, ComponentType::Basic);
    assert_relative_eq!(component.signature.magnitude(), signature.magnitude());
    assert_relative_eq!(component.scale, 1.0);
}

#[test]
fn test_ui_component_creation() {
    let signature = Multivector::scalar(1.0);
    let position = Vector::e2();
    let ui_component = TestComponent::ui_component(
        signature,
        position,
        UIComponentType::Button,
    );

    match ui_component.component_type {
        ComponentType::UIElement(UIComponentType::Button) => {}
        _ => panic!("Expected Button UI component"),
    }
}

#[test]
fn test_component_transformation() {
    let signature = Multivector::basis_vector(1);
    let position = Vector::e1();
    let orientation = Bivector::e12();

    let component = TestComponent::new(signature, position, ComponentType::Basic)
        .with_orientation(orientation)
        .with_scale(2.0);

    assert_relative_eq!(component.scale, 2.0);

    let transformed = component.transformed_signature();
    assert!(transformed.magnitude() > 0.0);
}

#[test]
fn test_component_connectivity() {
    let sig = Multivector::scalar(1.0);
    let pos = Vector::zero();

    let basic = TestComponent::new(sig.clone(), pos.clone(), ComponentType::Basic);
    let corner = TestComponent::new(sig.clone(), pos.clone(), ComponentType::Corner);
    let edge = TestComponent::new(sig.clone(), pos.clone(), ComponentType::Edge);
    let junction = TestComponent::new(sig, pos, ComponentType::Junction);

    // Basic components can connect to anything
    assert!(basic.can_connect_to(&corner));
    assert!(basic.can_connect_to(&edge));
    assert!(basic.can_connect_to(&junction));

    // Corner connects to edge and junction
    assert!(corner.can_connect_to(&edge));
    assert!(corner.can_connect_to(&junction));

    // Edge connects to junction
    assert!(edge.can_connect_to(&junction));
}

#[test]
fn test_ui_component_connectivity() {
    let sig = Multivector::scalar(1.0);
    let pos = Vector::zero();

    let button = TestComponent::ui_component(sig.clone(), pos.clone(), UIComponentType::Button);
    let label = TestComponent::ui_component(sig.clone(), pos.clone(), UIComponentType::Label);
    let panel = TestComponent::ui_component(sig, pos, UIComponentType::Panel);

    // UI elements should be able to connect to each other
    assert!(button.can_connect_to(&label));
    assert!(button.can_connect_to(&panel));
    assert!(label.can_connect_to(&panel));
}

#[test]
fn test_assembly_creation() {
    let mut assembly = TestAssembly::new();

    assert_eq!(assembly.components.len(), 0);
    assert_eq!(assembly.connections.len(), 0);
    assert_relative_eq!(assembly.energy, 0.0);
    assert_relative_eq!(assembly.stability, 0.0);
}

#[test]
fn test_assembly_add_component() {
    let mut assembly = TestAssembly::new();
    let component = TestComponent::new(
        Multivector::basis_vector(0),
        Vector::e1(),
        ComponentType::Basic,
    );

    assembly.add_component(component);

    assert_eq!(assembly.components.len(), 1);
    assert_eq!(assembly.connections.len(), 1);
    assert!(assembly.connections[0].is_empty());
}

#[test]
fn test_assembly_connections() {
    let mut assembly = TestAssembly::new();

    // Add two components
    let comp1 = TestComponent::new(
        Multivector::basis_vector(0),
        Vector::e1(),
        ComponentType::Basic,
    );
    let comp2 = TestComponent::new(
        Multivector::basis_vector(1),
        Vector::e2(),
        ComponentType::Basic,
    );

    assembly.add_component(comp1);
    assembly.add_component(comp2);

    // Connect them
    assembly.connect(0, 1).unwrap();

    assert!(assembly.connections[0].contains(&1));
    assert!(assembly.connections[1].contains(&0));

    // Test invalid connections
    assert!(assembly.connect(0, 5).is_err());
}

#[test]
fn test_assembly_energy_calculation() {
    let mut assembly = TestAssembly::new();

    // Add components at different positions
    let comp1 = TestComponent::new(
        Multivector::scalar(1.0),
        Vector::zero(),
        ComponentType::Basic,
    );
    let comp2 = TestComponent::new(
        Multivector::scalar(1.0),
        Vector::e1(),
        ComponentType::Basic,
    );

    assembly.add_component(comp1);
    assembly.add_component(comp2);
    assembly.connect(0, 1).unwrap();

    assembly.calculate_energy();

    // Energy should be non-zero due to interaction
    assert!(assembly.energy != 0.0);
}

#[test]
fn test_assembly_stability() {
    let mut assembly = TestAssembly::new();

    // Add multiple interconnected components
    for i in 0..3 {
        let component = TestComponent::new(
            Multivector::basis_vector(i % 3),
            Vector::e1() * (i as f64),
            ComponentType::Basic,
        );
        assembly.add_component(component);
    }

    // Connect in a chain
    assembly.connect(0, 1).unwrap();
    assembly.connect(1, 2).unwrap();

    assembly.calculate_energy();
    assembly.calculate_stability();

    assert!(assembly.stability > 0.0);
}

#[test]
fn test_components_by_type() {
    let mut assembly = TestAssembly::new();

    // Add different types of components
    let basic = TestComponent::new(
        Multivector::scalar(1.0),
        Vector::zero(),
        ComponentType::Basic,
    );
    let corner = TestComponent::new(
        Multivector::scalar(1.0),
        Vector::e1(),
        ComponentType::Corner,
    );
    let button = TestComponent::ui_component(
        Multivector::scalar(1.0),
        Vector::e2(),
        UIComponentType::Button,
    );

    assembly.add_component(basic);
    assembly.add_component(corner);
    assembly.add_component(button);

    let basic_components = assembly.components_of_type(&ComponentType::Basic);
    let corner_components = assembly.components_of_type(&ComponentType::Corner);

    assert_eq!(basic_components.len(), 1);
    assert_eq!(corner_components.len(), 1);
    assert_eq!(basic_components[0], 0);
    assert_eq!(corner_components[0], 1);
}

#[test]
fn test_self_assembler_creation() {
    let config = AssemblyConfig::default();
    let assembler = TestAssembler::new(config);

    // Assembler should be properly initialized
    assert_eq!(assembler.config.temperature, 1.0);
    assert_eq!(assembler.config.max_iterations, 1000);
}

#[test]
fn test_assembler_bounds() {
    let config = AssemblyConfig::default();
    let mut assembler = TestAssembler::new(config);

    let min = Vector::zero();
    let max = Vector::e1() + Vector::e2() + Vector::e3();

    assembler.set_bounds(min.clone(), max.clone());

    assert_eq!(assembler.bounds.0.mv, min.mv);
    assert_eq!(assembler.bounds.1.mv, max.mv);
}

#[test]
fn test_affinity_computation() {
    let config = AssemblyConfig::default();
    let assembler = TestAssembler::new(config);

    let comp1 = TestComponent::new(
        Multivector::basis_vector(0),
        Vector::zero(),
        ComponentType::Basic,
    );
    let comp2 = TestComponent::new(
        Multivector::basis_vector(0),
        Vector::e1() * 0.5, // Close position
        ComponentType::Basic,
    );
    let comp3 = TestComponent::new(
        Multivector::basis_vector(1),
        Vector::e1() * 10.0, // Far position
        ComponentType::Basic,
    );

    let affinity_close = assembler.affinity(&comp1, &comp2);
    let affinity_far = assembler.affinity(&comp1, &comp3);

    // Closer components should have higher affinity
    assert!(affinity_close > affinity_far);
}

#[test]
fn test_assembly_process() {
    let config = AssemblyConfig {
        affinity_threshold: 0.01, // Low threshold for testing
        ..Default::default()
    };
    let assembler = TestAssembler::new(config);

    let components = vec![
        TestComponent::new(
            Multivector::scalar(1.0),
            Vector::zero(),
            ComponentType::Basic,
        ),
        TestComponent::new(
            Multivector::scalar(1.0),
            Vector::e1() * 0.1,
            ComponentType::Basic,
        ),
        TestComponent::new(
            Multivector::scalar(1.0),
            Vector::e2() * 0.1,
            ComponentType::Basic,
        ),
    ];

    let assembly = assembler.assemble(&components).unwrap();

    assert_eq!(assembly.components.len(), 3);
    // Should have some connections due to proximity and compatibility
    let total_connections: usize = assembly.connections.iter().map(|c| c.len()).sum();
    assert!(total_connections > 0);
}

#[test]
fn test_assembly_stability_check() {
    let config = AssemblyConfig {
        energy_threshold: 0.0, // Accept any energy for testing
        ..Default::default()
    };
    let assembler = TestAssembler::new(config);

    let components = vec![
        TestComponent::new(
            Multivector::scalar(1.0),
            Vector::zero(),
            ComponentType::Basic,
        ),
    ];

    let assembly = assembler.assemble(&components).unwrap();

    // Single component should be stable
    assert!(assembler.is_stable(&assembly));
}

#[test]
fn test_ui_specific_assembly() {
    let config = AssemblyConfig::default();
    let assembler = TestAssembler::new(config);

    let ui_components = vec![
        TestComponent::ui_component(
            Multivector::scalar(1.0),
            Vector::zero(),
            UIComponentType::Button,
        ),
        TestComponent::ui_component(
            Multivector::scalar(1.0),
            Vector::e1(),
            UIComponentType::Label,
        ),
        TestComponent::ui_component(
            Multivector::scalar(1.0),
            Vector::e2(),
            UIComponentType::Panel,
        ),
    ];

    let assembly = assembler.assemble(&ui_components).unwrap();

    assert_eq!(assembly.components.len(), 3);

    // UI components should be able to connect
    let ui_count = assembly
        .components
        .iter()
        .filter(|c| matches!(c.component_type, ComponentType::UIElement(_)))
        .count();
    assert_eq!(ui_count, 3);
}

#[test]
fn test_assembly_config_values() {
    let config = AssemblyConfig {
        temperature: 2.0,
        max_iterations: 500,
        energy_threshold: -2.0,
        affinity_threshold: 0.05,
    };

    assert_relative_eq!(config.temperature, 2.0);
    assert_eq!(config.max_iterations, 500);
    assert_relative_eq!(config.energy_threshold, -2.0);
    assert_relative_eq!(config.affinity_threshold, 0.05);
}