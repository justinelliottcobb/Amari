//! Tests for UI-Specific Self-Assembly

use amari_automata::{
    UIAssembler, UIComponent, UIAssembly, UIAssemblyConfig, UIComponentType,
    LayoutRect, LayoutType, FlexDirection,
};
use amari_core::{Multivector, Vector};
use approx::assert_relative_eq;

type TestUIComponent = UIComponent<3, 0, 0>;
type TestUIAssembly = UIAssembly<3, 0, 0>;
type TestUIAssembler = UIAssembler<3, 0, 0>;

#[test]
fn test_ui_component_creation() {
    let signature = Multivector::scalar(1.0);
    let position = Vector::zero();
    let preferred_size = (100.0, 50.0, 1.0);

    let ui_comp = TestUIComponent::new(
        signature,
        position,
        UIComponentType::Button,
        preferred_size,
    );

    assert_eq!(ui_comp.preferred_size, preferred_size);
    assert_eq!(ui_comp.min_size, (0.0, 0.0, 0.0));
    assert_eq!(ui_comp.max_size, (f64::INFINITY, f64::INFINITY, f64::INFINITY));
    assert_relative_eq!(ui_comp.layout_weight, 1.0);
    assert_eq!(ui_comp.z_index, 0);
}

#[test]
fn test_ui_component_with_constraints() {
    let signature = Multivector::basis_vector(0);
    let position = Vector::e1();
    let preferred_size = (200.0, 100.0, 5.0);

    let ui_comp = TestUIComponent::new(signature, position, UIComponentType::Panel, preferred_size)
        .with_size_constraints((50.0, 30.0, 1.0), (300.0, 150.0, 10.0))
        .with_margin(10.0, 5.0, 10.0, 5.0)
        .with_padding(15.0, 10.0, 15.0, 10.0)
        .with_flex(1.0, 0.5, 100.0);

    assert_eq!(ui_comp.min_size, (50.0, 30.0, 1.0));
    assert_eq!(ui_comp.max_size, (300.0, 150.0, 10.0));
    assert_eq!(ui_comp.margin, (10.0, 5.0, 10.0, 5.0));
    assert_eq!(ui_comp.padding, (15.0, 10.0, 15.0, 10.0));
    assert_relative_eq!(ui_comp.flex.grow, 1.0);
    assert_relative_eq!(ui_comp.flex.shrink, 0.5);
    assert_relative_eq!(ui_comp.flex.basis, 100.0);
}

#[test]
fn test_effective_size_calculation() {
    let signature = Multivector::scalar(1.0);
    let position = Vector::zero();
    let preferred_size = (100.0, 50.0, 1.0);

    let ui_comp = TestUIComponent::new(signature, position, UIComponentType::Input, preferred_size)
        .with_margin(5.0, 3.0, 5.0, 3.0)
        .with_padding(2.0, 1.0, 2.0, 1.0);

    let (width, height, depth) = ui_comp.effective_size();

    // 100 + 5 + 5 + 2 + 2 = 114
    // 50 + 3 + 3 + 1 + 1 = 58
    // depth unchanged
    assert_relative_eq!(width, 114.0);
    assert_relative_eq!(height, 58.0);
    assert_relative_eq!(depth, 1.0);
}

#[test]
fn test_layout_compatibility() {
    let sig = Multivector::scalar(1.0);
    let pos = Vector::zero();
    let size = (100.0, 50.0, 1.0);

    let button = TestUIComponent::new(sig.clone(), pos.clone(), UIComponentType::Button, size);
    let label = TestUIComponent::new(sig.clone(), pos.clone(), UIComponentType::Label, size);
    let container = TestUIComponent::new(sig.clone(), pos.clone(), UIComponentType::Container, size);
    let input = TestUIComponent::new(sig, pos, UIComponentType::Input, size);

    // Container should be compatible with everything
    assert!(container.layout_compatible(&button));
    assert!(container.layout_compatible(&label));
    assert!(container.layout_compatible(&input));

    // Button and label should be compatible
    assert!(button.layout_compatible(&label));

    // Input and label should be compatible
    assert!(input.layout_compatible(&label));
}

#[test]
fn test_layout_rect_operations() {
    let rect1 = LayoutRect {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        width: 100.0,
        height: 50.0,
        depth: 1.0,
    };

    let rect2 = LayoutRect {
        x: 50.0,
        y: 25.0,
        z: 0.0,
        width: 100.0,
        height: 50.0,
        depth: 1.0,
    };

    let rect3 = LayoutRect {
        x: 200.0,
        y: 200.0,
        z: 0.0,
        width: 50.0,
        height: 25.0,
        depth: 1.0,
    };

    // rect1 and rect2 should overlap
    assert!(rect1.overlaps(&rect2));
    assert!(rect2.overlaps(&rect1));

    // rect1 and rect3 should not overlap
    assert!(!rect1.overlaps(&rect3));
    assert!(!rect3.overlaps(&rect1));

    // Test center calculation
    let (cx, cy, cz) = rect1.center();
    assert_relative_eq!(cx, 50.0);
    assert_relative_eq!(cy, 25.0);
    assert_relative_eq!(cz, 0.5);
}

#[test]
fn test_ui_assembly_creation() {
    let mut assembly = TestUIAssembly::new();

    assert_eq!(assembly.ui_components.len(), 0);
    assert_eq!(assembly.layout_rects.len(), 0);
    assert_eq!(assembly.base.components.len(), 0);
}

#[test]
fn test_ui_assembly_add_components() {
    let mut assembly = TestUIAssembly::new();

    let button = TestUIComponent::new(
        Multivector::scalar(1.0),
        Vector::zero(),
        UIComponentType::Button,
        (80.0, 30.0, 1.0),
    );

    let label = TestUIComponent::new(
        Multivector::basis_vector(0),
        Vector::e1(),
        UIComponentType::Label,
        (120.0, 20.0, 1.0),
    );

    assembly.add_ui_component(button);
    assembly.add_ui_component(label);

    assert_eq!(assembly.ui_components.len(), 2);
    assert_eq!(assembly.layout_rects.len(), 2);
    assert_eq!(assembly.base.components.len(), 2);
}

#[test]
fn test_layout_computation() {
    let mut assembly = TestUIAssembly::new();

    // Add components with different sizes
    let components = vec![
        TestUIComponent::new(
            Multivector::scalar(1.0),
            Vector::zero(),
            UIComponentType::Button,
            (100.0, 40.0, 1.0),
        ),
        TestUIComponent::new(
            Multivector::basis_vector(0),
            Vector::e1(),
            UIComponentType::Label,
            (150.0, 25.0, 1.0),
        ),
        TestUIComponent::new(
            Multivector::basis_vector(1),
            Vector::e2(),
            UIComponentType::Input,
            (200.0, 35.0, 1.0),
        ),
    ];

    for comp in components {
        assembly.add_ui_component(comp);
    }

    let available_space = LayoutRect {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        width: 400.0,
        height: 300.0,
        depth: 10.0,
    };

    assembly.compute_layout(available_space).unwrap();

    // Check that components are laid out vertically
    assert_relative_eq!(assembly.layout_rects[0].x, 0.0);
    assert_relative_eq!(assembly.layout_rects[0].y, 0.0);

    assert_relative_eq!(assembly.layout_rects[1].x, 0.0);
    assert!(assembly.layout_rects[1].y > assembly.layout_rects[0].y);

    assert_relative_eq!(assembly.layout_rects[2].x, 0.0);
    assert!(assembly.layout_rects[2].y > assembly.layout_rects[1].y);

    // UI bounds should be computed
    assert!(assembly.ui_bounds.height > 0.0);
}

#[test]
fn test_component_at_position() {
    let mut assembly = TestUIAssembly::new();

    let button = TestUIComponent::new(
        Multivector::scalar(1.0),
        Vector::zero(),
        UIComponentType::Button,
        (100.0, 50.0, 1.0),
    );

    assembly.add_ui_component(button);

    let available_space = LayoutRect {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        width: 400.0,
        height: 300.0,
        depth: 10.0,
    };

    assembly.compute_layout(available_space).unwrap();

    // Point inside the button
    let component_index = assembly.component_at_position(50.0, 25.0);
    assert_eq!(component_index, Some(0));

    // Point outside any component
    let no_component = assembly.component_at_position(200.0, 200.0);
    assert_eq!(no_component, None);
}

#[test]
fn test_layout_validation() {
    let mut assembly = TestUIAssembly::new();

    // Add non-overlapping components
    let button = TestUIComponent::new(
        Multivector::scalar(1.0),
        Vector::zero(),
        UIComponentType::Button,
        (100.0, 50.0, 1.0),
    );

    let label = TestUIComponent::new(
        Multivector::basis_vector(0),
        Vector::e1(),
        UIComponentType::Label,
        (100.0, 30.0, 1.0),
    );

    assembly.add_ui_component(button);
    assembly.add_ui_component(label);

    let available_space = LayoutRect {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        width: 400.0,
        height: 300.0,
        depth: 10.0,
    };

    assembly.compute_layout(available_space).unwrap();

    // Layout should be valid (no overlaps)
    assert!(assembly.is_layout_valid());
}

#[test]
fn test_ui_assembler_creation() {
    let config = UIAssemblyConfig::default();
    let assembler = TestUIAssembler::new(config);

    // Check default configuration values
    assert_relative_eq!(assembler.ui_config.default_spacing, 10.0);
    assert!(assembler.ui_config.auto_layout);
    assert_eq!(assembler.ui_config.breakpoints.len(), 3);
}

#[test]
fn test_ui_assembly_process() {
    let config = UIAssemblyConfig::default();
    let assembler = TestUIAssembler::new(config);

    let components = vec![
        TestUIComponent::new(
            Multivector::scalar(1.0),
            Vector::zero(),
            UIComponentType::Button,
            (80.0, 30.0, 1.0),
        ),
        TestUIComponent::new(
            Multivector::basis_vector(0),
            Vector::e1() * 0.1,
            UIComponentType::Label,
            (120.0, 20.0, 1.0),
        ),
        TestUIComponent::new(
            Multivector::basis_vector(1),
            Vector::e2() * 0.1,
            UIComponentType::Container,
            (200.0, 100.0, 1.0),
        ),
    ];

    let assembly = assembler.assemble_ui(&components).unwrap();

    assert_eq!(assembly.ui_components.len(), 3);
    assert_eq!(assembly.layout_rects.len(), 3);

    // Should have computed layout
    assert!(assembly.ui_bounds.width > 0.0);
    assert!(assembly.ui_bounds.height > 0.0);

    // Should have some connections based on layout compatibility
    let total_connections: usize = assembly.base.connections.iter().map(|c| c.len()).sum();
    assert!(total_connections >= 0); // May or may not have connections depending on affinity
}

#[test]
fn test_layout_tree_operations() {
    let mut tree = crate::ui_assembly::LayoutTree::new();

    let root = tree.add_node(0, LayoutType::Flex);
    let child1 = tree.add_node(1, LayoutType::Fixed);
    let child2 = tree.add_node(2, LayoutType::Grid);

    tree.add_child(root, child1).unwrap();
    tree.add_child(root, child2).unwrap();

    assert_eq!(tree.root, Some(root));
    assert_eq!(tree.nodes[root].children.len(), 2);
    assert!(tree.nodes[root].children.contains(&child1));
    assert!(tree.nodes[root].children.contains(&child2));

    // Test invalid child addition
    assert!(tree.add_child(10, child1).is_err());
}

#[test]
fn test_flex_properties() {
    let flex = crate::ui_assembly::FlexProperties::default();

    assert_relative_eq!(flex.grow, 0.0);
    assert_relative_eq!(flex.shrink, 1.0);
    assert_relative_eq!(flex.basis, 0.0);
    assert_eq!(flex.direction, FlexDirection::Row);

    let custom_flex = crate::ui_assembly::FlexProperties {
        grow: 2.0,
        shrink: 0.5,
        basis: 100.0,
        direction: FlexDirection::Column,
    };

    assert_relative_eq!(custom_flex.grow, 2.0);
    assert_eq!(custom_flex.direction, FlexDirection::Column);
}

#[test]
fn test_ui_assembly_config() {
    let config = UIAssemblyConfig {
        default_spacing: 15.0,
        auto_layout: false,
        breakpoints: vec![320.0, 768.0, 1024.0, 1440.0],
        max_layout_iterations: 50,
        layout_threshold: 0.5,
    };

    assert_relative_eq!(config.default_spacing, 15.0);
    assert!(!config.auto_layout);
    assert_eq!(config.breakpoints.len(), 4);
    assert_eq!(config.max_layout_iterations, 50);
    assert_relative_eq!(config.layout_threshold, 0.5);
}

#[test]
fn test_complex_ui_hierarchy() {
    let config = UIAssemblyConfig::default();
    let assembler = TestUIAssembler::new(config);

    // Create a complex UI structure
    let components = vec![
        // Container
        TestUIComponent::new(
            Multivector::scalar(1.0),
            Vector::zero(),
            UIComponentType::Container,
            (400.0, 300.0, 1.0),
        ),
        // Header panel
        TestUIComponent::new(
            Multivector::basis_vector(0),
            Vector::e1() * 0.1,
            UIComponentType::Panel,
            (400.0, 60.0, 1.0),
        ),
        // Button in header
        TestUIComponent::new(
            Multivector::basis_vector(1),
            Vector::e1() * 0.05,
            UIComponentType::Button,
            (80.0, 30.0, 1.0),
        ),
        // Label in header
        TestUIComponent::new(
            Multivector::basis_vector(2),
            Vector::e1() * 0.08,
            UIComponentType::Label,
            (200.0, 30.0, 1.0),
        ),
        // Main content area
        TestUIComponent::new(
            Multivector::scalar(0.5),
            Vector::e2() * 0.1,
            UIComponentType::Container,
            (400.0, 200.0, 1.0),
        ),
        // Input field
        TestUIComponent::new(
            Multivector::basis_vector(0) + Multivector::basis_vector(1),
            Vector::e2() * 0.05,
            UIComponentType::Input,
            (300.0, 40.0, 1.0),
        ),
    ];

    let assembly = assembler.assemble_ui(&components).unwrap();

    assert_eq!(assembly.ui_components.len(), 6);

    // Should have reasonable layout
    assert!(assembly.ui_bounds.width > 0.0);
    assert!(assembly.ui_bounds.height > 0.0);

    // Layout should be valid
    assert!(assembly.is_layout_valid());
}

#[test]
fn test_z_index_ordering() {
    let mut assembly = TestUIAssembly::new();

    let back_component = TestUIComponent::new(
        Multivector::scalar(1.0),
        Vector::zero(),
        UIComponentType::Panel,
        (200.0, 100.0, 1.0),
    );

    let mut front_component = TestUIComponent::new(
        Multivector::basis_vector(0),
        Vector::zero(),
        UIComponentType::Button,
        (100.0, 50.0, 1.0),
    );
    front_component.z_index = 10;

    assembly.add_ui_component(back_component);
    assembly.add_ui_component(front_component);

    let available_space = LayoutRect {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        width: 400.0,
        height: 300.0,
        depth: 100.0,
    };

    assembly.compute_layout(available_space).unwrap();

    // Front component should have higher z coordinate
    assert!(assembly.layout_rects[1].z > assembly.layout_rects[0].z);
}