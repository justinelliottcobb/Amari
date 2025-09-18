//! Comprehensive Tests for UI Assembly

use amari_automata::{UIComponent, UIAssembler, LayoutConstraint, Layout};
use amari_core::Multivector;
use approx::assert_relative_eq;

#[test]
fn test_ui_components_as_polyominoes() {
    let button = UIComponent::button(2, 1);
    let panel = UIComponent::panel(3, 3);

    let button_mv = button.to_multivector();
    let panel_mv = panel.to_multivector();

    let affinity = button_mv.inner_product(&panel_mv);
    assert!(affinity > 0.0);
}

#[test]
fn test_ui_self_assembly_from_constraints() {
    let constraints = vec![
        LayoutConstraint::MustInclude(UIComponent::navigation()),
        LayoutConstraint::MustInclude(UIComponent::content()),
        LayoutConstraint::MaxHeight(800),
        LayoutConstraint::PreferHorizontal,
    ];

    let assembler = UIAssembler::new();
    let layout = assembler.assemble_from_constraints(&constraints);

    assert!(layout.contains(&UIComponent::navigation()));
    assert!(layout.height() <= 800);
    assert!(layout.is_primarily_horizontal());
}

#[test]
fn test_responsive_ui_phase_transitions() {
    let mut assembler = UIAssembler::new();

    // Mobile phase
    assembler.set_viewport(375, 667);
    let mobile = assembler.assemble();

    // Desktop phase
    assembler.set_viewport(1920, 1080);
    let desktop = assembler.assemble();

    // Fundamentally different layouts
    assert_ne!(mobile.topology(), desktop.topology());
}

#[test]
fn test_ui_evolution_toward_target() {
    let target = Layout::optimal_dashboard();
    let mut ui = UIAssembler::with_evolution();
    ui.set_target(&target);

    let initial_fitness = ui.fitness();
    for _ in 0..100 {
        ui.evolution_step();
    }

    assert!(ui.fitness() > initial_fitness * 2.0);
    assert!(ui.similarity_to(&target) > 0.9);
}

#[test]
fn test_geometric_ui_composition() {
    // UI components compose via geometric algebra
    let header = UIComponent::header().with_signature(Multivector::e1());
    let sidebar = UIComponent::sidebar().with_signature(Multivector::e2());

    let composition = header.geometric_compose(&sidebar);

    // Should produce a bivector signature (e1 ∧ e2)
    assert!(composition.signature().is_bivector());
    assert_relative_eq!(composition.signature().magnitude(), 1.0, epsilon = 1e-10);
}

#[test]
fn test_adaptive_layout_constraints() {
    // UI adapts to changing constraints
    let mut assembler = UIAssembler::adaptive();

    // Initial constraints
    assembler.add_constraint(LayoutConstraint::accessibility_aa());
    assembler.add_constraint(LayoutConstraint::min_touch_target(44));

    let initial_layout = assembler.assemble();

    // Add new constraint
    assembler.add_constraint(LayoutConstraint::dark_mode_support());

    let adapted_layout = assembler.assemble();

    assert!(adapted_layout.supports_dark_mode());
    assert!(adapted_layout.maintains_accessibility());
}

#[test]
fn test_ui_component_affinities() {
    // Components have geometric affinities
    let search_box = UIComponent::search().with_position(Multivector::e1());
    let results_list = UIComponent::list().with_position(Multivector::e2());
    let pagination = UIComponent::pagination().with_position(Multivector::e3());

    // Search box and results should have high affinity
    let search_results_affinity = search_box.affinity(&results_list);
    let search_pagination_affinity = search_box.affinity(&pagination);

    assert!(search_results_affinity > search_pagination_affinity);
}

#[test]
fn test_layout_topology_invariants() {
    // UI layouts preserve topological properties
    let components = vec![
        UIComponent::navigation(),
        UIComponent::main_content(),
        UIComponent::sidebar(),
        UIComponent::footer(),
    ];

    let assembler = UIAssembler::topology_preserving();
    let layout = assembler.assemble_with_topology(&components);

    // Navigation should be topologically connected to all others
    assert!(layout.is_connected(&UIComponent::navigation(), &UIComponent::main_content()));
    assert!(layout.is_connected(&UIComponent::navigation(), &UIComponent::sidebar()));
    assert!(layout.is_connected(&UIComponent::navigation(), &UIComponent::footer()));
}

#[test]
fn test_ui_symmetry_groups() {
    // UI layouts respect symmetry groups
    let grid_layout = UIAssembler::grid_layout(3, 3);
    let components = vec![UIComponent::card(); 9];

    let assembled = grid_layout.assemble(&components);

    // Should have D4 symmetry (4-fold rotation + reflection)
    let symmetries = assembled.symmetry_group();
    assert_eq!(symmetries.order(), 8); // |D4| = 8
}

#[test]
fn test_information_density_optimization() {
    // Optimize information density using geometric measures
    let high_density_target = 0.8;
    let components = vec![
        UIComponent::data_table(10, 5),
        UIComponent::chart("line"),
        UIComponent::summary_stats(),
        UIComponent::filters(),
    ];

    let assembler = UIAssembler::with_density_optimization(high_density_target);
    let layout = assembler.assemble(&components);

    assert!(layout.information_density() >= high_density_target);
    assert!(layout.is_readable());
}

#[test]
fn test_ui_flow_fields() {
    // UI layouts create flow fields for user navigation
    let components = vec![
        UIComponent::landing_hero(),
        UIComponent::feature_cards(),
        UIComponent::call_to_action(),
        UIComponent::testimonials(),
    ];

    let assembler = UIAssembler::with_flow_optimization();
    let layout = assembler.assemble(&components);

    let flow_field = layout.compute_flow_field();

    // Flow should guide users from hero to CTA
    let hero_to_cta_flow = flow_field.flow_strength(
        &UIComponent::landing_hero(),
        &UIComponent::call_to_action()
    );

    assert!(hero_to_cta_flow > 0.7);
}

#[test]
fn test_multi_device_consistency() {
    // UI maintains consistency across devices
    let components = vec![
        UIComponent::profile_header(),
        UIComponent::action_buttons(),
        UIComponent::content_feed(),
    ];

    let phone_assembler = UIAssembler::for_phone();
    let tablet_assembler = UIAssembler::for_tablet();
    let desktop_assembler = UIAssembler::for_desktop();

    let phone_layout = phone_assembler.assemble(&components);
    let tablet_layout = tablet_assembler.assemble(&components);
    let desktop_layout = desktop_assembler.assemble(&components);

    // Layouts should be topologically equivalent
    assert!(phone_layout.topologically_equivalent(&tablet_layout));
    assert!(tablet_layout.topologically_equivalent(&desktop_layout));
}

#[test]
fn test_accessibility_optimization() {
    // UI automatically optimizes for accessibility
    let components = vec![
        UIComponent::form_with_inputs(),
        UIComponent::submit_button(),
        UIComponent::error_messages(),
    ];

    let assembler = UIAssembler::with_accessibility();
    let layout = assembler.assemble(&components);

    assert!(layout.meets_wcag_aa());
    assert!(layout.has_proper_focus_order());
    assert!(layout.has_semantic_structure());
}

#[test]
fn test_cultural_adaptation() {
    // UI adapts to cultural reading patterns
    let components = vec![
        UIComponent::article_title(),
        UIComponent::article_body(),
        UIComponent::author_info(),
        UIComponent::related_articles(),
    ];

    let ltr_assembler = UIAssembler::for_culture("en-US");
    let rtl_assembler = UIAssembler::for_culture("ar-SA");
    let vertical_assembler = UIAssembler::for_culture("ja-JP");

    let ltr_layout = ltr_assembler.assemble(&components);
    let rtl_layout = rtl_assembler.assemble(&components);
    let vertical_layout = vertical_assembler.assemble(&components);

    assert!(ltr_layout.reading_direction() == ReadingDirection::LeftToRight);
    assert!(rtl_layout.reading_direction() == ReadingDirection::RightToLeft);
    assert!(vertical_layout.reading_direction() == ReadingDirection::TopToBottom);
}

#[test]
fn test_ui_emergence() {
    // Complex UI behaviors emerge from simple rules
    let simple_components = vec![
        UIComponent::text_block(),
        UIComponent::clickable_area(),
        UIComponent::visual_separator(),
    ];

    let assembler = UIAssembler::emergent();
    assembler.set_simple_rules(vec![
        EmergenceRule::proximity_groups(),
        EmergenceRule::similar_elements_align(),
        EmergenceRule::contrast_creates_hierarchy(),
    ]);

    let layout = assembler.assemble(&simple_components);

    // Should exhibit complex behaviors
    assert!(layout.has_emergent_navigation());
    assert!(layout.has_visual_hierarchy());
    assert!(layout.has_grouping_patterns());
}

#[test]
fn test_ui_performance_constraints() {
    // UI respects performance constraints
    let heavy_components = vec![
        UIComponent::high_res_image_gallery(),
        UIComponent::data_visualization(),
        UIComponent::real_time_updates(),
    ];

    let assembler = UIAssembler::performance_aware();
    assembler.set_performance_budget(PerformanceBudget {
        max_render_time: 16, // 60fps
        max_memory_mb: 100,
        max_network_requests: 10,
    });

    let layout = assembler.assemble(&heavy_components);

    assert!(layout.meets_performance_budget());
    assert!(layout.uses_lazy_loading());
    assert!(layout.optimizes_critical_path());
}

#[test]
fn test_ui_self_repair() {
    // UI can repair itself when components fail
    let components = vec![
        UIComponent::primary_navigation(),
        UIComponent::search_functionality(),
        UIComponent::user_account_menu(),
    ];

    let mut assembler = UIAssembler::self_repairing();
    let mut layout = assembler.assemble(&components);

    // Simulate component failure
    layout.fail_component(&UIComponent::search_functionality());

    // UI should automatically repair
    layout.self_repair();

    assert!(layout.is_functional());
    assert!(layout.has_fallback_navigation());
}

// Helper enums and structs for tests
#[derive(PartialEq)]
enum ReadingDirection {
    LeftToRight,
    RightToLeft,
    TopToBottom,
}

struct PerformanceBudget {
    max_render_time: u32,
    max_memory_mb: u32,
    max_network_requests: u32,
}

struct EmergenceRule {
    name: String,
}

impl EmergenceRule {
    fn proximity_groups() -> Self {
        Self { name: "proximity_groups".to_string() }
    }

    fn similar_elements_align() -> Self {
        Self { name: "similar_elements_align".to_string() }
    }

    fn contrast_creates_hierarchy() -> Self {
        Self { name: "contrast_creates_hierarchy".to_string() }
    }
}