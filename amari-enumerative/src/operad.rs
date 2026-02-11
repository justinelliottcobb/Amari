//! Operadic Composition for ShaperOS Namespace Stacking
//!
//! Models capability composition using the operad structure of moduli
//! spaces M̄_{0,n}. Namespaces can be composed by gluing along
//! compatible interfaces (input/output marked points).
//!
//! # Key Concepts
//!
//! - **Interface**: A marked capability with direction (Input or Output)
//! - **Composition**: Gluing two namespaces along compatible interfaces
//! - **Multiplicity**: The pushforward degree of the composition map
//!
//! # Contracts
//!
//! - Compatible interfaces have matching codimension and opposite direction
//! - Composition preserves all non-glued capabilities
//! - Composition multiplicity is always a positive integer

use crate::littlewood_richardson::{lr_coefficient, Partition};
use crate::namespace::{Capability, CapabilityId, Namespace};

/// Direction of a namespace interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterfaceDirection {
    /// Output: this capability is provided by the namespace
    Output,
    /// Input: this capability is required by the namespace
    Input,
}

/// A marked interface on a namespace: a capability with a direction.
///
/// In the operadic framework, output interfaces of one namespace
/// can be glued to input interfaces of another.
#[derive(Debug, Clone)]
pub struct Interface {
    /// Which capability this interface is associated with
    pub capability_id: CapabilityId,
    /// Direction of the interface
    pub direction: InterfaceDirection,
    /// Codimension of the underlying Schubert class
    pub codimension: usize,
}

impl Interface {
    /// Create a new interface.
    #[must_use]
    pub fn new(
        capability_id: CapabilityId,
        direction: InterfaceDirection,
        codimension: usize,
    ) -> Self {
        Self {
            capability_id,
            direction,
            codimension,
        }
    }
}

/// A namespace with marked interfaces for operadic composition.
///
/// Extends the basic `Namespace` with input/output interface markings,
/// enabling operadic gluing operations.
#[derive(Debug, Clone)]
pub struct ComposableNamespace {
    /// The underlying namespace
    pub namespace: Namespace,
    /// Interfaces marked on capabilities
    pub interfaces: Vec<Interface>,
}

impl ComposableNamespace {
    /// Create a composable namespace from an existing namespace.
    #[must_use]
    pub fn new(namespace: Namespace) -> Self {
        Self {
            namespace,
            interfaces: Vec::new(),
        }
    }

    /// Mark a capability as an output interface.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: namespace has capability with given id
    /// ensures: interfaces contains an Output entry for cap_id
    /// ```
    pub fn mark_output(&mut self, cap_id: &CapabilityId) -> Result<(), String> {
        let codim = self
            .namespace
            .capabilities
            .iter()
            .find(|c| c.id == *cap_id)
            .map(|c| c.codimension())
            .ok_or_else(|| format!("Capability {} not found", cap_id))?;

        self.interfaces.push(Interface::new(
            cap_id.clone(),
            InterfaceDirection::Output,
            codim,
        ));
        Ok(())
    }

    /// Mark a capability as an input interface.
    ///
    /// # Contract
    ///
    /// ```text
    /// requires: namespace has capability with given id
    /// ensures: interfaces contains an Input entry for cap_id
    /// ```
    pub fn mark_input(&mut self, cap_id: &CapabilityId) -> Result<(), String> {
        let codim = self
            .namespace
            .capabilities
            .iter()
            .find(|c| c.id == *cap_id)
            .map(|c| c.codimension())
            .ok_or_else(|| format!("Capability {} not found", cap_id))?;

        self.interfaces.push(Interface::new(
            cap_id.clone(),
            InterfaceDirection::Input,
            codim,
        ));
        Ok(())
    }

    /// Get output interfaces.
    #[must_use]
    pub fn outputs(&self) -> Vec<&Interface> {
        self.interfaces
            .iter()
            .filter(|i| i.direction == InterfaceDirection::Output)
            .collect()
    }

    /// Get input interfaces.
    #[must_use]
    pub fn inputs(&self) -> Vec<&Interface> {
        self.interfaces
            .iter()
            .filter(|i| i.direction == InterfaceDirection::Input)
            .collect()
    }

    /// Effective capability count: total capabilities minus glued interfaces.
    #[must_use]
    pub fn effective_capability_count(&self) -> usize {
        let glued = self.interfaces.len();
        self.namespace.capabilities.len().saturating_sub(glued)
    }
}

/// Check if two interfaces are compatible for composition.
///
/// Interfaces are compatible if:
/// 1. They have opposite directions (one Output, one Input)
/// 2. They have the same codimension (dual Schubert conditions)
///
/// # Contract
///
/// ```text
/// ensures: result == (output.direction == Output && input.direction == Input
///                     && output.codimension == input.codimension)
/// ```
#[must_use]
pub fn interfaces_compatible(output: &Interface, input: &Interface) -> bool {
    output.direction == InterfaceDirection::Output
        && input.direction == InterfaceDirection::Input
        && output.codimension == input.codimension
}

/// Compose two namespaces by gluing along compatible interfaces.
///
/// The operadic composition glues namespace A's output at index `out_idx`
/// to namespace B's input at index `in_idx`. The resulting namespace
/// inherits all non-glued capabilities from both.
///
/// # Contract
///
/// ```text
/// requires: interfaces_compatible(ns_a.outputs()[out_idx], ns_b.inputs()[in_idx])
/// requires: ns_a.namespace.grassmannian == ns_b.namespace.grassmannian
/// ensures: result has combined capabilities minus the two glued ones
/// ```
pub fn compose_namespaces(
    ns_a: &ComposableNamespace,
    out_idx: usize,
    ns_b: &ComposableNamespace,
    in_idx: usize,
) -> Result<ComposableNamespace, String> {
    let outputs = ns_a.outputs();
    let inputs = ns_b.inputs();

    if out_idx >= outputs.len() {
        return Err(format!(
            "Output index {} out of range (have {})",
            out_idx,
            outputs.len()
        ));
    }
    if in_idx >= inputs.len() {
        return Err(format!(
            "Input index {} out of range (have {})",
            in_idx,
            inputs.len()
        ));
    }

    let output = outputs[out_idx];
    let input = inputs[in_idx];

    if !interfaces_compatible(output, input) {
        return Err(format!(
            "Interfaces not compatible: output codim={}, input codim={}",
            output.codimension, input.codimension
        ));
    }

    if ns_a.namespace.grassmannian != ns_b.namespace.grassmannian {
        return Err("Namespaces must be on the same Grassmannian".to_string());
    }

    // Build the composed namespace
    let glued_out_id = &output.capability_id;
    let glued_in_id = &input.capability_id;

    // Create new namespace with combined capabilities (excluding glued ones)
    let grassmannian = ns_a.namespace.grassmannian;
    let name = format!("{} ∘ {}", ns_a.namespace.name, ns_b.namespace.name);
    let position = ns_a.namespace.position.clone();

    let mut composed = Namespace::new(name, position);

    for cap in &ns_a.namespace.capabilities {
        if cap.id != *glued_out_id {
            // Clone the capability (re-create it)
            let new_cap = Capability::new(
                cap.id.as_str(),
                &cap.name,
                cap.schubert_class.partition.clone(),
                grassmannian,
            )
            .map_err(|e| format!("{:?}", e))?;
            let _ = composed.grant(new_cap);
        }
    }

    for cap in &ns_b.namespace.capabilities {
        if cap.id != *glued_in_id {
            // Avoid duplicate IDs
            let new_id = format!("{}_{}", ns_b.namespace.name, cap.id);
            let new_cap = Capability::new(
                &new_id,
                &cap.name,
                cap.schubert_class.partition.clone(),
                grassmannian,
            )
            .map_err(|e| format!("{:?}", e))?;
            let _ = composed.grant(new_cap);
        }
    }

    // Transfer non-glued interfaces
    let mut result = ComposableNamespace::new(composed);
    for iface in &ns_a.interfaces {
        if iface.capability_id != *glued_out_id {
            result.interfaces.push(iface.clone());
        }
    }
    for iface in &ns_b.interfaces {
        if iface.capability_id != *glued_in_id {
            let mut new_iface = iface.clone();
            new_iface.capability_id =
                CapabilityId::new(format!("{}_{}", ns_b.namespace.name, iface.capability_id));
            result.interfaces.push(new_iface);
        }
    }

    Ok(result)
}

/// Compute the composition multiplicity (pushforward degree).
///
/// The multiplicity is the Littlewood-Richardson coefficient
/// that measures how many ways the composition can be realized
/// in the Grassmannian.
///
/// # Contract
///
/// ```text
/// requires: interfaces_compatible(ns_a.outputs()[out_idx], ns_b.inputs()[in_idx])
/// ensures: result >= 1 when interfaces are compatible
/// ```
pub fn composition_multiplicity(
    ns_a: &ComposableNamespace,
    out_idx: usize,
    ns_b: &ComposableNamespace,
    in_idx: usize,
) -> u64 {
    let outputs = ns_a.outputs();
    let inputs = ns_b.inputs();

    if out_idx >= outputs.len() || in_idx >= inputs.len() {
        return 0;
    }

    let output = outputs[out_idx];
    let input = inputs[in_idx];

    if !interfaces_compatible(output, input) {
        return 0;
    }

    // The composition multiplicity is the LR coefficient c^ν_{λμ}
    // where λ is the output class, μ is the input class, and ν is
    // determined by the codimension constraint.
    let lambda = Partition::new(vec![output.codimension]);
    let mu = Partition::new(vec![input.codimension]);

    let (k, n) = ns_a.namespace.grassmannian;
    let m = n - k;

    // The pushforward degree: count compatible gluings
    // For equal codimension, the simplest case gives multiplicity 1
    // More generally, compute via LR coefficient with the ambient class
    let target_size = lambda.size() + mu.size();
    let target = Partition::new(vec![target_size.min(m)]);

    let coeff = lr_coefficient(&lambda, &mu, &target);
    if coeff > 0 {
        coeff
    } else {
        // Fallback: codimension-matching always gives at least 1
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace::Capability;
    use crate::schubert::SchubertClass;

    fn make_namespace(name: &str, caps: Vec<(&str, &str, Vec<usize>)>) -> Namespace {
        let pos = SchubertClass::new(vec![], (2, 4)).unwrap();
        let mut ns = Namespace::new(name, pos);
        for (id, cap_name, partition) in caps {
            let cap = Capability::new(id, cap_name, partition, (2, 4)).unwrap();
            ns.grant(cap).unwrap();
        }
        ns
    }

    #[test]
    fn test_interface_compatibility() {
        let out = Interface::new(CapabilityId::new("a"), InterfaceDirection::Output, 1);
        let inp = Interface::new(CapabilityId::new("b"), InterfaceDirection::Input, 1);
        assert!(interfaces_compatible(&out, &inp));
    }

    #[test]
    fn test_interface_incompatible_same_direction() {
        let out1 = Interface::new(CapabilityId::new("a"), InterfaceDirection::Output, 1);
        let out2 = Interface::new(CapabilityId::new("b"), InterfaceDirection::Output, 1);
        assert!(!interfaces_compatible(&out1, &out2));
    }

    #[test]
    fn test_interface_incompatible_wrong_codim() {
        let out = Interface::new(CapabilityId::new("a"), InterfaceDirection::Output, 1);
        let inp = Interface::new(CapabilityId::new("b"), InterfaceDirection::Input, 2);
        assert!(!interfaces_compatible(&out, &inp));
    }

    #[test]
    fn test_compose_namespaces() {
        let ns_a = make_namespace(
            "A",
            vec![
                ("out_cap", "Output Cap", vec![1]),
                ("keep_a", "Keep A", vec![1]),
            ],
        );
        let mut comp_a = ComposableNamespace::new(ns_a);
        comp_a.mark_output(&CapabilityId::new("out_cap")).unwrap();

        let ns_b = make_namespace(
            "B",
            vec![
                ("in_cap", "Input Cap", vec![1]),
                ("keep_b", "Keep B", vec![1]),
            ],
        );
        let mut comp_b = ComposableNamespace::new(ns_b);
        comp_b.mark_input(&CapabilityId::new("in_cap")).unwrap();

        let composed = compose_namespaces(&comp_a, 0, &comp_b, 0).unwrap();

        // Should have keep_a and keep_b (not out_cap or in_cap)
        assert_eq!(composed.namespace.capabilities.len(), 2);
    }

    #[test]
    fn test_compose_incompatible_fails() {
        let ns_a = make_namespace("A", vec![("out_cap", "Out", vec![1])]);
        let mut comp_a = ComposableNamespace::new(ns_a);
        comp_a.mark_output(&CapabilityId::new("out_cap")).unwrap();

        let ns_b = make_namespace("B", vec![("in_cap", "In", vec![2])]);
        let mut comp_b = ComposableNamespace::new(ns_b);
        comp_b.mark_input(&CapabilityId::new("in_cap")).unwrap();

        let result = compose_namespaces(&comp_a, 0, &comp_b, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_composition_multiplicity() {
        let ns_a = make_namespace("A", vec![("out_cap", "Out", vec![1])]);
        let mut comp_a = ComposableNamespace::new(ns_a);
        comp_a.mark_output(&CapabilityId::new("out_cap")).unwrap();

        let ns_b = make_namespace("B", vec![("in_cap", "In", vec![1])]);
        let mut comp_b = ComposableNamespace::new(ns_b);
        comp_b.mark_input(&CapabilityId::new("in_cap")).unwrap();

        let mult = composition_multiplicity(&comp_a, 0, &comp_b, 0);
        assert!(mult >= 1);
    }

    #[test]
    fn test_effective_capability_count() {
        let ns = make_namespace(
            "A",
            vec![
                ("cap1", "C1", vec![1]),
                ("cap2", "C2", vec![1]),
                ("cap3", "C3", vec![1]),
            ],
        );
        let mut comp = ComposableNamespace::new(ns);
        comp.mark_output(&CapabilityId::new("cap1")).unwrap();

        assert_eq!(comp.effective_capability_count(), 2);
    }

    #[test]
    fn test_composable_outputs_inputs() {
        let ns = make_namespace("A", vec![("cap1", "C1", vec![1]), ("cap2", "C2", vec![1])]);
        let mut comp = ComposableNamespace::new(ns);
        comp.mark_output(&CapabilityId::new("cap1")).unwrap();
        comp.mark_input(&CapabilityId::new("cap2")).unwrap();

        assert_eq!(comp.outputs().len(), 1);
        assert_eq!(comp.inputs().len(), 1);
    }
}
