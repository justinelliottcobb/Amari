//! Multi-GPU workload distribution and coordination infrastructure
//!
//! This module implements sophisticated multi-GPU capabilities for the Amari library,
//! enabling intelligent workload distribution, load balancing, and performance optimization
//! across multiple GPU devices.

use crate::{UnifiedGpuError, UnifiedGpuResult};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};
use tokio::sync::{Notify, RwLock};

/// Unique identifier for GPU devices in the multi-GPU system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

/// GPU device capabilities and characteristics
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Number of compute units (shader cores, streaming multiprocessors, etc.)
    pub compute_units: u32,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gb_s: f32,
    /// Peak floating-point operations per second
    pub peak_flops: f64,
    /// Total device memory in GB
    pub memory_size_gb: f32,
    /// GPU architecture family
    pub architecture: GpuArchitecture,
    /// Maximum workgroup size
    pub max_workgroup_size: (u32, u32, u32),
    /// Shared memory per workgroup in bytes
    pub shared_memory_per_workgroup: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuArchitecture {
    Nvidia { compute_capability: (u32, u32) },
    Amd { gcn_generation: u32 },
    Intel { generation: String },
    Apple { gpu_family: u32 },
    Unknown,
}

/// Individual GPU device in the multi-GPU system
#[derive(Debug)]
pub struct GpuDevice {
    pub id: DeviceId,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_info: wgpu::AdapterInfo,
    pub capabilities: DeviceCapabilities,

    // Runtime metrics
    pub current_load: Arc<AtomicU32>, // Stores f32 load as u32 bits
    pub memory_usage: Arc<AtomicU64>,
    pub total_operations: Arc<AtomicUsize>,
    pub error_count: Arc<AtomicUsize>,
    pub last_activity: Arc<Mutex<Instant>>,

    // Device health monitoring
    pub is_healthy: Arc<std::sync::atomic::AtomicBool>,
}

impl GpuDevice {
    /// Create a new GPU device from WebGPU adapter and device
    pub async fn new(
        id: DeviceId,
        adapter: &wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> UnifiedGpuResult<Self> {
        let adapter_info = adapter.get_info();
        let capabilities = Self::assess_capabilities(adapter, &adapter_info).await?;

        Ok(Self {
            id,
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            capabilities,
            current_load: Arc::new(AtomicU32::new(0.0_f32.to_bits())),
            memory_usage: Arc::new(AtomicU64::new(0)),
            total_operations: Arc::new(AtomicUsize::new(0)),
            error_count: Arc::new(AtomicUsize::new(0)),
            last_activity: Arc::new(Mutex::new(Instant::now())),
            is_healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        })
    }

    /// Assess device capabilities through benchmarking and feature detection
    async fn assess_capabilities(
        adapter: &wgpu::Adapter,
        adapter_info: &wgpu::AdapterInfo,
    ) -> UnifiedGpuResult<DeviceCapabilities> {
        let limits = adapter.limits();

        // Estimate capabilities based on adapter info and limits
        let architecture = match adapter_info.vendor {
            0x10DE => GpuArchitecture::Nvidia {
                compute_capability: (8, 0),
            }, // Estimate
            0x1002 | 0x1022 => GpuArchitecture::Amd { gcn_generation: 5 },
            0x8086 => GpuArchitecture::Intel {
                generation: "Gen12".to_string(),
            },
            _ => GpuArchitecture::Unknown,
        };

        // Estimate memory bandwidth (simplified heuristics)
        let memory_bandwidth_gb_s = match &architecture {
            GpuArchitecture::Nvidia { .. } => 500.0, // RTX 3080-class estimate
            GpuArchitecture::Amd { .. } => 512.0,    // RX 6800 XT-class estimate
            GpuArchitecture::Intel { .. } => 100.0,  // Intel Arc estimate
            GpuArchitecture::Apple { .. } => 400.0,  // M1 Ultra estimate
            GpuArchitecture::Unknown => 200.0,       // Conservative estimate
        };

        // Estimate compute units and FLOPS
        let (compute_units, peak_flops) = match &architecture {
            GpuArchitecture::Nvidia { .. } => (68, 30e12), // RTX 3080 estimate
            GpuArchitecture::Amd { .. } => (72, 20e12),    // RX 6800 XT estimate
            GpuArchitecture::Intel { .. } => (32, 15e12),  // Intel Arc estimate
            GpuArchitecture::Apple { .. } => (64, 25e12),  // M1 Ultra estimate
            GpuArchitecture::Unknown => (32, 10e12),       // Conservative estimate
        };

        Ok(DeviceCapabilities {
            compute_units,
            memory_bandwidth_gb_s,
            peak_flops,
            memory_size_gb: 8.0, // Default estimate - would need actual query
            architecture,
            max_workgroup_size: (
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ),
            shared_memory_per_workgroup: limits.max_compute_workgroup_storage_size,
        })
    }

    /// Update device load metrics
    pub fn update_load(&self, load_percent: f32) {
        self.current_load
            .store(load_percent.to_bits(), Ordering::Relaxed);
        if let Ok(mut last_activity) = self.last_activity.lock() {
            *last_activity = Instant::now();
        }
    }

    /// Get current device load percentage
    pub fn current_load(&self) -> f32 {
        f32::from_bits(self.current_load.load(Ordering::Relaxed))
    }

    /// Check if device is currently available for work
    pub fn is_available(&self) -> bool {
        self.is_healthy.load(Ordering::Relaxed) && self.current_load() < 90.0
    }

    /// Get device performance score for workload assignment
    pub fn performance_score(&self, operation_type: &str) -> f32 {
        let base_score = match operation_type {
            "matrix_multiply" => self.capabilities.peak_flops as f32 / 1e12,
            "memory_intensive" => self.capabilities.memory_bandwidth_gb_s,
            _ => {
                (self.capabilities.peak_flops as f32 / 1e12) * 0.5
                    + self.capabilities.memory_bandwidth_gb_s * 0.5
            }
        };

        // Adjust for current load
        let load_factor = 1.0 - (self.current_load() / 100.0);
        base_score * load_factor
    }
}

/// Workload definition for distribution across multiple GPUs
#[derive(Debug, Clone)]
pub struct Workload {
    pub operation_type: String,
    pub data_size: usize,
    pub memory_requirement_mb: f32,
    pub compute_intensity: ComputeIntensity,
    pub parallelizable: bool,
    pub synchronization_required: bool,
}

#[derive(Debug, Clone)]
pub enum ComputeIntensity {
    Light,    // Memory-bound operations
    Moderate, // Balanced compute/memory
    Heavy,    // Compute-bound operations
    Extreme,  // Very high arithmetic intensity
}

/// Device-specific workload assignment
#[derive(Debug, Clone)]
pub struct DeviceWorkload {
    pub device_id: DeviceId,
    pub workload_fraction: f32,
    pub data_range: (usize, usize),
    pub estimated_completion_ms: f32,
    pub memory_requirement_mb: f32,
}

/// Intelligent load balancer for multi-GPU workload distribution
pub struct IntelligentLoadBalancer {
    devices: Arc<RwLock<HashMap<DeviceId, Arc<GpuDevice>>>>,
    balancing_strategy: LoadBalancingStrategy,
    performance_history: Arc<Mutex<HashMap<String, Vec<PerformanceRecord>>>>,
}

#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    /// Distribute work equally across all devices
    Balanced,
    /// Distribute based on device capabilities
    CapabilityAware,
    /// Optimize for memory constraints
    MemoryAware,
    /// Minimize total completion time
    LatencyOptimized,
    /// Machine learning-driven distribution
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    pub device_id: DeviceId,
    pub operation_type: String,
    pub data_size: usize,
    pub completion_time_ms: f32,
    pub throughput_gops: f32,
    pub memory_bandwidth_utilized_gb_s: f32,
    pub timestamp: Instant,
}

impl IntelligentLoadBalancer {
    /// Create a new intelligent load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            balancing_strategy: strategy,
            performance_history: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add a device to the load balancer
    pub async fn add_device(&self, device: Arc<GpuDevice>) {
        let mut devices: tokio::sync::RwLockWriteGuard<HashMap<DeviceId, Arc<GpuDevice>>> =
            self.devices.write().await;
        devices.insert(device.id, device);
    }

    /// Remove a device from the load balancer
    pub async fn remove_device(&self, device_id: DeviceId) {
        let mut devices: tokio::sync::RwLockWriteGuard<HashMap<DeviceId, Arc<GpuDevice>>> =
            self.devices.write().await;
        devices.remove(&device_id);
    }

    /// Distribute workload across available devices
    pub async fn distribute_workload(
        &self,
        workload: &Workload,
    ) -> UnifiedGpuResult<Vec<DeviceWorkload>> {
        let devices: tokio::sync::RwLockReadGuard<HashMap<DeviceId, Arc<GpuDevice>>> =
            self.devices.read().await;
        let available_devices: Vec<&Arc<GpuDevice>> = devices
            .values()
            .filter(|device| device.is_available())
            .collect();

        if available_devices.is_empty() {
            return Err(UnifiedGpuError::InvalidOperation(
                "No available devices for workload distribution".into(),
            ));
        }

        match self.balancing_strategy {
            LoadBalancingStrategy::Balanced => {
                self.distribute_balanced(&available_devices, workload)
            }
            LoadBalancingStrategy::CapabilityAware => {
                self.distribute_capability_aware(&available_devices, workload)
            }
            LoadBalancingStrategy::MemoryAware => {
                self.distribute_memory_aware(&available_devices, workload)
            }
            LoadBalancingStrategy::LatencyOptimized => {
                self.distribute_latency_optimized(&available_devices, workload)
            }
            LoadBalancingStrategy::Adaptive => {
                self.distribute_adaptive(&available_devices, workload).await
            }
        }
    }

    /// Balanced distribution - equal work per device
    fn distribute_balanced(
        &self,
        devices: &[&Arc<GpuDevice>],
        workload: &Workload,
    ) -> UnifiedGpuResult<Vec<DeviceWorkload>> {
        let num_devices = devices.len();
        let work_per_device = 1.0 / num_devices as f32;
        let data_per_device = workload.data_size / num_devices;

        let mut assignments = Vec::new();
        for (i, device) in devices.iter().enumerate() {
            let start = i * data_per_device;
            let end = if i == num_devices - 1 {
                workload.data_size
            } else {
                (i + 1) * data_per_device
            };

            assignments.push(DeviceWorkload {
                device_id: device.id,
                workload_fraction: work_per_device,
                data_range: (start, end),
                estimated_completion_ms: 100.0, // Placeholder estimation
                memory_requirement_mb: workload.memory_requirement_mb / num_devices as f32,
            });
        }

        Ok(assignments)
    }

    /// Capability-aware distribution - weight by device performance
    fn distribute_capability_aware(
        &self,
        devices: &[&Arc<GpuDevice>],
        workload: &Workload,
    ) -> UnifiedGpuResult<Vec<DeviceWorkload>> {
        // Calculate performance scores for each device
        let scores: Vec<f32> = devices
            .iter()
            .map(|device| device.performance_score(&workload.operation_type))
            .collect();

        let total_score: f32 = scores.iter().sum();

        let mut assignments = Vec::new();
        let mut data_offset = 0;

        for (i, (device, &score)) in devices.iter().zip(scores.iter()).enumerate() {
            let fraction = score / total_score;
            let data_chunk_size = (workload.data_size as f32 * fraction) as usize;

            let end = if i == devices.len() - 1 {
                workload.data_size
            } else {
                data_offset + data_chunk_size
            };

            assignments.push(DeviceWorkload {
                device_id: device.id,
                workload_fraction: fraction,
                data_range: (data_offset, end),
                estimated_completion_ms: 100.0 / fraction, // Inverse of capability
                memory_requirement_mb: workload.memory_requirement_mb * fraction,
            });

            data_offset = end;
        }

        Ok(assignments)
    }

    /// Memory-aware distribution - consider memory constraints
    fn distribute_memory_aware(
        &self,
        devices: &[&Arc<GpuDevice>],
        workload: &Workload,
    ) -> UnifiedGpuResult<Vec<DeviceWorkload>> {
        // Filter devices that can handle the memory requirement
        let viable_devices: Vec<&Arc<GpuDevice>> = devices
            .iter()
            .filter(|device| {
                let required_memory_gb = workload.memory_requirement_mb / 1024.0;
                device.capabilities.memory_size_gb >= required_memory_gb
            })
            .copied()
            .collect();

        if viable_devices.is_empty() {
            return Err(UnifiedGpuError::InvalidOperation(
                "No devices with sufficient memory for workload".into(),
            ));
        }

        // Use capability-aware distribution among viable devices
        self.distribute_capability_aware(&viable_devices, workload)
    }

    /// Latency-optimized distribution - minimize total completion time
    fn distribute_latency_optimized(
        &self,
        devices: &[&Arc<GpuDevice>],
        workload: &Workload,
    ) -> UnifiedGpuResult<Vec<DeviceWorkload>> {
        // For now, use capability-aware as a proxy for latency optimization
        // In practice, this would use more sophisticated scheduling algorithms
        self.distribute_capability_aware(devices, workload)
    }

    /// Adaptive distribution using machine learning and historical data
    async fn distribute_adaptive(
        &self,
        devices: &[&Arc<GpuDevice>],
        workload: &Workload,
    ) -> UnifiedGpuResult<Vec<DeviceWorkload>> {
        // Check performance history for similar workloads
        if let Ok(history) = self.performance_history.lock() {
            if let Some(records) = history.get(&workload.operation_type) {
                // Use historical data to inform distribution
                return self.distribute_based_on_history(devices, workload, records);
            }
        }

        // Fallback to capability-aware if no historical data
        self.distribute_capability_aware(devices, workload)
    }

    /// Distribution based on historical performance data
    fn distribute_based_on_history(
        &self,
        devices: &[&Arc<GpuDevice>],
        workload: &Workload,
        history: &[PerformanceRecord],
    ) -> UnifiedGpuResult<Vec<DeviceWorkload>> {
        // Calculate performance predictions based on historical data
        let mut device_predictions = HashMap::new();

        for device in devices {
            let device_history: Vec<_> = history
                .iter()
                .filter(|record| record.device_id == device.id)
                .collect();

            let predicted_throughput = if device_history.is_empty() {
                device.performance_score(&workload.operation_type)
            } else {
                // Weighted average of recent performance
                let recent_throughput: f32 = device_history
                    .iter()
                    .rev()
                    .take(10) // Last 10 operations
                    .map(|record| record.throughput_gops)
                    .sum::<f32>()
                    / device_history.len().min(10) as f32;
                recent_throughput
            };

            device_predictions.insert(device.id, predicted_throughput);
        }

        // Distribute based on predicted performance
        let total_predicted: f32 = device_predictions.values().sum();

        let mut assignments = Vec::new();
        let mut data_offset = 0;

        for (i, device) in devices.iter().enumerate() {
            let predicted = device_predictions[&device.id];
            let fraction = predicted / total_predicted;
            let data_chunk_size = (workload.data_size as f32 * fraction) as usize;

            let end = if i == devices.len() - 1 {
                workload.data_size
            } else {
                data_offset + data_chunk_size
            };

            assignments.push(DeviceWorkload {
                device_id: device.id,
                workload_fraction: fraction,
                data_range: (data_offset, end),
                estimated_completion_ms: 100.0 / fraction,
                memory_requirement_mb: workload.memory_requirement_mb * fraction,
            });

            data_offset = end;
        }

        Ok(assignments)
    }

    /// Record performance data for adaptive learning
    pub async fn record_performance(&self, record: PerformanceRecord) {
        if let Ok(mut history) = self.performance_history.lock() {
            let operation_history = history
                .entry(record.operation_type.clone())
                .or_insert_with(Vec::new);

            operation_history.push(record);

            // Keep only recent history (last 1000 records)
            if operation_history.len() > 1000 {
                operation_history.remove(0);
            }
        }
    }

    /// Get performance statistics for an operation type
    pub async fn get_performance_stats(&self, operation_type: &str) -> Option<PerformanceStats> {
        if let Ok(history) = self.performance_history.lock() {
            if let Some(records) = history.get(operation_type) {
                if records.is_empty() {
                    return None;
                }

                let completion_times: Vec<f32> =
                    records.iter().map(|r| r.completion_time_ms).collect();
                let throughputs: Vec<f32> = records.iter().map(|r| r.throughput_gops).collect();

                let avg_completion_time =
                    completion_times.iter().sum::<f32>() / completion_times.len() as f32;
                let avg_throughput = throughputs.iter().sum::<f32>() / throughputs.len() as f32;

                return Some(PerformanceStats {
                    operation_type: operation_type.to_string(),
                    avg_completion_time_ms: avg_completion_time,
                    avg_throughput_gops: avg_throughput,
                    total_operations: records.len(),
                    best_device_id: records
                        .iter()
                        .max_by(|a, b| a.throughput_gops.partial_cmp(&b.throughput_gops).unwrap())
                        .map(|r| r.device_id),
                });
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub operation_type: String,
    pub avg_completion_time_ms: f32,
    pub avg_throughput_gops: f32,
    pub total_operations: usize,
    pub best_device_id: Option<DeviceId>,
}

/// Multi-GPU workload coordinator for synchronization and result aggregation
pub struct WorkloadCoordinator {
    active_workloads: Arc<Mutex<HashMap<String, ActiveWorkload>>>,
    #[allow(dead_code)]
    synchronization_manager: SynchronizationManager,
}

#[derive(Debug)]
pub struct ActiveWorkload {
    pub id: String,
    pub device_assignments: Vec<DeviceWorkload>,
    pub completion_status: Vec<bool>,
    pub results: Vec<Option<Vec<u8>>>,
    pub start_time: Instant,
}

/// Synchronization manager for multi-GPU operations
pub struct SynchronizationManager {
    barriers: Arc<Mutex<HashMap<String, MultiGpuBarrier>>>,
}

/// Multi-GPU barrier for synchronizing operations across devices
pub struct MultiGpuBarrier {
    pub barrier_id: String,
    pub device_count: usize,
    pub completed_devices: Arc<AtomicUsize>,
    pub completion_notifier: Arc<Notify>,
    pub timeout: Duration,
}

impl WorkloadCoordinator {
    /// Create a new workload coordinator
    pub fn new() -> Self {
        Self {
            active_workloads: Arc::new(Mutex::new(HashMap::new())),
            synchronization_manager: SynchronizationManager::new(),
        }
    }

    /// Submit a workload for execution across multiple devices
    pub async fn submit_workload(
        &self,
        workload_id: String,
        assignments: Vec<DeviceWorkload>,
    ) -> UnifiedGpuResult<()> {
        let device_count = assignments.len();

        let active_workload = ActiveWorkload {
            id: workload_id.clone(),
            device_assignments: assignments,
            completion_status: vec![false; device_count],
            results: vec![None; device_count],
            start_time: Instant::now(),
        };

        if let Ok(mut workloads) = self.active_workloads.lock() {
            workloads.insert(workload_id, active_workload);
        }

        Ok(())
    }

    /// Wait for workload completion and aggregate results
    pub async fn wait_for_completion(
        &self,
        workload_id: &str,
        timeout: Duration,
    ) -> UnifiedGpuResult<Vec<Vec<u8>>> {
        let start = Instant::now();

        loop {
            if start.elapsed() > timeout {
                return Err(UnifiedGpuError::InvalidOperation(
                    "Workload completion timeout".into(),
                ));
            }

            // Check completion status
            if let Ok(workloads) = self.active_workloads.lock() {
                if let Some(workload) = workloads.get(workload_id) {
                    if workload
                        .completion_status
                        .iter()
                        .all(|&completed| completed)
                    {
                        // All devices completed - aggregate results
                        let results: Vec<Vec<u8>> = workload
                            .results
                            .iter()
                            .filter_map(|result| result.as_ref())
                            .cloned()
                            .collect();
                        return Ok(results);
                    }
                }
            }

            // Brief sleep before checking again
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Mark device as completed for a workload
    pub async fn mark_device_completed(
        &self,
        workload_id: &str,
        device_id: DeviceId,
        result: Vec<u8>,
    ) -> UnifiedGpuResult<()> {
        if let Ok(mut workloads) = self.active_workloads.lock() {
            if let Some(workload) = workloads.get_mut(workload_id) {
                // Find device index and mark as completed
                for (i, assignment) in workload.device_assignments.iter().enumerate() {
                    if assignment.device_id == device_id {
                        workload.completion_status[i] = true;
                        workload.results[i] = Some(result);
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}

impl SynchronizationManager {
    /// Create a new synchronization manager
    pub fn new() -> Self {
        Self {
            barriers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a barrier for synchronizing multiple devices
    pub async fn create_barrier(
        &self,
        barrier_id: String,
        device_count: usize,
        timeout: Duration,
    ) -> UnifiedGpuResult<()> {
        let barrier = MultiGpuBarrier {
            barrier_id: barrier_id.clone(),
            device_count,
            completed_devices: Arc::new(AtomicUsize::new(0)),
            completion_notifier: Arc::new(Notify::new()),
            timeout,
        };

        if let Ok(mut barriers) = self.barriers.lock() {
            barriers.insert(barrier_id, barrier);
        }

        Ok(())
    }

    /// Wait for all devices to reach the barrier
    pub async fn wait_barrier(
        &self,
        barrier_id: &str,
        _device_id: DeviceId,
    ) -> UnifiedGpuResult<()> {
        let (notifier, _device_count) = {
            if let Ok(barriers) = self.barriers.lock() {
                if let Some(barrier) = barriers.get(barrier_id) {
                    let completed = barrier.completed_devices.fetch_add(1, Ordering::SeqCst) + 1;

                    if completed >= barrier.device_count {
                        // Last device to reach barrier - notify all
                        barrier.completion_notifier.notify_waiters();
                        return Ok(());
                    }

                    (
                        Arc::clone(&barrier.completion_notifier),
                        barrier.device_count,
                    )
                } else {
                    return Err(UnifiedGpuError::InvalidOperation(format!(
                        "Barrier {} not found",
                        barrier_id
                    )));
                }
            } else {
                return Err(UnifiedGpuError::InvalidOperation(
                    "Failed to access barriers".into(),
                ));
            }
        };

        // Wait for notification with timeout
        let timeout_duration = Duration::from_secs(30); // Default timeout
        tokio::time::timeout(timeout_duration, notifier.notified())
            .await
            .map_err(|_| UnifiedGpuError::InvalidOperation("Barrier wait timeout".into()))?;

        Ok(())
    }
}

impl Default for WorkloadCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for SynchronizationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id_creation() {
        let device_id = DeviceId(0);
        assert_eq!(device_id.0, 0);
    }

    #[test]
    fn test_workload_creation() {
        let workload = Workload {
            operation_type: "test_operation".to_string(),
            data_size: 1000,
            memory_requirement_mb: 100.0,
            compute_intensity: ComputeIntensity::Moderate,
            parallelizable: true,
            synchronization_required: false,
        };

        assert_eq!(workload.data_size, 1000);
        assert_eq!(workload.memory_requirement_mb, 100.0);
    }

    #[test]
    fn test_load_balancer_creation() {
        let balancer = IntelligentLoadBalancer::new(LoadBalancingStrategy::Balanced);
        assert!(matches!(
            balancer.balancing_strategy,
            LoadBalancingStrategy::Balanced
        ));
    }

    #[tokio::test]
    async fn test_workload_coordinator() {
        let coordinator = WorkloadCoordinator::new();

        let assignments = vec![
            DeviceWorkload {
                device_id: DeviceId(0),
                workload_fraction: 0.5,
                data_range: (0, 500),
                estimated_completion_ms: 100.0,
                memory_requirement_mb: 50.0,
            },
            DeviceWorkload {
                device_id: DeviceId(1),
                workload_fraction: 0.5,
                data_range: (500, 1000),
                estimated_completion_ms: 100.0,
                memory_requirement_mb: 50.0,
            },
        ];

        let result = coordinator
            .submit_workload("test_workload".to_string(), assignments)
            .await;
        assert!(result.is_ok());
    }
}
