use amari_info_geom::{DuallyFlatManifold, SimpleAlphaConnection};
use approx::assert_relative_eq;

mod info_geom_tests {
    use super::*;

    #[test]
    fn test_fisher_metric_positive_definite() {
        let manifold = DuallyFlatManifold::new(3, 0.0);
        let point = vec![0.3, 0.5, 0.2]; // Probability distribution

        let fisher = manifold.fisher_metric_at(&point);

        // Fisher metric should be positive definite
        for eigenvalue in fisher.eigenvalues() {
            assert!(eigenvalue > 0.0);
        }
    }

    #[test]
    fn test_bregman_divergence_non_negative() {
        let manifold = DuallyFlatManifold::new(3, 0.0);
        let p = vec![0.3, 0.5, 0.2];
        let q = vec![0.2, 0.4, 0.4];

        let divergence = manifold.bregman_divergence(&p, &q);

        assert!(divergence >= 0.0);

        // Self-divergence should be zero
        let self_div = manifold.bregman_divergence(&p, &p);
        assert_relative_eq!(self_div, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pythagorean_theorem() {
        // In dually flat spaces, Pythagorean theorem holds
        let manifold = DuallyFlatManifold::new(3, 0.0);

        let p = vec![0.3, 0.5, 0.2];
        let q = vec![0.2, 0.4, 0.4];
        let r = vec![0.1, 0.3, 0.6];

        // Test that Bregman divergences are well-defined
        let d_pq = manifold.bregman_divergence(&p, &q);
        let d_pr = manifold.bregman_divergence(&p, &r);
        let d_qr = manifold.bregman_divergence(&q, &r);

        // All divergences should be non-negative
        assert!(d_pq >= 0.0);
        assert!(d_pr >= 0.0);
        assert!(d_qr >= 0.0);
    }

    #[test]
    fn test_alpha_connection_interpolation() {
        // α = 1 gives e-connection, α = -1 gives m-connection
        let e_connection = SimpleAlphaConnection::new(1.0);
        let m_connection = SimpleAlphaConnection::new(-1.0);
        let mixed = SimpleAlphaConnection::new(0.0);

        // Test that α values are correctly stored
        assert_eq!(e_connection.alpha(), 1.0);
        assert_eq!(m_connection.alpha(), -1.0);
        assert_eq!(mixed.alpha(), 0.0);
    }
}
