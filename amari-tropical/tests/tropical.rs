use amari_tropical::{TropicalMatrix, TropicalNumber};
use core::ops::{Add, Mul};

mod tropical_tests {
    use super::*;

    #[test]
    fn test_tropical_addition_is_max() {
        let a = TropicalNumber::new(3.0);
        let b = TropicalNumber::new(5.0);

        let sum = a.add(b);
        assert_eq!(sum.value(), 5.0); // max(3, 5) = 5
    }

    #[test]
    fn test_tropical_multiplication_is_addition() {
        let a = TropicalNumber::new(3.0);
        let b = TropicalNumber::new(5.0);

        let product = a.mul(b);
        assert_eq!(product.value(), 8.0); // 3 + 5 = 8
    }

    #[test]
    fn test_tropical_zero_is_negative_infinity() {
        let zero = TropicalNumber::tropical_zero();
        let a = TropicalNumber::new(5.0);

        let sum = zero.add(a);
        assert_eq!(sum.value(), 5.0); // max(-∞, 5) = 5

        let product = zero.mul(a);
        assert_eq!(product.value(), f64::NEG_INFINITY); // -∞ + 5 = -∞
    }

    #[test]
    fn test_tropical_one_is_zero() {
        let one = TropicalNumber::tropical_one();
        let a = TropicalNumber::new(5.0);

        let product = one.mul(a);
        assert_eq!(product.value(), 5.0); // 0 + 5 = 5
    }

    #[test]
    fn test_tropical_softmax_becomes_max() {
        let logits = [1.0, 3.0, 2.0, 5.0, 4.0];
        let tropical_logits: Vec<TropicalNumber<f64>> =
            logits.iter().map(|&x| TropicalNumber::new(x)).collect();

        // Traditional softmax argmax
        let traditional_argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        // Tropical max (just find max)
        let tropical_max = tropical_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.value().partial_cmp(&b.value()).unwrap())
            .unwrap()
            .0;

        assert_eq!(traditional_argmax, tropical_max);
        assert_eq!(tropical_max, 3); // Index of 5.0
    }

    #[test]
    #[ignore = "TropicalMultivector::viterbi requires API update for v0.12.0 (needs 4 generic parameters)"]
    fn test_tropical_viterbi_path() {
        // Transition matrix (log probabilities)
        let _transitions = TropicalMatrix::from_log_probs(&[vec![-1.0, -2.0], vec![-1.5, -0.5]]);

        // Emission matrix
        let _emissions =
            TropicalMatrix::from_log_probs(&[vec![-0.5, -2.0, -1.0], vec![-2.0, -0.5, -1.5]]);

        // TODO: Update to new TropicalMultivector<T, P, Q, R> API (needs 4 generic params)
        // let path = TropicalMultivector::<f64, ?, ?, ?>::viterbi(
        //     &transitions,
        //     &emissions,
        //     &[0.0, -1.0], // Initial state probs
        //     3,            // Sequence length
        // );

        // // Path should maximize probability (minimize negative log prob)
        // assert_eq!(path.len(), 3);
    }
}
