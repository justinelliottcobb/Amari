use amari_tropical::{TropicalMatrix, TropicalMultivector, TropicalNumber};
use approx::assert_relative_eq;
use core::ops::{Add, Mul};

mod tropical_tests {
    use super::*;

    #[test]
    fn test_tropical_addition_is_max() {
        let a = TropicalNumber(3.0);
        let b = TropicalNumber(5.0);

        let sum = a.add(b);
        assert_eq!(sum.0, 5.0); // max(3, 5) = 5
    }

    #[test]
    fn test_tropical_multiplication_is_addition() {
        let a = TropicalNumber(3.0);
        let b = TropicalNumber(5.0);

        let product = a.mul(b);
        assert_eq!(product.0, 8.0); // 3 + 5 = 8
    }

    #[test]
    fn test_tropical_zero_is_negative_infinity() {
        let zero = TropicalNumber::ZERO;
        let a = TropicalNumber(5.0);

        let sum = zero.add(a);
        assert_eq!(sum.0, 5.0); // max(-∞, 5) = 5

        let product = zero.mul(a);
        assert_eq!(product.0, f64::NEG_INFINITY); // -∞ + 5 = -∞
    }

    #[test]
    fn test_tropical_one_is_zero() {
        let one = TropicalNumber::ONE;
        let a = TropicalNumber(5.0);

        let product = one.mul(a);
        assert_eq!(product.0, 5.0); // 0 + 5 = 5
    }

    #[test]
    fn test_tropical_softmax_becomes_max() {
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let tropical_logits: Vec<TropicalNumber<f64>> =
            logits.iter().map(|&x| TropicalNumber(x)).collect();

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
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .0;

        assert_eq!(traditional_argmax, tropical_max);
        assert_eq!(tropical_max, 3); // Index of 5.0
    }

    #[test]
    fn test_tropical_viterbi_path() {
        // Transition matrix (log probabilities)
        let transitions = TropicalMatrix::from_log_probs(&[vec![-1.0, -2.0], vec![-1.5, -0.5]]);

        // Emission matrix
        let emissions =
            TropicalMatrix::from_log_probs(&[vec![-0.5, -2.0, -1.0], vec![-2.0, -0.5, -1.5]]);

        let path = TropicalMultivector::<f64, 2>::viterbi(
            &transitions,
            &emissions,
            &[0.0, -1.0], // Initial state probs
            3,            // Sequence length
        );

        // Path should maximize probability (minimize negative log prob)
        assert_eq!(path.len(), 3);
    }
}
