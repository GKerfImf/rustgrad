#![allow(dead_code)]
pub mod core;
pub mod mlp;

#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod value { 
        use crate::core::Value;
        use crate::core::{top_sort, backward, forward, update_weights};
        use rand::Rng;
        use more_asserts as ma;
        use approx;

        #[test]
        fn basic() {
            let v = Value::new(1.0);
            assert_eq!(v.get_data(), 1.0);
            assert_eq!(v.get_grad(), 0.0);
        }

        #[test]
        fn sum1() {
            let a = Value::new(10.0);
            let b = Value::new(1.0);
            let s = a.clone() + b.clone();

            let top_sort = top_sort(s.clone());
            backward(s.clone(), &top_sort);

            assert_eq!(s.get_data(), 11.0);
            assert_eq!(s.get_grad(), 1.0);
            assert_eq!(a.get_grad(), 1.0);
            assert_eq!(b.get_grad(), 1.0);
        }

        #[test]
        fn sum2() {
            let a = Value::new(10.0);
            let b = Value::new(1.0);
            let s = a.clone() + b.clone() + b.clone();

            let top_sort = top_sort(s.clone());
            backward(s.clone(), &top_sort);

            assert_eq!(s.get_grad(), 1.0);
            assert_eq!(a.get_grad(), 1.0);
            assert_eq!(b.get_grad(), 2.0);
        }

        #[test]
        fn sub() {
            let a = Value::new(10.0);
            let b = Value::new(1.0);
            let s = a.clone() - b.clone();
            
            let top_sort = top_sort(s.clone());
            backward(s.clone(), &top_sort);
            
            assert_eq!(s.get_data(), 9.0);
            assert_eq!(s.get_grad(), 1.0);
            assert_eq!(a.get_grad(), 1.0);
            assert_eq!(b.get_grad(), -1.0);
        }

        #[test]
        fn mul1() {
            let a = Value::new(10.0);
            let b = Value::new(2.0);
            let s = a.clone() * b.clone();

            assert_eq!(s.get_data(), 20.0);

            let top_sort = top_sort(s.clone());
            backward(s.clone(), &top_sort);

            assert_eq!(s.get_grad(), 1.0);
            assert_eq!(a.get_grad(), 2.0);
            assert_eq!(b.get_grad(), 10.0);
        }

        #[test]
        fn mul2() {
            let a = Value::new(10.0);
            let s = a.clone() * a.clone();

            assert_eq!(s.get_data(), 100.0);

            let top_sort = top_sort(s.clone());
            backward(s.clone(), &top_sort);

            assert_eq!(s.get_grad(), 1.0);
            assert_eq!(a.get_grad(), 20.0);
        }

        #[test]
        fn mul_min() {
            let mut rng = rand::thread_rng();
            let a = Value::new((rng.gen::<f64>() - 0.5)*100.0);
            let s = a.clone() * a.clone();

            let top_sort = top_sort(s.clone());

            let mut old_val = s.get_data();
            for _ in 0..50 { 
                backward(s.clone(), &top_sort);
                update_weights(&vec![s.clone(), a.clone()]);
                forward(&top_sort);

                let new_val = s.get_data();
                ma::assert_le!(new_val, old_val);   // Value always decreases
                old_val = new_val;
            }
            approx::relative_eq!(s.get_data(), 0.0, epsilon = 0.001);
        }

        #[test]
        fn squared_diff() {
            let mut rng = rand::thread_rng();

            let a = Value::new((rng.gen::<f64>() - 0.5)*100.0);
            let b = Value::new((rng.gen::<f64>() - 0.5)*100.0);
            let s = (a.clone() - b.clone()) * (a.clone() - b.clone());

            let top_sort = top_sort(s.clone());
            for _ in 0..50 { 
                backward(s.clone(), &top_sort);
                update_weights(&vec![s.clone(), a.clone()]);
                forward(&top_sort);
            }
            approx::relative_eq!(a.get_data(), b.get_data(), epsilon = 0.001);
        }
    }

    #[cfg(test)]
    mod neurons { 
        use crate::core::{Value};
        use crate::mlp::{Neuron};
        use crate::core::{top_sort, backward, forward, update_weights};

        #[test]
        fn basic() {
            let a = Value::new(1.0);
            let b = Value::new(2.0);
            let c = Value::new(3.0);
            let n = Neuron::new(vec![a,b,c], vec![1.0,2.0,3.0], 1.0, false);

            let top_sort = top_sort(n.out.clone());
            forward(&top_sort);

            assert_eq!(1.0 + 4.0 + 9.0 + 1.0, n.out.get_data());
        }

        #[test]
        fn basic2() {
            let a = Value::new(-4.02704);
            let b = Value::new(2.0);
            let n = Neuron::new(vec![a,b], vec![1.0,2.0], 1.0, true);

            let top_sort = top_sort(n.out.clone());
            forward(&top_sort);
            approx::relative_eq!(n.out.get_data(), 0.75, epsilon = 0.001);
        }

        #[test]
        fn backprop() {
            let a = Value::new(1.0);
            let b = Value::new(2.0);
            let c = Value::new(3.0);
            let n = Neuron::new(vec![a.clone(),b.clone(),c.clone()], vec![11.0,22.0,33.0], 1.0, false);

            let top_sort = top_sort(n.out.clone());

            forward(&top_sort);
            backward(n.out.clone(), &top_sort);
            assert_eq!(a.get_grad(), 11.0);
            assert_eq!(b.get_grad(), 22.0);
            assert_eq!(c.get_grad(), 33.0);
        }
    }
}