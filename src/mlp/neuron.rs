#![allow(unused_imports)]
#![allow(dead_code)]

use rand::Rng;
use rand_distr::{Normal, Distribution};
use core::slice::Iter;
use std::{iter, fmt};

use crate::core::nonlinearity::NonLinearity;
use crate::core::core::{Value, RefValue};
use crate::core::core::{topological_sort, backward, forward, update_weights};

#[derive(Debug)]
pub struct Neuron {
    ins: Vec<RefValue>,         // Input variables
    out: RefValue,              // Output variable

    w: Vec<RefValue>,           // Weight variables
    b: RefValue,                // Bias variable
    parameters: Vec<RefValue>,  // All parameters (weights + bias)
}

impl Neuron {

    // The first argument is a vector of input variables
    // The second argument is a vector of parameters (bias + weights)
    pub fn from_vec(ins: Vec<RefValue>, parameters: Vec<f64>) -> Neuron {
        if ins.len() != parameters.len() - 1 {
            panic!("Number of inputs does not match the number of [parameters - 1]!")
        }

        // Number of weights
        let nweights = parameters.len() - 1;

        // Create a vector of weights and a bias variable
        let mut weights: Vec<RefValue> = Vec::with_capacity(nweights);
        for i in 1..=nweights {
            weights.push(Value::new(parameters[i]));
        }
        let bias: RefValue = Value::new(parameters[0]);

        // Create a vector of all parameters
        let params = weights.iter()
            .map( |rv| rv.clone() )
            .chain(iter::once(bias.clone()))
            .collect::<Vec<RefValue>>();

        // [out = ins * weights + bias]
        let out = ins.iter().zip(weights.iter())
            .map( |(i,w)| i.clone() * w.clone() )
            .sum::<RefValue>() + bias.clone();

        Neuron {
            ins: ins, out: out,
            w: weights, b: bias,
            parameters: params
        }
    }

    // Create a neuron with random weights and 0.0 bias
    pub fn new(ins: Vec<RefValue>) -> Neuron {
        let len = ins.len();
        let normal = Normal::new(0.0, 1.0).unwrap();
        return Neuron::from_vec(
            ins,
            // append 0.0 (bias) to vector of random gaussians (weights)
            iter::once(0.0).chain(
                (0..len).map( |_| normal.sample(&mut rand::thread_rng()) )
            ).collect::<Vec<f64>>()
        )
    }

    pub fn get_weights(&self) -> Vec<f64> {
        return self.w.iter().map( |rv| rv.get_data() ).collect::<Vec<f64>>()
    }

    pub fn get_bias(&self) -> f64 {
        return self.b.get_data()
    }

    fn update_weights(&self, rate: f64) {
        update_weights(&self.parameters, rate);
    }

    pub fn get_parameters(&self) -> Iter<RefValue> {
        return self.parameters.iter()
    }

    pub fn get_output_variable(&self) -> RefValue {
        return self.out.clone();
    }

}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f," [")?;
        for w in 0..self.w.len() {
            write!(f, "{val:>8.3} ", val=self.w[w].get_data())?;
        }
        write!(f,"]")?;
        write!(f, " + ({val:>8.3}) ", val=self.b.get_data())?;
        write!(f, " ==> {val:>8.3} \n", val=self.get_output_variable().get_data())?;

        return Ok(())
    }
}


// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
//                                      Tests                                      //
// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod neurons {
        use more_asserts as ma;
        use crate::core::nonlinearity::*;
        use crate::core::core::*;
        use crate::mlp::neuron::Neuron;

        #[test]
        fn basic() {
            let a = Value::new(1.0);
            let b = Value::new(2.0);
            let c = Value::new(3.0);
            let n = Neuron::from_vec(vec![a,b,c], vec![1.0, 1.0,2.0,3.0]);
            //                                         ^^^ -- bias

            let top_sort = topological_sort(n.out.clone());
            forward(&top_sort);

            assert_eq!(1.0 + 4.0 + 9.0 + 1.0, n.out.get_data());
        }

        #[test]
        fn basic2() {
            let a = Value::new(-4.02704492547);
            let b = Value::new(2.0);
            let n = Neuron::from_vec(vec![a,b], vec![1.0, 1.0, 2.0]);

            // [o = tanh(1.0 + 1.0*a + 2.0*b)]
            let o = n.out.clone().tanh();

            let top_sort = topological_sort(o.clone());
            forward(&top_sort);
            ma::assert_le!((o.get_data() - 0.75).abs(), 1e-6);
        }

        #[test]
        fn backprop() {
            let a = Value::new(1.0);
            let b = Value::new(2.0);
            let c = Value::new(3.0);
            let n = Neuron::from_vec(vec![a.clone(),b.clone(),c.clone()], vec![1.0, 11.0,22.0,33.0]);

            let top_sort = topological_sort(n.out.clone());

            forward(&top_sort);
            backward(n.out.clone(), &top_sort);
            assert_eq!(a.get_grad(), 11.0);
            assert_eq!(b.get_grad(), 22.0);
            assert_eq!(c.get_grad(), 33.0);
        }
    }
}