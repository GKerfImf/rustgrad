use rand::Rng;
use rand_distr::{Normal, Distribution};
use core::slice::Iter;
use std::iter;

use crate::core::nonlin::NonLin;
use crate::core::core::{Value, RefValue};
use crate::core::core::{topological_sort, backward, forward, update_weights};


#[derive(Debug)]
pub struct Neuron {
    ins: Vec<RefValue>,         // Input variables          // TODO: should be [&Vec<RefValue>]
    pub out: RefValue,          // Output variable

    pub w: Vec<RefValue>,       // Weight variables
    pub b: RefValue,            // Bias variable
    parameters: Vec<RefValue>,  // All parameters (weights + bias)
    pub nlin: NonLin            // Apply non-linearity (None/ReLu/Tanh)
}

impl Neuron { 
    pub fn new(ins: Vec<RefValue>, ws: Vec<f64>, b: f64, nlin: NonLin) -> Neuron { 
        if ins.len() != ws.len() { 
            panic!("Number of inputs does not match the number of weights!")
        }

        // Create a vector of weights and a bias variable
        let mut weights: Vec<RefValue> = Vec::with_capacity(ws.len());
        for i in 0..ws.len() {
            weights.push(Value::new(ws[i]));
        }
        let bias: RefValue = Value::new(b);

        let params = weights.iter()
            .map( |rv| rv.clone() )
            .chain(iter::once(bias.clone()))
            .collect::<Vec<RefValue>>();

        // [act = ins * weights + bias]
        let act = ins.iter().zip(weights.iter())
            .map( |(i,w)| i.clone() * w.clone() )
            .sum::<RefValue>() + bias.clone();

        let out = match nlin { 
            NonLin::None => act,
            NonLin::ReLu => act.relu(),
            NonLin::Tanh => act.tanh()
        };

        Neuron { 
            ins: ins, out: out, 
            w: weights, b: bias,
            parameters: params,
            nlin: nlin       
        }
    }

    pub fn with_rand_weights(ins: Vec<RefValue>, nlin: NonLin) -> Neuron {
        let mut rng = rand::thread_rng();
        let len = ins.len();
        let normal = Normal::new(0.0, (1.0 / len as f64).sqrt() ).unwrap();
        return Neuron::new(
            ins, 
            (0..len).map( |_| normal.sample(&mut rand::thread_rng()) ).collect::<Vec<f64>>(),
            nlin
            0.0
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
}


// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
//                                      Tests                                      //
// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod neurons { 
        use crate::core::nonlin::*;       
        use crate::core::core::*;       
        use crate::mlp::neuron::Neuron;

        #[test]
        fn basic() {
            let a = Value::new(1.0);
            let b = Value::new(2.0);
            let c = Value::new(3.0);
            let n = Neuron::new(vec![a,b,c], vec![1.0,2.0,3.0], 1.0, NonLin::None);

            let top_sort = topological_sort(n.out.clone());
            forward(&top_sort);

            assert_eq!(1.0 + 4.0 + 9.0 + 1.0, n.out.get_data());
        }

        #[test]
        fn basic2() {
            let a = Value::new(-4.02704);
            let b = Value::new(2.0);
            let n = Neuron::new(vec![a,b], vec![1.0,2.0], 1.0, NonLin::Tanh);

            let top_sort = topological_sort(n.out.clone());
            forward(&top_sort);
            approx::relative_eq!(n.out.get_data(), 0.75, epsilon = 0.001);
        }

        #[test]
        fn backprop() {
            let a = Value::new(1.0);
            let b = Value::new(2.0);
            let c = Value::new(3.0);
            let n = Neuron::new(vec![a.clone(),b.clone(),c.clone()], vec![11.0,22.0,33.0], 1.0, NonLin::None);

            let top_sort = topological_sort(n.out.clone());

            forward(&top_sort);
            backward(n.out.clone(), &top_sort);
            assert_eq!(a.get_grad(), 11.0);
            assert_eq!(b.get_grad(), 22.0);
            assert_eq!(c.get_grad(), 33.0);
        }
    }
}