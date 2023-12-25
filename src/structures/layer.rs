#![allow(dead_code)]
#![allow(clippy::new_ret_no_self)]

use crate::autograd::core::update_weights;
use crate::autograd::core::{RefValue, Value};
use crate::autograd::nonlinearity::NonLinearity;
use crate::structures::neuron::Neuron;
use crate::util::itermax::IterMaxExt;
use core::slice::Iter;
use std::fmt;

#[derive(Debug, Copy, Clone)]
pub enum LayerSpec {
    FullyConnected(u32),
    NonLinear(NonLinearity),
    SoftMax,
}

#[derive(Debug)]
pub struct Layer {
    ins: Vec<RefValue>,        // Input variables
    outs: Vec<RefValue>,       // Output variables
    neurons: Vec<Neuron>,      // Neurons
    parameters: Vec<RefValue>, // All parameters
}

pub struct LayerBuilder {
    ins: Vec<RefValue>,
    spec: LayerSpec,
    params: Option<Vec<Vec<f64>>>,
}

impl LayerBuilder {
    fn set_params(&mut self, params: Vec<Vec<f64>>) -> &mut Self {
        match self.spec {
            LayerSpec::FullyConnected(_) => {
                self.params = Some(params);
                self
            }
            _ => {
                panic!("Parameters can be set only for fully connected layer!");
            }
        }
    }

    fn nonlinear_layer(&self, nonlinearity: NonLinearity) -> Layer {
        let mut outs: Vec<RefValue> = Vec::with_capacity(self.ins.len());

        for i in self.ins.iter() {
            let out = match nonlinearity {
                NonLinearity::None => panic!("Non-linearity cannot be None!"),
                NonLinearity::ReLu => i.relu(),
                NonLinearity::Tanh => i.tanh(),
            };
            outs.push(out);
        }

        Layer {
            ins: self.ins.clone(),
            outs,
            neurons: vec![],
            parameters: vec![],
        }
    }

    fn softmax_layer(&self) -> Layer {
        let max: RefValue = self.ins.clone().into_iter().iter_max();
        let small_ins = self.ins.iter().map(|i| i.clone() - max.clone());

        let exps = small_ins.clone().map(|i| i.exp());
        let exp_sum = exps.clone().sum::<RefValue>();
        let exp_sum_inv = exp_sum.pow(Value::new(-1.0));

        let mut outs: Vec<RefValue> = Vec::with_capacity(self.ins.len());
        for i in small_ins {
            let out = i.exp() * exp_sum_inv.clone();
            outs.push(out);
        }
        Layer {
            ins: self.ins.clone(),
            outs,
            neurons: vec![],
            parameters: vec![],
        }
    }

    fn fully_connected_layer(&self, nout: u32) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(nout as usize);
        let mut outs: Vec<RefValue> = Vec::with_capacity(nout as usize);

        if let Some(params) = &self.params {
            for param in params.iter() {
                let neuron = Neuron::from_vec(self.ins.clone(), param.clone());
                outs.push(neuron.get_output_variable());
                neurons.push(neuron);
            }
        } else {
            for _ in 0..nout {
                let neuron = Neuron::new(self.ins.clone());
                outs.push(neuron.get_output_variable());
                neurons.push(neuron);
            }
        }

        let params = neurons
            .iter()
            .flat_map(|n| n.get_parameters())
            .cloned()
            .collect::<Vec<RefValue>>();

        Layer {
            ins: self.ins.clone(),
            outs,
            neurons,
            parameters: params,
        }
    }

    pub fn build(&mut self) -> Layer {
        match self.spec {
            LayerSpec::NonLinear(nonlin) => self.nonlinear_layer(nonlin),
            LayerSpec::SoftMax => self.softmax_layer(),
            LayerSpec::FullyConnected(nout) => self.fully_connected_layer(nout),
        }
    }
}

impl Layer {
    pub fn new(ins: Vec<RefValue>, spec: LayerSpec) -> LayerBuilder {
        LayerBuilder {
            ins,
            spec,
            params: None,
        }
    }

    pub fn update_weights(&self, rate: f64) {
        update_weights(&self.parameters, rate);
    }

    pub fn get_parameters(&self) -> Iter<RefValue> {
        self.parameters.iter()
    }

    pub fn get_out_variables(&self) -> Vec<RefValue> {
        self.outs.clone()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for n in 0..self.neurons.len() {
            write!(f, "{}", self.neurons[n])?;
        }
        Ok(())
    }
}

// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
//                                      Tests                                      //
// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod layers {
        use crate::autograd::core::*;
        use crate::structures::layer::*;

        #[test]
        fn basic() {
            let a = Value::new(-1.0);
            let b = Value::new(1.0);
            let l = Layer::new(vec![a, b], LayerSpec::FullyConnected(3))
                .set_params(vec![
                    vec![0.0, 5.0, 2.0],
                    vec![1.0, 3.0, 3.0],
                    vec![2.0, 1.0, 4.0],
                ])
                .build();

            let o = l.outs.clone().iter().map(|i| i.clone()).sum::<RefValue>();

            let top_sort = topological_sort(o.clone());
            forward(&top_sort);

            assert_eq!(3.0, o.get_data());
        }

        #[test]
        fn softmax1() {
            let a = Value::new(-1.0);
            let b = Value::new(1.0);
            let c = Value::new(2.0);
            let d = Value::new(0.5);
            let l = Layer::new(
                vec![a.clone(), b.clone(), c.clone(), d.clone()],
                LayerSpec::SoftMax,
            )
            .build();
            let o = l.outs[2].clone();

            let top_sort = topological_sort(o.clone());
            forward(&top_sort);
            assert_eq!(
                vec![
                    0.03034322855941622,
                    0.2242078180482011,
                    0.609460037598877,
                    0.1359889157935055
                ],
                l.outs.iter().map(|i| i.get_data()).collect::<Vec<f64>>()
            );

            backward(o.clone(), &top_sort);
            assert_eq!(a.get_grad(), -0.01849298521869313); // 0.030 * (0 - 0.609) = -0.018
            assert_eq!(b.get_grad(), -0.13664570521761885); // 0.224 * (0 - 0.609) = -0.136
            assert_eq!(c.get_grad(), 0.2380185001688524); // 0.609 * (1 - 0.609) =  0.238
            assert_eq!(d.get_grad(), -0.08287980973254037); // 0.135 * (0 - 0.609) = -0.082
        }
    }
}
