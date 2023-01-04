use core::slice::Iter;

use crate::core::nonlin::NonLin;
use crate::core::core::RefValue;
use crate::core::core::update_weights;

use crate::mlp::neuron::Neuron;

#[derive(Debug)]
pub struct Layer {
    ins: Vec<RefValue>,             // Input variables
    pub outs: Vec<RefValue>,        // Output variables
    pub neurons: Vec<Neuron>,       // Neurons
    parameters: Vec<RefValue>       // All parameters
}

impl Layer {
    pub fn new(ins: Vec<RefValue>, nout: u32, nlin: NonLin) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(nout as usize);
        let mut outs: Vec<RefValue> = Vec::with_capacity(nout as usize);

        for _ in 0..nout {
            let neuron = Neuron::with_rand_weights(ins.clone(), nlin);
            outs.push(neuron.out.clone());
            neurons.push(neuron);
        }

        let params = neurons.iter().flat_map( |n| n.get_parameters() )
            .map( |rv| rv.clone() ).collect::<Vec<RefValue>>();
        Layer { ins: ins, outs: outs, neurons: neurons, parameters: params }
    }
    pub fn update_weights(&self, rate: f64) {
        update_weights(&self.parameters, rate);
    }
    
    pub fn get_parameters(&self) -> Iter<RefValue> {
        return self.parameters.iter()
    }
}


