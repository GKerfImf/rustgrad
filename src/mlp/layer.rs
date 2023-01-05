use core::slice::Iter;

use crate::core::nonlinearity::NonLinearity;
use crate::core::core::RefValue;
use crate::core::core::update_weights;

use crate::mlp::neuron::Neuron;

pub enum LayerSpec {
    FullyConnected(u32),
    NonLinear(NonLinearity)
}


#[derive(Debug)]
pub struct Layer {
    ins: Vec<RefValue>,             // Input variables
    pub outs: Vec<RefValue>,        // Output variables
    pub neurons: Vec<Neuron>,       // Neurons                  // TODO: remove?
    parameters: Vec<RefValue>       // All parameters
}

impl Layer {
    pub fn new_fully_connected(ins: Vec<RefValue>, nout: u32) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(nout as usize);
        let mut outs: Vec<RefValue> = Vec::with_capacity(nout as usize);

        for _ in 0..nout {
            let neuron = Neuron::with_rand_weights(ins.clone());
            outs.push(neuron.out.clone());
            neurons.push(neuron);
        }

        let params = neurons.iter().flat_map( |n| n.get_parameters() )
            .map( |rv| rv.clone() ).collect::<Vec<RefValue>>();

        Layer { ins: ins, outs: outs, neurons: neurons, parameters: params }
    }
    
    pub fn new_non_linearity(ins: Vec<RefValue>, nonlinearity: NonLinearity) -> Layer {
        let mut outs: Vec<RefValue> = Vec::with_capacity(ins.len());

        for i in ins.iter() {
            let out = match nonlinearity { 
                NonLinearity::None => { panic!("Non-linearity cannot be None!") },
                NonLinearity::ReLu => i.relu(),
                NonLinearity::Tanh => i.tanh()
            };
            outs.push(out);
        }

        Layer { ins: ins, outs: outs, neurons: vec![], parameters: vec![] }
    }
}

impl Layer {

    pub fn update_weights(&self, rate: f64) {
        update_weights(&self.parameters, rate);
    }
    
    pub fn get_parameters(&self) -> Iter<RefValue> {
        return self.parameters.iter()
    }

}