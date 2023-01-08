use core::slice::Iter;

use crate::core::nonlinearity::NonLinearity;
use crate::core::core::RefValue;
use crate::core::core::update_weights;

use crate::mlp::neuron::Neuron;

#[derive(Debug, Copy, Clone)]
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

    pub fn from_vec(ins: Vec<RefValue>, parameters: Vec<Vec<f64>>, lspec: LayerSpec) -> Layer {
        match lspec {
            LayerSpec::NonLinear(nonlin) => {
                // Non-linearity layer has no parameters, hence initialization
                // from a vector is the same as the default one
                return Layer::new_non_linearity(ins, nonlin)
            },
            LayerSpec::FullyConnected(n) => {
                if (parameters.len() as u32) != n {
                    panic!("Number of parameters does not match!")
                }
                return Layer::from_vec_fully_connected(ins, parameters)
            }
        }
    }

    pub fn new(ins: Vec<RefValue>, lspec: LayerSpec) -> Layer {
        return match lspec {
            LayerSpec::FullyConnected(n) => {
                Layer::new_fully_connected(ins, n)
            },
            LayerSpec::NonLinear(nonlin) => {
                Layer::new_non_linearity(ins, nonlin)
            }
        }
    }


    fn from_vec_fully_connected(ins: Vec<RefValue>, parameters: Vec<Vec<f64>>) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(parameters.len());
        let mut outs: Vec<RefValue> = Vec::with_capacity(parameters.len());

        for p in parameters.iter() {
            let neuron = Neuron::from_vec(ins.clone(), p.clone());
            outs.push(neuron.out.clone());
            neurons.push(neuron);
        }

        let params = neurons.iter().flat_map( |n| n.get_parameters() )
            .map( |rv| rv.clone() ).collect::<Vec<RefValue>>();

        Layer { ins: ins, outs: outs, neurons: neurons, parameters: params }
    }

    fn new_fully_connected(ins: Vec<RefValue>, nout: u32) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(nout as usize);
        let mut outs: Vec<RefValue> = Vec::with_capacity(nout as usize);

        for _ in 0..nout {
            let neuron = Neuron::new(ins.clone());
            outs.push(neuron.out.clone());
            neurons.push(neuron);
        }

        let params = neurons.iter().flat_map( |n| n.get_parameters() )
            .map( |rv| rv.clone() ).collect::<Vec<RefValue>>();

        Layer { ins: ins, outs: outs, neurons: neurons, parameters: params }
    }
    
    fn new_non_linearity(ins: Vec<RefValue>, nonlinearity: NonLinearity) -> Layer {
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
        
    fn new_softmax(ins: Vec<RefValue>) -> Layer {
        // Not implemented yet
        todo!()
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


// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
//                                      Tests                                      //
// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

#[cfg(test)]
mod tests {

    #[cfg(test)]
    mod layers {
        use crate::core::core::*;
        use crate::mlp::layer::*;

        #[test]
        fn basic() {
            let a = Value::new(-1.0);
            let b = Value::new( 1.0);
            let l = Layer::from_vec(vec![a,b],
                vec![
                    vec![0.0, 5.0,2.0],
                    vec![1.0, 3.0,3.0],
                    vec![2.0, 1.0,4.0],
                ],
                LayerSpec::FullyConnected(3)
            );

            let o = l.outs.clone().iter().map( |i| i.clone() ).sum::<RefValue>();

            let top_sort = topological_sort(o.clone());
            forward(&top_sort);

            assert_eq!(3.0, o.get_data());
        }

    }
}