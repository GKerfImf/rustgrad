#![allow(unused_imports)]
#![allow(dead_code)]

use rand_distr::Distribution;
use std::fmt;

use crate::core::nonlinearity::NonLinearity;
use crate::core::core::{Value, RefValue};
use crate::core::core::{topological_sort, backward, forward};
use crate::mlp::layer::{LayerSpec, Layer};


// MultiLayer Perceptron
#[derive(Debug)]
pub struct MLP {
    pub ins: Vec<RefValue>,     // Input variables
    pub outs: Vec<RefValue>,    // Output variables
    layers: Vec<Layer>,         // 

    uni_out: RefValue,          // TODO: remove?
    top_sort: Vec<RefValue>     // 
}

impl MLP { 
    pub fn new(nins: u32, spec: Vec<LayerSpec>) -> MLP {
        let mut ins: Vec<RefValue> = Vec::with_capacity(nins as usize);
        (0..nins).for_each( |_| ins.push(Value::new(0.0)) );

        let mut layers: Vec<Layer> = Vec::with_capacity(spec.len());

        let mut outs = ins.clone();
        for lspec in spec.iter() {
            let l = Layer::new(outs.clone(), *lspec);
            outs = l.outs.clone();
            layers.push(l);
        }
        let uni_out = outs.clone().iter().map( |i| i.clone() ).sum::<RefValue>();
        let top_sort = topological_sort(uni_out.clone());

        MLP { ins: ins, outs: outs, layers: layers, uni_out: uni_out, top_sort: top_sort }
    }

    pub fn update_weights(&self, rate: f64) {
        for l in self.layers.iter() { 
            l.update_weights(rate)
        }
    }
    
    pub fn get_parameters(&self) -> Vec<RefValue> {
        return self.layers.iter().flat_map( |l| l.get_parameters() )
                .map( |rv| rv.clone() ).collect::<Vec<RefValue>>();
    }

    fn forward(&self, xs: &Vec<f64>) { 
        if xs.len() != self.ins.len() { 
            panic!("Number of inputs does not match!")
        }
        // Update input variables
        for (i,x) in self.ins.iter().zip(xs.iter()) { 
            i.set_data(*x)
        }
        forward(&self.top_sort)
    }
    pub fn eval(&self, xs: &Vec<f64>) -> Vec<f64> { 
        self.forward(xs);
        return self.outs.iter().map( |rv| rv.get_data() ).collect()
    }

    fn backward(&self) { 
        backward(self.uni_out.clone(), &self.top_sort)
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        write!(f, "================================================================\n")?;
        write!(f, "=                         State of MLP                         =\n")?;
        write!(f, "================================================================\n")?;

        // Print [ins]
        write!(f, "Inputs:\n")?;
        for i in self.ins.iter() { 
            write!(f, " [{val:>8.3}]\n", val=i.get_data())?;
        }
        write!(f, "\n")?;

        write!(f, "Weights:\n")?;
        for l in self.layers.iter() {
            for n in 0..l.neurons.len() {
                write!(f, "{}", l.neurons[n])?;
            }
            write!(f,"\n")?;
        }
        write!(f,"\n")?;

        // Print [outs]
        write!(f, "Outputs:\n")?;
        for o in self.outs.iter() { 
            write!(f, " [{name:>8.3}]\n", name=o.get_data())?;
        }
        return Ok(())
    }
}


// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
//                                      Tests                                      //
// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

// TODO: add tests