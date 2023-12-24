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

pub struct MPLBuilder {
    nins: u32,
    spec: Vec<LayerSpec>,
}

impl MPLBuilder {

    pub fn add_layer(&mut self, spec: LayerSpec) -> &mut Self {
        self.spec.push(spec);
        self
    }

    pub fn build(&self) -> MLP {
        let mut ins: Vec<RefValue> = Vec::with_capacity(self.nins as usize);
        (0..self.nins).for_each(
            |_| ins.push(Value::new(0.0))
        );

        let mut layers: Vec<Layer> = Vec::with_capacity(self.spec.len());

        let mut outs = ins.clone();
        for lspec in self.spec.iter() {
            let l = Layer::new(outs.clone(), *lspec).build();
            outs = l.get_out_variables();
            layers.push(l);
        }
        let uni_out = outs.clone().iter().map( |i| i.clone() ).sum::<RefValue>();
        let top_sort = topological_sort(uni_out.clone());

        MLP { ins, outs, layers, uni_out, top_sort }
    }

}

impl MLP {

    pub fn new(nins: u32) -> MPLBuilder {
        MPLBuilder {
            nins,
            spec: vec![],
        }
    }

    pub fn update_weights(&self, rate: f64) {
        for l in self.layers.iter() {
            l.update_weights(rate)
        }
    }

    pub fn get_parameters(&self) -> Vec<RefValue> {
        self.layers.iter().flat_map( |l| l.get_parameters() )
                .map( |rv| rv.clone() ).collect::<Vec<RefValue>>()
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
        self.outs.iter().map( |rv| rv.get_data() ).collect()
    }

    fn backward(&self) {
        backward(self.uni_out.clone(), &self.top_sort)
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        writeln!(f, "================================================================")?;
        writeln!(f, "=                         State of MLP                         =")?;
        writeln!(f, "================================================================")?;

        // Print [ins]
        writeln!(f, "Inputs:")?;
        for i in self.ins.iter() { 
            writeln!(f, " [{val:>8.3}]", val=i.get_data())?;
        }
        writeln!(f)?;

        writeln!(f, "Weights:")?;
        for l in self.layers.iter() {
            write!(f, "{}", l)?;
            writeln!(f)?;
        }
        writeln!(f)?;

        // Print [outs]
        writeln!(f, "Outputs:")?;
        for o in self.outs.iter() { 
            writeln!(f, " [{name:>8.3}]", name=o.get_data())?;
        }
        Ok(())
    }
}


// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
//                                      Tests                                      //
// ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //

// TODO: add tests