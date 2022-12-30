use rand::Rng;
use std::fmt;
use core::slice::Iter;
use core::iter::{Once,Chain};
use std::iter;

use crate::nonlin::NonLin;
use crate::core::{Value, RefValue};
use crate::core::{top_sort, backward, forward, update_weights};

#[derive(Debug)]
pub struct Neuron {
    ins: Vec<RefValue>,     // Input variables          // TODO: should be [&Vec<RefValue>]
    out: RefValue,          // Output variable

    w: Vec<RefValue>,       // Weight variables
    b: RefValue,            // Bias variable
    nlin: Option<NonLin>    // Apply non-linearity (true/false)
}

impl Neuron { 
    pub fn new(ins: Vec<RefValue>, ws: Vec<f64>, b: f64, nlin: Option<NonLin>) -> Neuron { 
        if ins.len() != ws.len() { 
            panic!("Number of inputs does not match the number of weights!")
        }

        // Create a vector of weights and a bias variable
        let mut weights: Vec<RefValue> = Vec::with_capacity(ws.len());
        for i in 0..ws.len() {
            weights.push(Value::new(ws[i]));
        }
        let bias: RefValue = Value::new(b);

        // [act = ins * weights + bias]
        let act = ins.iter().zip(weights.iter())
            .map( |(i,w)| i.clone() * w.clone() )
            .sum::<RefValue>() + bias.clone();

        let out = match nlin { 
            None => act,
            Some(NonLin::ReLu) => act.tanh(),
            Some(NonLin::Tanh) => act.relu()
        };

        Neuron { 
            ins: ins, out: out, 
            w: weights, b: bias, nlin: nlin       
        }
    }

    pub fn with_rand_weights(ins: Vec<RefValue>, nlin: Option<NonLin>) -> Neuron {
        let mut rng = rand::thread_rng();
        let len = ins.len();
        return Neuron::new(
            ins, 
            (0..len).map( |_| 2.0 * rng.gen::<f64>() - 1.0 ).collect::<Vec<f64>>(),
            2.0 * rng.gen::<f64>() - 1.0,
            nlin
        )
    }
    
    pub fn get_weights(&self) -> Vec<f64> { 
        return self.w.iter().map( |rv| rv.get_data() ).collect::<Vec<f64>>()
    }

    pub fn get_bias(&self) -> f64 { 
        return self.b.get_data()
    }

    fn update_weights(&self, rate: f64) { 
        update_weights(&self.w, rate);
        update_weights(&vec![self.b.clone()], rate);
    }

    fn get_parameters(&self) -> Chain<Iter<RefValue>,Once<&RefValue>> { 
        return self.w.iter().chain(iter::once(&self.b)) //self.w.iter()
    }
}

#[derive(Debug)]
struct Layer {
    ins: Vec<RefValue>,         // Input variables
    outs: Vec<RefValue>,        // Output variables
    neurons: Vec<Neuron>        // Neurons
}


impl Layer { 
    // [ins]  -- vector of inputs
    // [nout] -- number of output variables, essentially the number of neurons
    // [nlin] -- Apply ReLU to [outs] (true/false)
    fn new(ins: Vec<RefValue>, nout: u32, nlin: Option<NonLin>) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(nout as usize);
        let mut outs: Vec<RefValue> = Vec::with_capacity(nout as usize);

        for _ in 0..nout {
            let neuron = Neuron::with_rand_weights(ins.clone(), nlin);
            outs.push(neuron.out.clone());
            neurons.push(neuron);
        }
        Layer { ins: ins, outs: outs, neurons: neurons }
    }
    fn update_weights(&self, rate: f64) {
        for n in self.neurons.iter() { 
            n.update_weights(rate)
        }
    }
    
    fn get_parameters(&self) -> Vec<RefValue> {
        return self.neurons.iter().flat_map( |n| n.get_parameters() )
                .map( |rv| rv.clone() ).collect::<Vec<RefValue>>();
    }
}

// MultiLayer Perceptron
#[derive(Debug)]
pub struct MLP {
    ins: Vec<RefValue>,     // Input variables
    outs: Vec<RefValue>,    // Output variables
    layers: Vec<Layer>,     // 

    uni_out: RefValue,      // TODO: remove?
    top_sort: Vec<RefValue> // 
}

impl MLP { 
    pub fn new(lsizes: Vec<u32>) -> MLP {
        let mut ins: Vec<RefValue> = Vec::with_capacity(lsizes[0] as usize);
        for _ in 0..lsizes[0] {
            ins.push(Value::new(0.0));
        }

        let mut layers: Vec<Layer> = Vec::with_capacity(lsizes.len());
        let mut outs = ins.clone(); 

        for i in 1..lsizes.len() { 
            let l = Layer::new(
                outs.clone(), 
                lsizes[i],
                if i == lsizes.len() - 1 { None } else { Some(NonLin::Tanh) } );
            outs = l.outs.clone();
            layers.push(l);
        }
        let uni_out = outs.clone().iter().map( |i| i.clone() ).sum::<RefValue>();  // TODO: improve? 
        let top_sort = top_sort(uni_out.clone());

        MLP { ins: ins, outs: outs, layers: layers, uni_out: uni_out, top_sort: top_sort }
    }
    fn update_weights(&self, rate: f64) {
        for l in self.layers.iter() { 
            l.update_weights(rate)
        }
    }
    fn get_parameters(&self) -> Vec<RefValue> {
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
                write!(f," [")?;
                for w in 0..l.neurons[0].w.len() { 
                    // write!(f, "{val:>8.3} [{grad:>8.3}, {bgrad:>8.3}]", val=l.neurons[n].w[w].get_data(),grad=l.neurons[n].w[w].get_grad(),bgrad=l.neurons[n].w[w].get_batch_grad())?;
                    write!(f, "{val:>8.3} ", val=l.neurons[n].w[w].get_data())?;
                }
                write!(f,"]")?;
                // write!(f, " + ({val:>8.3}) [{grad:>8.3}, {bgrad:>8.3}]", val=l.neurons[n].b.get_data(), grad=l.neurons[n].b.get_grad(),bgrad=l.neurons[n].b.get_batch_grad())?;
                write!(f, " + ({val:>8.3}) ", val=l.neurons[n].b.get_data())?;
                match l.neurons[n].nlin {
                    None => {}
                    Some(nlin) => { write!(f, " --> {} ", nlin)? }
                };
                write!(f, " ==> {} \n", l.neurons[n].out.get_data())?;
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


#[derive(Debug)]
pub struct Loss {
    ins: Vec<RefValue>,         // Input variables

    mlp_outs: Vec<RefValue>,    // 
    exp_outs: Vec<RefValue>,    // 
    pub loss: RefValue,             //

    top_sort: Vec<RefValue>     //
}
impl Loss {
    pub fn new(mlp: &MLP) -> Loss { 
        let ins = mlp.ins.clone();

        let mlp_outs: Vec<RefValue> = mlp.outs.clone();
        let mut exp_outs: Vec<RefValue> = Vec::with_capacity(mlp.outs.len());
        for _ in 0..mlp.outs.len() {
            exp_outs.push(Value::new(0.0));
        }

        // let data_loss = mlp_outs.iter().zip(exp_outs.iter())
        //     .map( |(sci,yi)| (sci.clone() - yi.clone()) * (sci.clone() - yi.clone()) )
        //     .sum::<RefValue>();
        
        let data_loss = mlp_outs.iter().zip(exp_outs.iter())
            .map( |(sci,yi)| (Value::new(-1.0) * sci.clone() * yi.clone() + Value::new(1.0)).relu() )
            .sum::<RefValue>();

        let reg_loss = mlp.get_parameters().iter().map( |rv| rv.clone() * rv.clone() ).sum::<RefValue>();
        let loss = data_loss + Value::new(0.001) * reg_loss; 

        Loss { ins: ins, mlp_outs: mlp_outs, exp_outs: exp_outs, loss: loss.clone(), top_sort: top_sort(loss) }
    }

    fn compute_grads(&self, xs: &Vec<f64>, ys: &Vec<f64>) { 
        // Update input variables
        for (i,x) in self.ins.iter().zip(xs.iter()) { 
            i.set_data(*x)
        }
        // Update output variables
        for (o,y) in self.exp_outs.iter().zip(ys.iter()) { 
            o.set_data(*y)
        }
        forward(&self.top_sort);
        backward(self.loss.clone(), &self.top_sort);
    }

    pub fn batch_train(&self, mlp: &MLP, xss: &Vec<Vec<f64>>, yss: &Vec<Vec<f64>>, rate: f64) {
        if xss.len() != yss.len() { 
            panic!("Number of inputs and outputs examples do not match!")
        }
        for xs in xss { 
            if xs.len() != self.ins.len() { 
                panic!("Number of inputs does not match!")
            }
        }
        for ys in yss { 
            if ys.len() != self.exp_outs.len() { 
                panic!("Number of outputs does not match!")
            }
        }

        for (xs,ys) in xss.iter().zip(yss.iter()) {
            self.compute_grads(&xs, &ys);
        }
        mlp.update_weights(rate / xss.len() as f64);
    }

}
