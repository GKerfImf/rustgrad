use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::fmt;
use core::slice::Iter;
use core::iter::{Once,Chain};
use std::iter;

use crate::core::nonlin::NonLin;
use crate::core::core::{Value, RefValue};
use crate::core::core::{topological_sort, backward, forward, update_weights};

#[derive(Debug)]
pub struct Neuron {
    ins: Vec<RefValue>,     // Input variables          // TODO: should be [&Vec<RefValue>]
    out: RefValue,          // Output variable

    w: Vec<RefValue>,       // Weight variables
    b: RefValue,            // Bias variable
    nlin: NonLin            // Apply non-linearity (None/ReLu/Tanh)
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
            w: weights, b: bias, nlin: nlin       
        }
    }

    pub fn with_rand_weights(ins: Vec<RefValue>, nlin: NonLin) -> Neuron {
        let mut rng = rand::thread_rng();
        let len = ins.len();
        let normal = Normal::new(0.0, 1.0).unwrap();
        return Neuron::new(
            ins, 
            (0..len).map( |_| normal.sample(&mut rand::thread_rng()) ).collect::<Vec<f64>>(),
            normal.sample(&mut rand::thread_rng()),
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
    fn new(ins: Vec<RefValue>, nout: u32, nlin: NonLin) -> Layer {
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
                if i == lsizes.len() - 1 { NonLin::None } else { NonLin::Tanh } );
            outs = l.outs.clone();
            layers.push(l);
        }
        let uni_out = outs.clone().iter().map( |i| i.clone() ).sum::<RefValue>();  // TODO: improve? 
        let top_sort = topological_sort(uni_out.clone());

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
                    NonLin::None => {}
                    nlin => { write!(f, " --> {} ", nlin)? }
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

    pub fn with_hinge_loss(mlp: &MLP) -> Loss {

        let mut exp_outs: Vec<RefValue> = Vec::with_capacity(mlp.outs.len());
        for _ in 0..mlp.outs.len() {
            exp_outs.push(Value::new(0.0));
        }

        let data_loss = mlp.outs.iter().zip(exp_outs.iter())
            .map( |(sci,yi)| (Value::new(-1.0) * sci.clone() * yi.clone() + Value::new(1.0)).relu() )
            .sum::<RefValue>();

        let reg_loss = mlp.get_parameters().iter().map( |rv| rv.clone() * rv.clone() ).sum::<RefValue>();
        let loss = data_loss + Value::new(0.001) * reg_loss; 

        Loss { 
            ins: mlp.ins.clone(), 
            mlp_outs: mlp.outs.clone(), 
            exp_outs: exp_outs, 
            loss: loss.clone(), 
            top_sort: topological_sort(loss) 
        }
    }

    pub fn with_squared_loss(mlp: &MLP) -> Loss {

        let mut exp_outs: Vec<RefValue> = Vec::with_capacity(mlp.outs.len());
        for _ in 0..mlp.outs.len() {
            exp_outs.push(Value::new(0.0));
        }

        let data_loss = mlp.outs.iter().zip(exp_outs.iter())
            .map( |(sci,yi)| (sci.clone() - yi.clone()) * (sci.clone() - yi.clone()) )
            .sum::<RefValue>();

        let reg_loss = mlp.get_parameters().iter().map( |rv| rv.clone() * rv.clone() ).sum::<RefValue>();
        let loss = data_loss + Value::new(0.001) * reg_loss; 

        Loss { 
            ins: mlp.ins.clone(), 
            mlp_outs: mlp.outs.clone(), 
            exp_outs: exp_outs, 
            loss: loss.clone(), 
            top_sort: topological_sort(loss) 
        }
    }

    pub fn get_loss(&self) -> f64 { 
        return self.loss.get_data()
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

    pub fn rand_batch_train(&self, mlp: &MLP, xss: &Vec<Vec<f64>>, yss: &Vec<Vec<f64>>, batch_size: u64, rate: f64) -> f64 {
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

        let mut rng = rand::thread_rng();
        for _ in 0..batch_size {
            let i = rng.gen_range(1..xss.len());
            self.compute_grads(&xss[i], &yss[i]);
        }
        mlp.update_weights(rate / batch_size as f64);
        return self.get_loss()
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
        use crate::mlp::Neuron;

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