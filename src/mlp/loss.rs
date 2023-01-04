use rand::Rng;
use rand_distr::Distribution;

use crate::core::core::{Value, RefValue};
use crate::core::core::{topological_sort, backward, forward};

// pub mod mlp;
use crate::mlp::mlp::MLP;


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