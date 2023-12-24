#![allow(unused_imports)]
#![allow(dead_code)]

use rand::Rng;
use rand_distr::Distribution;
use crate::core::core::*;
use crate::mlp::mlp::MLP;

pub enum LossSpec {
    CrossEntropy,
    BinaryHinge,
    MultiHinge,
    Squared,
}

pub enum RegSpec {
    L2(f64)
}

#[derive(Debug)]
pub struct Loss {
    ins: Vec<RefValue>,         // Input variables

    mlp_outs: Vec<RefValue>,    // Output produced by MLP
    exp_outs: Vec<RefValue>,    // Expected output
    pub loss: RefValue,         //

    top_sort: Vec<RefValue>     //
}

pub struct LossBuilder<'a> {
    mlp: &'a MLP,
    loss_spec: Option<LossSpec>,
    reg_spec: Option<RegSpec>
}


impl <'a> LossBuilder<'_> {

    fn cross_entropy_loss(&self, exp_outs: Vec<RefValue>) -> RefValue {
        self.mlp.outs.iter().zip(exp_outs.iter())
            .map( |(sci,yi)| yi.clone() * sci.clone().log() )
            .sum::<RefValue>() * Value::new(-1.0)
    }

    fn with_binary_hinge_loss(&self, exp_outs: Vec<RefValue>) -> RefValue {
        self.mlp.outs.iter().zip(exp_outs.iter())
            .map( |(sci,yi)| (Value::new(-1.0) * sci.clone() * yi.clone() + Value::new(1.0)).relu() )
            .sum::<RefValue>()
    }

    fn with_multi_class_hinge_loss(&self, exp_outs: Vec<RefValue>) -> RefValue {
        // Assuming that expected outputs is a vector of 0s and 1s,
        // [o_star = oi * yi for (oi,yi) in zip(mlp.outs, exp_outs)]
        // is the value that [mlp] assignes to the correct class.
        let o_star = self.mlp.outs.iter().zip(exp_outs.iter())
            .map( |(oi,yi)| oi.clone() * yi.clone() ).sum::<RefValue>();

        // Term [1 - yi] allows to filter the correct class from the sum.
        // Term [max(0, 1 + oi - o_star)] is a standard multi-class hinge loss.
       self.mlp.outs.iter().zip(exp_outs.iter())
            .map( |(oi,yi)|
                (Value::new(1.0) - yi.clone()) * (Value::new(1.0) + oi.clone() - o_star.clone()).relu()
            ).sum::<RefValue>()
    }

    fn with_squared_loss(&self, exp_outs: Vec<RefValue>) -> RefValue {
        self.mlp.outs.iter().zip(exp_outs.iter())
            .map( |(sci,yi)| (sci.clone() - yi.clone()) * (sci.clone() - yi.clone()) )
            .sum::<RefValue>()
    }

    pub fn add_loss(&mut self, spec: LossSpec) -> &mut Self {
        if self.loss_spec.is_none() {
            self.loss_spec = Some(spec);
            self
        } else {
            panic!("Loss has been specified!")
        }
    }

    pub fn add_regularization(&mut self, spec: RegSpec) -> &mut Self {
        if self.reg_spec.is_none() {
            self.reg_spec = Some(spec);
            self
        } else {
            panic!("Regularization has been specified!")
        }
    }

    pub fn build(&mut self) -> Loss {
        let mut exp_outs: Vec<RefValue> = Vec::with_capacity(self.mlp.outs.len());
        for _ in 0..self.mlp.outs.len() {
            exp_outs.push(Value::new(0.0));
        }

        let data_loss = match self.loss_spec {
            None => Value::new(0.0),
            Some(LossSpec::CrossEntropy) => self.cross_entropy_loss(exp_outs.clone()),
            Some(LossSpec::BinaryHinge) => self.with_binary_hinge_loss(exp_outs.clone()),
            Some(LossSpec::MultiHinge) => self.with_multi_class_hinge_loss(exp_outs.clone()),
            Some(LossSpec::Squared) => self.with_squared_loss(exp_outs.clone()),
        };

        let reg_loss = match self.reg_spec {
            None => Value::new(0.0),
            Some(RegSpec::L2(c)) => {
                Value::new(c)
                    * self
                        .mlp
                        .get_parameters()
                        .iter()
                        .map(|rv| rv.clone() * rv.clone())
                        .sum::<RefValue>()
            }
        };

        let loss = data_loss + reg_loss;

        Loss {
            ins:        self.mlp.ins.clone(),
            mlp_outs:   self.mlp.outs.clone(),
            exp_outs:   exp_outs,
            loss:       loss.clone(),
            top_sort:   topological_sort(loss)
        }
    }
}

impl Loss {

    pub fn new(mlp: &MLP) -> LossBuilder {
        LossBuilder {
            mlp: mlp,
            loss_spec: None,
            reg_spec: None
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
    mod loss {
        use rand::Rng;

        use crate::core::core::*;
        use crate::mlp::layer::*;

        use crate::mlp::layer::LayerSpec::*;
        use crate::mlp::mlp::MLP;
        use crate::mlp::loss::{Loss, LossSpec};
        use crate::core::nonlinearity::NonLinearity::{Tanh, ReLu};

        // TODO: move to a separate module
        // TODO: remove duplication with tests::mnist::one_hot;
        fn one_hot(x: f64) -> Vec<f64> {
            let mut v = vec![0.0; 10];
            v[x as usize] = 1.0;
            return v
        }

        fn relative_error(a: f64, b: f64) -> f64 {
            if a == 0.0 && b == 0.0 {
                return 0.0
            } else {
                return (a - b).abs() / (a.abs()).max(b.abs())
            }
        }

        #[test]
        fn with_multi_class_hinge_loss() {
            let mut rng = rand::thread_rng();

            for _ in 0..100 {
                let nins = 1;
                let nouts = 5;
                let mlp =
                    MLP::new(nins)
                        .add_layer(FullyConnected(16)).add_layer(NonLinear(ReLu))
                        .add_layer(FullyConnected(16)).add_layer(NonLinear(ReLu))
                        .add_layer(FullyConnected(nouts))
                        .build();
                let loss = Loss::new(&mlp).add_loss(LossSpec::MultiHinge).build();


                let x = rng.gen::<f64>();
                let y = one_hot(rng.gen_range(0..=nouts) as f64);

                loss.compute_grads(&vec![x], &y);
                let grad_an = mlp.ins[0].get_grad();

                loss.compute_grads(&vec![x + 1e-6], &y);
                let a = loss.get_loss();
                loss.compute_grads(&vec![x - 1e-6], &y);
                let b = loss.get_loss();

                let grad_num = (a - b) / 2e-6;
                let relative_error = relative_error(grad_an, grad_num);

                assert!(relative_error < 1e-6);
            }
        }

        #[test]
        fn with_squared_loss() {
            let mut rng = rand::thread_rng();

            for _ in 0..100 {
                let nins = 1;
                let mlp =
                    MLP::new(nins)
                        .add_layer(FullyConnected(2)).add_layer(NonLinear(ReLu))
                        .add_layer(FullyConnected(4)).add_layer(NonLinear(Tanh))
                        .add_layer(FullyConnected(8)).add_layer(NonLinear(ReLu))
                        .add_layer(FullyConnected(16)).add_layer(NonLinear(ReLu))
                        .add_layer(FullyConnected(1))
                        .build();
                let loss = Loss::new(&mlp).add_loss(LossSpec::Squared).build();

                let x = rng.gen::<f64>();
                let y = rng.gen::<f64>();

                loss.compute_grads(&vec![x], &vec![y]);
                let grad_an = mlp.ins[0].get_grad();

                loss.compute_grads(&vec![x + 1e-6], &vec![y]);
                let a = loss.get_loss();
                loss.compute_grads(&vec![x - 1e-6], &vec![y]);
                let b = loss.get_loss();

                let grad_num = (a - b) / 2e-6;
                let relative_error = relative_error(grad_an, grad_num);

                assert!(relative_error < 1e-6);
            }
        }


        #[test]
        fn with_cross_entropy_loss() {
            let mut rng = rand::thread_rng();

            for _ in 0..100 {
                let nins = 1;
                let nouts = 5;
                let mlp =
                    MLP::new(nins)
                        .add_layer(FullyConnected(16)).add_layer(NonLinear(ReLu))
                        .add_layer(FullyConnected(16)).add_layer(NonLinear(ReLu))
                        .add_layer(FullyConnected(nouts)).add_layer(SoftMax)
                        .build();
                let loss = Loss::new(&mlp).add_loss(LossSpec::CrossEntropy).build();

                let x = rng.gen::<f64>();
                let y = one_hot(rng.gen_range(0..=nouts) as f64);

                loss.compute_grads(&vec![x], &y);
                let grad_an = mlp.ins[0].get_grad();

                loss.compute_grads(&vec![x + 1e-6], &y);
                let a = loss.get_loss();
                loss.compute_grads(&vec![x - 1e-6], &y);
                let b = loss.get_loss();

                let grad_num = (a - b) / 2e-6;
                let relative_error = relative_error(grad_an, grad_num);

                assert!(relative_error < 1e-6);
            }
        }


    }
}
