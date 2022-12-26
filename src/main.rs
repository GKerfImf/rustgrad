#![allow(dead_code)]
// #![allow(unused_imports)]

pub mod core;
pub mod mlp;

use rand::Rng;

fn main() {

    let mlp = mlp::MLP::new(vec![2,4,4,1]);
    let loss = mlp::Loss::new(&mlp);

    let mut rng = rand::thread_rng();
    let f = 100;
    for i in 0..10*f {
        
        if i % f == 0 {
            println!("{}", &mlp);
        }

        let mut temp = vec![0.0,0.0,0.0,0.0];
        if rng.gen::<f64>() < 0.5 { 
            loss.train(&mlp, &vec![2.0, 2.0], &vec![10.0]);
            temp[0] = loss.loss.get_data();
        }
        if rng.gen::<f64>() < 0.5 { 
            loss.train(&mlp, &vec![-1.0, 3.0], &vec![4.0]);
            temp[1] = loss.loss.get_data();
        }
        if rng.gen::<f64>() < 0.5 { 
            loss.train(&mlp, &vec![-10.0, 10.0], &vec![-10.0]);
            temp[2] = loss.loss.get_data();
        }       
        if rng.gen::<f64>() < 0.5 { 
            loss.train(&mlp, &vec![14.0, -10.0], &vec![-4.0]);
            temp[3] = loss.loss.get_data();
        }       
    }

    println!("{:?}", mlp.eval(&vec![2.0, 2.0])); //     ==> [ 99.227]
    println!("{:?}", mlp.eval(&vec![-1.0, 3.0])); //    ==> [ 39.491]
    println!("{:?}", mlp.eval(&vec![-10.0, 10.0])); //  ==> [-98.771]
    println!("{:?}", mlp.eval(&vec![14.0, -10.0])); //  ==> [-39.191]

}