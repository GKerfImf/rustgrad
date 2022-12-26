#![allow(dead_code)]
// #![allow(unused_imports)]

pub mod core;
pub mod mlp;

use rand::Rng;

fn main() {

    let mlp = mlp::MLP::new(vec![2,5,5,1]);
    let loss = mlp::Loss::new(&mlp);

    let mut rng = rand::thread_rng();

    let f = 1000;
    for i in 0..100*f {
        let mut temp = vec![0.0,0.0,0.0];

        if i % f == 0 { 
            println!("{}", &mlp);
        }

        if rng.gen::<f64>() < 0.5 { 
            loss.train(&mlp, &vec![2.0, 2.0], &vec![1.0]);
            temp[0] = loss.loss.get_data();

            if i % f == 0 { 
                println!("{}", &mlp);
            }

        }
        if rng.gen::<f64>() < 0.5 { 
            loss.train(&mlp, &vec![-1.0, 3.0], &vec![4.0]);
            temp[1] = loss.loss.get_data();
            if i % f == 0 { 
                println!("{}", &mlp);
            }

        }
        if rng.gen::<f64>() < 0.5 { 
            loss.train(&mlp, &vec![-10.0, 10.0], &vec![-1.0]);
            temp[2] = loss.loss.get_data();
            if i % f == 0 { 
                println!("{}", &mlp);
            }
        }       
    }

    println!("{:?}", mlp.eval(&vec![2.0, 2.0]));
    println!("{:?}", mlp.eval(&vec![-1.0, 3.0]));
    println!("{:?}", mlp.eval(&vec![-10.0, 10.0]));
    // println!("{}", &mlp);

}