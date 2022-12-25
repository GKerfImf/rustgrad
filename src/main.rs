#![allow(dead_code)]

mod core;

fn main() {

    let mlp = core::MLP::new(vec![2,4,1]);
    let loss = core::Loss::new(&mlp);

    let mut acc = 100.0;
    while acc > 0.01 {
        let mut temp = vec![0.0,0.0];
        loss.train(&mlp, &vec![2.0, 2.0], &vec![1.0]);
        temp[0] = loss.loss.clone().borrow().data;
        loss.train(&mlp, &vec![-1.0, 3.0], &vec![4.0]);
        temp[1] = loss.loss.clone().borrow().data;

        acc = temp.iter().sum();
        println!("{:?} {}", temp, acc);
    }
    println!("{:?}", mlp.eval(&vec![2.0, 2.0]));    // ==> [1.0711436410627702]
    println!("{:?}", mlp.eval(&vec![-1.0, 3.0]));   // ==> [3.999999460143359]
}