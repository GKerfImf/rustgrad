#![allow(dead_code)]
// #![allow(unused_imports)]
#![allow(unused_macros)]

pub mod core;
pub mod mlp;
use crate::mlp::mlp::MLP;
use crate::mlp::loss::Loss;

use rand::Rng;
use plotly::{Contour, HeatMap, Layout, Plot, Scatter};
use plotly::plot::ImageFormat;
use plotly::common::{ColorScale, ColorScalePalette, Title, Mode, Marker};
use plotly::common::color::{Color, NamedColor};
use plotly::contour::Contours;
use std::f64::consts::PI;
use std::io::{self, Write};

fn simple_plot() {
    let mut rng = rand::thread_rng();
    let mut generate_circle = |x_sh: f64, y_sh: f64, r_sm: f64, r_lr: f64| { 
        loop { 
            let r = r_sm + (r_lr - r_sm) * rng.gen::<f64>().sqrt();
            let theta = 2.0 * PI * rng.gen::<f64>();
            return (x_sh + r * theta.cos(), y_sh + r * theta.sin())
        }
    };


    // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
    //                                 Generate Input                                  //
    // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
    let mut x_train = Vec::<Vec<f64>>::new();
    let mut y_train = Vec::<Vec<f64>>::new();

    let n1 = 100;
    for _ in 0..n1 {
        let (x,y) = generate_circle(7.0, 0.0, 6.0, 6.0);
        x_train.push(vec![x,y]);
        y_train.push(vec![-1.0]);
    }
    let n2 = 100;
    for _ in 0..n2 {
        let (x,y) = generate_circle(-7.0, 0.0, 3.0, 3.0);
        x_train.push(vec![x,y]);
        y_train.push(vec![-1.0]);
    }
    let m1 = 100;
    for _ in 0..m1 {
        let (x,y) = generate_circle(7.0, 0.0, 3.0, 3.0);
        x_train.push(vec![x,y]);
        y_train.push(vec![1.0]);
    }
    let m2 = 100;
    for _ in 0..m2 {
        let (x,y) = generate_circle(-7.0, 0.0, 6.0, 6.0);
        x_train.push(vec![x,y]);
        y_train.push(vec![1.0]);
    }
   
    // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
    //                                  Train the MLP                                  //
    // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
    let mlp = MLP::new(vec![2,16,16,1]);
    let loss = Loss::with_hinge_loss(&mlp);

    let mut iterations = 0;
    let mut acc = 0.0;
    while acc < 1.0 {
        loss.rand_batch_train(&mlp, &x_train, &y_train, 32, 0.01); 
    
        acc = x_train.iter().zip(y_train.iter())
            .map( |(xs,ys)|  mlp.eval(&vec![xs[0], xs[1]])[0] * ys[0] )
            .filter( |y| y >= &0.0 )
            .count() as f64 / ((n1 + n2 + m1 + m2) as f64);
        print!("[{:>6.3} accuracy at {:>3}'th iteration ]\n", acc, { iterations += 1; iterations }); 
        io::stdout().flush().unwrap();
    }

    // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
    //                                   Plot Stuff                                    //
    // ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- //
    let n = 250;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut z: Vec<Vec<f64>> = Vec::new();

    for index in 0..n {
        let value = -20.0 + 40.0 * (index as f64) / (n as f64);
        x.push(value);
        y.push(value);
    }

    for xi in 0..n {
        let mut row = Vec::<f64>::new();
        for yi in 0..n {
            let zv = mlp.eval(&vec![y[yi], x[xi]])[0];
            row.push(zv);
        }
        z.push(row);
    }

    let mut plot = Plot::new();
    
    let trace = Contour::new(x,y,z)
        .auto_contour(false)
        .show_scale(false)
        .contours(Contours::new().start(-0.1).end(0.1));
    plot.add_trace(trace);

    let scatter_pos = Scatter::new(
            x_train.iter().zip(y_train.iter()).filter( |(_,y)| y[0] < 0.0 ).map( |(x,_)| x[0] ).collect::<Vec<f64>>(),
            x_train.iter().zip(y_train.iter()).filter( |(_,y)| y[0] < 0.0 ).map( |(x,_)| x[1] ).collect::<Vec<f64>>()
        )
        .name("Positive Example")
        .mode(Mode::Markers)
        .marker(Marker::new().size(10).color(NamedColor::Blue));
    plot.add_trace(scatter_pos);

    let scatter_neg = Scatter::new(
            x_train.iter().zip(y_train.iter()).filter( |(_,y)| y[0] > 0.0 ).map( |(x,_)| x[0] ).collect::<Vec<f64>>(),
            x_train.iter().zip(y_train.iter()).filter( |(_,y)| y[0] > 0.0 ).map( |(x,_)| x[1] ).collect::<Vec<f64>>()
        )
        .name("Negative Example")
        .mode(Mode::Markers)
        .marker(Marker::new().size(10).color(NamedColor::Red));
    plot.add_trace(scatter_neg);

    plot.show_image(ImageFormat::PNG, 1024, 680);    
    println!("{}", mlp);
}

fn main() {
    simple_plot();
}
