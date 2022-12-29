#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use plotly::{Contour, HeatMap, Layout, Plot, Scatter};
use plotly::plot::ImageFormat;
use plotly::common::{ColorScale, ColorScalePalette, Title, Mode, Marker};
use plotly::common::color::{Color, NamedColor};
use plotly::contour::Contours;
use std::f64::consts::PI;
use rand::Rng;

pub mod op;
pub mod core;
pub mod mlp;

fn simple_plot() {
    let mut rng = rand::thread_rng();
    let mut generate_circle = |x_sh: f64, y_sh: f64, r_sm: f64, r_lr: f64| { 
        loop { 
            let r = r_sm + (r_lr - r_sm) * rng.gen::<f64>().sqrt();
            let theta = 2.0 * PI * rng.gen::<f64>();
            return (x_sh + r * theta.cos(), y_sh + r * theta.sin())
        }
    };

    // Generate examples
    let n1 = 50;
    let mut examples = Vec::<(Vec<f64>, bool)>::new();
    for _ in 0..n1 {
        let (x,y) = generate_circle(7.0, 0.0, 6.0, 6.0);
        examples.push( (vec![x,y], true) );
    }
    let n2 = 50;
    for _ in 0..n2 {
        let (x,y) = generate_circle(-7.0, 0.0, 3.0, 3.0);
        examples.push( (vec![x,y], true) );
    }
    let m1 = 50;
    for _ in 0..m1 {
        let (x,y) = generate_circle(7.0, 0.0, 3.0, 3.0);
        examples.push( (vec![x,y], false) );
    }
    let m2 = 50;
    for _ in 0..m2 {
        let (x,y) = generate_circle(-7.0, 0.0, 6.0, 6.0);
        examples.push( (vec![x,y], false) );
    }

    // Train the MLP
    let mlp = mlp::MLP::new(vec![2,10,10,1]);
    let loss = mlp::Loss::new(&mlp);

    let iterations = 1000;
    for i in 0..iterations {
        for ex in 0..(n1 + n2 + m1 + m2) { 
            let ins = &examples[ex].0;
            let out = if examples[ex].1 { vec![-10.0] } else { vec![10.0] };
            loss.train(&mlp, &ins, &out, 0.0005); 
        }
    }

    // Plot stuff 
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
            examples.iter().filter( |cb| cb.1 ).map( |cb| cb.0[0]).collect::<Vec<f64>>(), 
            examples.iter().filter( |cb| cb.1 ).map( |cb| cb.0[1]).collect::<Vec<f64>>()
        )
        .name("Positive Example")
        .mode(Mode::Markers)
        .marker(Marker::new().size(10).color(NamedColor::Blue));
    plot.add_trace(scatter_pos);

    let scatter_neg = Scatter::new(
            examples.iter().filter( |cb| !cb.1 ).map( |cb| cb.0[0]).collect::<Vec<f64>>(), 
            examples.iter().filter( |cb| !cb.1 ).map( |cb| cb.0[1]).collect::<Vec<f64>>()
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