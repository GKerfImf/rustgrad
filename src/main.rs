#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]

use plotly::{Contour, HeatMap, Layout, Plot, Scatter};
use plotly::common::{ColorScale, ColorScalePalette, Title, Mode, Marker};
use plotly::common::color::{Color, NamedColor};
use plotly::contour::Contours;
use std::f64::consts::PI;
use rand::Rng;

pub mod core;
pub mod mlp;

fn simple_plot() {
    let mut rng = rand::thread_rng();

    // Generate examples
    let n = 1;
    let mut examples = Vec::<(Vec<f64>, bool)>::new();
    for _ in 0..n {
        let x = 5.0 * rng.gen::<f64>() + 10.0;
        let y = 10.0 * rng.gen::<f64>();
        examples.push( (vec![x,y], true) );
    }
    let m = 0;
    for _ in 0..m {
        let x = 5.0 * rng.gen::<f64>() - 10.0;
        let y = 10.0 * rng.gen::<f64>();
        examples.push( (vec![x,y], false) );
    }

    // Train the MLP
    let mlp = mlp::MLP::new(vec![2,4,4,1]);
    let loss = mlp::Loss::new(&mlp);

    let mut rng = rand::thread_rng();
    for i in 0..100000 {
        let ex = rng.gen::<usize>() % (n + m);
        let ins = &examples[ex].0;
        let out = if examples[ex].1 { vec![10.0] } else { vec![-10.0] };
        loss.train(&mlp, &ins, &out, 0.01);
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
            let zv = mlp.eval(&vec![x[xi], y[yi]])[0];
            row.push(zv);
        }
        z.push(row);
    }
    
    let trace = Contour::new(x,y,z)
        .color_scale(ColorScale::Palette(ColorScalePalette::Jet))
        .auto_contour(false)
        .contours(Contours::new().start(-3.0).end(3.0));

    let layout = Layout::new().title(Title::new("Customizing Size and Range of Contours"));
    
    let mut plot = Plot::new();
    plot.set_layout(layout);
    plot.add_trace(trace);

    let scatter_pos = Scatter::new(
            examples.iter().filter( |cb| cb.1 ).map( |cb| cb.0[0]).collect::<Vec<f64>>(), 
            examples.iter().filter( |cb| cb.1 ).map( |cb| cb.0[1]).collect::<Vec<f64>>()
        )
        .name("Positive Example")
        .mode(Mode::Markers)
        .marker(Marker::new().size(10).color(NamedColor::Blue));

    let scatter_neg = Scatter::new(
            examples.iter().filter( |cb| !cb.1 ).map( |cb| cb.0[0]).collect::<Vec<f64>>(), 
            examples.iter().filter( |cb| !cb.1 ).map( |cb| cb.0[1]).collect::<Vec<f64>>()
        )
        .name("Negative Example")
        .mode(Mode::Markers)
        .marker(Marker::new().size(10).color(NamedColor::Red));
        
    plot.add_trace(scatter_pos);
    plot.add_trace(scatter_neg);
    plot.show();
    
    println!("{}", mlp);
}

fn main() {
    simple_plot();
}