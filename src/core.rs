// TODO: split further

use std::{cell::RefCell, rc::Rc};
use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::ops::Deref;
use rand::Rng;
use std::ops::{Add,Sub,Mul};
use std::clone::Clone;
use std::iter::Sum;
use std::fmt;

// ------------------------------------------------
// https://stackoverflow.com/a/57955092/8125485

// Disable warnings
#[allow(unused_macros)]

// The debug version
#[cfg(debug_assertions)]
macro_rules! log {
    ($( $args:expr ),*) => { println!( $( $args ),* ); }
}

// Non-debug version
#[cfg(not(debug_assertions))]
macro_rules! log {
    ($( $args:expr ),*) => {()}
}

// ------------------------------------------------

#[derive(Debug,PartialEq,Clone,Copy)]
enum Op {
    Leaf,
    Add,
    Mul,
    ReLu,
    Tanh
}
impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Leaf => { write!(f, "") }
            Op::Add => { write!(f, "+") }
            Op::Mul => { write!(f, "*") }
            Op::ReLu => { write!(f, "ReLu") }
            Op::Tanh => { write!(f, "Tanh") }
        }   
    }
}

impl Default for Op {
    fn default() -> Self { Op::Leaf }
}

// https://stackoverflow.com/a/71564648/8125485
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

// https://github.com/nrc/r4cppp/blob/master/graphs/src/rc_graph.rs
#[derive(Default,Debug)]
pub struct Value {
    id: u64,
    pub data: f64,
    grad: f64,
    op: Op,
    children: Vec<RefValue> 
}
#[derive(Debug)]
pub struct RefValue(Rc<RefCell<Value>>);

impl Deref for RefValue {
    type Target = Rc<RefCell<Value>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Clone for RefValue {
    fn clone(&self) -> Self {
        RefValue(self.0.clone())
    }
}
impl Value { 
    pub fn new(data: f64) -> RefValue {
        // log!("New [Leaf] node ID={}", NEXT_ID.load(Ordering::Relaxed));
        RefValue(Rc::new(RefCell::new(
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: data, 
                ..Default::default() 
            }
        )))
    }
}
impl Add for RefValue {
    type Output = RefValue;

    fn add(self, other: RefValue) -> RefValue{
        // log!("New [+] node ID={}", NEXT_ID.load(Ordering::Relaxed));
        // log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), self.borrow().id);
        // log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), other.borrow().id);
    
        return RefValue(Rc::new(RefCell::new(
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: self.borrow().data + other.borrow().data,
                grad: 0.0,
                op: Op::Add,
                children: vec![self.clone(), other.clone()]
            }
        )))
    }
}
impl Mul for RefValue {
    type Output = RefValue;

    fn mul(self, other: RefValue) -> RefValue{
        // log!("New [*] node ID={}", NEXT_ID.load(Ordering::Relaxed));
        // log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), self.borrow().id);
        // log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), other.borrow().id);
    
        return RefValue(Rc::new(RefCell::new(
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: self.borrow().data * other.borrow().data,
                grad: 0.0,
                op: Op::Mul,
                children: vec![self.clone(), other.clone()]
            }
        )))
    }
}
impl Sub for RefValue {
    type Output = RefValue;

    fn sub(self, other: RefValue) -> RefValue {
        return self + Value::new(-1.0) * other; 
    }
}
fn relu(a: RefValue) -> RefValue { 
    // log!("New [ReLu] node ID={}", NEXT_ID.load(Ordering::Relaxed));
    // log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), a.borrow().id);
    
    return RefValue(Rc::new(RefCell::new(        
        Value { 
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: if a.borrow().data < 0.0 { 0.0 } else { a.borrow().data },
            grad: 0.0,
            op: Op::ReLu,
            children: vec![a.clone()]
        }
    )))
}
fn tanh(a: RefValue) -> RefValue { 
    // log!("New [Tanh] node ID={}", NEXT_ID.load(Ordering::Relaxed));
    // log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), a.borrow().id);
    
    return RefValue(Rc::new(RefCell::new(        
        Value { 
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: a.borrow().data.tanh(),
            grad: 0.0,
            op: Op::Tanh,
            children: vec![a.clone()]
        }
    )))
}

fn forwprop_node(value: RefValue) {
    let op = value.borrow().op;
    value.borrow_mut().grad = 0.0;

    match op { 
        Op::Leaf => { 
        }
        Op::Add => {
            let l_data = value.borrow().children[0].borrow().data; 
            let r_data = value.borrow().children[1].borrow().data; 
            
            value.borrow_mut().data = l_data + r_data;
        }
        Op::Mul => {
            let l_data = value.borrow().children[0].borrow().data; 
            let r_data = value.borrow().children[1].borrow().data; 
            
            value.borrow_mut().data = l_data * r_data;
        }
        Op::ReLu => { 
            let data = value.borrow().children[0].borrow().data;
            value.borrow_mut().data = if data < 0.0 { 0.0 } else { data }
        }
        Op::Tanh => {
            let data = value.borrow().children[0].borrow().data;
            value.borrow_mut().data = data.tanh();
        }
    }
}
fn backprop_node(value: RefValue) {
    match value.borrow().op { 
        Op::Leaf => { }
        Op::Add => {
            let grad = value.borrow().grad; 
            value.borrow().children[0].borrow_mut().grad += grad;
            value.borrow().children[1].borrow_mut().grad += grad;
        }
        Op::Mul => {
            let grad = value.borrow().grad;
            let l_data = value.borrow().children[0].borrow().data; 
            let r_data = value.borrow().children[1].borrow().data; 
            value.borrow().children[0].borrow_mut().grad += r_data * grad;
            value.borrow().children[1].borrow_mut().grad += l_data * grad;
        }
        Op::ReLu => { 
            let grad = value.borrow().grad;
            value.borrow().children[0].borrow_mut().grad += if grad > 0.0 { grad } else { 0.0 };
        }
        Op::Tanh => {
            let grad = value.borrow().grad;
            value.borrow().children[0].borrow_mut().grad += 1.0 - grad.tanh().powi(2);
        }
    }
}

fn top_sort(root: RefValue) -> Vec<RefValue>{
    let mut result = vec![];
    let mut visited = HashSet::new();

    fn dfs(result: &mut Vec<RefValue>, visited: &mut HashSet<u64>, value: RefValue) {
        if visited.contains(&value.borrow().id) { 
            return
        } 
        visited.insert(value.borrow().id);
        for ch in value.borrow().children.iter() {
            dfs(result, visited, ch.clone());
        }
        result.push(value.clone())
    }
    dfs(&mut result, &mut visited, root);
    return result
}
fn forward(nodes: &Vec<RefValue>) { 
    for node in nodes.iter() { 
        forwprop_node(node.clone());
    }
}
fn backward(root: RefValue, nodes: &Vec<RefValue>) { 
    root.borrow_mut().grad = 1.0;
    for node in nodes.iter().rev() {
        backprop_node(node.clone());
    }
}
fn update_weights(variables: &Vec<RefValue>) {
    let rate = 0.1; 

    for var in variables.iter() {
        let grad = var.borrow_mut().grad;
        var.borrow_mut().data -= rate * grad;
    }
}

// ------------------------------------------------

#[derive(Debug)]
struct Neuron {
    ins: Vec<RefValue>,     // Input variables          // TODO: should be [&Vec<RefValue>]
    out: RefValue,          // Output variable

    w: Vec<RefValue>,       // Weight variables
    b: RefValue,            // Bias variable
    nlin: bool              // Apply non-linearity (true/false)
}
impl Sum<RefValue> for RefValue {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = RefValue>,
    {
        let mut result = Value::new(0.0);
        for v in iter {
            result = result + v; // TODO: [_ = _ + _] --> [_ += _]
        }
        result
    }
}
impl Neuron { 
    fn new(ins: Vec<RefValue>, nlin: bool) -> Neuron { 
        let mut rng = rand::thread_rng();

        // Create a vector of weights and a bias variable
        let mut weights: Vec<RefValue> = Vec::with_capacity(ins.len() as usize);
        for _ in 0..ins.len() {
            // weights.push(Value::new(0.0));
            weights.push(Value::new(2.0* rng.gen::<f64>() - 1.0));
        }
        let bias: RefValue = Value::new(0.0);

        // [act = ins * weights + bias]
        let act = ins.iter().zip(weights.iter())
            .map( |(i,w)| i.clone() * w.clone() )
            .sum::<RefValue>() + bias.clone();

        // If [nlin = true], add Tanh non-linearity
        let out = if nlin { tanh(act) } else { act };

        Neuron { 
            ins: ins, out: out, 
            w: weights, b: bias, nlin: nlin       
        }
    }
    fn update_weights(&self) { 
        update_weights(&self.w);
        update_weights(&vec![self.b.clone()]);
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
    fn new(ins: Vec<RefValue>, nout: u32, nlin: bool) -> Layer {
        let mut neurons: Vec<Neuron> = Vec::with_capacity(nout as usize);
        let mut outs: Vec<RefValue> = Vec::with_capacity(nout as usize);

        for _ in 0..nout {
            let neuron = Neuron::new(ins.clone(), nlin);
            outs.push(neuron.out.clone());
            neurons.push(neuron);
        }
        Layer { ins: ins, outs: outs, neurons: neurons }
    }
    fn update_weights(&self) {
        for n in self.neurons.iter() { 
            n.update_weights()
        }
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
            let l = Layer::new(outs.clone(), lsizes[i], i != lsizes.len() - 1);
            outs = l.outs.clone();
            layers.push(l);
        }
        let uni_out = outs.clone().iter().map( |i| i.clone() ).sum::<RefValue>();  // TODO: improve? 
        let top_sort = top_sort(uni_out.clone());

        MLP { ins: ins, outs: outs, layers: layers, uni_out: uni_out, top_sort: top_sort }
    }
    fn update_weights(&self) {
        for l in self.layers.iter() { 
            l.update_weights()
        }
    }
    fn forward(&self, xs: &Vec<f64>) { 
        if xs.len() != self.ins.len() { 
            panic!("Number of inputs does not match!")
        }
        // Update input variables
        for (i,x) in self.ins.iter().zip(xs.iter()) { 
            i.borrow_mut().data = *x;
        }
        forward(&self.top_sort)
    }
    pub fn eval(&self, xs: &Vec<f64>) -> Vec<f64> { 
        self.forward(xs);
        return self.outs.iter().map( |rv| rv.borrow().data ).collect()
    }

    fn backward(&self) { 
        backward(self.uni_out.clone(), &self.top_sort)
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
    // TODO: implement batching
    // TODO: implement regularization

    pub fn new(mlp: &MLP) -> Loss { 
        let ins = mlp.ins.clone();

        let mlp_outs: Vec<RefValue> = mlp.outs.clone();
        let mut exp_outs: Vec<RefValue> = Vec::with_capacity(mlp.outs.len());
        for _ in 0..mlp.outs.len() {
            exp_outs.push(Value::new(0.0));
        }

        let loss = mlp_outs.iter().zip(exp_outs.iter())
            .map( |(sci,yi)| (sci.clone() - yi.clone()) * (sci.clone() - yi.clone()) )
            .sum::<RefValue>();

        let top_sort = top_sort(loss.clone());
    
        Loss { ins: ins, mlp_outs: mlp_outs, exp_outs: exp_outs, loss: loss, top_sort: top_sort }
    }

    pub fn train(&self, mlp: &MLP, xs: &Vec<f64>, ys: &Vec<f64>) { 
        if xs.len() != self.ins.len() { 
            panic!("Number of inputs does not match!")
        }
        if ys.len() != self.exp_outs.len() { 
            panic!("Number of outputs does not match!")
        }

        // Update input variables
        for (i,x) in self.ins.iter().zip(xs.iter()) { 
            i.borrow_mut().data = *x;
        }
        // Update output variables
        for (o,y) in self.exp_outs.iter().zip(ys.iter()) { 
            o.borrow_mut().data = *y;
        }
        forward(&self.top_sort);
        backward(self.loss.clone(), &self.top_sort);
        mlp.update_weights();
    }
}


