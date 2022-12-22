#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};
use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::ops::Deref;
use rand::Rng;
use std::ops::{Add,Mul};
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
    ReLu
}
impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Leaf => { write!(f, "") }
            Op::Add => { write!(f, "+") }
            Op::Mul => { write!(f, "*") }
            Op::ReLu => { write!(f, "RL") }
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
struct Value {
    id: u64,
    data: f64,
    grad: f64,
    op: Op,
    children: Vec<RefValue> 
}
#[derive(Debug)]
struct RefValue(Rc<RefCell<Value>>);

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
    fn new(data: f64) -> RefValue {
        log!("New [Leaf] node ID={}", NEXT_ID.load(Ordering::Relaxed));
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
        log!("New [+] node ID={}", NEXT_ID.load(Ordering::Relaxed));
        log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), self.borrow().id);
        log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), other.borrow().id);
    
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
        log!("New [*] node ID={}", NEXT_ID.load(Ordering::Relaxed));
        log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), self.borrow().id);
        log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), other.borrow().id);
    
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
fn relu(a: RefValue) -> RefValue { 
    log!("New [ReLu] node ID={}", NEXT_ID.load(Ordering::Relaxed));
    log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), a.borrow().id);
    
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
            todo!()
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
fn forward(root: RefValue) { 
    let nodes = top_sort(root.clone());
    for node in nodes.iter() { 
        forwprop_node(node.clone());
    }
}
fn backward(root: RefValue) { 
    let nodes = top_sort(root.clone());
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
    ins: Vec<RefValue>, // Input variables
    out: RefValue,      // Output variable

    w: Vec<RefValue>,   // Weight variables
    b: RefValue,        // Bias variable
    nlin: bool          // Apply ReLU [true/false]
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
            weights.push(Value::new(2.0* rng.gen::<f64>() - 1.0));
        }
        let bias: RefValue = Value::new(0.0);

        // [act = ins * weights + bias]
        let act = ins.iter().zip(weights.iter())
            .map( |(i,w)| i.clone() * w.clone() )
            .sum::<RefValue>() + bias.clone();

        // If [nlin = true], add ReLu non-linearity
        let out = if nlin { relu(act) } else { act };

        Neuron { 
            ins: ins, out: out, 
            w: weights, b: bias, nlin: nlin       
        }
    }
    fn update_weights(&self) { 
        update_weights(&self.w);
        update_weights(&vec![self.b.clone()]);
    }

    // TODO: del 
    fn forward_temp(&self, xs: &Vec<f64>) { 
        // Update input variables
        for (i,x) in self.ins.iter().zip(xs.iter()) { 
            i.borrow_mut().data = *x;
        }
        // Run forward pass
        forward(self.out.clone());
    }
    // TODO: del 
    fn backward_temp(&self) { 
        backward(self.out.clone());
    }
}

fn main() {

    // Create a vector of input variables
    let nin = 3;
    let mut input: Vec<RefValue> = Vec::with_capacity(nin as usize);
    for _ in 0..nin {
        input.push(Value::new(0.0));
    }
    let n = Neuron::new(input, false);

    for _ in 0..100 { 
        n.forward_temp(&vec![1.0, 2.0, 3.0]);
        n.backward_temp();
        n.update_weights();
        println!("{:?}", n.out.clone().borrow().data);
    }
    println!("{:?}", n.w);
}
    }
    println!("a={} b={}", a.borrow().data, b.borrow().data);
}
