#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};
use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use rand::Rng;

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
impl Default for Op {
    fn default() -> Self { Op::Leaf }
}

// https://stackoverflow.com/a/71564648/8125485
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

// https://github.com/nrc/r4cppp/blob/master/graphs/src/rc_graph.rs
#[derive(Debug,Default)]
struct Value {
    id: u64,
    data: f64,
    grad: f64,
    op: Op,
    children: Vec<Rc<RefCell<Value>>> 
}
type RefValue = Rc<RefCell<Value>>;

impl Value { 
    fn new(data: f64) -> RefValue {
        log!("New [Leaf] node ID={}", NEXT_ID.load(Ordering::Relaxed));
        Rc::new(RefCell::new(
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: data, 
                ..Default::default() 
            }
        ))
    }
}

fn add(a: RefValue, b: RefValue) -> RefValue{ 
    log!("New [+] node ID={}", NEXT_ID.load(Ordering::Relaxed));
    log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), a.borrow().id);
    log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), b.borrow().id);

    return Rc::new(RefCell::new(
        Value { 
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: a.borrow().data + b.borrow().data,
            grad: 0.0,
            op: Op::Add,
            children: vec![a.clone(), b.clone()]
        }
    ))
}
fn mul(a: RefValue, b: RefValue) -> RefValue{ 
    log!("New [*] node ID={}", NEXT_ID.load(Ordering::Relaxed));
    log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), a.borrow().id);
    log!("  {} --> {}", NEXT_ID.load(Ordering::Relaxed), b.borrow().id);

    return Rc::new(RefCell::new(
        Value { 
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: a.borrow().data * b.borrow().data,
            grad: 0.0,
            op: Op::Mul,
            children: vec![a.clone(), b.clone()]
        }
    ))
}
fn relu(a: RefValue) -> RefValue { 
    return Rc::new(RefCell::new(
        Value { 
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            data: if a.borrow().data < 0.0 { 0.0 } else { a.borrow().data },
            grad: 0.0,
            op: Op::ReLu,
            children: vec![a.clone()]
        }
    ))
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

fn main() {

    let a = Value::new(12.0);
    let b = Value::new(-22.0);
    let variables = vec![a.clone(),b.clone()];

    let loss = add(mul(a.clone(),a.clone()), mul(b.clone(),b.clone()));

    for _ in 0..100 { 
        backward(loss.clone());
        update_weights(&variables);
        forward(loss.clone());
        println!("{}", loss.borrow().data);
    }
    println!("a={} b={}", a.borrow().data, b.borrow().data);
}
