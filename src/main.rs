#![allow(dead_code)]

use std::{cell::RefCell, rc::Rc};
use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::ops::Deref;
use rand::Rng;
use std::ops::{Add,Mul};
use std::clone::Clone;


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
#[derive(Default)]
struct Value {
    id: u64,
    data: f64,
    grad: f64,
    op: Op,
    children: Vec<RefValue> 
}
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

fn main() {

    let a = Value::new(12.0);
    let b = Value::new(-22.0);
    let variables = vec![a.clone(),b.clone()];

    let loss = a.clone() * a.clone() + b.clone() * b.clone(); 

    for _ in 0..100 { 
        backward(loss.clone());
        update_weights(&variables);
        forward(loss.clone());
        println!("{}", loss.borrow().data);
    }
    println!("a={} b={}", a.borrow().data, b.borrow().data);
}
