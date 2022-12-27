use std::{cell::RefCell, rc::Rc};
use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::ops::Deref;
use std::ops::{Add,Sub,Mul};
use std::clone::Clone;
use std::fmt;
use std::iter::Sum;

// ------------------------------------------------
// https://stackoverflow.com/a/57955092/8125485

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
pub enum Op {
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
    data: f64,
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

pub fn relu(a: RefValue) -> RefValue { 
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
pub fn tanh(a: RefValue) -> RefValue { 
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
impl RefValue { 
    pub fn get_type(&self) -> Op { 
        return self.borrow().op
    }
    pub fn get_data(&self) -> f64 { 
        return self.borrow().data
    }
    pub fn set_data(&self, x: f64) { 
        self.borrow_mut().data = x
    }
    pub fn get_grad(&self) -> f64 { 
        return self.borrow().grad
    }
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
            let data = value.borrow().data;
            let grad = value.borrow().grad;
            value.borrow().children[0].borrow_mut().grad += if data > 0.0 { grad } else { 0.0 };
        }
        Op::Tanh => {
            let c_data = value.borrow().children[0].borrow().data; 
            let grad = value.borrow().grad;
            value.borrow().children[0].borrow_mut().grad += (1.0 - c_data.tanh().powi(2)) * grad;
        }
    }
}

pub fn top_sort(root: RefValue) -> Vec<RefValue>{
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
pub fn forward(nodes: &Vec<RefValue>) { 
    for node in nodes.iter() { 
        forwprop_node(node.clone());
    }
}
pub fn backward(root: RefValue, nodes: &Vec<RefValue>) { 
    root.borrow_mut().grad = 1.0;
    for node in nodes.iter().rev() {
        backprop_node(node.clone());
    }
}
pub fn update_weights(variables: &Vec<RefValue>) {
    let rate = 0.01; 

    for var in variables.iter() {
        let grad = var.borrow_mut().grad;
        var.borrow_mut().data -= rate * grad;
    }
}