use std::{cell::RefCell, rc::Rc};
use std::collections::HashSet;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::ops::Deref;
use std::ops::{Add,Sub,Mul};
use std::clone::Clone;
use std::iter::Sum;

use crate::core::op::Op;

// ------------------------------------------------

// https://stackoverflow.com/a/71564648/8125485
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

// https://github.com/nrc/r4cppp/blob/master/graphs/src/rc_graph.rs
#[derive(Default,Debug)]
pub struct Value {
    id: u64,
    data: f64,
    grad: f64,
    batch_grad: f64,
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
        return RefValue(Rc::new(RefCell::new(
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: self.borrow().data + other.borrow().data,
                grad: 0.0,
                batch_grad: 0.0,
                op: Op::Add,
                children: vec![self.clone(), other.clone()]
            }
        )))
    }
}
impl Mul for RefValue {
    type Output = RefValue;

    fn mul(self, other: RefValue) -> RefValue{
        return RefValue(Rc::new(RefCell::new(
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: self.borrow().data * other.borrow().data,
                grad: 0.0,
                batch_grad: 0.0,
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
    pub fn get_batch_grad(&self) -> f64 { 
        return self.borrow().batch_grad
    }

    pub fn relu(&self) -> RefValue { 
        return RefValue(Rc::new(RefCell::new(        
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: if self.borrow().data < 0.0 { 0.0 } else { self.borrow().data },
                grad: 0.0,
                batch_grad: 0.0,
                op: Op::ReLu,
                children: vec![self.clone()]
            }
        )))
    }

    pub fn tanh(&self) -> RefValue { 
        return RefValue(Rc::new(RefCell::new(        
            Value { 
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
                data: self.borrow().data.tanh(),
                grad: 0.0,
                batch_grad: 0.0,
                op: Op::Tanh,
                children: vec![self.clone()]
            }
        )))
    }

    fn update_grads(&self, grad: f64) {
        self.borrow_mut().grad += grad;
        self.borrow_mut().batch_grad += grad;
    }

    fn reset_grad(&self) {
        self.borrow_mut().grad = 0.0
    }

    fn reset_batch_grad(&self) { 
        self.borrow_mut().batch_grad = 0.0
    }

    fn evaluate_forward(&self) {
        let op = self.borrow().op;
        self.reset_grad();
    
        match op { 
            Op::Leaf => { }
            Op::Add => {
                let l_data = self.borrow().children[0].borrow().data; 
                let r_data = self.borrow().children[1].borrow().data; 
                self.borrow_mut().data = l_data + r_data;
            }
            Op::Mul => {
                let l_data = self.borrow().children[0].borrow().data; 
                let r_data = self.borrow().children[1].borrow().data; 
                self.borrow_mut().data = l_data * r_data;
            }
            Op::ReLu => { 
                let data = self.borrow().children[0].borrow().data;
                self.borrow_mut().data = if data < 0.0 { 0.0 } else { data }
            }
            Op::Tanh => {
                let data = self.borrow().children[0].borrow().data;
                self.borrow_mut().data = data.tanh();
            }
        }
    }

    fn evaluate_backward(&self) {
        match self.borrow().op { 
            Op::Leaf => { }
            Op::Add => {
                let grad = self.borrow().grad; 
                self.borrow().children[0].update_grads(grad);
                self.borrow().children[1].update_grads(grad);
            }
            Op::Mul => {
                let grad = self.borrow().grad;
                let l_data = self.borrow().children[0].borrow().data; 
                let r_data = self.borrow().children[1].borrow().data; 
                self.borrow().children[0].update_grads(r_data * grad);
                self.borrow().children[1].update_grads(l_data * grad);
            }
            Op::ReLu => { 
                let data = self.borrow().data;
                let grad = self.borrow().grad;
                self.borrow().children[0].update_grads(if data > 0.0 { grad } else { 0.0 });
            }
            Op::Tanh => {
                let c_data = self.borrow().children[0].borrow().data; 
                let grad = self.borrow().grad;
                self.borrow().children[0].update_grads((1.0 - c_data.tanh().powi(2)) * grad);
            }
        }
    }
}

// ------------------------------------------------

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
        node.evaluate_forward();
    }
}

pub fn backward(root: RefValue, nodes: &Vec<RefValue>) { 
    root.borrow_mut().grad = 1.0;
    for node in nodes.iter().rev() {
        node.evaluate_backward();
    }
}

pub fn update_weights(variables: &Vec<RefValue>, rate: f64) {
    for var in variables.iter() {
        let batch_grad = var.borrow_mut().batch_grad;
        var.borrow_mut().data -= rate * batch_grad;
        var.reset_grad();
        var.reset_batch_grad();
    }
}