use std::fmt;
use std::clone::Clone;

#[derive(Debug,PartialEq,Clone,Copy)]
pub enum Op {
    Leaf,
    Add,
    Mul,
    ReLu,
    Tanh,
    Softmax
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Leaf => { write!(f, "") }
            Op::Add => { write!(f, "+") }
            Op::Mul => { write!(f, "*") }
            Op::ReLu => { write!(f, "ReLu") }
            Op::Tanh => { write!(f, "Tanh") }
            Op::Softmax => { write!(f, "Softmax") }
        }   
    }
}

impl Default for Op {
    fn default() -> Self { Op::Leaf }
}