use std::fmt;
use std::clone::Clone;

#[derive(Debug,PartialEq,Clone,Copy)]
pub enum Op {
    Leaf,
    Exp,
    Add,
    Mul,
    ReLu,
    Tanh,
    Max
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Leaf => { write!(f, "") }
            Op::Exp => { write!(f, "Exp") }
            Op::Add => { write!(f, "+") }
            Op::Mul => { write!(f, "*") }
            Op::ReLu => { write!(f, "ReLu") }
            Op::Tanh => { write!(f, "Tanh") }
            Op::Max => { write!(f, "Max") }
        }
    }
}

impl Default for Op {
    fn default() -> Self { Op::Leaf }
}