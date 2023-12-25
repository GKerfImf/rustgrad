use std::clone::Clone;
use std::fmt;

#[derive(Debug, Default, PartialEq, Clone, Copy)]
pub enum Op {
    #[default]
    Leaf,
    Exp,
    Log,
    Pow,
    Add,
    Mul,
    ReLu,
    Tanh,
    Max,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Leaf => write!(f, ""),
            Op::Exp => write!(f, "Exp"),
            Op::Log => write!(f, "Log"),
            Op::Pow => write!(f, "Pow"),
            Op::Add => write!(f, "+"),
            Op::Mul => write!(f, "*"),
            Op::ReLu => write!(f, "ReLu"),
            Op::Tanh => write!(f, "Tanh"),
            Op::Max => write!(f, "Max"),
        }
    }
}
