use std::fmt;

#[derive(Debug,PartialEq,Clone,Copy)]
pub enum NonLin {
    None,
    ReLu,
    Tanh
}

impl fmt::Display for NonLin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonLin::None => { write!(f, "") }
            NonLin::ReLu => { write!(f, "ReLu") }
            NonLin::Tanh => { write!(f, "Tanh") }
        }   
    }
}