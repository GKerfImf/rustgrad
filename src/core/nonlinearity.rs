use std::fmt;

#[derive(Debug,PartialEq,Clone,Copy)]
pub enum NonLinearity {
    None,
    ReLu,
    Tanh
}

impl fmt::Display for NonLinearity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonLinearity::None => { write!(f, "") }
            NonLinearity::ReLu => { write!(f, "ReLu") }
            NonLinearity::Tanh => { write!(f, "Tanh") }
        }   
    }
}