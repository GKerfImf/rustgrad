pub trait IterMaxExt: Iterator {
    fn iter_max<M>(self) -> M
    where
        M: IterMax<Self::Item>,
        Self: Sized,
    {
        M::iter_max(self)
    }
}

impl<I: Iterator> IterMaxExt for I {}

pub trait IterMax<A = Self> {
    fn iter_max<I>(iter: I) -> Self
    where
        I: Iterator<Item = A>;
}
