use crate::math::vector::Vector;

#[derive(Debug)]
pub struct DataPoint {
    pub input: Vector,
    pub exp_output: Vector,
}

impl DataPoint {
    pub fn new(input: Vector, exp_output: Vector) -> Self {
        Self { input, exp_output }
    }
}
