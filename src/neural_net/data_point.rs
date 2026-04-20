use crate::math::vector::Vector;

#[derive(Debug)]
pub struct DataPoint {
    input: Vector,
    exp_output: Vector,
}

impl DataPoint {
    fn new(input: Vector, exp_output: Vector) -> Self {
        Self { input, exp_output }
    }
}
