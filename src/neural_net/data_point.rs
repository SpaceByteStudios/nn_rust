use crate::neural_net::matrix::Matrix;

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub input: Matrix,
    pub exp_output: Matrix,
}

impl DataPoint {
    pub fn new(input: Matrix, exp_output: Matrix) -> Self {
        Self { input, exp_output }
    }
}
