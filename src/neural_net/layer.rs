use crate::{
    math::{matrix::Matrix, vector::Vector},
    neural_net::activation,
};
use rand::random;

#[derive(Debug)]
pub struct Layer {
    pub size: usize,
    pub weights: Matrix,
    pub bias: Vector,
    pub act_func: fn(f64) -> f64,
}

impl Layer {
    pub fn new(size: usize, prev_size: usize, act_func: fn(f64) -> f64) -> Self {
        let weights_data: Vec<f64> = (0..size * prev_size).map(|_| random()).collect();
        let weights: Matrix = Matrix::new(size, prev_size, weights_data);

        let bias_data: Vec<f64> = (0..size).map(|_| random()).collect();
        let bias: Vector = Vector::new(bias_data);

        Self {
            size,
            weights,
            bias,
            act_func,
        }
    }

    pub fn calc_layer(&self, input: &Vector) -> Vector {
        assert_eq!(self.weights.cols, input.data.len());

        let mut out: Vector = self.weights.mul_vector(&input);
        out = out.vecadd(&self.bias);

        for o in &mut out.data {
            *o = (self.act_func)(*o);
        }

        return out;
    }
}
