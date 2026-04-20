use crate::math::{matrix::Matrix, vector::Vector};
use rand::random;

#[derive(Debug)]
pub struct Layer {
    pub size: usize,
    pub weights: Matrix,
    pub bias: Vector,
    pub activation: fn(f64) -> f64,
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
            activation: act_func,
        }
    }
}
