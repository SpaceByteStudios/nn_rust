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
    pub der_act_func: fn(f64) -> f64,

    cached_x: Vector,
    cached_z: Vector,
    cached_a: Vector,

    weights_grad: Matrix,
    bias_grad: Vector,
}

impl Layer {
    pub fn new(
        size: usize,
        prev_size: usize,
        act_func: fn(f64) -> f64,
        der_act_func: fn(f64) -> f64,
    ) -> Self {
        let weights_data: Vec<f64> = (0..size * prev_size).map(|_| random()).collect();
        let weights: Matrix = Matrix::new(size, prev_size, weights_data);

        let bias_data: Vec<f64> = (0..size).map(|_| random()).collect();
        let bias: Vector = Vector::new(bias_data);

        Self {
            size,
            weights,
            bias,
            act_func,
            der_act_func,

            cached_x: Vector::zeros(prev_size),
            cached_z: Vector::zeros(size),
            cached_a: Vector::zeros(size),

            weights_grad: Matrix::zeros(size, prev_size),
            bias_grad: Vector::zeros(size),
        }
    }

    pub fn calc_layer(&mut self, input: &Vector) -> Vector {
        assert_eq!(self.weights.cols, input.data.len());

        let mut out: Vector = self.weights.mul_vector(&input);
        out = out.vecadd(&self.bias);

        self.cached_x = input.clone();
        self.cached_z = out.clone();

        for o in &mut out.data {
            *o = (self.act_func)(*o);
        }

        self.cached_a = out.clone();

        return out;
    }

    pub fn back_prop_layer(&mut self) {}
}
