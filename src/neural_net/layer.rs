#![allow(dead_code)]

use crate::neural_net::matrix::Matrix;
use crate::neural_net::{functions::Activation, matrix::Vector};
use rand::{RngExt, rng};

#[derive(Debug)]
pub struct Layer {
    size: usize,
    prev_size: usize,

    weights: Matrix,
    bias: Vector,
    activation: Activation,

    cached_x: Vector,
    cached_z: Vector,

    weights_grad: Matrix,
    bias_grad: Vector,
}

impl Layer {
    pub fn new(size: usize, prev_size: usize, activation: Activation) -> Self {
        let weights_data: Vec<f64> = (0..size * prev_size)
            .map(|_| rng().random_range(-1.0..1.0))
            .collect();
        let weights: Matrix = Matrix::new(size, prev_size, weights_data);

        let bias_data: Vec<f64> = (0..size).map(|_| rng().random_range(-1.0..1.0)).collect();
        let bias: Vector = Vector::new(bias_data);

        Self {
            size,
            prev_size,
            weights,
            bias,
            activation,

            cached_x: Vector::zeros(prev_size),
            cached_z: Vector::zeros(size),

            weights_grad: Matrix::zeros(size, prev_size),
            bias_grad: Vector::zeros(size),
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn prev_size(&self) -> usize {
        self.prev_size
    }

    pub fn calc_layer(&mut self, input: &Vector) -> Vector {
        assert_eq!(self.weights.size()[1], input.len());

        let mut out: Vector = self.weights.mul_vec(input);
        out.add_mut(&self.bias);

        self.cached_x = input.clone();
        self.cached_z = out.clone();

        self.activation.apply_vec_mut(&mut out);

        out
    }

    //Change back propagated error term and calculate gradients
    pub fn back_prop_layer(&mut self, error_term: &mut Vector) {
        assert_eq!(error_term.len(), self.cached_z.len());

        self.activation.der_apply_vec_mut(&mut self.cached_z);
        error_term.elem_mul_mut(&self.cached_z);

        self.bias_grad.add_mut(error_term);
        self.weights_grad.add_mut(&error_term.outer(&self.cached_x));

        *error_term = self.weights.transpose().mul_vec(error_term);
    }

    //Update the weights and biases based on gradients with learn rate
    pub fn update_layer(&mut self, data_amount: usize, learn_rate: f64) {
        let rate: f64 = -learn_rate / (data_amount) as f64;

        self.weights_grad.scale_mut(rate);
        self.weights.add_mut(&self.weights_grad);

        self.bias_grad.scale_mut(rate);
        self.bias.add_mut(&self.bias_grad);

        self.weights_grad = Matrix::zeros(self.weights.size()[0], self.weights.size()[1]);
        self.bias_grad = Vector::zeros(self.bias.len());
    }
}
