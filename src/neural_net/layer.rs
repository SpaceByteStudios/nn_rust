use crate::neural_net::matrix::Matrix;
use crate::neural_net::{activation::Activation, matrix::Vector};
use rand::{RngExt, rng};

#[derive(Debug)]
pub struct Layer {
    pub weights: Matrix,
    pub bias: Vector,
    pub activation: Activation,

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
            weights,
            bias,
            activation,

            cached_x: Vector::zeros(prev_size),
            cached_z: Vector::zeros(size),

            weights_grad: Matrix::zeros(size, prev_size),
            bias_grad: Vector::zeros(size),
        }
    }

    pub fn calc_layer(&mut self, input: &Vector) -> Vector {
        assert_eq!(self.weights.size()[1], input.len());

        let mut out: Vector = self.weights.mul_vec(input).add(&self.bias);

        self.cached_x = input.clone();
        self.cached_z = out.clone();

        for i in 0..out.len() {
            out.set(i, self.activation.apply(out.get(i)));
        }

        out
    }

    pub fn back_prop_layer(&mut self, error_term: &mut Vector) -> Vector {
        assert_eq!(error_term.len(), self.cached_z.len());

        let mut delta = Vector::zeros(error_term.len());

        for i in 0..error_term.len() {
            delta.set(
                i,
                error_term.get(i) * self.activation.der_apply(self.cached_z.get(i)),
            );
        }

        self.weights_grad = self.weights_grad.add(&delta.outer(&self.cached_x));
        self.bias_grad = self.bias_grad.add(&delta);

        let new_error_term: Vector = self.weights.transpose().mul_vec(&delta);
        new_error_term
    }

    pub fn update_layer(&mut self, data_amount: usize, learn_rate: f64) {
        let rate: f64 = -learn_rate / (data_amount) as f64;

        self.weights = self.weights.add(&self.weights_grad.scale(rate));
        self.bias = self.bias.add(&self.bias_grad.scale(rate));

        self.weights_grad = Matrix::zeros(self.weights.size()[0], self.weights.size()[1]);
        self.bias_grad = Vector::zeros(self.bias.len());
    }
}
