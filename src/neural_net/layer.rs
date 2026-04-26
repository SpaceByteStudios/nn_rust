use crate::math::matrix::Matrix;
use rand::{RngExt, rng};

#[derive(Debug)]
pub struct Layer {
    pub weights: Matrix,
    pub bias: Matrix,
    pub act_func: fn(f64) -> f64,
    pub der_act_func: fn(f64) -> f64,

    cached_x: Matrix,
    cached_z: Matrix,

    weights_grad: Matrix,
    bias_grad: Matrix,
}

impl Layer {
    pub fn new(
        size: usize,
        prev_size: usize,
        act_func: fn(f64) -> f64,
        der_act_func: fn(f64) -> f64,
    ) -> Self {
        let weights_data: Vec<f64> = (0..size * prev_size)
            .map(|_| rng().random_range(-1.0..1.0))
            .collect();
        let weights: Matrix = Matrix::new(size, prev_size, weights_data);

        let bias_data: Vec<f64> = (0..size).map(|_| rng().random_range(-1.0..1.0)).collect();
        let bias: Matrix = Matrix::new(size, 1, bias_data);

        Self {
            weights,
            bias,
            act_func,
            der_act_func,

            cached_x: Matrix::zeros(prev_size, 1),
            cached_z: Matrix::zeros(size, 1),

            weights_grad: Matrix::zeros(size, prev_size),
            bias_grad: Matrix::zeros(size, 1),
        }
    }

    pub fn calc_layer(&mut self, input: &Matrix) -> Matrix {
        assert_eq!(self.weights.cols, input.data.len());

        let mut out: Matrix = self.weights.matmul(&input);
        out = out.matadd(&self.bias);

        self.cached_x = input.clone();
        self.cached_z = out.clone();

        for o in &mut out.data {
            *o = (self.act_func)(*o);
        }

        out
    }

    pub fn back_prop_layer(&mut self, error_term: &mut Matrix) -> Matrix {
        assert_eq!(error_term.rows, self.cached_z.rows);
        assert_eq!(error_term.cols, 1);

        let mut delta = Matrix::zeros(error_term.rows, 1);

        for r in 0..error_term.rows {
            delta.data[r] = error_term.data[r] * (self.der_act_func)(self.cached_z.data[r]);
        }

        self.weights_grad = self
            .weights_grad
            .matadd(&delta.matmul(&self.cached_x.transpose()));
        self.bias_grad = self.bias_grad.matadd(&delta);

        let new_error_term: Matrix = self.weights.transpose().matmul(&delta);
        new_error_term
    }

    pub fn update_layer(&mut self, data_amount: usize, learn_rate: f64) {
        let rate: f64 = (-1.0 * learn_rate) / data_amount as f64;

        self.weights = self.weights.matadd(&self.weights_grad.scale(rate));
        self.bias = self.bias.matadd(&self.bias_grad.scale(rate));

        self.weights_grad = Matrix::zeros(self.weights.rows, self.weights.cols);
        self.bias_grad = Matrix::zeros(self.bias.rows, self.bias.cols);
    }
}
