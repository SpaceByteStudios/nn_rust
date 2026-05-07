use crate::neural_net::matrix::Vector;

#[derive(Debug, Clone)]
pub enum Activation {
    Linear,
    ReLu,
    LeakyReLu,
    Sigmoid,
    Tanh,
}

impl Activation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::Linear => x,
            Activation::ReLu => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
            Activation::LeakyReLu => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
        }
    }

    pub fn der_apply(&self, x: f64) -> f64 {
        match self {
            Activation::Linear => 1.0,
            Activation::ReLu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Sigmoid => {
                let s: f64 = self.apply(x);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - (x.tanh()).powi(2),
        }
    }

    pub fn apply_vec(&self, v: &Vector) -> Vector {
        let mut result: Vector = Vector::zeros(v.len());

        for i in 0..v.len() {
            result.set(i, self.apply(v.get(i)));
        }

        result
    }

    pub fn apply_vec_mut(&self, v: &mut Vector) {
        for i in 0..v.len() {
            v.set(i, self.apply(v.get(i)));
        }
    }

    pub fn der_apply_vec(&self, v: &Vector) -> Vector {
        let mut result: Vector = Vector::zeros(v.len());

        for i in 0..v.len() {
            result.set(i, self.der_apply(v.get(i)));
        }

        result
    }

    pub fn der_apply_vec_mut(&self, v: &mut Vector) {
        for i in 0..v.len() {
            v.set(i, self.der_apply(v.get(i)));
        }
    }
}
