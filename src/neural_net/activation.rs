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
}
