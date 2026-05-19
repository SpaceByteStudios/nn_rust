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

#[derive(Debug, Clone)]
pub enum OutputActivation {
    Linear,
    Softmax,
}

impl OutputActivation {
    pub fn apply(&self, x: &Vector) -> Vector {
        match self {
            OutputActivation::Linear => x.clone(),
            OutputActivation::Softmax => {
                let mut v = x.clone();
                let n = x.len();

                let mut max = f64::NEG_INFINITY;
                for i in 0..n {
                    let val = v.get(i);
                    if val > max {
                        max = val;
                    }
                }

                let mut sum = 0.0;

                for i in 0..n {
                    let exp_val = (v.get(i) - max).exp();
                    v.set(i, exp_val);
                    sum += exp_val;
                }

                for i in 0..n {
                    let val = v.get(i) / sum;
                    v.set(i, val);
                }

                v
            }
        }
    }

    pub fn apply_mut(&self, x: &mut Vector) {
        match self {
            OutputActivation::Linear => {}
            OutputActivation::Softmax => {
                let n = x.len();

                let mut max = f64::NEG_INFINITY;
                for i in 0..n {
                    let val = x.get(i);
                    if val > max {
                        max = val;
                    }
                }

                let mut sum = 0.0;

                for i in 0..n {
                    let exp_val = (x.get(i) - max).exp();
                    x.set(i, exp_val);
                    sum += exp_val;
                }

                for i in 0..n {
                    let val = x.get(i) / sum;
                    x.set(i, val);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Loss {
    MeanSquaredError,
    CrossEntropy,
}

impl Loss {
    pub fn apply(&self, y: &Vector, p: &Vector) -> f64 {
        match self {
            Loss::MeanSquaredError => {
                let mut loss: f64 = 0.0;

                for i in 0..p.len() {
                    let pre_value: f64 = p.get(i);
                    let exp_value: f64 = y.get(i);
                    loss += (pre_value - exp_value).powi(2);
                }

                loss /= p.len() as f64;

                loss
            }

            Loss::CrossEntropy => {
                let mut loss: f64 = 0.0;

                for i in 0..p.len() {
                    let prediction = p.get(i).max(1e-12);

                    loss += y.get(i) * prediction.ln();
                }

                -loss
            }
        }
    }
}
