pub fn linear(x: f64) -> f64 {
    x
}

pub fn der_linear(_: f64) -> f64 {
    1.0
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn der_relu(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn leaky_relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.1 * x }
}

pub fn der_leaky_relu(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.1 }
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn der_sigmoid(x: f64) -> f64 {
    let s: f64 = sigmoid(x);
    s * (1.0 - s)
}

pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

pub fn der_tanh(x: f64) -> f64 {
    1.0 - (x.tanh()).powi(2)
}
