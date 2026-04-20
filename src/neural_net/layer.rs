use crate::math::{matrix::Matrix, vector::Vector};

#[derive(Debug)]
struct Layer {
    weights: Matrix,
    bias: Vector,
    activation: fn(f64) -> f64,
}
