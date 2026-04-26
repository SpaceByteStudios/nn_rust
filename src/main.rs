#![allow(dead_code)]
#![allow(unused_imports)]

use crate::{
    math::matrix::Matrix,
    neural_net::{
        activation::{
            der_leaky_relu, der_linear, der_relu, der_sigmoid, leaky_relu, linear, relu, sigmoid,
            tanh,
        },
        data_point::DataPoint,
        network::Network,
    },
};
use rand::seq::SliceRandom;
use std::time::Instant;

mod math;
mod neural_net;

fn main() {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = xor_dataset();
    let test_data: Vec<DataPoint> = xor_dataset();

    //Specify Layer Sizes
    let layers_sizes: Vec<usize> = vec![2, 3, 1];

    //Specify Activation Function
    let act_func: fn(f64) -> f64 = sigmoid;
    let der_act_func: fn(f64) -> f64 = der_sigmoid;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, act_func, der_act_func);

    let test_score: f64 = network.test_network(&test_data);
    println!("Starting Score: {}", test_score);

    let start: Instant = Instant::now();

    for _ in 0..10 {
        //Train Neural Network
        let _train_score: Vec<f64> = network.train_network(&mut train_data, 10000, 4, 0.1);

        //Test Neural Network
        let test_score: f64 = network.test_network(&test_data);
        println!("Test Score: {}", test_score);

        //Plot Neural Network output
    }

    let seconds: f64 = start.elapsed().as_secs_f64();
    println!("Training took {:.3} seconds", seconds);

    println!();
    println!("Testing Neural Network Performance");
    for data in &test_data {
        let pred: Matrix = network.calc_network(&data.input);

        println!(
            "pred: {:.5}, true: {:.5}",
            pred.data[0], data.exp_output.data[0]
        );
    }
}

fn generate_data(amount: i32) -> Vec<DataPoint> {
    let mut data: Vec<DataPoint> = vec![];

    for _ in 0..amount {
        let a: f64 = rand::random();
        let b: f64 = rand::random();

        let out: f64 = (a + b) - 2.0 * a * b;

        let input: Matrix = Matrix::new(2, 1, vec![a, b]);
        let exp_output: Matrix = Matrix::new(1, 1, vec![out]);

        data.push(DataPoint { input, exp_output });
    }

    data
}

fn xor_dataset() -> Vec<DataPoint> {
    let mut data: Vec<DataPoint> = vec![];

    data.push(DataPoint {
        input: Matrix::new(2, 1, vec![0.0, 0.0]),
        exp_output: Matrix::new(1, 1, vec![0.0]),
    });

    data.push(DataPoint {
        input: Matrix::new(2, 1, vec![1.0, 0.0]),
        exp_output: Matrix::new(1, 1, vec![1.0]),
    });

    data.push(DataPoint {
        input: Matrix::new(2, 1, vec![0.0, 1.0]),
        exp_output: Matrix::new(1, 1, vec![1.0]),
    });

    data.push(DataPoint {
        input: Matrix::new(2, 1, vec![1.0, 1.0]),
        exp_output: Matrix::new(1, 1, vec![0.0]),
    });

    data
}
