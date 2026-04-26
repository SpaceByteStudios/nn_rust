#![allow(dead_code)]
#![allow(unused_imports)]

use crate::{
    math::matrix::Matrix,
    neural_net::{
        activation::{der_linear, der_relu, der_sigmoid, linear, relu, sigmoid, tanh},
        data_point::DataPoint,
        network::Network,
    },
};
use rand::seq::SliceRandom;

mod math;
mod neural_net;

fn main() {
    //Specify Training & Test Data
    let train_data: Vec<DataPoint> = generate_data(2048);
    let test_data: Vec<DataPoint> = generate_data(256);

    let test_data2: Vec<DataPoint> = generate_data(16);

    //Specify Layer Sizes
    let layers_sizes: Vec<usize> = vec![1, 2, 2, 1];

    //Specify Activation Function
    let act_func: fn(f64) -> f64 = relu;
    let der_act_func: fn(f64) -> f64 = relu;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, act_func, der_act_func);

    let test_score: f64 = network.test_network(&test_data);
    println!("Test Score: {}", test_score);

    for i in 0..1000 {
        //Train Neural Network
        let _train_score: Vec<f64> = network.train_network(&train_data, 1, 0.01);

        //Test Neural Network
        if i % 25 == 0 {
            let test_score: f64 = network.test_network(&test_data);
            println!("Test Score: {}", test_score);
        }

        //Plot Neural Network output
    }

    for data in &test_data2 {
        let pred: Matrix = network.calc_network(&data.input);

        println!("pred: {}, true: {}", pred.data[0], data.exp_output.data[0]);
    }
}

fn generate_data(amount: i32) -> Vec<DataPoint> {
    let mut data: Vec<DataPoint> = vec![];

    for _ in 0..amount {
        let num1: f64 = rand::random();
        //let num2: f64 = rand::random();
        //let out: f64 = 2.0 * num1 + 3.0 * num2;
        let out: f64 = 2.0 * num1;

        //let input: Matrix = Matrix::new(2, 1, vec![num1, num2]);
        let input: Matrix = Matrix::new(1, 1, vec![num1]);
        let exp_output: Matrix = Matrix::new(1, 1, vec![out]);

        data.push(DataPoint { input, exp_output });
    }

    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    data
}
