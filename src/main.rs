#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::{
    math::vector::Vector,
    neural_net::{
        activation::{linear, relu, sigmoid, tanh},
        data_point::DataPoint,
        network::Network,
    },
};
use rand::random;

mod math;
mod neural_net;

fn main() {
    //Specify Training Data
    let train_data: Vec<DataPoint> = generate_data(10000);

    //Specify Test Data
    let test_data: Vec<DataPoint> = generate_data(1000);

    //Specify Layer Sizes
    let layers_sizes: Vec<usize> = vec![2, 2, 1];

    //Specify Activation Function
    let act_func: fn(f64) -> f64 = relu;

    //Create Neural Network
    let network: Network = Network::new(layers_sizes, act_func);

    //Test Calculation
    for i in 0..1 {
        let input: Vector = Vector::new(vec![random(), random()]);
        let test_out: Vector = network.calc_network(&input);

        println!("Input: {:#?}", input);
        println!("Output: {:#?}", test_out);
    }

    for (i, layer) in network.layers.iter().enumerate() {
        println!("Layer {}:", i + 1);
        println!("Weights: {:#?}", layer.weights);
        println!("Biases: {:#?}", layer.bias);
    }

    //Train Neural Network

    //Test Neural Network
}

fn generate_data(amount: i32) -> Vec<DataPoint> {
    let mut data: Vec<DataPoint> = vec![];

    for i in 0..amount {
        let num1: f64 = rand::random();
        let num2: f64 = rand::random();
        let out: f64 = 2.0 * num1 + 3.0 * num2;

        let input: Vector = Vector::new(vec![num1, num2]);
        let exp_output: Vector = Vector::new(vec![out]);

        data.push(DataPoint { input, exp_output });
    }

    return data;
}
