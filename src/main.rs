#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::{
    math::vector::Vector,
    neural_net::{
        activation::{der_relu, linear, relu, sigmoid, tanh},
        data_point::DataPoint,
        network::Network,
    },
};
use rand::{random, seq::SliceRandom};

mod math;
mod neural_net;

fn main() {
    //Specify Training Data
    let train_data: Vec<DataPoint> = generate_data(1024);

    //Specify Test Data
    let test_data: Vec<DataPoint> = generate_data(128);

    //Specify Layer Sizes
    let layers_sizes: Vec<usize> = vec![2, 1, 1];

    //Specify Activation Function
    let act_func: fn(f64) -> f64 = relu;
    let der_act_func: fn(f64) -> f64 = der_relu;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, act_func, der_act_func);

    //Test Calculation
    for i in 0..1 {
        //let input: Vector = Vector::new(vec![random(), random()]);
        let data: Vec<DataPoint> = generate_data(1);
        let test_out: Vector = network.calc_network(&data[0].input);

        println!("Input: {:#?}", data[0].input);
        println!("Output: {:#?}", test_out);

        println!("Expected Output: {:#?}", &data[0].exp_output);
        println!(
            "Cost: {}",
            network.calc_cost(&test_out, &data[0].exp_output)
        )
    }

    for i in 0..5 {
        //Train Neural Network
        let train_score: i32 = network.train_network(&train_data, 1);

        //Test Neural Network
        let test_score: i32 = network.test_network(&test_data);

        //Plot Neural Network output
    }
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

    let mut rng = rand::rng();
    data.shuffle(&mut rng);

    return data;
}
