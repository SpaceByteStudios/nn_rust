use nn_rust::neural_net::{
    data_point::DataPoint,
    functions::{Activation, OutputActivation},
    matrix::Vector,
    network::Network,
};

use std::time::Instant;

use rand::seq::SliceRandom;

fn main() {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = xor_dataset();
    let test_data: Vec<DataPoint> = xor_dataset();

    //Specify Network
    let layers_sizes: Vec<usize> = vec![2, 10, 1];
    let activation: Activation = Activation::Tanh;
    let out_activation: OutputActivation = OutputActivation::Softmax;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, activation, out_activation);

    let test_score: f64 = network.test_network(&test_data);
    println!("Starting Score: {}", test_score);

    let start: Instant = Instant::now();
    let mut performance: Vec<f64> = vec![];

    for _ in 0..25 {
        //Train Neural Network
        let mut rng = rand::rng();
        train_data.shuffle(&mut rng);

        let mut train_score: Vec<f64> = network.train_network(&mut train_data, 100, 4, 0.01);
        performance.append(&mut train_score);

        //Test Neural Network
        let test_score: f64 = network.test_network(&test_data);
        println!("Test Score: {}", test_score);
    }

    let seconds: f64 = start.elapsed().as_secs_f64();
    println!("Training took {:.3} seconds", seconds);
    println!();
}

fn xor_dataset() -> Vec<DataPoint> {
    let data: Vec<DataPoint> = vec![
        DataPoint {
            input: Vector::new(vec![0.0, 0.0]),
            exp_output: Vector::new(vec![0.0]),
        },
        DataPoint {
            input: Vector::new(vec![1.0, 0.0]),
            exp_output: Vector::new(vec![1.0]),
        },
        DataPoint {
            input: Vector::new(vec![0.0, 1.0]),
            exp_output: Vector::new(vec![1.0]),
        },
        DataPoint {
            input: Vector::new(vec![1.0, 1.0]),
            exp_output: Vector::new(vec![0.0]),
        },
    ];

    data
}
