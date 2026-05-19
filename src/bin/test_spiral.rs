use nn_rust::neural_net::{
    data_point::DataPoint,
    functions::{Activation, OutputActivation},
    matrix::Vector,
    network::Network,
};

use rand::RngExt;
use std::{f64::consts::PI, time::Instant};

use rand::seq::SliceRandom;

fn main() {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = spiral_data(256);
    let test_data: Vec<DataPoint> = spiral_data(64);

    //Specify Network
    let layers_sizes: Vec<usize> = vec![2, 10, 1];
    let activation: Activation = Activation::Tanh;
    let out_activation: OutputActivation = OutputActivation::Linear;

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

        let mut train_score: Vec<f64> = network.train_network(&mut train_data, 67, 4, 0.01);
        performance.append(&mut train_score);

        //Test Neural Network
        let test_score: f64 = network.test_network(&test_data);
        println!("Test Score: {}", test_score);
    }

    let seconds: f64 = start.elapsed().as_secs_f64();
    println!("Training took {:.3} seconds", seconds);
    println!();
}

fn spiral_data(amount: i32) -> Vec<DataPoint> {
    let mut rng = rand::rng();
    let mut data: Vec<DataPoint> = vec![];

    let noise: f64 = 0.2;
    let turns: f64 = 1.0;

    for i in 0..amount {
        let t = i as f64 / amount as f64;
        let theta = t * turns * 2.0 * PI;

        let r = theta;
        let x = r * theta.cos() + rng.random_range(-noise..noise);
        let y = r * theta.sin() + rng.random_range(-noise..noise);

        data.push(DataPoint {
            input: Vector::new(vec![x, y]),
            exp_output: Vector::new(vec![0.0]),
        });

        let theta2 = theta + PI;
        let x2 = r * theta2.cos() + rng.random_range(-noise..noise);
        let y2 = r * theta2.sin() + rng.random_range(-noise..noise);

        data.push(DataPoint {
            input: Vector::new(vec![x2, y2]),
            exp_output: Vector::new(vec![1.0]),
        });
    }

    data
}
