use nn_rust::neural_net::{
    data_point::DataPoint,
    functions::{Activation, OutputActivation},
    matrix::Vector,
    network::Network,
};

use nn_rust::plotting;

use rand::seq::SliceRandom;
use std::io::Write;
use std::time::Instant;

fn main() {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = xor_dataset();
    let test_data: Vec<DataPoint> = xor_dataset();

    //Specify Network
    let layers_sizes: Vec<usize> = vec![2, 8, 1];
    let activation: Activation = Activation::ReLu;
    let out_activation: OutputActivation = OutputActivation::Linear;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, activation, out_activation);

    let iterations: usize = 10;
    let epochs: usize = 25;
    println!(
        "Neural Network will train for {} epochs",
        iterations * epochs
    );
    println!();

    let test_score: f64 = network.test_network(&test_data);
    println!("Starting Score: {}", test_score);

    let start: Instant = Instant::now();
    let mut performance: Vec<f64> = vec![];

    for _ in 0..iterations {
        //Train Neural Network
        let mut rng = rand::rng();
        train_data.shuffle(&mut rng);

        let mut train_score: Vec<f64> = network.train_network(&mut train_data, epochs, 4, 0.05);
        performance.append(&mut train_score);

        //Test Neural Network
        let test_score: f64 = network.test_network(&test_data);
        println!("Test Score: {}", test_score);
    }

    let seconds: f64 = start.elapsed().as_secs_f64();
    println!("Training took {:.3} seconds", seconds);
    println!();

    let _ = plotting::plot_performance(performance, String::from("graphs/xor_performance.png"));
    println!("Performance plot saved to graphs/xor_performance.png");
    println!();

    println!("Program finished. Press Enter to exit...");
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
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
