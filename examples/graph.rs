use nn_rust::neural_net::{
    data_point::DataPoint,
    functions::{Activation, OutputActivation},
    matrix::Vector,
    network::Network,
};

mod plotting;

use rand::RngExt;
use std::io::Write;
use std::{f64::consts::PI, time::Instant};

use rand::seq::SliceRandom;

fn main() {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = graph_dataset(256);
    let test_data: Vec<DataPoint> = graph_dataset(64);

    //Specify Network
    let layers_sizes: Vec<usize> = vec![1, 32, 16, 1];
    let activation: Activation = Activation::LeakyReLu;
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

        let mut train_score: Vec<f64> = network.train_network(&mut train_data, epochs, 32, 0.05);
        performance.append(&mut train_score);

        //Test Neural Network
        let test_score: f64 = network.test_network(&test_data);
        println!("Test Score: {}", test_score);
    }

    let seconds: f64 = start.elapsed().as_secs_f64();
    println!("Training took {:.3} seconds", seconds);
    println!();

    //Plot Neural Network Performance
    let _ = plotting::plot_performance(performance, String::from("graphs/graph_performance.png"));
    println!("Performance plot saved to graphs/graph_performance.png");

    //Plot Neural Network Graph
    let mut predictions: Vec<DataPoint> = vec![];

    for data in &train_data {
        let out = network.calc_network(&data.input);
        let pred_point: DataPoint = DataPoint {
            input: data.input.clone(),
            exp_output: out,
        };

        predictions.push(pred_point);
    }

    let _ = plotting::plot_2d_graph(
        &train_data,
        &predictions,
        String::from("graphs/graph_plot.png"),
    );
    println!("Graph plot saved to graphs/graph_plot.png");
    println!();

    println!("Program finished. Press Enter to exit...");
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}

fn graph_dataset(amount: i32) -> Vec<DataPoint> {
    let mut rng = rand::rng();
    let mut data: Vec<DataPoint> = vec![];

    for _ in 0..amount {
        let x: f64 = rng.random_range(-PI..PI);
        let y = x.sin();

        data.push(DataPoint {
            input: Vector::new(vec![x]),
            exp_output: Vector::new(vec![y]),
        });
    }

    data
}
