use nn_rust::neural_net::{
    data_point::DataPoint,
    functions::{Activation, OutputActivation},
    matrix::Vector,
    network::Network,
};

use nn_rust::plotting;

use rand::RngExt;
use std::{f64::consts::PI, time::Instant};

use std::io::Write;

use rand::seq::SliceRandom;

fn main() {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = spiral_dataset(512);
    let test_data: Vec<DataPoint> = spiral_dataset(64);

    //Specify Network
    let layers_sizes: Vec<usize> = vec![2, 16, 16, 2];
    let activation: Activation = Activation::ReLu;
    let out_activation: OutputActivation = OutputActivation::Softmax;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, activation, out_activation);

    let iterations: usize = 10;
    let epochs: usize = 250;
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
    let _ = plotting::plot_performance(performance, String::from("graphs/spiral_performance.png"));
    println!("Performance plot saved to graphs/spiral_performance.png");

    //Plot Neural Network Classification
    let mut predictions: Vec<DataPoint> = vec![];

    for data in &train_data {
        let out = network.calc_network(&data.input);
        let pred_point: DataPoint = DataPoint {
            input: data.input.clone(),
            exp_output: out,
        };

        predictions.push(pred_point);
    }

    let _ = plotting::plot_2d_classification(
        &train_data,
        &predictions,
        String::from("graphs/spiral_plot.png"),
    );
    println!("Classification plot saved to graphs/spiral_plot.png");
    println!();

    println!("Program finished. Press Enter to exit...");
    let _ = std::io::stdout().flush();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}

fn spiral_dataset(amount: i32) -> Vec<DataPoint> {
    let mut rng = rand::rng();
    let mut data: Vec<DataPoint> = vec![];

    let noise: f64 = 0.3;
    let turns: f64 = 2.0;

    for i in 0..amount {
        let t = i as f64 / amount as f64;
        let theta = t * turns * 2.0 * PI;

        let r = theta;
        let x = r * theta.cos() + rng.random_range(-noise..noise);
        let y = r * theta.sin() + rng.random_range(-noise..noise);

        data.push(DataPoint {
            input: Vector::new(vec![x, y]),
            exp_output: Vector::new(vec![0.0, 1.0]),
        });

        let theta2 = theta + PI;
        let x2 = r * theta2.cos() + rng.random_range(-noise..noise);
        let y2 = r * theta2.sin() + rng.random_range(-noise..noise);

        data.push(DataPoint {
            input: Vector::new(vec![x2, y2]),
            exp_output: Vector::new(vec![1.0, 0.0]),
        });
    }

    data
}
