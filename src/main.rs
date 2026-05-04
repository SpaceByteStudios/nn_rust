#![allow(dead_code)]
#![allow(unused_imports)]

use crate::{
    math::matrix::Matrix,
    neural_net::{
        activation::{
            der_leaky_relu, der_linear, der_relu, der_sigmoid, der_tanh, leaky_relu, linear, relu,
            sigmoid, tanh,
        },
        data_point::{self, DataPoint},
        network::Network,
    },
};
use rand::seq::SliceRandom;
use std::{f64::consts::PI, time::Instant};

use plotters::prelude::*;

mod math;
mod neural_net;

fn main() {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = generate_data(128);
    let test_data: Vec<DataPoint> = generate_data(16);

    //Specify Layer Sizes
    let layers_sizes: Vec<usize> = vec![1, 10, 1];

    //Specify Activation Function
    let act_func: fn(f64) -> f64 = tanh;
    let der_act_func: fn(f64) -> f64 = der_tanh;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, act_func, der_act_func);

    let test_score: f64 = network.test_network(&test_data);
    println!("Starting Score: {}", test_score);

    let start: Instant = Instant::now();
    let mut performance: Vec<f64> = vec![];

    for _ in 0..10 {
        //Train Neural Network
        let mut train_score: Vec<f64> = network.train_network(&mut train_data, 250, 8, 0.01);
        performance.append(&mut train_score);

        //Test Neural Network
        let test_score: f64 = network.test_network(&test_data);
        println!("Test Score: {}", test_score);
    }

    let seconds: f64 = start.elapsed().as_secs_f64();
    println!("Training took {:.3} seconds", seconds);
    println!();

    //Plot Neural Network Performance
    let _ = plot_performance(performance);

    let mut predictions: Vec<DataPoint> = vec![];

    for data in &train_data {
        let out = network.calc_network(&data.input);
        let pred_point: DataPoint = DataPoint {
            input: data.input.clone(),
            exp_output: out,
        };

        predictions.push(pred_point);
    }

    let _ = plot_2d_graph(&train_data, &predictions);
}

fn generate_data(amount: i32) -> Vec<DataPoint> {
    let mut data: Vec<DataPoint> = vec![];

    for _ in 0..amount {
        let mut a: f64 = rand::random();

        a = (a - 0.5) * PI * 2.0;

        let out: f64 = a.sin();

        let input: Matrix = Matrix::new(1, 1, vec![a]);
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

fn plot_performance(performance: Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("output.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min: f64 = performance.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max: f64 = performance
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Neural Network Cost", ("sans-serif", 20))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0..performance.len() as i32, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Epochs")
        .y_desc("Cost")
        .draw()?;

    chart.draw_series(LineSeries::new(
        performance.iter().enumerate().map(|(i, y)| (i as i32, *y)),
        &RED,
    ))?;

    root.present()?;

    println!("Plot saved to output.png");
    Ok(())
}

fn plot_2d_graph(
    train_data: &Vec<DataPoint>,
    predictions: &Vec<DataPoint>,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("graph.png", (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Neural Network Cost", ("sans-serif", 20))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(-PI..PI, -1.0..1.0)?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        train_data
            .iter()
            .map(|x| Circle::new((x.input.data[0], x.exp_output.data[0]), 5, BLUE.filled())),
    )?;

    chart.draw_series(
        predictions
            .iter()
            .map(|x| Circle::new((x.input.data[0], x.exp_output.data[0]), 5, RED.filled())),
    )?;

    root.present()?;

    println!("Plot saved to graph.png");
    Ok(())
}
