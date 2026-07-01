use nn_rust::neural_net::{
    data_point::DataPoint,
    functions::{Activation, OutputActivation},
    matrix::Vector,
    network::Network,
};

use std::fs::File;
use std::io;
use std::io::{Read, Write};
use std::time::Instant;

use rand::seq::SliceRandom;

use flate2::read::GzDecoder;

fn main() -> std::io::Result<()> {
    //Specify Training & Test Data
    let mut train_data: Vec<DataPoint> = mnist_dataset(
        "data/train-images-idx3-ubyte.gz",
        "data/train-labels-idx1-ubyte.gz",
    )?;

    let test_data: Vec<DataPoint> = mnist_dataset(
        "data/t10k-images-idx3-ubyte.gz",
        "data/t10k-labels-idx1-ubyte.gz",
    )?;

    //Specify Network
    let layers_sizes: Vec<usize> = vec![784, 128, 64, 10];
    let activation: Activation = Activation::ReLu;
    let out_activation: OutputActivation = OutputActivation::Softmax;

    //Create Neural Network
    let mut network: Network = Network::new(layers_sizes, activation, out_activation);

    print!("For how many epochs should the Neural Network train (default 1): ");
    io::stdout().flush().unwrap();
    let default: i32 = 1;

    let mut input = String::new();

    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    let number: i32 = if input.trim().is_empty() {
        default
    } else {
        input.trim().parse().expect("Please enter a valid integer")
    };

    let iterations: usize = 1;
    let epochs: usize = number as usize;
    println!(
        "Neural Network will train for {} epochs",
        iterations * epochs
    );
    println!();

    let test_score: f64 = network.test_network(&test_data);
    println!("Starting Score: {}", test_score);

    let accuraccy = mnist_accuracy(&test_data, &mut network);
    println!("Neural Network is correct {:.2}%", accuraccy * 100.0);

    let start: Instant = Instant::now();
    let mut performance: Vec<f64> = vec![];

    for _ in 0..iterations {
        //Train Neural Network
        let mut rng = rand::rng();
        train_data.shuffle(&mut rng);

        let mut train_score: Vec<f64> = network.train_network(&train_data, epochs, 64, 0.05);
        performance.append(&mut train_score);

        //Test Neural Network
        let test_score: f64 = network.test_network(&test_data);
        println!("Test Score: {}", test_score);

        let accuraccy = mnist_accuracy(&test_data, &mut network);
        println!("Neural Network is correct {:.2}%", accuraccy * 100.0);

        if accuraccy > 0.95 {
            println!("Accuraccy high enough!");
            break;
        }
    }

    let seconds: f64 = start.elapsed().as_secs_f64();
    println!("Training and Testing took {:.3} seconds", seconds);
    println!();

    Ok(())
}

fn read_u32_be(data: &[u8], start: usize) -> u32 {
    u32::from_be_bytes([
        data[start],
        data[start + 1],
        data[start + 2],
        data[start + 3],
    ])
}

fn mnist_dataset(image_path: &str, label_path: &str) -> std::io::Result<Vec<DataPoint>> {
    let mut data: Vec<DataPoint> = vec![];

    //Open Image File and init decoder
    let image_file = File::open(image_path)?;
    let mut image_decoder = GzDecoder::new(image_file);

    //Read Image Data
    let mut image_data = Vec::new();
    image_decoder.read_to_end(&mut image_data)?;

    //Open Label File and init decoder
    let label_file = File::open(label_path)?;
    let mut label_decoder = GzDecoder::new(label_file);

    //Read Label Data
    let mut label_data = Vec::new();
    label_decoder.read_to_end(&mut label_data)?;

    let images_num = read_u32_be(&image_data, 4);
    let rows = read_u32_be(&image_data, 8);
    let cols = read_u32_be(&image_data, 12);

    data.reserve(images_num as usize);

    let image_size: usize = (rows * cols) as usize;

    let images = &image_data[16..];
    let labels = &label_data[8..];

    for i in 0..images_num as usize {
        let image = images[i * image_size..(i + 1) * image_size].to_vec();
        let label = labels[i];

        let data_input: Vec<f64> = image.iter().map(|p| *p as f64 / 255.0).collect();
        let mut data_exp_out: Vec<f64> = vec![0.0; 10];
        data_exp_out[label as usize] = 1.0;

        let image_data_point: DataPoint =
            DataPoint::new(Vector::new(data_input), Vector::new(data_exp_out));

        data.push(image_data_point);
    }

    Ok(data)
}

fn mnist_accuracy(test_data: &Vec<DataPoint>, network: &mut Network) -> f64 {
    let mut correct_rate: f64 = 0.0;

    for data in test_data {
        let p = network.calc_network(&data.input);
        let y = &data.exp_output;

        let mut max_p_index = 0;
        let mut max_y_index = 0;

        for i in 1..p.len() {
            if p.get(i) > p.get(max_p_index) {
                max_p_index = i;
            }
        }

        for i in 1..y.len() {
            if y.get(i) > y.get(max_y_index) {
                max_y_index = i;
            }
        }

        if max_p_index == max_y_index {
            correct_rate += 1.0;
        }
    }

    correct_rate /= test_data.len() as f64;
    correct_rate
}
