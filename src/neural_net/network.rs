use crate::{
    math::matrix::Matrix,
    neural_net::{
        activation::{der_linear, linear},
        data_point::DataPoint,
        layer::{self, Layer},
    },
};

use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(
        layers_sizes: Vec<usize>,
        act_func: fn(f64) -> f64,
        der_act_func: fn(f64) -> f64,
    ) -> Self {
        assert!(layers_sizes.len() >= 2);

        let mut layers: Vec<Layer> = vec![];

        for i in 1..layers_sizes.len() {
            let size: usize = layers_sizes[i];
            let prev_size: usize = layers_sizes[i - 1];

            let layer: Layer;

            if i == layers_sizes.len() - 1 {
                layer = Layer::new(size, prev_size, linear, der_linear);
            } else {
                layer = Layer::new(size, prev_size, act_func, der_act_func);
            }

            layers.push(layer);
        }

        Self { layers }
    }

    pub fn calc_network(&mut self, input: &Matrix) -> Matrix {
        assert_eq!(self.layers[0].weights.cols, input.data.len());

        let mut result: Matrix = input.clone();

        for layer in &mut self.layers {
            result = layer.calc_layer(&result);
        }

        let output: Matrix = result;

        output
    }

    pub fn calc_cost(&self, prediction: &Matrix, expected_out: &Matrix) -> f64 {
        assert_eq!(prediction.data.len(), expected_out.data.len());

        let mut cost: f64 = 0.0;

        for i in 0..prediction.data.len() {
            let pre_value: f64 = prediction.data[i];
            let exp_value: f64 = expected_out.data[i];
            cost += (pre_value - exp_value).powi(2);
        }

        cost /= prediction.data.len() as f64;

        cost
    }

    pub fn back_prop(&mut self, prediction: &Matrix, exp_output: &Matrix) {
        let mut data: Vec<f64> = vec![0.0; prediction.rows];

        for r in 0..data.len() {
            data[r] = 2.0 * (prediction.get(r, 0) - exp_output.get(r, 0));
        }

        let mut error_term: Matrix = Matrix::new(prediction.rows, 1, data);

        for layer in &mut self.layers.iter_mut().rev() {
            error_term = layer.back_prop_layer(&mut error_term);
        }
    }

    pub fn train_network(
        &mut self,
        train_data: &mut [DataPoint],
        epochs: usize,
        batch_size: usize,
        learn_rate: f64,
    ) -> Vec<f64> {
        let mut cost_history: Vec<f64> = vec![];

        for _ in 0..epochs {
            let mut e_cost: f64 = 0.0;

            let mut rng = rand::rng();
            train_data.shuffle(&mut rng);

            let batches = train_data.chunks_exact_mut(batch_size);

            for batch in batches {
                for data in batch.iter() {
                    //Forward Propagation
                    let prediction: Matrix = self.calc_network(&data.input);

                    //Calculate cost
                    let batch_cost: f64 = self.calc_cost(&prediction, &data.exp_output);
                    e_cost += batch_cost;

                    //Backwards Propagation
                    self.back_prop(&prediction, &data.exp_output);
                }

                //Update Parameters
                for layer in &mut self.layers {
                    layer.update_layer(batch_size, learn_rate);
                }
            }

            cost_history.push(e_cost / train_data.len() as f64);
        }

        cost_history
    }

    pub fn test_network(&mut self, test_data: &Vec<DataPoint>) -> f64 {
        let mut cost: f64 = 0.0;

        for data in test_data {
            let prediction: Matrix = self.calc_network(&data.input);

            cost += self.calc_cost(&prediction, &data.exp_output);
        }

        cost / test_data.len() as f64
    }
}
