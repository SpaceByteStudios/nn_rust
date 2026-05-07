use crate::neural_net::{
    activation::Activation, data_point::DataPoint, layer::Layer, matrix::Vector,
};

use rand::seq::SliceRandom;

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Network {
    pub fn new(layers_sizes: Vec<usize>, activation: Activation) -> Self {
        assert!(layers_sizes.len() >= 2);

        let mut layers: Vec<Layer> = vec![];

        for i in 1..layers_sizes.len() {
            let size: usize = layers_sizes[i];
            let prev_size: usize = layers_sizes[i - 1];

            let layer: Layer = if i == layers_sizes.len() - 1 {
                Layer::new(size, prev_size, Activation::Linear)
            } else {
                Layer::new(size, prev_size, activation.clone())
            };

            layers.push(layer);
        }

        Self { layers }
    }

    pub fn calc_network(&mut self, input: &Vector) -> Vector {
        assert_eq!(self.layers[0].weights.size()[1], input.len());

        let mut result: Vector = input.clone();

        for layer in &mut self.layers {
            result = layer.calc_layer(&result);
        }

        result
    }

    pub fn calc_cost(&self, prediction: &Vector, expected_out: &Vector) -> f64 {
        assert_eq!(prediction.len(), expected_out.len());

        let mut cost: f64 = 0.0;

        for i in 0..prediction.len() {
            let pre_value: f64 = prediction.get(i);
            let exp_value: f64 = expected_out.get(i);
            cost += (pre_value - exp_value).powi(2);
        }

        cost /= prediction.len() as f64;

        cost
    }

    pub fn back_prop(&mut self, prediction: &Vector, expexted_out: &Vector) {
        let mut data: Vec<f64> = vec![0.0; prediction.len()];

        for (i, d) in data.iter_mut().enumerate() {
            *d = 2.0 * (prediction.get(i) - expexted_out.get(i));
        }

        let mut error_term: Vector = Vector::new(data);

        for layer in &mut self.layers.iter_mut().rev() {
            layer.back_prop_layer(&mut error_term);
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
                    let prediction: Vector = self.calc_network(&data.input);

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
            let prediction: Vector = self.calc_network(&data.input);

            cost += self.calc_cost(&prediction, &data.exp_output);
        }

        cost / test_data.len() as f64
    }
}
