use crate::{
    math::vector::Vector,
    neural_net::{
        data_point::DataPoint,
        layer::{self, Layer},
    },
};

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub act_func: fn(f64) -> f64,
}

impl Network {
    pub fn new(layers_sizes: Vec<usize>, act_func: fn(f64) -> f64) -> Self {
        assert!(layers_sizes.len() >= 2);

        let mut layers: Vec<Layer> = vec![];

        for i in 1..layers_sizes.len() {
            let size: usize = layers_sizes[i];
            let prev_size: usize = layers_sizes[i - 1];

            let layer: Layer = Layer::new(size, prev_size, act_func);

            layers.push(layer);
        }

        Self { layers, act_func }
    }

    pub fn calc_network(&self, input: &Vector) -> Vector {
        assert_eq!(self.layers[0].weights.cols, input.data.len());

        let mut result: Vector = input.clone();

        for layer in &self.layers {
            result = layer.calc_layer(&result);
        }

        let output: Vector = result;

        return output;
    }

    pub fn calc_cost(prediction: Vector, expected_out: Vector) -> f64 {
        assert_eq!(prediction.data.len(), expected_out.data.len());

        let mut cost: f64 = 0.0;

        for i in 0..prediction.data.len() {
            let pre_value: f64 = prediction.data[i];
            let exp_value: f64 = expected_out.data[i];
            cost +=
                -1.0 * (exp_value * pre_value.ln() + (1.0 - exp_value) * (1.0 - pre_value).ln());
        }

        cost /= prediction.data.len() as f64;

        return cost;
    }

    pub fn train_network(&mut self, train_data: &Vec<DataPoint>, num_epochs: i32) -> i32 {
        let mut train_score: i32 = 0;

        for e in 0..num_epochs {
            for data in train_data {
                //Forward Propagation
                let prediction: Vector = self.calc_network(&data.input);

                //Calculate cost

                //Backwards Propagation
                //Update Parameters
                //Repeat
            }
        }

        return train_score;
    }

    pub fn test_network(&mut self, test_data: &Vec<DataPoint>) -> i32 {
        return 0;
    }
}
