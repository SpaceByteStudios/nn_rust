use crate::neural_net::{
    data_point::DataPoint,
    functions::{Activation, Loss, OutputActivation},
    layer::Layer,
    matrix::{Matrix, Vector},
};

#[derive(Debug, Clone)]
pub struct Network {
    layers: Vec<Layer>,
    out_activation: OutputActivation,
    loss_function: Loss,
}

impl Network {
    pub fn new(
        layers_sizes: Vec<usize>,
        activation: Activation,
        out_activation: OutputActivation,
    ) -> Self {
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

        let loss_function: Loss;

        //Use correct Loss Function based on last Layer Activation Function
        match out_activation {
            OutputActivation::Linear => loss_function = Loss::MeanSquaredError,
            OutputActivation::Softmax => loss_function = Loss::CrossEntropy,
        }

        Self {
            layers,
            out_activation,
            loss_function,
        }
    }

    pub fn get_weights(&self) -> Vec<&Matrix> {
        let mut weights: Vec<&Matrix> = vec![];

        for layer in &self.layers {
            weights.push(layer.get_weights());
        }

        weights
    }

    pub fn get_bias(&self) -> Vec<&Vector> {
        let mut bias: Vec<&Vector> = vec![];

        for layer in &self.layers {
            bias.push(layer.get_bias());
        }

        bias
    }

    pub fn get_layers(&self) -> &[Layer] {
        &self.layers
    }

    pub fn calc_network(&mut self, input: &Vector) -> Vector {
        assert_eq!(input.len(), self.layers[0].prev_size());

        let mut result: Vector = input.clone();

        for layer in &mut self.layers {
            result = layer.calc_layer(&result);
        }

        self.out_activation.apply_mut(&mut result);

        result
    }

    fn calc_loss(&self, prediction: &Vector, expected_out: &Vector) -> f64 {
        assert_eq!(prediction.len(), expected_out.len());

        self.loss_function.apply(expected_out, prediction)
    }

    fn back_prop(&mut self, prediction: &Vector, expexted_out: &Vector) {
        let mut data: Vec<f64> = vec![0.0; prediction.len()];

        for i in 0..prediction.len() {
            data[i] = prediction.get(i) - expexted_out.get(i);
        }

        let mut error_term: Vector = Vector::new(data);

        //Backpropagating through the layers backwards
        for layer in &mut self.layers.iter_mut().rev() {
            layer.back_prop_layer(&mut error_term);
        }
    }

    pub fn train_network(
        &mut self,
        train_data: &[DataPoint],
        epochs: usize,
        batch_size: usize,
        learn_rate: f64,
    ) -> Vec<f64> {
        let mut cost_history: Vec<f64> = vec![];

        for _ in 0..epochs {
            let mut e_cost: f64 = 0.0;

            let batches = train_data.chunks_exact(batch_size);

            for batch in batches {
                for data in batch.iter() {
                    //Forward Propagation
                    let prediction: Vector = self.calc_network(&data.input);

                    //Calculate cost
                    let batch_cost: f64 = self.calc_loss(&prediction, &data.exp_output);
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

    pub fn test_network(&mut self, test_data: &[DataPoint]) -> f64 {
        let mut cost: f64 = 0.0;

        for data in test_data {
            let prediction: Vector = self.calc_network(&data.input);

            cost += self.calc_loss(&prediction, &data.exp_output);
        }

        cost / test_data.len() as f64
    }

    pub fn mutate(&mut self, rate: f64, strength: f64) {
        for layer in &mut self.layers {
            layer.mutate(rate, strength);
        }
    }
}
