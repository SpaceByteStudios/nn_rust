use crate::{
    math::vector::Vector,
    neural_net::layer::{self, Layer},
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

    pub fn train_network() {}
}
