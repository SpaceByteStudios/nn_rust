use crate::{
    math::vector::Vector,
    neural_net::layer::{self, Layer},
};

#[derive(Debug)]
struct Network {
    layers: Vec<Layer>,
    act_func: fn(f64) -> f64,
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

    pub fn calc_network(&self, input: Vector) -> Vector {
        assert_eq!(self.layers[0].size, input.data.len());

        let output: Vector = input;

        return output;
    }

    pub fn train_network() {}
}
