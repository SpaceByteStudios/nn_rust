#[derive(Debug)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    fn dot(&self, other: &Self) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}
