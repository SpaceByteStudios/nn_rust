#[derive(Debug, Clone)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    pub fn vecadd(&self, other: &Self) -> Self {
        let result: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Vector { data: result }
    }

    pub fn dot(&self, other: &Self) -> f64 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}
