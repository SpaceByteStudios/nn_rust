use crate::math::vector::Vector;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len());
        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, value: f64) {
        self.data[r * self.cols + c] = value;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }

        result
    }

    pub fn mul_vector(&self, v: &Vector) -> Vector {
        assert_eq!(self.cols, v.data.len(),);

        let result: Vec<f64> = (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| self.data[i * self.cols + j] * v.data[j])
                    .sum()
            })
            .collect();

        Vector { data: result }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);

        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        result
    }
}
