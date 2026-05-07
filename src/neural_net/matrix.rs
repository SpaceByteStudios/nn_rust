#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct Vector {
    vector: Matrix,
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

    pub fn size(&mut self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn scale(&self, scalar: f64) -> Self {
        let mut result = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(i, j, self.get(i, j) * scalar);
            }
        }

        result
    }

    pub fn scale_mut(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.set(i, j, self.get(i, j) * scalar);
            }
        }
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

    pub fn transpose_mut(&mut self) {
        let mut result = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }

        self.rows = result.rows;
        self.cols = result.cols;
        self.data = result.data;
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.cols);
        assert_eq!(self.rows, other.rows);

        let mut result: Matrix = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let sum: f64 = self.get(i, j) + other.get(i, j);
                result.set(i, j, sum);
            }
        }

        result
    }

    pub fn add_mut(&mut self, other: &Self) {
        assert_eq!(self.cols, other.cols);
        assert_eq!(self.rows, other.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                let sum: f64 = self.get(i, j) + other.get(i, j);
                self.set(i, j, sum);
            }
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
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

    pub fn mul_mut(&mut self, other: &Self) {
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

        self.rows = result.rows;
        self.cols = result.cols;
        self.data = result.data;
    }

    pub fn mul_vec(&self, other: &Vector) -> Vector {
        assert_eq!(self.cols, other.len());

        let mut result = Vector::zeros(self.rows);

        for i in 0..self.rows {
            let mut sum = 0.0;

            for j in 0..self.cols {
                sum += self.get(i, j) * other.get(j);
            }

            result.set(i, sum);
        }

        result
    }
}

impl Vector {
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            vector: Matrix {
                rows: data.len(),
                cols: 1,
                data,
            },
        }
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            vector: Matrix {
                rows: size,
                cols: 1,
                data: vec![0.0; size],
            },
        }
    }

    pub fn get(&self, index: usize) -> f64 {
        self.vector.data[index]
    }

    pub fn set(&mut self, index: usize, value: f64) {
        self.vector.data[index] = value
    }

    pub fn len(&self) -> usize {
        self.vector.rows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn scale(&self, scalar: f64) -> Self {
        let mut result: Vector = Vector::zeros(self.len());

        for i in 0..self.len() {
            result.set(i, scalar * self.get(i));
        }

        result
    }

    pub fn scale_mut(&mut self, scalar: f64) {
        for i in 0..self.len() {
            self.set(i, scalar * self.get(i));
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());

        let mut result: Vector = Vector::zeros(self.len());

        for i in 0..self.len() {
            result.set(i, self.get(i) + other.get(i));
        }

        result
    }

    pub fn add_mut(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len());

        for i in 0..self.len() {
            self.set(i, self.get(i) + other.get(i));
        }
    }

    pub fn elem_mul(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());

        let mut result: Vector = Vector::zeros(self.len());

        for i in 0..self.len() {
            result.set(i, self.get(i) * other.get(i));
        }

        result
    }

    pub fn elem_mul_mut(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len());

        for i in 0..self.len() {
            self.set(i, self.get(i) * other.get(i));
        }
    }

    pub fn outer(&self, other: &Self) -> Matrix {
        let mut result: Matrix = Matrix::zeros(self.len(), other.len());

        for i in 0..self.len() {
            for j in 0..other.len() {
                result.set(i, j, self.get(i) * other.get(j));
            }
        }

        result
    }
}
