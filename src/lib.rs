pub mod neural_net;
pub use neural_net::network::Network;

pub use neural_net::data_point::DataPoint;
pub use neural_net::matrix::Vector;

pub use neural_net::functions::Activation;
pub use neural_net::functions::OutputActivation;

pub mod prelude {
    pub use crate::neural_net::network::Network;

    pub use crate::neural_net::data_point::DataPoint;
    pub use crate::neural_net::matrix::Vector;

    pub use crate::neural_net::functions::Activation;
    pub use crate::neural_net::functions::OutputActivation;
}
