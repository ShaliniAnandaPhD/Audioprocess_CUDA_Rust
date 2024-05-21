use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rand::Rng;
use std::sync::Arc;
use tch::{nn, Device, Tensor};

// Define the audio generation model
struct AudioGenerationModel {
    model: nn::Sequential,
}

impl AudioGenerationModel {
    fn new(input_size: i64, hidden_size: i64, output_size: i64) -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = nn::seq()
            .add(nn::linear(&vs.root(), input_size, hidden_size, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root(), hidden_size, output_size, Default::default()));
        AudioGenerationModel { model }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.model.forward(input)
    }
}

// Function to train the audio generation model
fn train_model(
    model: &AudioGenerationModel,
    data: &[f32],
    epochs: i64,
    batch_size: i64,
    learning_rate: f64,
) -> f64 {
    let data_tensor = Tensor::of_slice(data).reshape(&[-1, data.len() as i64]);
    let dataset = data_tensor.to_device(Device::Cpu);

    let mut optimizer = nn::Adam::default().build(&model.model.vars(), learning_rate).unwrap();

    let mut loss_sum = 0.0;
    let num_batches = dataset.size()[0] / batch_size;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for i in 0..num_batches {
            let offset = i * batch_size;
            let batch = dataset.narrow(0, offset, batch_size);

            let output = model.forward(&batch);
            let loss = output.mse_loss(&batch, tch::Reduction::Mean);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.double_value(&[]);
        }

        let avg_loss = epoch_loss / num_batches as f64;
        loss_sum += avg_loss;

        println!("Epoch {}: Loss = {:.4}", epoch + 1, avg_loss);
    }

    loss_sum / epochs as f64
}

// Function to perform random search hyperparameter optimization
fn random_search_optimization(
    input_size: i64,
    output_size: i64,
    data: &[f32],
    num_trials: i64,
    epochs: i64,
    batch_size: i64,
) -> (i64, f64) {
    let mut best_hidden_size = 0;
    let mut best_learning_rate = 0.0;
    let mut best_loss = std::f64::MAX;

    let mut rng = rand::thread_rng();

    for _ in 0..num_trials {
        let hidden_size = rng.gen_range(32..=256);
        let learning_rate = 10.0_f64.powf(-rng.gen_range(1.0..=5.0));

        let model = AudioGenerationModel::new(input_size, hidden_size, output_size);
        let loss = train_model(&model, data, epochs, batch_size, learning_rate);

        if loss < best_loss {
            best_hidden_size = hidden_size;
            best_learning_rate = learning_rate;
            best_loss = loss;
        }
    }

    (best_hidden_size, best_learning_rate)
}

// Python module to expose hyperparameter optimization functions
#[pymodule]
fn hyperparameter_optimization(_py: Python, m: &PyModule) -> PyResult<()> {
    // Function to perform random search hyperparameter optimization
    #[pyfn(m)]
    #[pyo3(name = "perform_random_search")]
    fn perform_random_search_py(
        input_size: i64,
        output_size: i64,
        data: Vec<f32>,
        num_trials: i64,
        epochs: i64,
        batch_size: i64,
    ) -> PyResult<(i64, f64)> {
        let (best_hidden_size, best_learning_rate) =
            random_search_optimization(input_size, output_size, &data, num_trials, epochs, batch_size);

        Ok((best_hidden_size, best_learning_rate))
    }

    Ok(())
}