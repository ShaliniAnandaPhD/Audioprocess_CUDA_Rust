use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
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

// Function to compute the gradients of the model output with respect to the input
fn compute_gradients(model: &AudioGenerationModel, input: &Tensor) -> Tensor {
    let input_var = input.requires_grad_(true);
    let output = model.forward(&input_var);
    output.backward();
    input_var.grad().unwrap().clone()
}

// Function to compute the saliency map of the model
fn compute_saliency_map(model: &AudioGenerationModel, input: &Tensor) -> Tensor {
    let gradients = compute_gradients(model, input);
    gradients.abs()
}

// Function to compute the guided backpropagation of the model
fn compute_guided_backprop(model: &AudioGenerationModel, input: &Tensor) -> Tensor {
    let input_var = input.requires_grad_(true);
    let output = model.forward(&input_var);

    let mut gradients = Vec::new();

    for i in 0..output.size()[1] {
        let output_i = output.select(1, i);
        output_i.backward_with_grad(&Tensor::ones(&output_i.size(), (Kind::Float, Device::Cpu)), true, false);
        let gradient = input_var.grad().unwrap().clone();
        gradients.push(gradient);
        input_var.zero_grad();
    }

    Tensor::stack(&gradients, 0).mean_dim(true, &[0], false, Kind::Float)
}

// Function to compute the integrated gradients of the model
fn compute_integrated_gradients(model: &AudioGenerationModel, input: &Tensor, baseline: &Tensor, steps: i64) -> Tensor {
    let mut integrated_gradients = Tensor::zeros_like(input);
    let input_diff = input - baseline;

    for i in 0..steps {
        let alpha = (i as f64 + 1.0) / steps as f64;
        let interpolated_input = baseline + input_diff * alpha;
        let gradients = compute_gradients(model, &interpolated_input);
        integrated_gradients += gradients * input_diff;
    }

    integrated_gradients / steps as f64
}

// Python module to expose model interpretability tools
#[pymodule]
fn model_interpretability(_py: Python, m: &PyModule) -> PyResult<()> {
    // Function to compute the saliency map of the model
    #[pyfn(m)]
    #[pyo3(name = "compute_saliency_map")]
    fn compute_saliency_map_py(
        model_path: &str,
        input_size: i64,
        hidden_size: i64,
        output_size: i64,
        input_data: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
        let model = tch::CModule::load(model_path)?;
        let model = AudioGenerationModel {
            model: model.sequential(),
        };

        let input_tensor = Tensor::of_slice(&input_data).reshape(&[1, input_size]);
        let saliency_map = compute_saliency_map(&model, &input_tensor);
        Ok(saliency_map.flatten().into())
    }

    // Function to compute the guided backpropagation of the model
    #[pyfn(m)]
    #[pyo3(name = "compute_guided_backprop")]
    fn compute_guided_backprop_py(
        model_path: &str,
        input_size: i64,
        hidden_size: i64,
        output_size: i64,
        input_data: Vec<f32>,
    ) -> PyResult<Vec<f32>> {
        let model = tch::CModule::load(model_path)?;
        let model = AudioGenerationModel {
            model: model.sequential(),
        };

        let input_tensor = Tensor::of_slice(&input_data).reshape(&[1, input_size]);
        let guided_backprop = compute_guided_backprop(&model, &input_tensor);
        Ok(guided_backprop.flatten().into())
    }

    // Function to compute the integrated gradients of the model
    #[pyfn(m)]
    #[pyo3(name = "compute_integrated_gradients")]
    fn compute_integrated_gradients_py(
        model_path: &str,
        input_size: i64,
        hidden_size: i64,
        output_size: i64,
        input_data: Vec<f32>,
        baseline_data: Vec<f32>,
        steps: i64,
    ) -> PyResult<Vec<f32>> {
        let model = tch::CModule::load(model_path)?;
        let model = AudioGenerationModel {
            model: model.sequential(),
        };

        let input_tensor = Tensor::of_slice(&input_data).reshape(&[1, input_size]);
        let baseline_tensor = Tensor::of_slice(&baseline_data).reshape(&[1, input_size]);
        let integrated_gradients = compute_integrated_gradients(&model, &input_tensor, &baseline_tensor, steps);
        Ok(integrated_gradients.flatten().into())
    }

    Ok(())
}