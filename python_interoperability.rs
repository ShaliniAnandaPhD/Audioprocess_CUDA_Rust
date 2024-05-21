use numpy::ndarray::ArrayD;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::Arc;
use tch::{nn, Device, Tensor};

// Struct to represent an audio generator
struct AudioGenerator {
    model: Arc<tch::CModule>,
}

impl AudioGenerator {
    // Constructor to create a new audio generator with a pre-trained model
    fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = tch::CModule::load(model_path)?;
        Ok(Self {
            model: Arc::new(model),
        })
    }

    // Method to generate audio samples from a given input tensor
    fn generate(&self, input: &Tensor) -> Tensor {
        self.model.forward_ts(&[input.unsqueeze(0)]).unwrap().squeeze()
    }
}

// Python module to expose the audio generator functionality
#[pymodule]
fn python_interoperability(_py: Python, m: &PyModule) -> PyResult<()> {
    // Define a Python class for the audio generator
    #[pyo3(text_signature = "(model_path)")]
    #[pyclass(name = "AudioGenerator")]
    struct PyAudioGenerator {
        #[pyo3(get)]
        generator: AudioGenerator,
    }

    // Implement Python methods for the audio generator class
    #[pymethods]
    impl PyAudioGenerator {
        // Constructor to create a new audio generator from a model path
        #[new]
        fn new(model_path: &str) -> PyResult<Self> {
            let generator = AudioGenerator::new(model_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to create AudioGenerator: {}", e)))?;
            Ok(Self { generator })
        }

        // Method to generate audio samples from a NumPy array
        fn generate(&self, input: &PyArray1<f32>) -> PyResult<Py<PyArray1<f32>>> {
            let input_tensor = Tensor::of_slice(input.as_slice().unwrap()).to(Device::Cpu);
            let output_tensor = self.generator.generate(&input_tensor);
            let output_array = output_tensor.try_into().unwrap();
            Ok(output_array.into_pyarray(input.py()).to_owned())
        }
    }

    // Add the audio generator class to the Python module
    m.add_class::<PyAudioGenerator>()?;
    Ok(())
}