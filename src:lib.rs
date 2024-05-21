use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::time::Instant;
use tch::{nn, Device, Tensor};

/// Generates music using a pre-trained PyTorch model.
///
/// # Arguments
///
/// * `model_path` - The path to the pre-trained PyTorch model.
/// * `num_samples` - The number of music samples to generate.
/// * `device` - The device to use for computation ("cpu" or "cuda").
///
/// # Returns
///
/// * `Result<Vec<Tensor>, PyErr>` - The generated music samples or a Python exception.
#[pyfunction]
fn rust_generate_music(model_path: String, num_samples: i64, device: String) -> PyResult<Vec<Tensor>> {
    // Convert the device string to the corresponding device type
    let device = match device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        // Handle invalid device input
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid device")),
    };

    // Load the pre-trained PyTorch model
    let model = tch::CModule::load(&model_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load model: {}", e)))?;
    
    // Move the model to the specified device
    let model = model.to_device(device);

    let mut music_samples = Vec::new();

    // Generate music samples
    for _ in 0..num_samples {
        // Generate random noise as input for the model
        let random_noise = Tensor::rand(&[1, 128], kind::FLOAT_CPU).to_device(device);

        // Perform forward pass to generate music
        let output = model.forward_ts(&[random_noise])
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Model forward pass failed: {}", e)))?;
        
        music_samples.push(output);
    }

    Ok(music_samples)
}

/// Benchmarks the performance of Rust-PyTorch music generation.
///
/// # Arguments
///
/// * `model_path` - The path to the pre-trained PyTorch model.
/// * `num_samples` - The number of music samples to generate.
/// * `device` - The device to use for computation ("cpu" or "cuda").
///
/// # Returns
///
/// * `f64` - The execution time in seconds.
#[pyfunction]
fn rust_benchmark_music_generation(model_path: String, num_samples: i64, device: String) -> PyResult<f64> {
    // Record the start time
    let start_time = Instant::now();
    
    // Generate music samples using Rust-PyTorch integration
    let _ = rust_generate_music(model_path, num_samples, device)?;
    
    // Record the end time
    let end_time = Instant::now();
    
    // Calculate the execution time
    let execution_time = end_time.duration_since(start_time).as_secs_f64();
    
    Ok(execution_time)
}

/// Rust module exposed to Python.
#[pymodule]
fn rust_music_benchmarks(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_generate_music, m)?)?;
    m.add_function(wrap_pyfunction!(rust_benchmark_music_generation, m)?)?;
    Ok(())
}
