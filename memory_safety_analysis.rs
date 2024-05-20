use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::slice;

/// Generates audio samples using a Rust function.
///
/// # Arguments
///
/// * `sample_rate` - The sample rate of the audio.
/// * `num_samples` - The number of audio samples to generate.
///
/// # Returns
///
/// * `Vec<f32>` - The generated audio samples.
#[pyfunction]
fn generate_audio_samples_rust(sample_rate: u32, num_samples: usize) -> Vec<f32> {
    let mut samples = vec![0.0; num_samples];
    let frequency = 440.0; // Frequency of the generated tone (A4 note)

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        samples[i] = (2.0 * std::f32::consts::PI * frequency * t).sin();
    }

    samples
}

/// Generates audio samples using a Python function.
///
/// # Arguments
///
/// * `sample_rate` - The sample rate of the audio.
/// * `num_samples` - The number of audio samples to generate.
///
/// # Returns
///
/// * `&PyArray1<f32>` - The generated audio samples as a PyTorch tensor.
#[pyfunction]
fn generate_audio_samples_python(
    py: Python,
    sample_rate: u32,
    num_samples: usize,
) -> &PyArray1<f32> {
    let samples = PyArray1::zeros(py, num_samples, false);
    let frequency = 440.0; // Frequency of the generated tone (A4 note)

    let mut i = 0;
    while i < num_samples {
        let t = i as f32 / sample_rate as f32;
        samples.as_slice_mut().unwrap()[i] = (2.0 * std::f32::consts::PI * frequency * t).sin();
        i += 1;
    }

    samples
}

/// Demonstrates the memory safety benefits of using Rust for audio generation.
#[pyfunction]
fn analyze_memory_safety(py: Python) {
    let sample_rate = 44100;
    let num_samples = 1024;

    // Generate audio samples using Rust
    let rust_samples = generate_audio_samples_rust(sample_rate, num_samples);

    // Generate audio samples using Python
    let py_samples = generate_audio_samples_python(py, sample_rate, num_samples);

    // Print the generated audio samples
    println!("Rust samples: {:?}", rust_samples);
    println!("Python samples: {:?}", py_samples.as_slice().unwrap());
}

/// A Python module implemented in Rust.
#[pymodule]
fn memory_safety_analysis(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_audio_samples_rust, m)?)?;
    m.add_function(wrap_pyfunction!(generate_audio_samples_python, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_memory_safety, m)?)?;
    Ok(())
}