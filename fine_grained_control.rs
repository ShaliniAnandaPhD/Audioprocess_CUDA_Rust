use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::simd::{f32x4, SimdFloat};

/// Generates audio samples using Rust's SIMD instructions.
///
/// # Arguments
///
/// * `sample_rate` - The sample rate of the audio.
/// * `num_samples` - The number of audio samples to generate.
/// * `frequency` - The frequency of the generated sine wave.
///
/// # Returns
///
/// * `Vec<f32>` - The generated audio samples.
#[pyfunction]
fn generate_audio_samples_simd(sample_rate: u32, num_samples: usize, frequency: f32) -> Vec<f32> {
    let mut samples = vec![0.0; num_samples];
    let scale = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;

    // Process samples in chunks of 4 using SIMD instructions
    let num_chunks = num_samples / 4;
    for i in 0..num_chunks {
        let t = f32x4::from([
            (i * 4 + 0) as f32,
            (i * 4 + 1) as f32,
            (i * 4 + 2) as f32,
            (i * 4 + 3) as f32,
        ]);
        let phase = t * f32x4::splat(scale);
        let chunk = phase.sin();
        chunk.write_to_slice(&mut samples[i * 4..i * 4 + 4]);
    }

    // Process remaining samples individually
    for i in num_chunks * 4..num_samples {
        let t = i as f32;
        samples[i] = (t * scale).sin();
    }

    samples
}

/// Generates audio samples using Rust's standard floating-point instructions.
///
/// # Arguments
///
/// * `sample_rate` - The sample rate of the audio.
/// * `num_samples` - The number of audio samples to generate.
/// * `frequency` - The frequency of the generated sine wave.
///
/// # Returns
///
/// * `Vec<f32>` - The generated audio samples.
#[pyfunction]
fn generate_audio_samples_standard(sample_rate: u32, num_samples: usize, frequency: f32) -> Vec<f32> {
    let mut samples = vec![0.0; num_samples];
    let scale = 2.0 * std::f32::consts::PI * frequency / sample_rate as f32;

    for i in 0..num_samples {
        let t = i as f32;
        samples[i] = (t * scale).sin();
    }

    samples
}

/// Demonstrates the fine-grained control over system resources for audio generation.
#[pyfunction]
fn demonstrate_fine_grained_control(py: Python) -> PyResult<()> {
    let sample_rate = 44100;
    let num_samples = 44100; // 1 second of audio
    let frequency = 440.0; // A4 note

    // Generate audio samples using SIMD instructions
    let simd_samples = generate_audio_samples_simd(sample_rate, num_samples, frequency);

    // Generate audio samples using standard floating-point instructions
    let standard_samples = generate_audio_samples_standard(sample_rate, num_samples, frequency);

    // Print the generated audio samples
    println!("SIMD samples: {:?}", simd_samples);
    println!("Standard samples: {:?}", standard_samples);

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn fine_grained_control(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_audio_samples_simd, m)?)?;
    m.add_function(wrap_pyfunction!(generate_audio_samples_standard, m)?)?;
    m.add_function(wrap_pyfunction!(demonstrate_fine_grained_control, m)?)?;
    Ok(())
}