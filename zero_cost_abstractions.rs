// zero_cost_abstractions.rs

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::time::Instant;

// Trait for audio generation algorithms
trait AudioGenerator {
    // Generates audio samples based on the provided parameters
    // Returns a vector of audio samples
    fn generate(&self, sample_rate: u32, num_samples: usize) -> Vec<f32>;
}

// Struct representing a sine wave audio generator
struct SineWaveGenerator {
    frequency: f32,
}

impl AudioGenerator for SineWaveGenerator {
    fn generate(&self, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        let mut samples = vec![0.0; num_samples];

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            samples[i] = (2.0 * std::f32::consts::PI * self.frequency * t).sin();
        }

        samples
    }
}

// Struct representing a square wave audio generator
struct SquareWaveGenerator {
    frequency: f32,
}

impl AudioGenerator for SquareWaveGenerator {
    fn generate(&self, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        let mut samples = vec![0.0; num_samples];

        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let phase = (2.0 * std::f32::consts::PI * self.frequency * t) % (2.0 * std::f32::consts::PI);
            samples[i] = if phase < std::f32::consts::PI { 1.0 } else { -1.0 };
        }

        samples
    }
}

// Generates audio samples using the specified audio generator
// Returns the generated audio samples
fn generate_audio(generator: &dyn AudioGenerator, sample_rate: u32, num_samples: usize) -> Vec<f32> {
    generator.generate(sample_rate, num_samples)
}

// Demonstrates the usage of zero-cost abstractions for audio generation
#[pyfunction]
fn demonstrate_zero_cost_abstractions(py: Python) -> PyResult<()> {
    let sample_rate = 44100;
    let num_samples = 44100; // 1 second of audio

    // Create audio generators
    let sine_wave_generator = SineWaveGenerator { frequency: 440.0 };
    let square_wave_generator = SquareWaveGenerator { frequency: 440.0 };

    // Generate audio samples using sine wave generator
    let start_time = Instant::now();
    let sine_samples = generate_audio(&sine_wave_generator, sample_rate, num_samples);
    let sine_elapsed = start_time.elapsed();

    // Generate audio samples using square wave generator
    let start_time = Instant::now();
    let square_samples = generate_audio(&square_wave_generator, sample_rate, num_samples);
    let square_elapsed = start_time.elapsed();

    // Print the performance results
    println!("Sine wave generation time: {:?}", sine_elapsed);
    println!("Square wave generation time: {:?}", square_elapsed);

    // Compare performance with and without abstractions
    let start_time = Instant::now();
    let mut direct_sine_samples = vec![0.0; num_samples];
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        direct_sine_samples[i] = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
    }
    let direct_sine_elapsed = start_time.elapsed();

    println!("Direct sine wave generation time: {:?}", direct_sine_elapsed);

    Ok(())
}

// Python module to expose the zero-cost abstractions demonstration
#[pymodule]
fn zero_cost_abstractions_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(demonstrate_zero_cost_abstractions, m)?)?;
    Ok(())
}
