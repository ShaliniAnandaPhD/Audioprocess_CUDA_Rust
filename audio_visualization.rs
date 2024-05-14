// File: audio_visualization.rs

use audioprocess_cuda_rust::{AudioProcessor, CudaProcessor};
use cpal::Stream;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

struct AudioVisualizationSystem {
    audio_processor: AudioProcessor,
    cuda_processor: CudaProcessor,
    visualization_data_buffer: Arc<Mutex<VecDeque<Vec<f32>>>>,
}

impl AudioVisualizationSystem {
    fn new(sample_rate: f32, channels: usize, chunk_size: usize) -> Self {
        let audio_processor = AudioProcessor::new(sample_rate, channels, chunk_size);
        let cuda_processor = CudaProcessor::new(chunk_size);
        AudioVisualizationSystem {
            audio_processor,
            cuda_processor,
            visualization_data_buffer: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    fn process_audio(&mut self, audio_data: &[f32]) -> Vec<f32> {
        let mut fft_data = vec![0.0; self.cuda_processor.chunk_size];

        // Perform Fast Fourier Transform (FFT) on the audio data using the CudaProcessor
        self.cuda_processor.fft(audio_data, &mut fft_data);

        // Perform additional computations on the FFT data using the CudaProcessor
        self.calculate_frequency_spectrum(&mut fft_data);

        fft_data
    }

    fn calculate_frequency_spectrum(&self, fft_data: &mut [f32]) {
        // Example: Normalize the FFT data for visualization
        let max_val = fft_data.iter().cloned().fold(f32::MIN, f32::max);
        for val in fft_data.iter_mut() {
            *val /= max_val;
        }
    }

    fn run(&mut self) {
        let host = cpal::default_host();
        let input_device = host.default_input_device().expect("No input device available");

        let input_config = input_device
            .default_input_config()
            .expect("Failed to get default input config");

        let visualization_data_buffer = Arc::clone(&self.visualization_data_buffer);

        let input_stream = input_device
            .build_input_stream(
                &input_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let visualization_data = self.process_audio(data);

                    let mut buffer = visualization_data_buffer.lock().unwrap();
                    buffer.push_back(visualization_data);

                    // Limit buffer size to prevent memory overflow
                    if buffer.len() > 10 {
                        buffer.pop_front();
                    }
                },
                |err| eprintln!("Error occurred during audio input: {}", err),
            )
            .expect("Failed to build input stream");

        input_stream.play().expect("Failed to start audio input stream");

        // Run a separate thread for the visualization
        let visualization_data_buffer = Arc::clone(&self.visualization_data_buffer);
        std::thread::spawn(move || {
            loop {
                let mut buffer = visualization_data_buffer.lock().unwrap();
                if let Some(visualization_data) = buffer.pop_front() {
                    // Pass the visualization data to the graphics rendering system
                    // Example: Update shader uniforms, vertex buffers, or textures with the visualization data
                    println!("Visualization Data: {:?}", visualization_data);
                }
                std::thread::sleep(std::time::Duration::from_millis(16)); // Approx 60 FPS
            }
        });

        // Keep the program running until interrupted
        std::thread::park();

        drop(input_stream);
    }
}

fn main() {
    let sample_rate = 44100.0;
    let channels = 2;
    let chunk_size = 1024;

    let mut audio_visualization_system =
        AudioVisualizationSystem::new(sample_rate, channels, chunk_size);
    audio_visualization_system.run();
}
