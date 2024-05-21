// File: audio_visualization.rs

use audioprocess_cuda_rust::{AudioProcessor, CudaProcessor}; // Importing audio processing modules
use cpal::Stream; // Importing the Stream module from cpal
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait}; // Importing necessary traits from cpal
use std::sync::{Arc, Mutex}; // Importing Arc and Mutex for thread-safe reference counting and mutual exclusion
use std::collections::VecDeque; // Importing VecDeque for double-ended queue

// Struct representing the audio visualization system
struct AudioVisualizationSystem {
    audio_processor: AudioProcessor, // AudioProcessor instance
    cuda_processor: CudaProcessor, // CudaProcessor instance
    visualization_data_buffer: Arc<Mutex<VecDeque<Vec<f32>>>>, // Buffer for visualization data
}

impl AudioVisualizationSystem {
    // Constructor for initializing the audio visualization system with the given audio parameters
    fn new(sample_rate: f32, channels: usize, chunk_size: usize) -> Self {
        let audio_processor = AudioProcessor::new(sample_rate, channels, chunk_size); // Creating a new AudioProcessor
        let cuda_processor = CudaProcessor::new(chunk_size); // Creating a new CudaProcessor
        AudioVisualizationSystem {
            audio_processor,
            cuda_processor,
            visualization_data_buffer: Arc::new(Mutex::new(VecDeque::new())), // Initializing the buffer
        }
    }

    // Method for processing the audio data and generating visualization data
    fn process_audio(&mut self, audio_data: &[f32]) -> Vec<f32> {
        let mut fft_data = vec![0.0; self.cuda_processor.chunk_size];

        // Perform Fast Fourier Transform (FFT) on the audio data using the CudaProcessor
        self.cuda_processor.fft(audio_data, &mut fft_data);

        // Perform additional computations on the FFT data
        self.calculate_frequency_spectrum(&mut fft_data);

        fft_data
        
        // Possible error: FFT computation failure
        // Solution: Ensure the audio data is correctly formatted and the CUDA processor is properly initialized.
    }

    // Method to calculate the frequency spectrum from the FFT data
    fn calculate_frequency_spectrum(&self, fft_data: &mut [f32]) {
        // Normalize the FFT data for visualization
        let max_val = fft_data.iter().cloned().fold(f32::MIN, f32::max);
        for val in fft_data.iter_mut() {
            *val /= max_val;
        }

        // Possible error: Division by zero if max_val is zero
        // Solution: Add a check to ensure max_val is not zero before dividing.
    }

    // Method to run the audio visualization system
    fn run(&mut self) {
        // Get the default audio host
        let host = cpal::default_host();
        
        // Get the default input device
        let input_device = host.default_input_device().expect("No input device available");

        // Possible error: No input device available
        // Solution: Ensure that a microphone is connected and recognized by the system.

        // Get the default input configuration
        let input_config = input_device.default_input_config().expect("Failed to get default input config");

        // Possible error: Failed to get default input config
        // Solution: Verify that the audio device supports default configurations or manually specify configurations.

        let visualization_data_buffer = Arc::clone(&self.visualization_data_buffer); // Clone the buffer for use in the input callback

        // Build the input stream
        let input_stream = input_device.build_input_stream(
            &input_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let visualization_data = self.process_audio(data); // Process the audio data

                let mut buffer = visualization_data_buffer.lock().unwrap(); // Acquire lock on buffer
                buffer.push_back(visualization_data);

                // Limit buffer size to prevent memory overflow
                if buffer.len() > 10 {
                    buffer.pop_front();
                }

                // Possible error: Buffer lock issues or deadlock
                // Solution: Ensure locks are held for the shortest time necessary and check for deadlock scenarios.
            },
            |err| eprintln!("Error occurred during audio input: {}", err), // Error handling

            // Possible error: Error occurred during audio input
            // Solution: Log error details for debugging and ensure proper handling of microphone input errors.
        ).expect("Failed to build input stream"); // Expecting the input stream to be built successfully

        // Start the input stream
        input_stream.play().expect("Failed to start audio input stream");

        // Possible error: Failed to start audio input stream
        // Solution: Ensure no other application is using the microphone and that the microphone supports streaming.

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

        // Drop the input stream to stop it and free resources
        drop(input_stream);
    }
}

fn main() {
    let sample_rate = 44100.0; // Set the sample rate
    let channels = 2; // Set the number of channels
    let chunk_size = 1024; // Set the chunk size

    // Create a new instance of the audio visualization system
    let mut audio_visualization_system = AudioVisualizationSystem::new(sample_rate, channels, chunk_size);
    
    // Run the audio visualization system
    audio_visualization_system.run();
}
