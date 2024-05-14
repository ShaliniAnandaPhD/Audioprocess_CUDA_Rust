use audioprocess_cuda_rust::{AudioProcessor, CudaProcessor}; // Importing audio processing modules
use cpal::Stream; // Importing the Stream module from cpal
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait}; // Importing necessary traits from cpal
use std::sync::Arc; // Importing Arc for thread-safe reference counting

struct GuitarAmp {
    processor: AudioProcessor, // AudioProcessor instance
    cuda_processor: CudaProcessor, // CudaProcessor instance
}

impl GuitarAmp {
    fn new(sample_rate: f32, channels: usize, chunk_size: usize) -> Self {
        // Initialize the AudioProcessor with the provided audio parameters
        let processor = AudioProcessor::new(sample_rate, channels, chunk_size); // Creating a new AudioProcessor
        // Initialize the CudaProcessor with the chunk size
        let cuda_processor = CudaProcessor::new(chunk_size); // Creating a new CudaProcessor
        GuitarAmp {
            processor, // Assigning processor to the struct field
            cuda_processor, // Assigning cuda_processor to the struct field
        }
    }

    fn process_audio(&mut self, audio_data: &mut [f32], gain: f32, distortion: f32, delay: f32, reverb: f32) {
        // Apply gain to the audio data using the AudioProcessor
        self.processor.apply_gain(audio_data, gain); // Applying gain

        // Apply distortion to the audio data using the CudaProcessor
        // This assumes that the CudaProcessor has an 'apply_distortion' method
        self.cuda_processor.apply_distortion(audio_data, distortion); // Applying distortion

        // Apply delay to the audio data using the CudaProcessor
        // This assumes that the CudaProcessor has an 'apply_delay' method
        self.cuda_processor.apply_delay(audio_data, delay); // Applying delay

        // Apply reverb to the audio data using the CudaProcessor
        // This assumes that the CudaProcessor has an 'apply_reverb' method
        self.cuda_processor.apply_reverb(audio_data, reverb); // Applying reverb
    }

    fn run(&mut self) {
        // Get the default audio host
        let host = cpal::default_host(); // Getting the default audio host
        // Get the default input device
        let device = host.default_input_device().expect("No input device available"); // Getting the default input device
        // Get the default input config
        let config = device.default_input_config().expect("Failed to get default input config"); // Getting the default input config

        // Build an input stream using the default input config
        let stream = device.build_input_stream(
            &config.into(), // Converting the config into the correct type
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // Convert the input audio data to a mutable vector
                let mut audio_data = data.to_vec(); // Converting input data to a mutable vector
                // Process the audio data with the selected amp model and effects
                self.process_audio(&mut audio_data, 2.0, 5.0, 0.2, 0.3); // Processing the audio data
                // Write the processed audio data to the output buffer
                // You can use an audio output library like cpal to play the processed audio
                // TODO: Implement audio output
            },
            |err| eprintln!("Error occurred during audio processing: {}", err), // Error handling
        ).expect("Failed to build input stream"); // Expecting the input stream to be built successfully

        // Start the audio stream
        stream.play().expect("Failed to start audio stream"); // Starting the audio stream
        // Run the audio stream for 60 seconds
        std::thread::sleep(std::time::Duration::from_secs(60)); // Running the stream for 60 seconds
        // Drop the audio stream to stop it and free resources
        drop(stream); // Dropping the stream to stop it
    }
}

fn main() {
    // Set the desired audio parameters
    let sample_rate = 44100.0; // Setting the sample rate
    let channels = 1; // Setting the number of channels
    let chunk_size = 1024; // Setting the chunk size

    // Create a new GuitarAmp instance with the audio parameters
    let mut guitar_amp = GuitarAmp::new(sample_rate, channels, chunk_size); // Creating a new GuitarAmp
    // Run the guitar amp
    guitar_amp.run(); // Running the guitar amp
}
