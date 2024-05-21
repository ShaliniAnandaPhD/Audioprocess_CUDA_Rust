// File: voice_changer.rs

use audioprocess_cuda_rust::{AudioProcessor, CudaProcessor}; // Importing audio processing modules
use cpal::Stream; // Importing the Stream module from cpal
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait}; // Importing necessary traits from cpal
use std::sync::{Arc, Mutex}; // Importing Arc and Mutex for thread-safe reference counting and mutual exclusion
use std::collections::VecDeque; // Importing VecDeque for double-ended queue

// Struct representing the voice changer system
struct VoiceChangerSystem {
    audio_processor: AudioProcessor, // AudioProcessor instance
    cuda_processor: CudaProcessor, // CudaProcessor instance
    pitch_shift_factor: f32, // Factor for pitch shifting
    echo_delay: usize, // Delay for echo effect
    reverb_amount: f32, // Amount of reverb effect
    distortion_level: f32, // Level of distortion effect
    buffer: Arc<Mutex<VecDeque<Vec<f32>>>>, // Buffer for audio data synchronization
}

impl VoiceChangerSystem {
    // Constructor for initializing the voice changer system with the given audio parameters
    fn new(sample_rate: f32, channels: usize, chunk_size: usize) -> Self {
        let audio_processor = AudioProcessor::new(sample_rate, channels, chunk_size); // Creating a new AudioProcessor
        let cuda_processor = CudaProcessor::new(chunk_size); // Creating a new CudaProcessor
        VoiceChangerSystem {
            audio_processor,
            cuda_processor,
            pitch_shift_factor: 1.0,
            echo_delay: 4410, // 100ms delay at 44.1kHz sample rate
            reverb_amount: 0.5,
            distortion_level: 0.2,
            buffer: Arc::new(Mutex::new(VecDeque::new())), // Initializing buffer
        }
    }

    // Method for processing audio data
    fn process_audio(&mut self, input_data: &[f32], output_data: &mut [f32]) {
        // Apply pitch shifting using the CudaProcessor
        self.cuda_processor.apply_pitch_shift(input_data, output_data, self.pitch_shift_factor);

        // Apply echo effect
        self.apply_echo(output_data);

        // Apply reverb effect
        self.apply_reverb(output_data);

        // Apply distortion effect
        self.apply_distortion(output_data);
    }

    // Method to apply echo effect
    fn apply_echo(&self, data: &mut [f32]) {
        let delay_samples = self.echo_delay;
        let mut buffer = self.buffer.lock().unwrap(); // Acquire lock on buffer

        for (i, sample) in data.iter_mut().enumerate() {
            let delayed_sample = if let Some(delayed_chunk) = buffer.get(delay_samples) {
                delayed_chunk[i % delay_samples]
            } else {
                0.0
            };
            *sample += delayed_sample * 0.5; // Apply echo effect
        }

        buffer.push_back(data.to_vec());
        if buffer.len() > delay_samples {
            buffer.pop_front();
        }

        // Possible error: Buffer lock issues or deadlock
        // Solution: Ensure locks are held for the shortest time necessary and check for deadlock scenarios.
    }

    // Method to apply reverb effect
    fn apply_reverb(&self, data: &mut [f32]) {
        for sample in data.iter_mut() {
            *sample = *sample * (1.0 - self.reverb_amount) + *sample * self.reverb_amount;
        }

        // Possible error: Reverb parameters causing distortion
        // Solution: Fine-tune reverb parameters and test with different inputs.
    }

    // Method to apply distortion effect
    fn apply_distortion(&self, data: &mut [f32]) {
        for sample in data.iter_mut() {
            *sample = (*sample * self.distortion_level).tanh(); // Apply distortion
        }

        // Possible error: Distortion parameters causing excessive distortion
        // Solution: Adjust distortion parameters and test with different inputs.
    }

    // Method to set pitch shift factor
    fn set_pitch_shift_factor(&mut self, factor: f32) {
        self.pitch_shift_factor = factor;
    }

    // Method to set echo delay
    fn set_echo_delay(&mut self, delay: usize) {
        self.echo_delay = delay;
    }

    // Method to set reverb amount
    fn set_reverb_amount(&mut self, amount: f32) {
        self.reverb_amount = amount;
    }

    // Method to set distortion level
    fn set_distortion_level(&mut self, level: f32) {
        self.distortion_level = level;
    }

    // Method to run the voice changer system
    fn run(&mut self) {
        // Get the default audio host
        let host = cpal::default_host(); // Getting the default audio host

        // Get the default input device
        let input_device = host.default_input_device().expect("No input device available"); // Getting the default input device

        // Possible error: No input device available
        // Solution: Ensure that a microphone is connected and recognized by the system.

        // Get the default output device
        let output_device = host.default_output_device().expect("No output device available"); // Getting the default output device

        // Possible error: No output device available
        // Solution: Ensure that speakers or an output device are connected and recognized by the system.

        // Get the default input and output configurations
        let input_config = input_device.default_input_config().expect("Failed to get default input config"); // Getting the default input config
        let output_config = output_device.default_output_config().expect("Failed to get default output config"); // Getting the default output config

        // Possible error: Failed to get default input/output config
        // Solution: Verify that the audio devices support default configurations or manually specify configurations.

        let buffer = Arc::clone(&self.buffer); // Clone the buffer for use in the input callback

        // Build the input stream
        let input_stream = input_device.build_input_stream(
            &input_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let num_frames = data.len() / self.audio_processor.channels;
                let mut output_data = vec![0.0; num_frames * self.audio_processor.channels];

                self.process_audio(data, &mut output_data); // Process the audio data

                let mut buffer = buffer.lock().unwrap();
                buffer.push_back(output_data);
            },
            |err| eprintln!("Error occurred during audio input: {}", err), // Error handling

            // Possible error: Error occurred during audio input
            // Solution: Log error details for debugging and ensure proper handling of microphone input errors.
        ).expect("Failed to build input stream"); // Expecting the input stream to be built successfully

        // Build the output stream
        let output_stream = output_device.build_output_stream(
            &output_config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut buffer = buffer.lock().unwrap();
                if let Some(output_data) = buffer.pop_front() {
                    data.copy_from_slice(&output_data); // Copy the processed data to the output stream
                }
            },
            |err| eprintln!("Error occurred during audio output: {}", err), // Error handling

            // Possible error: Error occurred during audio output
            // Solution: Log error details for debugging and ensure proper handling of audio output errors.
        ).expect("Failed to build output stream"); // Expecting the output stream to be built successfully

        // Start the input and output streams
        input_stream.play().expect("Failed to start audio input stream"); // Starting the input stream

        // Possible error: Failed to start audio input stream
        // Solution: Ensure no other application is using the microphone and that the microphone supports streaming.

        output_stream.play().expect("Failed to start audio output stream"); // Starting the output stream

        // Possible error: Failed to start audio output stream
        // Solution: Ensure no other application is using the audio output device and that the device supports streaming.

        // Run the audio stream until the program is terminated
        std::thread::park(); // Park the thread to keep the streams running

        // Drop the input and output streams to stop them and free resources
        drop(input_stream); // Dropping the input stream
        drop(output_stream); // Dropping the output stream
    }
}

fn main() {
    let sample_rate = 44100.0; // Set the sample rate
    let channels = 1; // Set the number of channels
    let chunk_size = 1024; // Set the chunk size

    // Create a new instance of the voice changer system
    let mut voice_changer_system = VoiceChangerSystem::new(sample_rate, channels, chunk_size);

    // Set the desired audio effect parameters
    let pitch_shift_factor = 1.5;
    voice_changer_system.set_pitch_shift_factor(pitch_shift_factor);

    voice_changer_system.set_echo_delay(4410); // Set echo delay
    voice_changer_system.set_reverb_amount(0.5); // Set reverb amount
    voice_changer_system.set_distortion_level(0.2); // Set distortion level

    // Run the voice changer system
    voice_changer_system.run();
}
