// File: voice_changer.rs

use audioprocess_cuda_rust::{AudioProcessor, CudaProcessor};
use cpal::Stream;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

struct VoiceChangerSystem {
    audio_processor: AudioProcessor,
    cuda_processor: CudaProcessor,
    pitch_shift_factor: f32,
    echo_delay: usize,
    reverb_amount: f32,
    distortion_level: f32,
    buffer: Arc<Mutex<VecDeque<Vec<f32>>>>,
}

impl VoiceChangerSystem {
    fn new(sample_rate: f32, channels: usize, chunk_size: usize) -> Self {
        let audio_processor = AudioProcessor::new(sample_rate, channels, chunk_size);
        let cuda_processor = CudaProcessor::new(chunk_size);
        VoiceChangerSystem {
            audio_processor,
            cuda_processor,
            pitch_shift_factor: 1.0,
            echo_delay: 4410, // 100ms delay at 44.1kHz sample rate
            reverb_amount: 0.5,
            distortion_level: 0.2,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

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

    fn apply_echo(&self, data: &mut [f32]) {
        let delay_samples = self.echo_delay;
        let mut buffer = self.buffer.lock().unwrap();

        for (i, sample) in data.iter_mut().enumerate() {
            let delayed_sample = if let Some(delayed_chunk) = buffer.get(delay_samples) {
                delayed_chunk[i % delay_samples]
            } else {
                0.0
            };
            *sample += delayed_sample * 0.5;
        }

        buffer.push_back(data.to_vec());
        if buffer.len() > delay_samples {
            buffer.pop_front();
        }
    }

    fn apply_reverb(&self, data: &mut [f32]) {
        for sample in data.iter_mut() {
            *sample = *sample * (1.0 - self.reverb_amount) + *sample * self.reverb_amount;
        }
    }

    fn apply_distortion(&self, data: &mut [f32]) {
        for sample in data.iter_mut() {
            *sample = (*sample * self.distortion_level).tanh();
        }
    }

    fn set_pitch_shift_factor(&mut self, factor: f32) {
        self.pitch_shift_factor = factor;
    }

    fn set_echo_delay(&mut self, delay: usize) {
        self.echo_delay = delay;
    }

    fn set_reverb_amount(&mut self, amount: f32) {
        self.reverb_amount = amount;
    }

    fn set_distortion_level(&mut self, level: f32) {
        self.distortion_level = level;
    }

    fn run(&mut self) {
        let host = cpal::default_host();
        let input_device = host.default_input_device().expect("No input device available");
        let output_device = host.default_output_device().expect("No output device available");

        let input_config = input_device
            .default_input_config()
            .expect("Failed to get default input config");
        let output_config = output_device
            .default_output_config()
            .expect("Failed to get default output config");

        let buffer = Arc::clone(&self.buffer);

        let input_stream = input_device
            .build_input_stream(
                &input_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let num_frames = data.len() / self.audio_processor.channels;
                    let mut output_data = vec![0.0; num_frames * self.audio_processor.channels];

                    self.process_audio(data, &mut output_data);

                    let mut buffer = buffer.lock().unwrap();
                    buffer.push_back(output_data);
                },
                |err| eprintln!("Error occurred during audio input: {}", err),
            )
            .expect("Failed to build input stream");

        let output_stream = output_device
            .build_output_stream(
                &output_config.into(),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let mut buffer = buffer.lock().unwrap();
                    if let Some(output_data) = buffer.pop_front() {
                        data.copy_from_slice(&output_data);
                    }
                },
                |err| eprintln!("Error occurred during audio output: {}", err),
            )
            .expect("Failed to build output stream");

        input_stream.play().expect("Failed to start audio input stream");
        output_stream.play().expect("Failed to start audio output stream");

        std::thread::park();

        drop(input_stream);
        drop(output_stream);
    }
}

fn main() {
    let sample_rate = 44100.0;
    let channels = 1;
    let chunk_size = 1024;

    let mut voice_changer_system = VoiceChangerSystem::new(sample_rate, channels, chunk_size);
    
    let pitch_shift_factor = 1.5;
    voice_changer_system.set_pitch_shift_factor(pitch_shift_factor);

    voice_changer_system.set_echo_delay(4410);
    voice_changer_system.set_reverb_amount(0.5);
    voice_changer_system.set_distortion_level(0.2);

    voice_changer_system.run();
}
