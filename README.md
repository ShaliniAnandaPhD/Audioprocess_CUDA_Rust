# Real-Time Audio Processing with Rust and CUDA

## Overview

This repository contains a collection of real-time audio processing applications implemented using Rust and CUDA. The applications leverage Rust's safety and concurrency features alongside CUDA's parallel computing capabilities to efficiently handle various audio processing tasks. Each application focuses on a specific audio processing scenario and demonstrates how Rust and CUDA can be combined to achieve high-performance and low-latency audio processing.

## Applications

### 1. Audio Visualization System (`audio_visualization.rs`)

The Audio Visualization System processes audio input in real-time and generates visual representations of the audio data. It uses the `cpal` crate for cross-platform audio input and the `AudioProcessor` and `CudaProcessor` from the `audioprocess_cuda_rust` crate for audio processing. The application performs Fast Fourier Transform (FFT) on the audio data using the `CudaProcessor` and prepares the data for visualization. The processed audio data can be used to generate real-time graphics, such as audio waveforms or frequency spectrums.

### 2. Binaural Audio Simulator (`binaural_audio_simulator.rs`)

The Binaural Audio Simulator creates an immersive 3D audio experience by simulating binaural audio. It takes audio input, applies binaural audio processing using the `CudaProcessor`, and outputs the processed audio. The application can be extended to incorporate additional audio processing techniques, such as head-related transfer functions (HRTFs), room acoustics simulation, or audio spatialization, to enhance the realism of the 3D audio simulation.

### 3. Karaoke System (`karaoke_system.rs`)

The Karaoke System allows users to sing along with their favorite songs while removing the original vocals. It processes the audio input from both the song and the user's microphone. The `CudaProcessor` is used to remove the vocal track from the song and apply effects like reverb to the user's singing voice. The processed song and microphone audio are then mixed together using the `AudioProcessor` to create the karaoke experience. The application demonstrates real-time audio processing and mixing capabilities.

### 4. Virtual Guitar Amplifier (`virtual_guitar_amp.rs`)

The Virtual Guitar Amplifier simulates the sound of a guitar amplifier, allowing users to apply various effects to their guitar input. It processes the guitar audio input using the `AudioProcessor` and `CudaProcessor`. The application applies gain, distortion, delay, and reverb effects to the audio data in real-time. The processed audio is then output, creating a virtual guitar amplifier experience. The application showcases the use of Rust and CUDA for real-time audio effects processing.

### 5. Voice Call Noise Cancellation (`voice_call_noise_cancellation.rs`)

The Voice Call Noise Cancellation application demonstrates real-time noise cancellation for voice calls. It processes the microphone input to remove background noise and enhances the clarity of the user's voice. The noise cancellation algorithm can be implemented using signal processing techniques like spectral subtraction or adaptive filtering. The processed audio is then sent to the remote participant in the voice call. The application highlights the use of Rust and CUDA for real-time audio processing in communication scenarios.

### 6. Voice Changer System (`voice_changer.rs`)

The Voice Changer System allows users to modify their voice in real-time by applying various effects. It processes the audio input using the `AudioProcessor` and `CudaProcessor`. The application applies pitch shifting to change the pitch of the user's voice and can be extended to include other voice changing effects like voice distortion, echo, or reverb. The processed audio is then output, creating a fun and interactive voice changing experience.

# Leveraging LLM Capabilities in Audio Processing

This project explores the integration of Large Language Models (LLMs) into various audio processing applications to enhance functionality and user experience. The key areas of focus include:

- Automated documentation and real-time user support
- Semantic audio content analysis and tagging
- Real-time language translation and transcription
- Smart audio editing based on user descriptions
- Generative audio content creation and augmentation
- Natural language user interaction and command parsing

By leveraging the power of LLMs, this project aims to revolutionize audio processing applications, providing advanced features such as intelligent content analysis, real-time localization, creative assistance, and intuitive user interaction.

## Getting Started

To run any of the applications, ensure that you have Rust and CUDA installed on your system. Each application can be run independently using the Rust compiler.

1. Clone the repository:
   ```
   git clone https://github.com/ShaliniAnandaPhD/Audioprocess_CUDA_Rust.git
   ```

2. Navigate to the desired application directory:
   ```
   cd Audioprocess_CUDA_Rust/<application_directory>
   ```

3. Run the application:
   ```
   cargo run
   ```

Make sure to review the code and adjust any necessary configurations or parameters within each application file.

## Dependencies

The applications in this repository rely on the following dependencies:

- `audioprocess_cuda_rust`: A custom crate that provides audio processing functionalities using Rust and CUDA.
- `cpal`: A cross-platform audio library for Rust.

These dependencies are managed through the Rust package manager, Cargo, and will be automatically resolved when running the applications.

## Troubleshooting

If you encounter any issues while running the applications, consider the following:

- Ensure that you have the latest versions of Rust and CUDA installed.
- Verify that your system has a CUDA-capable GPU and the necessary drivers are installed.
- Check the audio device configurations and ensure that the correct input and output devices are selected.
- Refer to the Rust and CUDA documentation for any specific error messages or compilation issues.

## Contributing

Contributions to this repository are welcome! If you have any ideas for improvements, new features, or bug fixes, please feel free to submit a pull request. Make sure to follow the existing code style and provide appropriate documentation for your changes.

## License

This project is licensed under the [MIT License].

## Acknowledgments

- The Rust community for providing excellent resources and libraries for audio processing and cross-platform development.
- The CUDA community for their contributions to parallel computing and GPU acceleration.
- The developers of the `cpal` crate for simplifying cross-platform audio input and output in Rust.
- The Rust Programming Language by Steve Klabnik and Carol Nichols
- Programming Rust: Fast, Safe Systems Development" by Jim Blandy, Jason Orendorff, and Leonora F. S. Tindall
- Rust by Example by Steve Klabnik



