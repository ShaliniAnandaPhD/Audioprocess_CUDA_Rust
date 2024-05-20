import torch
from rust_music_benchmarks import rust_generate_music, rust_benchmark_music_generation
import time

def python_generate_music(model_path, num_samples, device):
    # Load the pre-trained PyTorch model
    model = torch.jit.load(model_path)
    model = model.to(device)
    
    music_samples = []
    
    # Generate music samples
    for _ in range(num_samples):
        random_noise = torch.rand(1, 128, device=device)
        output = model(random_noise)
        music_samples.append(output)
    
    return music_samples

def python_benchmark_music_generation(model_path, num_samples, device):
    # Record the start time
    start_time = time.time()
    
    # Generate music samples using pure Python
    _ = python_generate_music(model_path, num_samples, device)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the execution time
    execution_time = end_time - start_time
    
    return execution_time

# Set the parameters
model_path = "path/to/pretrained_model.pt"
num_samples = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Benchmark Rust-PyTorch music generation
rust_execution_time = rust_benchmark_music_generation(model_path, num_samples, device)
print(f"Rust-PyTorch execution time: {rust_execution_time:.2f} seconds")

# Benchmark pure Python music generation
python_execution_time = python_benchmark_music_generation(model_path, num_samples, device)
print(f"Pure Python execution time: {python_execution_time:.2f} seconds")

# Calculate the speedup
speedup = python_execution_time / rust_execution_time
print(f"Speedup: {speedup:.2f}x")