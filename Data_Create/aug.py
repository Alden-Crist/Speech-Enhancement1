import numpy as np
import soundfile as sf
import os
from scipy.signal import resample

def generate_variations(input_dir, output_dir, num_variations, duration, sample_rate):
    """
    Generate variations of all noise samples in a directory with a fixed sample rate.

    Args:
        input_dir (str): Path to the directory containing input noise files.
        output_dir (str): Path to the directory to save generated noise variations.
        num_variations (int): Number of noise variations to create for each file.
        duration (float): Desired duration of each output file in seconds.
        sample_rate (int): Sampling rate in Hz.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List all input noise files
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    for file_name in input_files:
        # Read the input file
        input_path = os.path.join(input_dir, file_name)
        noise, original_sample_rate = sf.read(input_path)
        
        # Resample the noise if the sample rate differs
        if original_sample_rate != sample_rate:
            noise = resample(noise, int(len(noise) * sample_rate / original_sample_rate))
        
        # Extend or trim the noise to match the desired duration
        num_samples = int(duration * sample_rate)
        if len(noise) < num_samples:
            # Loop the noise to extend it
            noise = np.tile(noise, (num_samples // len(noise) + 1))[:num_samples]
        else:
            # Trim the noise to the desired length
            noise = noise[:num_samples]
        
        # Create variations for the current file
        base_name = os.path.splitext(file_name)[0]
        for i in range(num_variations):
            # Apply variation (e.g., random volume scaling, noise addition)
            variation = noise.copy()
            variation *= np.random.uniform(0.8, 1.2)  # Random volume scaling
            variation += np.random.uniform(-0.1, 0.1, size=variation.shape)  # Add random noise
            
            # Save the variation
            variation_name = f"{base_name}_variation_{i+1}.wav"
            output_path = os.path.join(output_dir, variation_name)
            sf.write(output_path, variation, sample_rate)
            print(f"Generated variation: {output_path}")

# Parameters
input_directory = "/Users/alden/OneDrive/Desktop/Data/Train/noise"  # Replace with your input directory path
output_directory = "/Users/alden/OneDrive/Desktop/Data/Train/aug"  # Replace with your output directory path
number_of_variations = 10  # Number of variations per file
desired_duration = 2.0  # Each file will be 2 seconds long
sampling_rate = 8000  # Fixed sample rate (8 kHz)

# Generate variations for all files
generate_variations(input_directory, output_directory, number_of_variations, desired_duration, sampling_rate)
