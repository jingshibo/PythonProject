import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# Assuming your model definitions are in 'Transformer_GAN_Model.py'
# or that the classes are already defined in the current scope (e.g., in a Jupyter notebook)
from cGAN_Transformer.Models import Transformer_GAN_Model  # Make sure this path is correct
from cGAN_Transformer.Functions import Storage





def plot_signals(signals, num_to_plot=3, title="Generated Signals"):
    """
    Simple plotting utility for a few signals.
    Assumes signal shape is (num_signals, 1, C_spatial, T_orig) or (num_signals, C_spatial, T_orig)
    And plots the first spatial channel.
    """
    if signals.ndim == 4 and signals.shape[1] == 1:  # (N, 1, H, W)
        signals = signals.squeeze(1)  # (N, H, W)

    num_to_plot = min(num_to_plot, signals.shape[0])
    if num_to_plot == 0:
        print("No signals to plot.")
        return

    fig, axes = plt.subplots(num_to_plot, 1, figsize=(12, num_to_plot * 3), squeeze=False)
    fig.suptitle(title, fontsize=16)
    for i in range(num_to_plot):
        # Plotting the first spatial channel (index 0 of C_spatial dimension)
        if signals.shape[1] > 0:  # Check if C_spatial dimension exists and is not empty
            signal_to_plot = signals[i, 0, :].numpy()  # Assuming (N, C_spatial, T_orig)
            axes[i, 0].plot(signal_to_plot)
            axes[i, 0].set_title(f"Sample {i + 1}, Spatial Channel 0")
            axes[i, 0].set_xlabel("Time Steps")
            axes[i, 0].set_ylabel("Amplitude")
        else:
            axes[i, 0].set_title(f"Sample {i + 1} - No spatial channels to plot or incorrect shape")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for suptitle
    plt.show()


# --- Main Generation Script ---
if __name__ == "__main__":
    # Configuration
    subject = 'Number1'
    version = 0
    model_type = 'Transformer_Gan'
    model_name = ['gen', 'disc']
    storage_parameters = {'subject': subject, 'version': version, 'model_type': model_type, 'model_name': model_name, 'gan_result_set': 0}

    NUM_CONDITIONS = 4
    NUM_SAMPLES_TO_GENERATE = 5
    C_SPATIAL_ORIG = 65  # Original spatial dimension (e.g., EMG channels)
    T_ORIG = 1200  # Original time dimension

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Storage.loadGanModels(storage_parameters)

    # 2. Prepare Input Data for Generation
    all_generated_samples = []
    all_conditions_used = []

    for i in range(NUM_SAMPLES_TO_GENERATE):
        input_A_batch, input_B_batch = prepare_input_signals(1, C_SPATIAL_ORIG, T_ORIG, device)  # Generate for one sample at a time

        # Choose a condition for this sample
        # Example: generate one sample for each condition, then repeat
        condition_idx = i % NUM_CONDITIONS
        condition_label_batch = torch.tensor([condition_idx], dtype=torch.long).to(device)

        print(f"\nGenerating sample {i + 1}/{NUM_SAMPLES_TO_GENERATE} for condition {condition_idx}...")

        # 3. Generate Data
        fake_C_tensor = generate_samples(model['gen'], input_A_batch, input_B_batch, condition_label_batch, device)
        # fake_C_tensor shape is (1, 1, C_SPATIAL_ORIG, T_ORIG)

        all_generated_samples.append(fake_C_tensor.squeeze(0).numpy())  # Remove batch dim, convert to numpy
        all_conditions_used.append(condition_idx)

        # 4. Save or Process  # Example: Save each generated sample  # output_filename = os.path.join(output_dir, f"generated_sample_{
        # i+1}_condition_{condition_idx}.npy")  # np.save(output_filename, fake_C_tensor.squeeze(0).numpy()) # Squeeze batch and channel
        # dim if channel=1  # print(f"Saved: {output_filename}")

    # Combine all samples into one array for easier plotting/analysis if needed
    if all_generated_samples:
        all_generated_samples_np = np.stack(all_generated_samples)  # Shape: (NUM_SAMPLES, 1, C_SPATIAL, T_ORIG)
        print(f"\nShape of all_generated_samples_np: {all_generated_samples_np.shape}")

        # Example: Save all generated samples and their conditions
        np.save(os.path.join(output_dir, "all_generated_emg.npy"), all_generated_samples_np)
        np.save(os.path.join(output_dir, "all_conditions_used.npy"), np.array(all_conditions_used))
        print(f"Saved all generated samples and conditions to '{output_dir}'")

        # 5. Visualize (Optional)
        plot_signals(torch.from_numpy(all_generated_samples_np), num_to_plot=min(5, NUM_SAMPLES_TO_GENERATE), title="Generated EMG Samples")