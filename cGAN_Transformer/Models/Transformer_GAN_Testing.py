##
import torch
import itertools # For creating all pairs
import numpy as np


## Generates EMG signals for a given condition using ALL combinations of A and B.
def generateDataPerCondition(generator_model, condition_label, gen_data_1, gen_data_2, generation_batch_size):
    """
    Each A, B is a NumPy array of shape (Time, Features), e.g., (1200, 65).
    Output of generator is (Batch, 1, Features, Time), e.g., (Batch, 1, 65, 1200).
    """
    num_A = len(gen_data_1)
    num_B = len(gen_data_2)
    total_pairs = num_A * num_B
    all_pair_indices = list(itertools.product(range(num_A), range(num_B)))  # Create iterators for all combinations of A and B indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We prepare data for one full batch at a time
    all_generated_C_for_condition = []
    for i in range(0, total_pairs, generation_batch_size):
        current_batch_indices = all_pair_indices[i: i + generation_batch_size]
        if not current_batch_indices:
            continue

        # prepare test batch data
        batch_A_list_torch = []
        batch_B_list_torch = []
        for idx_a, idx_b in current_batch_indices:
            # Transpose, add channel dim, convert to tensor
            A_tensor = torch.tensor(gen_data_1[idx_a].T, dtype=torch.float32).unsqueeze(0)  # (1, Features, Time) e.g., (1, 65, 1200)
            B_tensor = torch.tensor(gen_data_2[idx_b].T, dtype=torch.float32).unsqueeze(0)  # (1, Features, Time)
            batch_A_list_torch.append(A_tensor)
            batch_B_list_torch.append(B_tensor)
        # Stack to create batch for generator
        batch_A = torch.cat(batch_A_list_torch, dim=0).unsqueeze(1).to(device)  # (batch_size_actual, 1, Features, Time)
        batch_B = torch.cat(batch_B_list_torch, dim=0).unsqueeze(1).to(device)  # (batch_size_actual, 1, Features, Time)
        batch_conditions = torch.full((batch_A.size(0),), condition_label, dtype=torch.long).to(device)

        generator = generator_model.to(device)
        generator.train(False)
        with torch.no_grad():
            fake_C_batch = generator(batch_A, batch_B, batch_conditions)
        all_generated_C_for_condition.append(fake_C_batch.cpu().numpy())
        print(f"  Generate {condition_label} batch {i // generation_batch_size + 1}/"
              f"{(total_pairs + generation_batch_size - 1) // generation_batch_size}, "
              f"Output shape for this batch: {fake_C_batch.shape}")

    if all_generated_C_for_condition:
        return np.concatenate(all_generated_C_for_condition, axis=0)
    else:
        return np.array([])


## Generates EMG signals for all transition types
def generateTransitionData(model, train_gan_data, condition_encoding, batch_size):
    all_generated_data_dict = {}  # To store all generated arrays keyed by condition name
    for condition_name_str, condition_label_int in condition_encoding.items():
        current_condition_input_data = train_gan_data[condition_name_str]

        gen_data_1 = current_condition_input_data['gen_data_1']
        gen_data_2 = current_condition_input_data['gen_data_2']
        all_generated_data_dict[condition_name_str] = generateDataPerCondition(model, condition_label_int, gen_data_1, gen_data_2, batch_size)

    return all_generated_data_dict