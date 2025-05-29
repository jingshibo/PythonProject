##
import torch
import itertools # For creating all pairs
import numpy as np
import random
from scipy.stats import truncnorm


## Generates EMG signals for a given condition using ALL combinations of A and B.
def generateDataPerCondition(generator_model, transition_label_int, gen_data_1, gen_data_2, sample_number, generation_batch_size,
        num_windows, window_length, window_increment, window_shift):

    # Step 1: Select fixed (A, B) pairs once
    num_A = len(gen_data_1)
    num_B = len(gen_data_2)
    all_pair_indices = list(itertools.product(range(num_A), range(num_B)))
    actual_num_to_generate = min(sample_number, len(all_pair_indices))
    selected_pair_indices = random.sample(all_pair_indices, actual_num_to_generate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator_model.to(device)
    generator.eval()

    def sample_shift(std, max_shift):
        """Sample a truncated normal shift between [-max_shift, +max_shift]."""
        return int(truncnorm.rvs(-max_shift / std, max_shift / std, loc=0, scale=std))

    # Step 2: For each time slice, process the same pairs
    generated_data_by_time_slice = {}
    for time_slice in range(num_windows):
        generated_batches = []
        blending_factor_batches = []
        # Initialize the keys first
        generated_data_by_time_slice[time_slice] = {'generated_images': None, 'blending_factors': None}
        print(f"\nGenerating for time_slice {time_slice} with {len(selected_pair_indices)} pairs:")

        for i in range(0, actual_num_to_generate, generation_batch_size):
            batch_indices = selected_pair_indices[i: i + generation_batch_size]
            batch_A, batch_B = [], []

            for idx_a, idx_b in batch_indices:
                A_raw = gen_data_1[idx_a].T  # (C, T)
                B_raw = gen_data_2[idx_b].T

                base_start = window_shift + time_slice * window_increment
                # time_shift = sample_shift(std=window_shift / 2, max_shift=window_shift)
                # start_A = base_start + time_shift
                # start_B = base_start + time_shift
                start_A = base_start
                start_B = base_start
                end_A = start_A + window_length
                end_B = start_B + window_length
                # Check bounds
                if end_A > A_raw.shape[1] or end_B > B_raw.shape[1]:
                    raise ValueError(f"Slice out of bounds: A[{start_A}:{end_A}] or B[{start_B}:{end_B}] exceeds sample length.")

                A_slice = torch.tensor(A_raw[:, start_A:end_A], dtype=torch.float32).unsqueeze(0)  # [C, T]
                B_slice = torch.tensor(B_raw[:, start_B:end_B], dtype=torch.float32).unsqueeze(0)
                batch_A.append(A_slice)
                batch_B.append(B_slice)

            batch_A = torch.cat(batch_A, dim=0).unsqueeze(1).to(device)  # [B, 1, C, T]
            batch_B = torch.cat(batch_B, dim=0).unsqueeze(1).to(device)
            condition_id = transition_label_int * num_windows + time_slice
            batch_conditions = torch.full((batch_A.size(0),), condition_id, dtype=torch.long).to(device)

            with torch.no_grad():
                fake_C, blending_factors = generator(batch_A, batch_B, batch_conditions)

            generated_batches.append(fake_C.cpu().numpy())
            blending_factor_batches.append(blending_factors.cpu().numpy())
            print(f" Transition {transition_label_int}, Time slice {time_slice}, batch {i // generation_batch_size + 1}: {fake_C.shape}")

        generated_data_by_time_slice[time_slice]['generated_images'] = np.concatenate(generated_batches, axis=0).transpose(0, 1, 3, 2)
        generated_data_by_time_slice[time_slice]['blending_factors'] = np.concatenate(blending_factor_batches, axis=0).transpose(0, 1, 3, 2)

    return generated_data_by_time_slice



## Generates EMG signals for all transition types
def generateTransitionData(model, train_gan_data, transition_encoding, num_windows, window_length, window_increment, window_shift,
        sample_number, batch_size):
    all_generated_data_dict = {}  # To store all generated arrays keyed by condition name
    for transition_name_str, transition_label_int in transition_encoding.items():
        current_condition_input_data = train_gan_data[transition_name_str]

        gen_data_1 = current_condition_input_data['gen_data_1']
        gen_data_2 = current_condition_input_data['gen_data_2']
        all_generated_data_dict[transition_name_str] = generateDataPerCondition(model, transition_label_int, gen_data_1, gen_data_2,
            sample_number, batch_size, num_windows, window_length, window_increment, window_shift)

    return all_generated_data_dict



## generation without considering time slice
# def generateDataPerCondition(generator_model, transition_label_int, gen_data_1, gen_data_2, sample_number, generation_batch_size):
#     """
#     Each A, B is a NumPy array of shape (Time, Features), e.g., (1200, 65).
#     Output of generator is (Batch, 1, Features, Time), e.g., (Batch, 1, 65, 1200).
#     """
#     num_A = len(gen_data_1)
#     num_B = len(gen_data_2)
#     total_pairs = num_A * num_B
#     all_pair_indices = list(itertools.product(range(num_A), range(num_B)))  # Create iterators for all combinations of A and B indices
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     actual_num_to_generate = min(sample_number, len(all_pair_indices))
#     selected_pair_indices = random.sample(all_pair_indices, actual_num_to_generate)
#
#     # We prepare data for one full batch at a time
#     generated_C_for_condition = []
#     for i in range(0, actual_num_to_generate, generation_batch_size):
#         current_batch_indices = selected_pair_indices[i: i + generation_batch_size]
#         if not current_batch_indices:
#             continue
#
#         # prepare test batch data
#         batch_A_list_torch = []
#         batch_B_list_torch = []
#         for idx_a, idx_b in current_batch_indices:
#             # Transpose, add channel dim, convert to tensor
#             A_tensor = torch.tensor(gen_data_1[idx_a].T, dtype=torch.float32).unsqueeze(0)  # (1, Features, Time) e.g., (1, 65, 1200)
#             B_tensor = torch.tensor(gen_data_2[idx_b].T, dtype=torch.float32).unsqueeze(0)  # (1, Features, Time)
#             batch_A_list_torch.append(A_tensor)
#             batch_B_list_torch.append(B_tensor)
#         # Stack to create batch for generator
#         batch_A = torch.cat(batch_A_list_torch, dim=0).unsqueeze(1).to(device)  # (batch_size_actual, 1, Features, Time)
#         batch_B = torch.cat(batch_B_list_torch, dim=0).unsqueeze(1).to(device)  # (batch_size_actual, 1, Features, Time)
#         batch_conditions = torch.full((batch_A.size(0),), transition_label_int, dtype=torch.long).to(device)
#
#         generator = generator_model.to(device)
#         generator.train(False)
#         with torch.no_grad():
#             fake_C_batch = generator(batch_A, batch_B, batch_conditions)
#         generated_C_for_condition.append(fake_C_batch.cpu().numpy())
#         print(f"  Generate {transition_label_int} batch {i // generation_batch_size + 1}/"
#               f"{(actual_num_to_generate + generation_batch_size - 1) // generation_batch_size}, "
#               f"Output shape for this batch: {fake_C_batch.shape}")
#
#     if generated_C_for_condition:
#         return np.concatenate(generated_C_for_condition, axis=0)
#     else:
#         return np.array([])



