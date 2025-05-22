import torch
import torch.nn as nn
import math


class MixedAdditive2DSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_dim1: int, max_dim2: int, dropout: float = 0.1):
        """
        Implements the 2D sinusoidal positional encoding based on the "mixed" additive formula.
        The input sequence to the Transformer is assumed to be a flattened representation
        of a 2D grid (dim1_coord, dim2_coord).

        Args:
            d_model (int): The dimensionality of the Transformer's input features and PEs.
                           Must be even.
            max_dim1 (int): Maximum size of the first dimension (e.g., EMG channels or height).
            max_dim2 (int): Maximum size of the second dimension (e.g., time steps or width).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be an even number, got {d_model}")

        num_positions = max_dim1 * max_dim2
        positional_encoding = torch.zeros(num_positions, d_model)  # Positional encoding table

        # Create coordinate vectors for the flattened 2D grid
        # These will correspond to 'c' and 't' in your formula.
        # dim1_coords will be for 'c', dim2_coords for 't'.
        # Flattening order: dim1 is outer loop, dim2 is inner loop.
        # e.g., (c0,t0), (c0,t1), ..., (c0,t_max-1), (c1,t0), (c1,t1), ...
        dim1_pos_flat = torch.arange(max_dim1, dtype=torch.float32).repeat_interleave(max_dim2)
        dim2_pos_flat = torch.arange(max_dim2, dtype=torch.float32).repeat(max_dim1)

        # Unsqueeze for broadcasting with div_term
        position_c = dim1_pos_flat.unsqueeze(1)  # Shape: (num_positions, 1)
        position_t = dim2_pos_flat.unsqueeze(1)  # Shape: (num_positions, 1)

        # Divisor term: 10000^(2i/d_model)
        # 'i' goes from 0 to d_model/2 - 1
        # So, 2i goes from 0 to d_model - 2 (even indices)
        div_term_indices = torch.arange(0, d_model, 2, dtype=torch.float32)  # Shape: (d_model/2)
        div_term = torch.pow(10000.0, div_term_indices / d_model)  # Shape: (d_model/2)

        # Calculate arguments for sin/cos
        # c / (10000^(2i/D))
        arg_c = position_c / div_term  # Broadcasting: (num_pos, 1) / (d_model/2) -> (num_pos, d_model/2)
        # t / (10000^(2i/D))
        arg_t = position_t / div_term  # Broadcasting: (num_pos, 1) / (d_model/2) -> (num_pos, d_model/2)

        # Apply the formula from your image
        # Even dimensions (j = 2i)
        positional_encoding[:, 0::2] = torch.sin(arg_c) + torch.cos(arg_t)
        # Odd dimensions (j = 2i+1)
        positional_encoding[:, 1::2] = torch.cos(arg_c) + torch.sin(arg_t)

        # Add a batch dimension for broadcasting: (1, num_positions, d_model)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer('pe', positional_encoding)  # Not a parameter, but part of the model state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
                              where seq_len = max_dim1 * max_dim2 (or less, if using a subsequence).
        Returns:
            torch.Tensor: Tensor with positional encodings added.
        """
        current_seq_len = x.size(1)
        if current_seq_len > self.pe.size(1):
            raise ValueError(f"Input sequence length {current_seq_len} is greater than "
                             f"the pre-calculated max PE sequence length {self.pe.size(1)} "
                             f"(max_dim1 * max_dim2).")

        # Add the positional encoding up to the length of the input sequence
        x = x + self.pe[:, :current_seq_len, :]
        return self.dropout(x)


# --- Example Usage ---
if __name__ == '__main__':
    d_model = 512
    max_emg_channels = 8  # Corresponds to 'c' or dim1
    max_time_steps = 10  # Corresponds to 't' or dim2
    batch_size = 4
    seq_len = max_emg_channels * max_time_steps  # 80

    # Instantiate the positional encoding layer
    pos_encoder = MixedAdditive2DSinusoidalPositionalEncoding(d_model=d_model, max_dim1=max_emg_channels, max_dim2=max_time_steps,
        dropout=0.0  # No dropout for this test
    )

    # Create a dummy input tensor (e.g., token embeddings)
    # Shape: (batch_size, seq_len, d_model)
    dummy_input_tokens = torch.randn(batch_size, seq_len, d_model)

    # Apply positional encoding
    output_with_pe = pos_encoder(dummy_input_tokens)
    print("Shape of output with PE:", output_with_pe.shape)  # Expected: (4, 80, 512)

    # Test uniqueness for (c,t) vs (t,c) if max_dim1 == max_dim2 (for simplicity of indexing)
    # Let's create a smaller example for this test
    d_model_test = 4
    max_d1_test = 3
    max_d2_test = 3
    pos_encoder_test = MixedAdditive2DSinusoidalPositionalEncoding(d_model_test, max_d1_test, max_d2_test, dropout=0.0)

    # Flattening order assumed: (c0,t0), (c0,t1), (c0,t2), (c1,t0), (c1,t1), (c1,t2), ...
    # Position (c=1, t=2)
    # c=1 is the second channel index (0-indexed)
    # t=2 is the third time step index (0-indexed)
    # Flattened index = c_idx * max_d2_test + t_idx = 1 * 3 + 2 = 5
    idx_c1_t2 = 1 * max_d2_test + 2
    pe_c1_t2 = pos_encoder_test.pe[0, idx_c1_t2, :]
    print(f"\nPE for (c=1, t=2) (index {idx_c1_t2}):\n{pe_c1_t2}")

    # Position (c=2, t=1)
    idx_c2_t1 = 2 * max_d2_test + 1
    pe_c2_t1 = pos_encoder_test.pe[0, idx_c2_t1, :]
    print(f"PE for (c=2, t=1) (index {idx_c2_t1}):\n{pe_c2_t1}")

    # Check if they are different (they should be)
    are_different = not torch.allclose(pe_c1_t2, pe_c2_t1)
    print(f"Are PE(c=1, t=2) and PE(c=2, t=1) different? {are_different}")

    # Compare with the symmetric sum case (what we want to avoid for the overall vector)
    # If we had PE_sym(2i)   = sin(c/W) + sin(t/W)
    #           PE_sym(2i+1) = cos(c/W) + cos(t/W)
    # Then PE_sym(c=1,t=2) would equal PE_sym(c=2,t=1).
    # But our formula is different.

    print("\nTest for your example (c=4, t=3) vs (c=3, t=4):")
    d_model_ex = 4
    c_coord1, t_coord1 = 4, 3
    c_coord2, t_coord2 = 3, 4
    max_dim1_ex = 5  # Must be >= max coordinate
    max_dim2_ex = 5  # Must be >= max coordinate

    pos_encoder_ex = MixedAdditive2DSinusoidalPositionalEncoding(d_model_ex, max_dim1_ex, max_dim2_ex, dropout=0.0)

    idx1 = c_coord1 * max_dim2_ex + t_coord1
    pe1 = pos_encoder_ex.pe[0, idx1, :]
    print(f"PE for (c={c_coord1}, t={t_coord1}) (index {idx1}):\n{pe1}")

    idx2 = c_coord2 * max_dim2_ex + t_coord2
    pe2 = pos_encoder_ex.pe[0, idx2, :]
    print(f"PE for (c={c_coord2}, t={t_coord2}) (index {idx2}):\n{pe2}")

    are_different_ex = not torch.allclose(pe1, pe2)
    print(f"Are PE(c={c_coord1}, t={t_coord1}) and PE(c={c_coord2}, t={t_coord2}) different? {are_different_ex}")
    assert are_different_ex, "The PEs for (c,t) and (t,c) should be different!"