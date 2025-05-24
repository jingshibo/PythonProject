import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchinfo import summary


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_channels, cond_embed_dim, use_adain=False):
        super().__init__()
        self.use_adain = use_adain
        if use_adain:
            self.norm = nn.InstanceNorm2d(num_channels, affine=False)
        else:
            self.norm = nn.BatchNorm2d(num_channels, affine=False)
        self.gamma = nn.Linear(cond_embed_dim, num_channels)
        self.beta = nn.Linear(cond_embed_dim, num_channels)

    def forward(self, x, cond_embed):
        out = self.norm(x)
        gamma = self.gamma(cond_embed).unsqueeze(2).unsqueeze(3)
        beta = self.beta(cond_embed).unsqueeze(2).unsqueeze(3)
        return gamma * out + beta


class ConvCBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_embed_dim, use_cbn=True, use_adain=False, use_spectral_norm=False, transpose=False,
            activation=nn.ReLU):
        super().__init__()
        self.use_cbn = use_cbn
        self.transpose = transpose  # still used to indicate upsampling intent

        # If transpose, use Upsample + Conv2d instead of ConvTranspose2d to increase spatial resolution
        if transpose:
            self.upsample = nn.Upsample(scale_factor=(1, 2), mode='bilinear')
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 9), stride=1, padding=(1, 4))  # can only use stride=1, not stride=(1, 2)
        else:
            self.upsample = None
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))

        self.conv = spectral_norm(conv) if use_spectral_norm else conv

        if use_cbn:
            self.norm = ConditionalBatchNorm2d(out_channels, cond_embed_dim, use_adain=use_adain)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        self.activation = activation() if isinstance(activation, type) else activation

    def forward(self, x, cond_embed):
        if self.transpose:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.norm(x, cond_embed) if self.use_cbn else self.norm(x)
        x = self.activation(x)
        return x


class Additive2DSinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_dim1: int, max_dim2: int, dropout: float = 0.1):
        """
        Implements the 2D sinusoidal positional encoding based on the "mixed" additive formula.
        The input sequence to the Transformer is assumed to be a flattened representation
        of a 2D grid (dim1_coord, dim2_coord).
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if d_model % 2 != 0:
            raise ValueError(f"d_model must be an even number, got {d_model}")

        num_positions = max_dim1 * max_dim2
        positional_encoding = torch.zeros(num_positions, d_model)  # Positional encoding table

        # Create coordinate vectors for the flattened 2D grid. ("C-style" or row-major order)
        # Flattening order: dim1 is outer loop, dim2 is inner loop.  e.g., (c0,t0), (c0,t1), ..., (c0,t_max-1), (c1,t0), (c1,t1), ...
        dim1_pos_flat = torch.arange(max_dim1, dtype=torch.float32).repeat_interleave(max_dim2)
        dim2_pos_flat = torch.arange(max_dim2, dtype=torch.float32).repeat(max_dim1)

        # Unsqueeze for broadcasting with div_term
        position_c = dim1_pos_flat.unsqueeze(1)  # Shape: (num_positions, 1)
        position_t = dim2_pos_flat.unsqueeze(1)  # Shape: (num_positions, 1)

        # Divisor term: 10000^(2i/d_model). 'i' goes from 0 to d_model/2 - 1. So, 2i goes from 0 to d_model - 2 (even indices)
        div_term_indices = torch.arange(0, d_model, 2, dtype=torch.float32)  # Shape: (d_model/2)
        div_term = torch.pow(10000.0, div_term_indices / d_model)  # Shape: (d_model/2)

        # Calculate arguments for sin/cos, c / (10000^(2i/D)), t / (10000^(2i/D))
        arg_c = position_c / div_term  # Broadcasting: (num_pos, 1) / (d_model/2) -> (num_pos, d_model/2)
        arg_t = position_t / div_term  # Broadcasting: (num_pos, 1) / (d_model/2) -> (num_pos, d_model/2)

        # Apply the formula from your image
        positional_encoding[:, 0::2] = torch.sin(arg_c) + torch.cos(arg_t)  # Even dimensions (j = 2i)
        positional_encoding[:, 1::2] = torch.cos(arg_c) + torch.sin(arg_t)  # Odd dimensions (j = 2i+1)

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

        # Add the pe up to the length of the input sequence
        pe_to_add = self.pe[:, :current_seq_len, :].to(x.device)          # Ensure self.pe is on the same device as x
        x = x + pe_to_add
        return self.dropout(x)


class EMGFusionGenerator(nn.Module):
    def __init__(self, num_conditions, cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False, hidden_channels=32,
            activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01}, initial_cond_map_channels=1):
        super().__init__()

        if activation_params:
            act_fn = activation_class(**activation_params)
        else:
            act_fn = activation_class()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)
        self.initial_cond_projector = nn.Linear(cond_embed_dim, initial_cond_map_channels)

        # reduce time dim to a smaller number while incresing the CNN channel dim
        self.encoder1 = ConvCBNBlock(2 + initial_cond_map_channels, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)
        self.encoder2 = ConvCBNBlock(hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)
        self.encoder3 = ConvCBNBlock(hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)
        self.encoder4 = ConvCBNBlock(hidden_channels * 4, hidden_channels * 8, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)

        self.d_model = hidden_channels * 8 + cond_embed_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=self.d_model * 2, batch_first=True, dropout=0.1,
                activation=nn.GELU()), num_layers=4)

        self.decoder1 = ConvCBNBlock(self.d_model, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=act_fn)
        self.decoder2 = ConvCBNBlock(hidden_channels * 4, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=act_fn)
        self.decoder3 = ConvCBNBlock(hidden_channels * 2, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=act_fn)
        self.decoder4 = ConvCBNBlock(hidden_channels, 1, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=act_fn)

        self.sig = torch.nn.Sigmoid()  # convert values to the range of [0, 1]
        self.pos_encoder = None  # will be initialized in the first forward pass

    def forward(self, A, B, condition):
        B_, _, C, T = A.shape

        cond_embed = self.condition_embedding(condition)  # Shape: [B, cond_embed_dim] (B: batch size)
        initial_cond_map_feats = self.initial_cond_projector(cond_embed)  # project condition embed vector to [B, initial_cond_map_channels]
        cond_for_cat = initial_cond_map_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, T) # to create shape: [B, initial_cond_map_channels, C, T]
        x = torch.cat([A, B, cond_for_cat], dim=1)  # [B, 3, C, T]

        x = self.encoder1(x, cond_embed)
        x = self.encoder2(x, cond_embed)
        x = self.encoder3(x, cond_embed)
        x = self.encoder4(x, cond_embed)

        B_, C_encoder, C_spatial, T_reduced = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B_, C_spatial * T_reduced, C_encoder)  # [batch, time_step, encoder_channels]
        cond_embed_seq = cond_embed.unsqueeze(1).expand(-1, C_spatial * T_reduced, -1)  # [batch, time_step, cond_embed_dim]
        x = torch.cat([x, cond_embed_seq], dim=-1)  # [batch, time_step, d_model=encoder_channels+cond_embed_dim]

        if self.pos_encoder is None:
            self.pos_encoder = Additive2DSinusoidalPositionalEncoding(self.d_model, C_spatial, T_reduced, dropout=0.0)
            self.pos_encoder = self.pos_encoder.to(x.device)  # <<< CRITICAL: Move to device
        x = self.pos_encoder(x)
        x = self.transformer(x)  # transformer input: [Batch, time_step, d_model]

        # Transformer typically has the same shape as its input [Batch, time_step, d_model]
        B_, time_step, d_model = x.shape
        x = x.reshape(B_, C_spatial, T_reduced, d_model).permute(0, 3, 1, 2)
        x = self.decoder1(x, cond_embed)
        x = self.decoder2(x, cond_embed)
        x = self.decoder3(x, cond_embed)
        x = self.decoder4(x, cond_embed)
        x = self.sig(x)
        return x


# ConditionalBatchNorm2d and ConvCBNBlock remain the same as you defined them
# (assuming ConvCBNBlock uses stride=(1,2) for downsampling in time)


# Conceptual EMGFusionPatchDiscriminator with Projection Principle
class EMGFusionPatchDiscriminator(nn.Module):
    def __init__(self, num_conditions, cond_embed_dim=64, use_cbn=False, use_adain=False, use_spectral_norm=True, hidden_channels=32,
            activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.2}):
        super().__init__()

        if activation_params:
            act_fn = activation_class(**activation_params)
        else:
            act_fn = activation_class()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)
        C_phi = hidden_channels * 8  # Output channels of the backbone

        # --- Backbone ---
        self.block1 = ConvCBNBlock(3, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)
        self.block2 = ConvCBNBlock(hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)
        self.block3 = ConvCBNBlock(hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)
        self.block4 = ConvCBNBlock(hidden_channels * 4, C_phi, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=act_fn)

        # --- Unconditional Patch Score Head ---
        final_conv_unconditional_layer = nn.Conv2d(C_phi, 1, kernel_size=(3, 3), stride=1, padding=1)
        self.final_conv_unconditional = spectral_norm(
            final_conv_unconditional_layer) if use_spectral_norm else final_conv_unconditional_layer
        # --- For Conditional Projection Term ---
        self.cond_projector_for_phi = nn.Linear(cond_embed_dim, C_phi)  # Project condition to match feature channels

    def forward(self, C_sample, A_ref, B_ref, condition_label):
        cond_embed = self.condition_embedding(condition_label)  # [B, cond_embed_dim]
        x = torch.cat([A_ref, B_ref, C_sample], dim=1)  # [B, 3, H, W]

        phi_features = self.block1(x, cond_embed)
        phi_features = self.block2(phi_features, cond_embed)
        phi_features = self.block3(phi_features, cond_embed)
        phi_features = self.block4(phi_features, cond_embed)  # [B, C_phi, Patch_H, Patch_W]

        # Unconditional part
        unconditional_patch_logits = self.final_conv_unconditional(phi_features)  # [B, 1, Patch_H, Patch_W]
        # Conditional part (Projection)
        projected_cond_embed = self.cond_projector_for_phi(cond_embed)  # [B, C_phi]
        projected_cond_embed_spatial = projected_cond_embed.unsqueeze(-1).unsqueeze(-1).expand_as(phi_features)  # [B, C_phi, Patch_H, Patch_W]

        # Inner product for each patch: element-wise product then sum over channels
        conditional_term_values = (phi_features * projected_cond_embed_spatial).sum(dim=1, keepdim=True)  # [B, 1, Patch_H, Patch_W]
        final_patch_logits = unconditional_patch_logits + conditional_term_values

        return final_patch_logits


# class ModelConfig:
#     def __init__(self, num_conditions, cond_embed_dim=64, hidden_channels_gen=32, hidden_channels_disc=64, use_cbn=True, use_adain=False,
#             use_spectral_norm=True, transformer_nhead=4, transformer_layers=2):
#         self.num_conditions = num_conditions
#         self.cond_embed_dim = cond_embed_dim
#         self.hidden_channels_gen = hidden_channels_gen
#         self.hidden_channels_disc = hidden_channels_disc
#         self.use_cbn = use_cbn
#         self.use_adain = use_adain
#         self.use_spectral_norm = use_spectral_norm
#         self.transformer_nhead = transformer_nhead
#         self.transformer_layers = transformer_layers


## model summary
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_conditions_example = 4
    batch_size_example = 64
    dummy_A = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_B = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_C = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_condition = torch.randint(0, num_conditions_example, (batch_size_example,), dtype=torch.long).to(device)
    model = EMGFusionGenerator(num_conditions=num_conditions_example).to(device)

    # Keras-like summary primarily shows Layer Name, Output Shape, and Param #
    print(f"--- Model Summary (Keras-like: Layer Name, Output Shape, Param #) ---")
    # 'input_size' is useful but not standard in Keras summary per layer, rather it's shown for the overall model.
    # 'kernel_size' is also not typically in the main Keras summary table per row.
    model_summary_obj = summary(model, input_data=(dummy_A, dummy_B, dummy_condition),
        col_names=["output_size", "num_params", "trainable"],  # We can also add "trainable" to distinguish trainable params
        # `row_settings=["var_names"]` will show variable names for layers if they have them (e.g. self.encoder1)
        row_settings=["var_names", "depth"],  # Adding depth can help with structure
        depth=3,  # Adjust depth to control nesting. For very nested models, a higher depth is informative.
        # For a Keras-like flat view, you might use depth=1 or 2 if top-level modules are simple.
        verbose=0  # Set to 0 to only return the object
    )
    print(model_summary_obj)

    # Print total parameters separately, as Keras does at the end
    print("================================================================")
    print(f"Total params: {model_summary_obj.total_params:,}")
    print(f"Trainable params: {model_summary_obj.trainable_params:,}")
    print(f"Non-trainable params: {model_summary_obj.total_params - model_summary_obj.trainable_params:,}")
    print("----------------------------------------------------------------")


