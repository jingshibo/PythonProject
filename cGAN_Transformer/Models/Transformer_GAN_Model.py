import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torchinfo import summary
import math

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
            activation=nn.ReLU):
        super().__init__()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)

        # reduce time dim to a smaller number while incresing the CNN channel dim
        self.encoder1 = ConvCBNBlock(2 + 1, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=activation)
        self.encoder2 = ConvCBNBlock(hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=activation)
        self.encoder3 = ConvCBNBlock(hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=activation)
        self.encoder4 = ConvCBNBlock(hidden_channels * 4, hidden_channels * 8, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=activation)

        self.d_model = hidden_channels * 8 + cond_embed_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=self.d_model * 2, batch_first=True, dropout=0.1),
            num_layers=2)

        self.decoder1 = ConvCBNBlock(hidden_channels * 8 + cond_embed_dim, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=True, activation=activation)
        self.decoder2 = ConvCBNBlock(hidden_channels * 4, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=activation)
        self.decoder3 = ConvCBNBlock(hidden_channels * 2, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=activation)
        self.decoder4 = ConvCBNBlock(hidden_channels, 1, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=nn.Tanh)

        self.sig = torch.nn.Sigmoid()  # convert values to the range of [0, 1]
        self.pos_encoder = None  # will be initialized in the first forward pass

    def forward(self, A, B, condition):
        B_, _, C, T = A.shape

        cond_embed = self.condition_embedding(condition)  # Shape: [B, cond_embed_dim] (B: batch size)
        cond_for_cat = cond_embed.mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, C, T)  # to create shape: [B, 1, C, T]
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


import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm  # Assuming this is at the top of your file


# ConditionalBatchNorm2d and ConvCBNBlock remain the same as you defined them
# (assuming ConvCBNBlock uses stride=(1,2) for downsampling in time)

class EMGFusionPatchDiscriminator(nn.Module):
    def __init__(self, num_conditions, cond_embed_dim=64, use_cbn_in_blocks=False,  # Typically False for D unless specific reason
            use_adain_in_blocks=False, use_spectral_norm=True, hidden_channels=64, activation_class=nn.LeakyReLU,
            activation_params={'negative_slope': 0.2}):
        super().__init__()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)

        if activation_params:
            act_fn = activation_class(**activation_params)
        else:
            act_fn = activation_class()

        # Input channels to first block is 3 (A_ref, B_ref, C_sample each 1 channel)
        # Let's define the blocks with consistent naming for clarity
        self.block1 = ConvCBNBlock(3, hidden_channels, cond_embed_dim, use_cbn=use_cbn_in_blocks, use_adain=use_adain_in_blocks,
            use_spectral_norm=use_spectral_norm, activation=act_fn, transpose=False)
        # Output H_out = H_in, W_out = W_in / 2
        # e.g., (B, 64, 65, 600)

        self.block2 = ConvCBNBlock(hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn=use_cbn_in_blocks,
            use_adain=use_adain_in_blocks, use_spectral_norm=use_spectral_norm, activation=act_fn, transpose=False)
        # e.g., (B, 128, 65, 300)

        self.block3 = ConvCBNBlock(hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn=use_cbn_in_blocks,
            use_adain=use_adain_in_blocks, use_spectral_norm=use_spectral_norm, activation=act_fn, transpose=False)
        # e.g., (B, 256, 65, 150)

        self.block4 = ConvCBNBlock(hidden_channels * 4, hidden_channels * 8, cond_embed_dim, use_cbn=use_cbn_in_blocks,
            use_adain=use_adain_in_blocks, use_spectral_norm=use_spectral_norm, activation=act_fn, transpose=False)
        # e.g., (B, 512, 65, 75)

        # Final convolutional layer to produce the patch scores.
        # It does not downsample further with stride=1.
        # The number of input channels is (hidden_channels * 8) + cond_embed_dim
        # if we concatenate the condition embedding spatially.
        # A common PatchGAN approach is to NOT concatenate the global condition embedding
        # spatially before this last conv, but rather let the CBN layers handle conditioning,
        # or make the final conv layer itself conditional if needed, or simply output
        # a grid of scores and the loss function handles the condition matching.

        # Option 1: Simpler final layer (no spatial condition concatenation here)
        # The conditioning is handled by CBN if active, or by the loss function's targets.
        # Output channels = 1 (for real/fake score per patch)
        last_conv_in_channels = hidden_channels * 8
        self.final_conv = nn.Conv2d(last_conv_in_channels, 1, kernel_size=(3, 3), stride=1, padding=1)
        # You might use spectral_norm here too:  # self.final_conv = spectral_norm(nn.Conv2d(last_conv_in_channels, 1,
        # kernel_size=4, stride=1, padding=1))  # Kernel size 4, stride 1, padding 1 is common in some PatchGANs (e.g., Pix2Pix)  # Let's
        # stick to 3x3 padding 1 for now to maintain dimensions.

        # No Sigmoid here if using BCEWithLogitsLoss or a least-squares GAN loss.

    def forward(self, C_sample, A_ref, B_ref, condition_label):
        current_device = C_sample.device
        A_ref = A_ref.to(current_device)
        B_ref = B_ref.to(current_device)
        condition_label = condition_label.to(current_device)

        cond_embed = self.condition_embedding(condition_label)  # [B, cond_embed_dim]

        x = torch.cat([A_ref, B_ref, C_sample], dim=1)  # [B, 3, C_spatial_orig, T_orig]

        x = self.block1(x, cond_embed)
        x = self.block2(x, cond_embed)
        x = self.block3(x, cond_embed)
        x = self.block4(x, cond_embed)  # Output: [B, hidden_channels*8, C_spatial_reduced, T_reduced]
        # e.g., [B, 512, 65, 75]

        # The output of self.block4 is a feature map.
        # Now, apply the final convolution to get a grid of scores.
        patch_scores = self.final_conv(x)
        # Output shape: [B, 1, C_spatial_reduced, T_reduced]
        # e.g., [B, 1, 65, 75]
        # Each element in the 65x75 grid is a logit predicting if that corresponding patch is real or fake.

        return patch_scores


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


# model summary
device = "cuda" if torch.cuda.is_available() else "cpu"
num_conditions_example = 4

model = EMGFusionGenerator(num_conditions=num_conditions_example).to(device)

batch_size_example = 64

dummy_A = torch.randn(batch_size_example, 1, 65, 1200).to(device)
dummy_B = torch.randn(batch_size_example, 1, 65, 1200).to(device)
dummy_condition = torch.randint(0, num_conditions_example, (batch_size_example,), dtype=torch.long).to(device)

print(f"--- Model Summary (Keras-like: Layer Name, Output Shape, Param #) ---")

# Keras-like summary primarily shows Layer Name, Output Shape, and Param #
# 'input_size' is useful but not standard in Keras summary per layer,
# rather it's shown for the overall model.
# 'kernel_size' is also not typically in the main Keras summary table per row.

model_summary_obj = summary(model, input_data=(dummy_A, dummy_B, dummy_condition), # To get closer to Keras, we focus on these columns:
    col_names=["output_size", "num_params"], # We can also add "trainable" to distinguish trainable params
    # col_names=["output_size", "num_params", "trainable"],

    # `row_settings=["var_names"]` will show variable names for layers if they have them (e.g. self.encoder1)
    # Default already shows layer type.
    row_settings=["var_names", "depth"],  # Adding depth can help with structure

    depth=3,  # Adjust depth to control nesting. Keras summary is usually flatter.
    # For very nested models, a higher depth is informative.
    # For a Keras-like flat view, you might use depth=1 or 2 if top-level modules are simple.
    # But your ConvCBNBlock is a module, so depth=2 or 3 is good.
    verbose=0  # Set to 0 to only return the object
)

print(model_summary_obj)

# Print total parameters separately, as Keras does at the end
print("================================================================")
print(f"Total params: {model_summary_obj.total_params:,}")
print(f"Trainable params: {model_summary_obj.trainable_params:,}")
print(f"Non-trainable params: {model_summary_obj.total_params - model_summary_obj.trainable_params:,}")
print("----------------------------------------------------------------")
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = EMGFusionGenerator(num_conditions=4).to(device)
# batch_size = 16
# summary(
#     model,
#     input_data=(
#         torch.randn(batch_size, 1, 65, 1200).to(device),
#         torch.randn(batch_size, 1, 65, 1200).to(device),
#         torch.randint(0, 4, (batch_size,), dtype=torch.long).to(device)
#     ),
#     device=device, col_names=["input_size", "output_size", "num_params"], depth=3
# )



# class ConvCBNBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, cond_embed_dim, use_cbn=True, use_adain=False, use_spectral_norm=False, transpose=False,
#             activation=nn.ReLU):
#         super().__init__()
#         conv_layer = nn.ConvTranspose2d if transpose else nn.Conv2d
#         conv_kwargs = dict(kernel_size=(3, 9), stride=(1, 2), padding=(1, 4))
#         if transpose:  # use ConvTranspose2d to increase spatial resolution
#             conv_kwargs['output_padding'] = (0, 1)
#         conv = conv_layer(in_channels, out_channels, **conv_kwargs)
#         self.conv = spectral_norm(conv) if use_spectral_norm else conv
#
#         self.use_cbn = use_cbn
#         if use_cbn:
#             self.norm = ConditionalBatchNorm2d(out_channels, cond_embed_dim, use_adain=use_adain)
#         else:
#             self.norm = nn.BatchNorm2d(out_channels)
#
#         self.activation = activation() if isinstance(activation, type) else activation
#
#     def forward(self, x, cond_embed):
#         x = self.conv(x)
#         x = self.norm(x, cond_embed) if self.use_cbn else self.norm(x)
#         x = self.activation(x)
#         return x


# # The positional encoding is additive and separable. Treats time and space independently
# class Sinusoidal2dPositionalEncoding(nn.Module):
#     def __init__(self, height, width, d_model):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.d_model = d_model
#
#         pe_h = torch.zeros(height, d_model // 2)
#         pe_w = torch.zeros(width, d_model // 2)
#
#         position_h = torch.arange(0, height).unsqueeze(1)
#         position_w = torch.arange(0, width).unsqueeze(1)
#
#         div_term = torch.exp(torch.arange(0, d_model // 2, 2) * -(torch.log(torch.tensor(10000.0)) / (d_model // 2)))
#
#         pe_h[:, 0::2] = torch.sin(position_h * div_term)
#         pe_h[:, 1::2] = torch.cos(position_h * div_term)
#         pe_w[:, 0::2] = torch.sin(position_w * div_term)
#         pe_w[:, 1::2] = torch.cos(position_w * div_term)
#
#         pe = pe_h.unsqueeze(1) + pe_w.unsqueeze(0)  # [H, W, d_model//2]
#         pe = pe.reshape(height * width, d_model)
#         self.register_buffer('pe', pe.unsqueeze(1))  # [H*W, 1, d_model]
#
#     def forward(self, x):
#         return x + self.pe[:x.size(0)]

# class EMGFusionGenerator(nn.Module):
#     def __init__(self, num_conditions, cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False, hidden_channels=32,
#             activation=nn.ReLU):
#         super().__init__()
#         self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)  # embed 4 conditions into a 64 dimension vector
#
#         self.encoder1 = ConvCBNBlock(2, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=activation)
#         self.encoder2 = ConvCBNBlock(hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, activation=activation)
#
#         d_model = hidden_channels * 2 + cond_embed_dim
#         self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 2),
#             num_layers=2)
#
#         self.decoder1 = ConvCBNBlock(hidden_channels * 2, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=True, activation=activation)
#         self.decoder2 = ConvCBNBlock(hidden_channels, 1, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=True, activation=nn.Tanh)
#
#     def forward(self, A, B, condition):
#         # In this generator architecture, the condition is not directly concatenated to the input or feature maps.
#         # Instead, it's passed through an embedding and used exclusively as input to the conditional normalization layers.
#         cond_embed = self.condition_embedding(condition)
#         x = torch.cat([A, B], dim=1)  # [B, 2, C, T]
#
#         x = self.encoder1(x, cond_embed)
#         x = self.encoder2(x, cond_embed)
#
#         B_, C_enc, C, T_red = x.shape
#         x = x.permute(3, 0, 2, 1).reshape(T_red, B_, C * C_enc)  # (time length * batch size * features per time step)
#
#         cond_seq = cond_embed.unsqueeze(0).repeat(T_red, 1, 1)
#         x = torch.cat([x, cond_seq], dim=-1)
#         x = self.transformer(x)
#
#         x = x[:, :, :C * C_enc].reshape(T_red, B_, C, C_enc).permute(1, 3, 2, 0)
#         x = self.decoder1(x, cond_embed)
#         x = self.decoder2(x)
#         return x
#
