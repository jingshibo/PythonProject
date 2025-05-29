import torch
import torch.nn as nn
import torch.nn.functional as F
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


class ConvCBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_embed_dim, use_cbn=True, use_adain=False, use_spectral_norm=False, transpose=False,
            activation=nn.ReLU, pooling=None, kernel_size=(5, 9), stride=None, padding=None, dilation=(1, 1), upsample_scale=(1, 2)):
        super().__init__()

        self.use_cbn = use_cbn
        self.transpose = transpose

        # Default stride and padding if not specified
        stride = stride if stride is not None else ((1, 2) if not transpose else (1, 1))
        if padding is None:
            padding = ((dilation[0] * (kernel_size[0] - 1)) // 2, (dilation[1] * (kernel_size[1] - 1)) // 2,)

        # Upsample only for transpose=True
        if transpose:
            self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=False)
        else:
            self.upsample = None

        # Conv layer
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            padding_mode='reflect')
        self.conv = spectral_norm(conv) if use_spectral_norm else conv

        # Normalization
        if use_cbn:
            self.norm = ConditionalBatchNorm2d(out_channels, cond_embed_dim, use_adain=use_adain)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

        self.activation = activation() if isinstance(activation, type) else activation

        # Pooling (optional)
        if pooling == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = None

    def forward(self, x_input, cond_embed):
        processed_x = x_input

        # Handle residual input (tuple = skip connection)
        if isinstance(x_input, (tuple, list)) and len(x_input) == 2:
            x_prev, x_skip = x_input
            x_prev = self.upsample(x_prev) if self.upsample else x_prev
            processed_x = torch.cat([x_prev, x_skip], dim=1)
        elif self.upsample:
            processed_x = self.upsample(x_input)

        out = self.conv(processed_x)
        out = self.norm(out, cond_embed) if self.use_cbn else self.norm(out)
        out = self.activation(out)
        if self.pool:
            out = self.pool(out)
        return out


class EMGFusionGenerator(nn.Module):
    def __init__(self, num_conditions, cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False, hidden_channels=32,
            activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01}):
        super().__init__()

        if activation_params:
            act_fn = activation_class(**activation_params)
        else:
            act_fn = activation_class()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)

        # Encoder
        self.encoder1 = ConvCBNBlock(2, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=False,
            activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
        self.encoder2 = ConvCBNBlock(hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
        self.encoder3 = ConvCBNBlock(hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
        self.encoder4 = ConvCBNBlock(hidden_channels * 4, hidden_channels * 8, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))

        # Transformer
        self.d_model_transformer = hidden_channels * 4
        self.memory_projector = nn.Linear(cond_embed_dim, self.d_model_transformer)
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model_transformer, nhead=4,
            dim_feedforward=self.d_model_transformer * 2, dropout=0.1, activation=nn.GELU(), batch_first=True)
        self.transformer = nn.TransformerDecoder(transformer_decoder_layer, num_layers=4)
        self.transformer_output_norm = nn.LayerNorm(self.d_model_transformer)
        self.pos_encoder = None

        # Decoder with skip connections
        self.decoder0 = ConvCBNBlock(hidden_channels * 8 + hidden_channels * 4, hidden_channels * 6, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
        self.decoder1 = ConvCBNBlock(hidden_channels * 6 + hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
        self.decoder2 = ConvCBNBlock(hidden_channels * 4 + hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
        self.decoder3 = ConvCBNBlock(hidden_channels * 2 + 2, 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=True,
            activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))

        self.sig = torch.nn.Sigmoid()

    def _init_pos_encoder(self, C_spatial_reduced, T_time_reduced, device):
        if self.pos_encoder is None or self.pos_encoder.pe.shape[1] != C_spatial_reduced * T_time_reduced or \
                self.pos_encoder.pe.shape[2] != self.d_model_transformer:
            self.pos_encoder = Additive2DSinusoidalPositionalEncoding(self.d_model_transformer, C_spatial_reduced, T_time_reduced,
                dropout=0.0).to(device)

    def forward(self, A, B, condition):
        cond_embed = self.condition_embedding(condition)
        x = torch.cat([A, B], dim=1)

        # Encoder
        e1_out = self.encoder1(x, cond_embed)
        e2_out = self.encoder2(e1_out, cond_embed)
        e3_out = self.encoder3(e2_out, cond_embed)
        e4_out = self.encoder4(e3_out, cond_embed)
        B_batch, C_encoder, C_spatial_reduced, T_time_reduced = e3_out.shape
        assert C_encoder == self.d_model_transformer

        # transformer_input = e3_out.permute(0, 2, 3, 1).reshape(B_batch, C_spatial_reduced * T_time_reduced, C_encoder)
        # self._init_pos_encoder(C_spatial_reduced, T_time_reduced, transformer_input.device)
        # transformer_input_pos_encoded = self.pos_encoder(transformer_input)
        # projected_cond_embed = self.memory_projector(cond_embed)
        # memory = projected_cond_embed.unsqueeze(1)
        #
        # transformed_features = self.transformer(tgt=transformer_input_pos_encoded, memory=memory)
        # transformer_output = transformer_input_pos_encoded + transformed_features
        # transformer_output = self.transformer_output_norm(transformer_output)
        # transformer_output = transformer_output.reshape(B_batch, C_spatial_reduced, T_time_reduced,
        #     self.d_model_transformer).permute(0, 3, 1, 2)

        # Pass a tuple: (previous_decoder_layer_output, encoder_skip_feature)
        d0_out = self.decoder0((e4_out, e3_out), cond_embed)
        d1_out = self.decoder1((d0_out, e2_out), cond_embed)
        d2_out = self.decoder2((d1_out, e1_out), cond_embed)
        d3_out = self.decoder3((d2_out, x), cond_embed)  # x is the original input cat(A,B)
        blending_factors = self.sig(d3_out)
        generated_image = torch.mean(blending_factors * x, 1, keepdim=True)

        return generated_image, blending_factors


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

        # --- Backbone ---
        C_phi = hidden_channels  # Output channels of the backbone

        self.block1 = ConvCBNBlock(in_channels=1, out_channels=hidden_channels, cond_embed_dim=cond_embed_dim, use_cbn=use_cbn,
            use_adain=use_adain, use_spectral_norm=use_spectral_norm, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1),
            pooling='max', transpose=False)

        self.block2 = ConvCBNBlock(in_channels=hidden_channels, out_channels=hidden_channels * 2, cond_embed_dim=cond_embed_dim,
            use_cbn=use_cbn, use_adain=use_adain, use_spectral_norm=use_spectral_norm, activation=act_fn, kernel_size=(5, 9), stride=(1, 2),
            dilation=(1, 1), pooling='max', transpose=False)

        self.block3 = ConvCBNBlock(in_channels=hidden_channels * 2, out_channels=C_phi, cond_embed_dim=cond_embed_dim, use_cbn=use_cbn,
            use_adain=use_adain, use_spectral_norm=use_spectral_norm, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1),
            pooling='max', transpose=False)

        # self.block4 = ConvCBNBlock(in_channels=hidden_channels * 4, out_channels=C_phi, cond_embed_dim=cond_embed_dim, use_cbn=use_cbn,
        #     use_adain=use_adain, use_spectral_norm=use_spectral_norm, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4),
        #     dilation=(1, 1), pooling='max', transpose=False)

        # --- Unconditional Patch Score Head ---
        final_conv_unconditional_layer = nn.Conv2d(C_phi, 1, kernel_size=(3, 3), stride=1, padding=1)
        self.final_conv_unconditional = spectral_norm(
            final_conv_unconditional_layer) if use_spectral_norm else final_conv_unconditional_layer
        # --- For Conditional Projection Term ---
        self.cond_projector_for_phi = nn.Linear(cond_embed_dim, C_phi)  # Project condition to match feature channels

    def forward(self, C_sample, condition_label):
        cond_embed = self.condition_embedding(condition_label)  # [B, cond_embed_dim]
        x = C_sample

        phi_features = self.block1(x, cond_embed)
        phi_features = self.block2(phi_features, cond_embed)
        phi_features = self.block3(phi_features, cond_embed)
        # phi_features = self.block4(phi_features, cond_embed)  # [B, C_phi, Patch_H, Patch_W]
        unconditional_patch_logits = self.final_conv_unconditional(phi_features)  # Unconditional part, [B, 1, Patch_H, Patch_W]

        # Conditional Projection
        projected_cond_embed = self.cond_projector_for_phi(cond_embed)  # [B, C_phi]
        projected_cond_embed_spatial = projected_cond_embed.unsqueeze(-1).unsqueeze(-1).expand_as(phi_features)  # [B, C_phi, Patch_H, Patch_W]

        # Inner product for each patch: element-wise product then sum over channels
        conditional_term_values = (phi_features * projected_cond_embed_spatial).sum(dim=1, keepdim=True)  # [B, 1, Patch_H, Patch_W]
        # The final discriminator output (logit) is the sum of the unconditional score and this conditional inner product term.
        final_patch_logits = unconditional_patch_logits + conditional_term_values

        return final_patch_logits


## model summary
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_conditions_example = 4
    batch_size_example = 20
    dummy_A = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_B = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_C = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_condition = torch.randint(0, num_conditions_example, (batch_size_example,), dtype=torch.long).to(device)
    model = EMGFusionGenerator(num_conditions=num_conditions_example).to(device)
    # model = EMGFusionPatchDiscriminator(num_conditions=num_conditions_example).to(device)

    # Keras-like summary primarily shows Layer Name, Output Shape, and Param #
    print(f"--- Model Summary (Keras-like: Layer Name, Output Shape, Param #) ---")
    # 'input_size' is useful but not standard in Keras summary per layer, rather it's shown for the overall model.
    # 'kernel_size' is also not typically in the main Keras summary table per row.
    model_summary_obj = summary(model, input_data=(dummy_C, dummy_C, dummy_condition),
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








