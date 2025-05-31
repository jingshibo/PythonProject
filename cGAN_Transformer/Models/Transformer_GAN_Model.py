import torch
import torch.nn as nn
import torch.nn.functional as F
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
            activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.02}):
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




# class EMGFusionSeparateGenerator(nn.Module):
#     def __init__(self, num_conditions, cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False,
#             initial_encoder_channels=32, hidden_channels=32, activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01}):
#         super().__init__()
#
#         if activation_params:
#             act_fn = activation_class(**activation_params)
#         else:
#             act_fn = activation_class()
#         self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)
#
#         # --- 1. Initial Separate Encoders for A and B ---
#         # These blocks should use padding to preserve H and W, as stride is (1,1)
#         self.initial_encoder_A = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1))
#         self.initial_encoder_B = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1))
#
#         # Encoder
#         self.encoder1 = ConvCBNBlock(initial_encoder_channels * 2, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
#         self.encoder2 = ConvCBNBlock(hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
#         self.encoder3 = ConvCBNBlock(hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
#         self.encoder4 = ConvCBNBlock(hidden_channels * 4, hidden_channels * 8, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
#
#         # Transformer
#         self.d_model_transformer = hidden_channels * 4
#         self.memory_projector = nn.Linear(cond_embed_dim, self.d_model_transformer)
#         transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model_transformer, nhead=4,
#             dim_feedforward=self.d_model_transformer * 2, dropout=0.1, activation=nn.GELU(), batch_first=True)
#         self.transformer = nn.TransformerDecoder(transformer_decoder_layer, num_layers=4)
#         self.transformer_output_norm = nn.LayerNorm(self.d_model_transformer)
#         self.pos_encoder = None
#
#         # Decoder with skip connections
#         self.decoder0 = ConvCBNBlock(hidden_channels * 8 + hidden_channels * 4, hidden_channels * 6, cond_embed_dim, use_cbn, use_adain,
#             use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
#         self.decoder1 = ConvCBNBlock(hidden_channels * 6 + hidden_channels * 2, hidden_channels * 4, cond_embed_dim, use_cbn, use_adain,
#             use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
#         self.decoder2 = ConvCBNBlock(hidden_channels * 4 + hidden_channels, hidden_channels * 2, cond_embed_dim, use_cbn, use_adain,
#             use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
#         self.decoder3 = ConvCBNBlock(hidden_channels * 2 + 2, 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=True,
#             activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
#
#         self.sig = torch.nn.Sigmoid()
#
#     def _init_pos_encoder(self, C_spatial_reduced, T_time_reduced, device):
#         if self.pos_encoder is None or self.pos_encoder.pe.shape[1] != C_spatial_reduced * T_time_reduced or \
#                 self.pos_encoder.pe.shape[2] != self.d_model_transformer:
#             self.pos_encoder = Additive2DSinusoidalPositionalEncoding(self.d_model_transformer, C_spatial_reduced, T_time_reduced,
#                 dropout=0.0).to(device)
#
#     def forward(self, A, B, condition):
#         cond_embed = self.condition_embedding(condition)
#         x = torch.cat([A, B], dim=1)
#
#         # 1. Initial Separate Feature Extraction
#         A_feat = self.initial_encoder_A(A, cond_embed)  # [B, init_enc_ch, H, W]
#         B_feat = self.initial_encoder_B(B, cond_embed)  # [B, init_enc_ch, H, W]
#         x_feat = torch.cat([A_feat, B_feat], dim=1)
#
#         # Encoder
#         e1_out = self.encoder1(x_feat, cond_embed)
#         e2_out = self.encoder2(e1_out, cond_embed)
#         e3_out = self.encoder3(e2_out, cond_embed)
#         e4_out = self.encoder4(e3_out, cond_embed)
#         B_batch, C_encoder, C_spatial_reduced, T_time_reduced = e3_out.shape
#         assert C_encoder == self.d_model_transformer
#
#         # transformer_input = e3_out.permute(0, 2, 3, 1).reshape(B_batch, C_spatial_reduced * T_time_reduced, C_encoder)
#         # self._init_pos_encoder(C_spatial_reduced, T_time_reduced, transformer_input.device)
#         # transformer_input_pos_encoded = self.pos_encoder(transformer_input)
#         # projected_cond_embed = self.memory_projector(cond_embed)
#         # memory = projected_cond_embed.unsqueeze(1)
#         #
#         # transformed_features = self.transformer(tgt=transformer_input_pos_encoded, memory=memory)
#         # transformer_output = transformer_input_pos_encoded + transformed_features
#         # transformer_output = self.transformer_output_norm(transformer_output)
#         # transformer_output = transformer_output.reshape(B_batch, C_spatial_reduced, T_time_reduced,
#         #     self.d_model_transformer).permute(0, 3, 1, 2)
#
#         # Pass a tuple: (previous_decoder_layer_output, encoder_skip_feature)
#         d0_out = self.decoder0((e4_out, e3_out), cond_embed)
#         d1_out = self.decoder1((d0_out, e2_out), cond_embed)
#         d2_out = self.decoder2((d1_out, e1_out), cond_embed)
#         d3_out = self.decoder3((d2_out, x), cond_embed)  # x is the original input cat(A,B)
#         blending_factors = self.sig(d3_out)
#         generated_image = torch.mean(blending_factors * x, 1, keepdim=True)
#
#         return generated_image, blending_factors





## model summary
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_conditions_example = 4
    batch_size_example = 20
    dummy_A = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_B = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_C = torch.randn(batch_size_example, 1, 65, 1200).to(device)
    dummy_condition = torch.randint(0, num_conditions_example, (batch_size_example,), dtype=torch.long).to(device)
    model = EMGFusionSeparateGenerator(num_conditions=num_conditions_example).to(device)
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















# # --- Helper: Positional Encoding (1D for time sequence in Cross-Attention) ---
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0) # For batch_first=True, shape [1, max_len, d_model]
#         self.register_buffer('pe', pe)
#
#     def forward(self, x): # x is [Batch, SeqLen, EmbedDim]
#         # Ensure pe is on the same device as x and sliced correctly
#         x = x + self.pe[:, :x.size(1), :].to(x.device)
#         return self.dropout(x)
#
# # --- Helper: Cross-Attention Module (Simplified Transformer Encoder Layer style) ---
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, dim_feedforward_factor=2, dropout=0.1, activation_fn=nn.GELU()):
#         super().__init__()
#         self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * dim_feedforward_factor), activation_fn,
#             nn.Dropout(dropout), nn.Linear(embed_dim * dim_feedforward_factor, embed_dim))
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.dropout_res = nn.Dropout(dropout) # Dropout for residual connections
#
#     def forward(self, query_seq, kv_seq, query_pos=None, kv_pos=None):
#         # query_seq: Sequence that queries (e.g., decoder embeddings) — shape [B, L_q, D].
#         # kv_seq: Sequence that provides key and value (e.g., encoder embeddings) — shape [B, L_kv, D].
#         # query_pos, kv_pos: Optional positional encodings (same shape as sequences).
#
#         # Adds positional encodings to the query and key (but not to value — which is a common design choice).
#         q_input = query_seq + query_pos if query_pos is not None else query_seq
#         k_input = kv_seq + kv_pos if kv_pos is not None else kv_seq
#         v_input = kv_seq  # Value often doesn't get positional encoding added before MHA
#
#         attn_output, _ = self.mha(q_input, k_input, v_input)
#         # Residual connection for attention output
#         query_seq_res = query_seq + self.dropout_res(attn_output)
#         query_seq_norm1 = self.norm1(query_seq_res)
#
#         # FFN part
#         ffn_output = self.ffn(query_seq_norm1)
#         # Residual connection for FFN output
#         query_seq_res2 = query_seq_norm1 + self.dropout_res(ffn_output)
#         final_output = self.norm2(query_seq_res2)
#         return final_output
#
# # --- Main Generator ---
# class EMGFusionTransformerGenerator(nn.Module):
#     def __init__(self, num_conditions, input_h=65, input_w=1200, cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False,
#             initial_encoder_channels=32, hidden_channels_main_enc=32, activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01},
#             cross_attn_heads=4, cross_attn_layers=1, cross_attn_ff_factor=2):
#         super().__init__()
#
#         if activation_params:
#             act_fn = activation_class(**activation_params)
#         else:
#             act_fn = activation_class()
#         self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)
#         self.input_h = input_h
#         self.input_w = input_w
#
#         # --- 1. Initial Separate Encoders for A and B ---
#         # These blocks should use padding to preserve H and W, as stride is (1,1)
#         # For kernel (5,9), padding should be (2,4)
#         self.initial_encoder_A = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
#         self.initial_encoder_B = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
#
#         # --- 2. Cross-Attention Modules ---
#         self.embed_dim_ca = initial_encoder_channels * self.input_h
#
#         self.cross_attention_A_queries_B = nn.ModuleList()
#         for _ in range(cross_attn_layers):
#             self.cross_attention_A_queries_B.append(
#                 CrossAttentionBlock(embed_dim=self.embed_dim_ca, num_heads=cross_attn_heads, dim_feedforward_factor=cross_attn_ff_factor,
#                     activation_fn=act_fn))
#
#         # Add symmetric cross-attention: B queries A
#         self.cross_attention_B_queries_A = nn.ModuleList()
#         for _ in range(cross_attn_layers):
#             self.cross_attention_B_queries_A.append(
#                 CrossAttentionBlock(embed_dim=self.embed_dim_ca, num_heads=cross_attn_heads, dim_feedforward_factor=cross_attn_ff_factor,
#                     activation_fn=act_fn))
#
#         self.ca_pos_encoder_time = PositionalEncoding(d_model=self.embed_dim_ca, dropout=0.0, max_len=self.input_w + 10)
#
#         # --- 3. Main Encoder (Processes Fused/Attended Features) ---
#         main_encoder_in_channels = initial_encoder_channels + initial_encoder_channels  # attended_A_feat + B_feat
#         # Each ConvCBNBlock in the main encoder will need padding based on its kernel and stride
#         self.encoder1 = ConvCBNBlock(main_encoder_in_channels, hidden_channels_main_enc, cond_embed_dim, use_cbn, use_adain,
#             use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))
#         self.encoder2 = ConvCBNBlock(hidden_channels_main_enc, hidden_channels_main_enc * 2, cond_embed_dim, use_cbn, use_adain,
#             use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))
#         self.encoder3 = ConvCBNBlock(hidden_channels_main_enc * 2, hidden_channels_main_enc * 4, cond_embed_dim, use_cbn, use_adain,
#             use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))
#         # self.encoder4 = ConvCBNBlock(hidden_channels_main_enc * 4, hidden_channels_main_enc * 8, cond_embed_dim, use_cbn, use_adain,
#         #     use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))
#
#         # --- 4. Decoder with skip connections ---
#         # Each ConvCBNBlock (transpose=True) needs appropriate padding and output_padding
#         # self.decoder0 = ConvCBNBlock(hidden_channels_main_enc * 8 + hidden_channels_main_enc * 4, hidden_channels_main_enc * 6,
#         #     cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1),
#         #     dilation=(1, 1), upsample_scale=(1, 2))
#         self.decoder1 = ConvCBNBlock(hidden_channels_main_enc * 4 + hidden_channels_main_enc * 2, hidden_channels_main_enc * 4,
#             cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1),
#             dilation=(1, 1), upsample_scale=(1, 2))
#         self.decoder2 = ConvCBNBlock(hidden_channels_main_enc * 4 + hidden_channels_main_enc, hidden_channels_main_enc * 2, cond_embed_dim,
#             use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1),
#             upsample_scale=(1, 2))
#         self.decoder3 = ConvCBNBlock(hidden_channels_main_enc * 2 + initial_encoder_channels * 2, hidden_channels_main_enc * 2, cond_embed_dim, use_cbn, use_adain,
#             use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
#
#         self.final_decoder = ConvCBNBlock(hidden_channels_main_enc * 2, 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
#             transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))
#
#         self.sig = torch.nn.Sigmoid()
#
#     def forward(self, A, B, condition):  # A, B: [Batch, 1, H, W]
#         cond_embed = self.condition_embedding(condition)
#         original_AB_concat = torch.cat([A, B], dim=1)  # Used for final skip connection
#
#         # 1. Initial Separate Feature Extraction
#         A_feat = self.initial_encoder_A(A, cond_embed)  # [B, init_enc_ch, H, W]
#         B_feat = self.initial_encoder_B(B, cond_embed)  # [B, init_enc_ch, H, W]
#         # At this point, H_curr=self.input_h, W_curr=self.input_w due to padding in initial_encoders
#         B_batch, C_init_enc, H_curr, W_curr = A_feat.shape
#
#         # 2. Prepare for Cross-Attention
#         seq_A = A_feat.permute(0, 3, 1, 2).reshape(B_batch, W_curr, self.embed_dim_ca)
#         seq_B = B_feat.permute(0, 3, 1, 2).reshape(B_batch, W_curr, self.embed_dim_ca)
#
#         pos_enc_time = self.ca_pos_encoder_time(seq_A)  # Same PE for A and B sequences
#
#         # 2a. A queries B
#         attended_A_seq = seq_A
#         for ca_block in self.cross_attention_A_queries_B:
#             attended_A_seq = ca_block(attended_A_seq, seq_B, query_pos=pos_enc_time, kv_pos=pos_enc_time)
#         attended_A_feat = attended_A_seq.reshape(B_batch, W_curr, C_init_enc, H_curr).permute(0, 2, 3, 1)
#
#         # 2b. B queries A
#         attended_B_seq = seq_B
#         for ca_block in self.cross_attention_B_queries_A:
#             attended_B_seq = ca_block(attended_B_seq, seq_A, query_pos=pos_enc_time, kv_pos=pos_enc_time)  # Note: kv_seq is seq_A here
#         attended_B_feat = attended_B_seq.reshape(B_batch, W_curr, C_init_enc, H_curr).permute(0, 2, 3, 1)
#
#         # 3. Fuse features for the main encoder
#         fused_features = torch.cat([attended_A_feat, attended_B_feat], dim=1)  # [B, 2*initial_encoder_channels, H_curr, W_curr]
#
#         # 4. Main Encoder Path
#         e1_out = self.encoder1(fused_features, cond_embed)  # [B, hc_main, H_in, W_in/2]
#         e2_out = self.encoder2(e1_out, cond_embed)  # [B, hc_main*2, H_in, W_in/4]
#         e3_out = self.encoder3(e2_out, cond_embed)  # [B, hc_main*4, H_in, W_in/8]
#         # e4_out = self.encoder4(e3_out, cond_embed)  # [B, hc_main*8, H_in, W_in/16]
#
#         bottleneck_features = e3_out  # This is the input to the decoder
#
#         # 5. Decoder Path
#         # d0_out = self.decoder0((bottleneck_features, e3_out), cond_embed)
#         d1_out = self.decoder1((bottleneck_features, e2_out), cond_embed)
#         d2_out = self.decoder2((d1_out, e1_out), cond_embed)
#         d3_out = self.decoder3((d2_out, fused_features), cond_embed)
#
#         blending_factors = self.sig(self.final_decoder(d3_out, cond_embed))  # [B, 2, H_in, W_in]
#         generated_image = torch.mean(blending_factors * original_AB_concat, 1, keepdim=True)  # [B, 1, H_in, W_in]
#
#         return generated_image, blending_factors