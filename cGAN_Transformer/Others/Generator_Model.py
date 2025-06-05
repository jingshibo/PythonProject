import torch
import torch.nn as nn


##
# --- Helper: Positional Encoding (1D for time sequence in Cross-Attention) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # For batch_first=True, shape [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x): # x is [Batch, SeqLen, EmbedDim]
        # Ensure pe is on the same device as x and sliced correctly
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# --- Helper: Cross-Attention Module (Simplified Transformer Encoder Layer style) ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward_factor=2, dropout=0.1, activation_fn=nn.GELU()):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * dim_feedforward_factor), activation_fn,
            nn.Dropout(dropout), nn.Linear(embed_dim * dim_feedforward_factor, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_res = nn.Dropout(dropout) # Dropout for residual connections

    def forward(self, query_seq, kv_seq, query_pos=None, kv_pos=None):
        # query_seq: Sequence that queries (e.g., decoder embeddings) — shape [B, L_q, D].
        # kv_seq: Sequence that provides key and value (e.g., encoder embeddings) — shape [B, L_kv, D].
        # query_pos, kv_pos: Optional positional encodings (same shape as sequences).

        # Adds positional encodings to the query and key (but not to value — which is a common design choice).
        q_input = query_seq + query_pos if query_pos is not None else query_seq
        k_input = kv_seq + kv_pos if kv_pos is not None else kv_seq
        v_input = kv_seq  # Value often doesn't get positional encoding added before MHA

        attn_output, _ = self.mha(q_input, k_input, v_input)
        # Residual connection for attention output
        query_seq_res = query_seq + self.dropout_res(attn_output)
        query_seq_norm1 = self.norm1(query_seq_res)

        # FFN part
        ffn_output = self.ffn(query_seq_norm1)
        # Residual connection for FFN output
        query_seq_res2 = query_seq_norm1 + self.dropout_res(ffn_output)
        final_output = self.norm2(query_seq_res2)
        return final_output

# --- Main Generator ---
class EMGFusionTransformerGenerator(nn.Module):
    def __init__(self, num_conditions, input_h=65, input_w=1200, cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False,
            initial_encoder_channels=32, hidden_channels_main_enc=32, activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01},
            cross_attn_heads=4, cross_attn_layers=1, cross_attn_ff_factor=2):
        super().__init__()

        if activation_params:
            act_fn = activation_class(**activation_params)
        else:
            act_fn = activation_class()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)
        self.input_h = input_h
        self.input_w = input_w

        # --- 1. Initial Separate Encoders for A and B ---
        # These blocks should use padding to preserve H and W, as stride is (1,1)
        # For kernel (5,9), padding should be (2,4)
        self.initial_encoder_A = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))
        self.initial_encoder_B = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), dilation=(1, 1))

        # --- 2. Cross-Attention Modules ---
        self.embed_dim_ca = initial_encoder_channels * self.input_h

        self.cross_attention_A_queries_B = nn.ModuleList()
        for _ in range(cross_attn_layers):
            self.cross_attention_A_queries_B.append(
                CrossAttentionBlock(embed_dim=self.embed_dim_ca, num_heads=cross_attn_heads, dim_feedforward_factor=cross_attn_ff_factor,
                    activation_fn=act_fn))

        # Add symmetric cross-attention: B queries A
        self.cross_attention_B_queries_A = nn.ModuleList()
        for _ in range(cross_attn_layers):
            self.cross_attention_B_queries_A.append(
                CrossAttentionBlock(embed_dim=self.embed_dim_ca, num_heads=cross_attn_heads, dim_feedforward_factor=cross_attn_ff_factor,
                    activation_fn=act_fn))

        self.ca_pos_encoder_time = PositionalEncoding(d_model=self.embed_dim_ca, dropout=0.0, max_len=self.input_w + 10)

        # --- 3. Main Encoder (Processes Fused/Attended Features) ---
        main_encoder_in_channels = initial_encoder_channels + initial_encoder_channels  # attended_A_feat + B_feat
        # Each ConvCBNBlock in the main encoder will need padding based on its kernel and stride
        self.encoder1 = ConvCBNBlock(main_encoder_in_channels, hidden_channels_main_enc, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))
        self.encoder2 = ConvCBNBlock(hidden_channels_main_enc, hidden_channels_main_enc * 2, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))
        self.encoder3 = ConvCBNBlock(hidden_channels_main_enc * 2, hidden_channels_main_enc * 4, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))
        # self.encoder4 = ConvCBNBlock(hidden_channels_main_enc * 4, hidden_channels_main_enc * 8, cond_embed_dim, use_cbn, use_adain,
        #     use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2))

        # --- 4. Decoder with skip connections ---
        # Each ConvCBNBlock (transpose=True) needs appropriate padding and output_padding
        # self.decoder0 = ConvCBNBlock(hidden_channels_main_enc * 8 + hidden_channels_main_enc * 4, hidden_channels_main_enc * 6,
        #     cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1),
        #     dilation=(1, 1), upsample_scale=(1, 2))
        self.decoder1 = ConvCBNBlock(hidden_channels_main_enc * 4 + hidden_channels_main_enc * 2, hidden_channels_main_enc * 4,
            cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1),
            dilation=(1, 1), upsample_scale=(1, 2))
        self.decoder2 = ConvCBNBlock(hidden_channels_main_enc * 4 + hidden_channels_main_enc, hidden_channels_main_enc * 2, cond_embed_dim,
            use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1),
            upsample_scale=(1, 2))
        self.decoder3 = ConvCBNBlock(hidden_channels_main_enc * 2 + initial_encoder_channels * 2, hidden_channels_main_enc * 2, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))

        self.final_decoder = ConvCBNBlock(hidden_channels_main_enc * 2, 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), dilation=(1, 1), upsample_scale=(1, 2))

        self.sig = torch.nn.Sigmoid()

    def forward(self, A, B, condition):  # A, B: [Batch, 1, H, W]
        cond_embed = self.condition_embedding(condition)
        original_AB_concat = torch.cat([A, B], dim=1)  # Used for final skip connection

        # 1. Initial Separate Feature Extraction
        A_feat = self.initial_encoder_A(A, cond_embed)  # [B, init_enc_ch, H, W]
        B_feat = self.initial_encoder_B(B, cond_embed)  # [B, init_enc_ch, H, W]
        # At this point, H_curr=self.input_h, W_curr=self.input_w due to padding in initial_encoders
        B_batch, C_init_enc, H_curr, W_curr = A_feat.shape

        # 2. Prepare for Cross-Attention
        seq_A = A_feat.permute(0, 3, 1, 2).reshape(B_batch, W_curr, self.embed_dim_ca)
        seq_B = B_feat.permute(0, 3, 1, 2).reshape(B_batch, W_curr, self.embed_dim_ca)

        pos_enc_time = self.ca_pos_encoder_time(seq_A)  # Same PE for A and B sequences

        # 2a. A queries B
        attended_A_seq = seq_A
        for ca_block in self.cross_attention_A_queries_B:
            attended_A_seq = ca_block(attended_A_seq, seq_B, query_pos=pos_enc_time, kv_pos=pos_enc_time)
        attended_A_feat = attended_A_seq.reshape(B_batch, W_curr, C_init_enc, H_curr).permute(0, 2, 3, 1)

        # 2b. B queries A
        attended_B_seq = seq_B
        for ca_block in self.cross_attention_B_queries_A:
            attended_B_seq = ca_block(attended_B_seq, seq_A, query_pos=pos_enc_time, kv_pos=pos_enc_time)  # Note: kv_seq is seq_A here
        attended_B_feat = attended_B_seq.reshape(B_batch, W_curr, C_init_enc, H_curr).permute(0, 2, 3, 1)

        # 3. Fuse features for the main encoder
        fused_features = torch.cat([attended_A_feat, attended_B_feat], dim=1)  # [B, 2*initial_encoder_channels, H_curr, W_curr]

        # 4. Main Encoder Path
        e1_out = self.encoder1(fused_features, cond_embed)  # [B, hc_main, H_in, W_in/2]
        e2_out = self.encoder2(e1_out, cond_embed)  # [B, hc_main*2, H_in, W_in/4]
        e3_out = self.encoder3(e2_out, cond_embed)  # [B, hc_main*4, H_in, W_in/8]
        # e4_out = self.encoder4(e3_out, cond_embed)  # [B, hc_main*8, H_in, W_in/16]

        bottleneck_features = e3_out  # This is the input to the decoder

        # 5. Decoder Path
        # d0_out = self.decoder0((bottleneck_features, e3_out), cond_embed)
        d1_out = self.decoder1((bottleneck_features, e2_out), cond_embed)
        d2_out = self.decoder2((d1_out, e1_out), cond_embed)
        d3_out = self.decoder3((d2_out, fused_features), cond_embed)

        blending_factors = self.sig(self.final_decoder(d3_out, cond_embed))  # [B, 2, H_in, W_in]
        generated_image = torch.mean(blending_factors * original_AB_concat, 1, keepdim=True)  # [B, 1, H_in, W_in]

        return generated_image, blending_factors




## # Assuming BilateralGatedFusionUnit_Conv and ConvCBNBlock are defined
import torch
import torch.nn as nn

class BilateralGatedFusionUnit_Conv(nn.Module):
    def __init__(self, channels_A, channels_B, output_channels,
                 transform_hidden_channels=None, gate_hidden_channels=None,
                 activation_fn=nn.GELU()):  # Using GELU as an example, can be LeakyReLU
        super(BilateralGatedFusionUnit_Conv, self).__init__()

        if transform_hidden_channels is None:
            transform_hidden_channels = output_channels
        if gate_hidden_channels is None:
            gate_hidden_channels = max(16, output_channels // 4)  # Smaller projection for gates

        # Transformation path for A (T_A)
        self.transform_A = nn.Sequential(
            nn.Conv2d(channels_A, transform_hidden_channels, kernel_size=1),  # Optional: Add a BatchNorm/LayerNorm here if not in ConvCBNBlock
            activation_fn,  # An activation for the transformation
            nn.Conv2d(transform_hidden_channels, output_channels, kernel_size=1)
            # The output of this path will be multiplied by g_A
        )

        # Transformation path for B (T_B)
        self.transform_B = nn.Sequential(
            nn.Conv2d(channels_B, transform_hidden_channels, kernel_size=1),
            activation_fn,
            nn.Conv2d(transform_hidden_channels, output_channels, kernel_size=1)
        )

        # Gate generation path for A (g_A)
        # The gate should be influenced by both A and B
        self.gate_A_conv1 = nn.Conv2d(channels_A + channels_B, gate_hidden_channels, kernel_size=1)
        self.gate_A_activation = activation_fn
        self.gate_A_conv2 = nn.Conv2d(gate_hidden_channels, output_channels, kernel_size=1)  # Output channels match transformed features

        # Gate generation path for B (g_B)
        self.gate_B_conv1 = nn.Conv2d(channels_A + channels_B, gate_hidden_channels, kernel_size=1)
        self.gate_B_activation = activation_fn
        self.gate_B_conv2 = nn.Conv2d(gate_hidden_channels, output_channels, kernel_size=1)

    def forward(self, feat_A, feat_B):  # feat_A/B are [B, C_A/B, H, W]
        # Transformed features
        T_A_feat = self.transform_A(feat_A)  # [B, output_channels, H, W]
        T_B_feat = self.transform_B(feat_B)  # [B, output_channels, H, W]

        # Concatenate original features for gate calculation
        # This allows gates to be conditioned on the joint information of A and B
        feat_AB_concat = torch.cat([feat_A, feat_B], dim=1)

        # Calculate gates
        g_A_intermediate = self.gate_A_activation(self.gate_A_conv1(feat_AB_concat))
        g_A = torch.sigmoid(self.gate_A_conv2(g_A_intermediate))  # [B, output_channels, H, W]

        g_B_intermediate = self.gate_B_activation(self.gate_B_conv1(feat_AB_concat))
        g_B = torch.sigmoid(self.gate_B_conv2(g_B_intermediate))  # [B, output_channels, H, W]

        # Gated fusion
        fused_output = g_A * T_A_feat + g_B * T_B_feat
        # Alternative: Some might normalize gates, e.g., softmax across g_A, g_B for each feature if they should sum to 1.
        # However, independent sigmoids allow both to be high or low.
        # If output_channels of T_A/T_B are high, this sum can also be high.
        # Another option for fusion:
        # fused_output = g_A * T_A_feat + (1 - g_A) * T_B_feat # If g_A and g_B are complementary for each channel
        # The current sum is more general.

        return fused_output


class ShallowFusionGenerator(nn.Module):
    def __init__(self, num_conditions, input_h=65, input_w=320, cond_embed_dim=64, use_cbn=True, use_adain=False,
            use_spectral_norm=False, initial_encoder_channels=32, bilateral_gate_transform_hidden=32, bilateral_gate_gate_hidden=16,
            fused_feature_channels=64,  # Output of BilateralGatedFusionUnit
            # Blending predictor head parameters
            blend_predictor_hidden_channels=[64, 32],  # Channels for layers in the predictor head
            activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01}):
        super().__init__()

        if activation_params:
            main_act_fn = activation_class(**activation_params)
        else:
            main_act_fn = activation_class()
        fusion_act_fn = nn.GELU()  # Or main_act_fn

        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)
        self.input_h = input_h
        self.input_w = input_w

        # === STAGE 1: Initial Separate Feature Extraction ===
        self.initial_encoder_A = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=main_act_fn, kernel_size=(5, 9), stride=(1, 1), padding=(2, 4))
        self.initial_encoder_B = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=main_act_fn, kernel_size=(5, 9), stride=(1, 1), padding=(2, 4))

        # === STAGE 2: Bilateral Gated Fusion ===
        self.bilateral_fusion = BilateralGatedFusionUnit_Conv(channels_A=initial_encoder_channels, channels_B=initial_encoder_channels,
            output_channels=fused_feature_channels, transform_hidden_channels=bilateral_gate_transform_hidden,
            gate_hidden_channels=bilateral_gate_gate_hidden, activation_fn=fusion_act_fn)

        # === STAGE 3: Shallow Blending Factor Predictor Head ===
        # This head operates at the same resolution as fused_features (input HxW)
        # It does not use downsampling/upsampling like a U-Net.
        predictor_layers = []
        current_channels = fused_feature_channels
        for h_dim in blend_predictor_hidden_channels:
            predictor_layers.append(
                ConvCBNBlock(current_channels, h_dim, cond_embed_dim, use_cbn, use_adain, use_spectral_norm, transpose=False,
                    activation=main_act_fn, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            current_channels = h_dim

        # Final convolution to get 2 channels for blending factors
        predictor_layers.append(nn.Conv2d(current_channels, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.blending_predictor_head = nn.Sequential(*predictor_layers)

        self.sig = torch.nn.Sigmoid()

    def forward(self, A, B, condition):
        assert A.shape[2] == self.input_h and A.shape[3] == self.input_w
        assert B.shape[2] == self.input_h and B.shape[3] == self.input_w

        cond_embed = self.condition_embedding(condition)
        original_AB_concat = torch.cat([A, B], dim=1)

        # STAGE 1: Initial Separate Feature Extraction
        A_feat = self.initial_encoder_A(A, cond_embed)  # [B, initial_encoder_channels, H, W]
        B_feat = self.initial_encoder_B(B, cond_embed)  # [B, initial_encoder_channels, H, W]

        # STAGE 2: Bilateral Gated Fusion
        fused_features = self.bilateral_fusion(A_feat, B_feat)  # [B, fused_feature_channels, H, W]

        # STAGE 3: Shallow Blending Factor Predictor Head
        # Pass cond_embed to each ConvCBNBlock if they are designed to take it.
        # If Sequential doesn't handle that, you'll need to call layers individually.
        # For simplicity, assuming ConvCBNBlock can work without explicit cond_embed if use_cbn=False,
        # or if conditional params are injected globally or not strictly needed in this shallow head.
        # A more robust way is to loop through layers and pass cond_embed if the layer is a ConvCBNBlock.

        # Let's assume for now the head takes fused_features and produces raw_blending_factors
        # If ConvCBNBlock needs cond_embed and is used in Sequential, this needs adjustment.
        # A simpler head without ConvCBNBlock:
        # temp_feat = fused_features
        # for layer in self.blending_predictor_head[:-1]: # All but the last conv
        #     if isinstance(layer, ConvCBNBlock): # This check is problematic with Sequential
        #         temp_feat = layer(temp_feat, cond_embed)
        #     else:
        #         temp_feat = layer(temp_feat)
        # raw_blending_factors = self.blending_predictor_head[-1](temp_feat)

        # Simpler to ensure blending_predictor_head layers don't *require* cond_embed,
        # or implement it as a Module that handles passing cond_embed.
        # For this example, let's make the head very simple for now:
        # (You'd build this with ConvCBNBlock or simpler nn.Conv2d + act_fn layers)

        # --- Revised Head (if ConvCBNBlock is hard to use in Sequential with extra arg) ---
        # Option: Make a small custom head module
        # For now, let's assume a simplified Sequential or that ConvCBNBlock
        # in the head doesn't strictly need cond_embed (e.g. if use_cbn=False in head)
        raw_blending_factors = self.blending_predictor_head(fused_features)  # [B, 2, H, W]

        blending_factors = self.sig(raw_blending_factors)
        generated_image = torch.mean(blending_factors * original_AB_concat, 1, keepdim=True)

        return generated_image, blending_factors




import torch
import torch.nn as nn

# Assume ConvCBNBlock is defined and handles padding for stride=(1,1) to maintain size.
# Example: kernel_size=(5,9) needs padding=(2,4)

class EMGFusionGFUGeneratorImproved(nn.Module):
    def __init__(self, num_conditions, cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False, hidden_channels=32,
            gate_intermediate_channels=16,  # Channels for intermediate gate computation
            transform_channels=32,  # Channels for T_A(A_feat), T_B(B_feat)
            blend_predictor_hidden_channels=32,  # Hidden channels in the blending factor predictor
            activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01}):
        super().__init__()

        act_fn = activation_class(**activation_params) if activation_params else activation_class()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)

        # --- STAGE 1: Initial Separate Feature Extractors for A and B ---
        self.initial_A_feat_extractor = ConvCBNBlock(1, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 1))
        self.initial_B_feat_extractor = ConvCBNBlock(1, hidden_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 1))

        # --- STAGE 2: Transformation Paths (Optional but Recommended) T_A, T_B ---
        # These transform A_feat and B_feat into representations ready for gated summation.
        # Output channels should be 'transform_channels'.
        self.transform_A = ConvCBNBlock(hidden_channels, transform_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(1, 1))  # 1x1 conv for transformation
        self.transform_B = ConvCBNBlock(hidden_channels, transform_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(1, 1))  # 1x1 conv

        # --- STAGE 3: Gate Generation (g_A, g_B) ---
        # Gates are functions of both A_feat and B_feat.
        # The output channels of these gate paths should match 'transform_channels' for element-wise product.
        self.gate_common_intermediate = ConvCBNBlock(hidden_channels * 2, gate_intermediate_channels, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(3, 3), padding=(1, 1))

        # Path for g_A
        self.gate_A_final_conv = nn.Conv2d(gate_intermediate_channels, transform_channels,
            kernel_size=(1, 1))  # No activation, sigmoid later
        # Path for g_B
        self.gate_B_final_conv = nn.Conv2d(gate_intermediate_channels, transform_channels,
            kernel_size=(1, 1))  # No activation, sigmoid later

        # --- STAGE 4: Gated Fusion & Blending Factor Prediction ---
        # The fused features will have 'transform_channels'.
        # This head predicts the 2 blending factors.
        self.blend_predictor_conv1 = ConvCBNBlock(transform_channels, blend_predictor_hidden_channels, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(3, 3), padding=(1, 1))
        self.blend_predictor_final_conv = nn.Conv2d(blend_predictor_hidden_channels, 2, kernel_size=(1, 1))  # Final output: 2 channels

        # Final sigmoid for blending factors is applied in forward
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, A, B, condition):  # A, B are [Batch, 1, H, W]
        cond_embed = self.condition_embedding(condition)

        # STAGE 1: Separate feature extraction
        A_feat = self.initial_A_feat_extractor(A, cond_embed)  # [B, hidden_channels, H, W]
        B_feat = self.initial_B_feat_extractor(B, cond_embed)  # [B, hidden_channels, H, W]

        # STAGE 2: Transform features for gated sum
        T_A_feat = self.transform_A(A_feat, cond_embed)  # [B, transform_channels, H, W]
        T_B_feat = self.transform_B(B_feat, cond_embed)  # [B, transform_channels, H, W]

        # STAGE 3: Calculate gates
        # Gates are based on the initial (untransformed) A_feat and B_feat concatenation
        # for richer contextual information for the gating decision.
        concat_for_gates = torch.cat([A_feat, B_feat], dim=1)  # [B, hidden_channels*2, H, W]
        gate_intermediate_out = self.gate_common_intermediate(concat_for_gates, cond_embed)  # [B, gate_intermediate_channels, H, W]

        g_A_logits = self.gate_A_final_conv(gate_intermediate_out)
        g_A = torch.sigmoid(g_A_logits)  # [B, transform_channels, H, W]

        g_B_logits = self.gate_B_final_conv(gate_intermediate_out)
        g_B = torch.sigmoid(g_B_logits)  # [B, transform_channels, H, W]

        # STAGE 4: Gated Fusion
        fused_features = g_A * T_A_feat + g_B * T_B_feat  # [B, transform_channels, H, W]

        # STAGE 5: Predict Blending Factors from fused_features
        blend_feat_hidden = self.blend_predictor_conv1(fused_features, cond_embed)
        raw_blending_factors = self.blend_predictor_final_conv(blend_feat_hidden)  # [B, 2, H, W]
        blending_factors = self.final_sigmoid(raw_blending_factors)  # Apply sigmoid here

        # Final image generation using original A and B
        generated_image = blending_factors[:, 0:1, :, :] * A + blending_factors[:, 1:2, :, :] * B
        # Used slicing blending_factors[:, 0:1, :, :] to keepdim for broadcasting if A/B are [B,1,H,W]

        return generated_image, blending_factors
