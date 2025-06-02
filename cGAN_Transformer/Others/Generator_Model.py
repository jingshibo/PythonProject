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



##
# Assuming GatedFusionUnit_Conv is defined as in the previous example
class EMGFusionGFUGenerator(nn.Module):
    def __init__(self, num_conditions, input_h=65, input_w=320,
                 cond_embed_dim=64, use_cbn=True, use_adain=False, use_spectral_norm=False,
                 initial_encoder_channels=32, hidden_channels_main_enc=32, # Renamed
                 gmu_output_channels=32, # Channels out of GMU, can be same as initial_encoder_channels
                 activation_class=nn.LeakyReLU, activation_params={'negative_slope': 0.01}):
        super().__init__()

        if activation_params:
            act_fn = activation_class(**activation_params)
        else:
            act_fn = activation_class()
        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)
        self.input_h = input_h
        self.input_w = input_w

        # === STAGE 1: Initial Separate Feature Extraction ===
        # These should use padding to preserve HxW for GMU input if GMU expects full res
        self.initial_encoder_A = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), padding=(2,4), dilation=(1, 1))
        self.initial_encoder_B = ConvCBNBlock(1, initial_encoder_channels, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), padding=(2,4), dilation=(1, 1))

        # === STAGE 2: Gated Multi-Modal Fusion ===
        self.gmu_fusion = GatedFusionUnit_Conv(
            channels_A=initial_encoder_channels,
            channels_B=initial_encoder_channels,
            output_channels=gmu_output_channels, # This will be the input channels for encoder1
            common_hidden_channels=gmu_output_channels # Or another intermediate dim for GMU
        )

        # === STAGE 3: Main Shared Encoder (Processes Fused Features) ===
        # Input channels for encoder1 is now gmu_output_channels
        self.encoder1 = ConvCBNBlock(gmu_output_channels, hidden_channels_main_enc, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), padding=(2,4))
        self.encoder2 = ConvCBNBlock(hidden_channels_main_enc, hidden_channels_main_enc * 2, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), padding=(2,4))
        self.encoder3 = ConvCBNBlock(hidden_channels_main_enc * 2, hidden_channels_main_enc * 4, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), padding=(2,4))
        self.encoder4 = ConvCBNBlock(hidden_channels_main_enc * 4, hidden_channels_main_enc * 8, cond_embed_dim, use_cbn, use_adain,
            use_spectral_norm, transpose=False, activation=act_fn, kernel_size=(5, 9), stride=(1, 2), padding=(2,4))

        # === STAGE 4: Decoder (Takes bottleneck features and skip connections) ===
        # Skip connections come from the main shared encoder
        self.decoder0 = ConvCBNBlock(hidden_channels_main_enc * 8 + hidden_channels_main_enc * 4, hidden_channels_main_enc * 6, cond_embed_dim,
            use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1),
            padding=(2,4), dilation=(1, 1), upsample_scale=(1, 2)) # Skip from encoder3
        self.decoder1 = ConvCBNBlock(hidden_channels_main_enc * 6 + hidden_channels_main_enc * 2, hidden_channels_main_enc * 4, cond_embed_dim,
            use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1),
            padding=(2,4), dilation=(1, 1), upsample_scale=(1, 2)) # Skip from encoder2
        self.decoder2 = ConvCBNBlock(hidden_channels_main_enc * 4 + hidden_channels_main_enc, hidden_channels_main_enc * 2, cond_embed_dim,
            use_cbn, use_adain, use_spectral_norm, transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1),
            padding=(2,4), dilation=(1, 1), upsample_scale=(1, 2)) # Skip from encoder1
        # Final skip from original A,B concatenated
        self.decoder3 = ConvCBNBlock(hidden_channels_main_enc * 2 + 2, 2, cond_embed_dim, use_cbn, use_adain, use_spectral_norm,
            transpose=True, activation=act_fn, kernel_size=(5, 9), stride=(1, 1), padding=(2,4), dilation=(1, 1), upsample_scale=(1, 2))

        self.sig = torch.nn.Sigmoid()

    def forward(self, A, B, condition): # A, B: [Batch, 1, H, W]
        assert A.shape[2] == self.input_h and A.shape[3] == self.input_w
        assert B.shape[2] == self.input_h and B.shape[3] == self.input_w

        cond_embed = self.condition_embedding(condition)
        original_AB_concat = torch.cat([A, B], dim=1)

        # STAGE 1: Initial Separate Feature Extraction
        A_feat = self.initial_encoder_A(A, cond_embed) # [B, initial_encoder_channels, H, W]
        B_feat = self.initial_encoder_B(B, cond_embed) # [B, initial_encoder_channels, H, W]

        # STAGE 2: Gated Multi-Modal Fusion
        fused_features = self.gmu_fusion(A_feat, B_feat) # [B, gmu_output_channels, H, W]

        # STAGE 3: Main Shared Encoder Path
        e1_out = self.encoder1(fused_features, cond_embed) # [B, hc_main, H, W/2]
        e2_out = self.encoder2(e1_out, cond_embed)         # [B, hc_main*2, H, W/4]
        e3_out = self.encoder3(e2_out, cond_embed)         # [B, hc_main*4, H, W/8]
        e4_out = self.encoder4(e3_out, cond_embed)         # [B, hc_main*8, H, W/16]

        bottleneck_features = e4_out

        # STAGE 4: Decoder Path
        d0_out = self.decoder0((bottleneck_features, e3_out), cond_embed)
        d1_out = self.decoder1((d0_out, e2_out), cond_embed)
        d2_out = self.decoder2((d1_out, e1_out), cond_embed)
        d3_out = self.decoder3((d2_out, original_AB_concat), cond_embed)

        blending_factors = self.sig(d3_out)
        generated_image = torch.mean(blending_factors * original_AB_concat, 1, keepdim=True) * 2

        return generated_image, blending_factors

# --- GatedFusionUnit using 1x1 Convolutions for feature maps (Assumed to be defined) ---
class GatedFusionUnit_Conv(nn.Module):
    def __init__(self, channels_A, channels_B, output_channels, common_hidden_channels=None):
        super(GatedFusionUnit_Conv, self).__init__()
        if common_hidden_channels is None:
            common_hidden_channels = output_channels # Or some other factor like output_channels // 2

        self.proj_A_htilde = nn.Conv2d(channels_A, common_hidden_channels, kernel_size=1, bias=False)
        self.proj_B_htilde = nn.Conv2d(channels_B, common_hidden_channels, kernel_size=1, bias=True) # One bias is enough for the sum

        self.proj_A_g = nn.Conv2d(channels_A, output_channels, kernel_size=1, bias=False)
        self.proj_B_g = nn.Conv2d(channels_B, output_channels, kernel_size=1, bias=True) # One bias is enough

        self.htilde_final_proj = nn.Conv2d(common_hidden_channels, output_channels, kernel_size=1)
        self.transform_A_residual = nn.Conv2d(channels_A, output_channels, kernel_size=1)
        # It's common for the residual path to also be learnable, or it could be an identity if channels_A == output_channels

    def forward(self, feat_A, feat_B):
        h_tilde_intermediate = torch.tanh(self.proj_A_htilde(feat_A) + self.proj_B_htilde(feat_B))
        h_tilde = self.htilde_final_proj(h_tilde_intermediate)

        gate = torch.sigmoid(self.proj_A_g(feat_A) + self.proj_B_g(feat_B))
        transformed_A_res = self.transform_A_residual(feat_A)

        fused_output = gate * h_tilde + (1 - gate) * transformed_A_res
        return fused_output



# --- GatedFusionUnit using 1x1 Convolutions for feature maps ---
class GatedFusionUnit_Conv(nn.Module):
    def __init__(self, channels_A, channels_B, output_channels, common_hidden_channels=None):
        super(GatedFusionUnit_Conv, self).__init__()
        if common_hidden_channels is None:
            common_hidden_channels = output_channels

        # Project A and B to a common hidden dimension for h_tilde and gate calculation
        self.proj_A_htilde = nn.Conv2d(channels_A, common_hidden_channels, kernel_size=1)
        self.proj_B_htilde = nn.Conv2d(channels_B, common_hidden_channels, kernel_size=1)

        self.proj_A_g = nn.Conv2d(channels_A, output_channels, kernel_size=1) # Gate dim = output_channels
        self.proj_B_g = nn.Conv2d(channels_B, output_channels, kernel_size=1)

        # To produce h_tilde of output_channels
        self.htilde_final_proj = nn.Conv2d(common_hidden_channels, output_channels, kernel_size=1)

        # Optional: transform for the (1-g) * feat_A_transformed part
        # Ensure this output matches output_channels
        self.transform_A_residual = nn.Conv2d(channels_A, output_channels, kernel_size=1)

    def forward(self, feat_A, feat_B): # feat_A/B are [B, C_A/B, H, W]
        # Candidate h_tilde path
        projected_A_h = self.proj_A_htilde(feat_A)
        projected_B_h = self.proj_B_htilde(feat_B)
        h_tilde_intermediate = torch.tanh(projected_A_h + projected_B_h) # [B, common_hidden_channels, H, W]
        h_tilde = self.htilde_final_proj(h_tilde_intermediate) # [B, output_channels, H, W]

        # Gate path
        projected_A_g = self.proj_A_g(feat_A)
        projected_B_g = self.proj_B_g(feat_B)
        gate = torch.sigmoid(projected_A_g + projected_B_g) # [B, output_channels, H, W]

        # Residual path (using feat_A as the one being preserved when gate is low)
        transformed_A_res = self.transform_A_residual(feat_A) # [B, output_channels, H, W]

        fused_output = gate * h_tilde + (1 - gate) * transformed_A_res
        return fused_output

