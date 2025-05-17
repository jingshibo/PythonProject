import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchinfo import summary


# ------------------------------
# Device setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Generator: (A, B, condition) -> C using 2D Convs + condition embedding
# ------------------------------
class EMGFusionGenerator(nn.Module):
    def __init__(self, num_channels=65, time=1200, num_conditions=2, cond_embed_dim=64):
        super(EMGFusionGenerator, self).__init__()

        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        d_model = 64 + cond_embed_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model*2), num_layers=2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), output_padding=(0,1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4), output_padding=(0,1)),
            nn.Tanh()
        )

    def forward(self, A, B, condition):
        x = torch.cat([A, B], dim=1)  # [B, 2, C, T]
        x = self.encoder(x)           # [B, 64, C, T//4]

        B_, C_enc, C, T_red = x.shape
        x = x.permute(3, 0, 2, 1).reshape(T_red, B_, C * C_enc)  # [T, B, CC_enc]

        cond_embed = self.condition_embedding(condition)  # [B, cond_embed_dim]
        cond_embed = cond_embed.unsqueeze(0).repeat(T_red, 1, 1)  # [T, B, cond_embed_dim]

        x = torch.cat([x, cond_embed], dim=-1)  # [T, B, CC_enc + cond_embed_dim]
        x = self.transformer(x)       # [T, B, CC_enc + cond_embed_dim]

        x = x[:, :, :C * C_enc].reshape(T_red, B_, C, C_enc).permute(1, 3, 2, 0)  # [B, C_enc, C, T_red]
        x = self.decoder(x)           # [B, 1, C, T]
        return x

# ------------------------------
# Discriminator: (A, B, C, condition) -> real/fake using 2D Convs + condition embedding
# ------------------------------
class EMGFusionDiscriminator(nn.Module):
    def __init__(self, num_channels=65, time=1200, num_conditions=2, cond_embed_dim=64):
        super(EMGFusionDiscriminator, self).__init__()

        self.condition_embedding = nn.Embedding(num_conditions, cond_embed_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 + cond_embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, A, B, C, condition):
        x = torch.cat([A, B, C], dim=1)  # [B, 3, C, T]
        x = self.conv(x)                # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 256]

        cond_embed = self.condition_embedding(condition)  # [B, cond_embed_dim]
        x = torch.cat([x, cond_embed], dim=1)             # [B, 256 + cond_embed_dim]

        return self.classifier(x)                         # [B, 1]

# ------------------------------
# Dataset class to load EMG fusion data
# ------------------------------
class EMGFusionDataset(Dataset):
    def __init__(self, data_dict, condition_map, batch_size, repetition, time=1200):
        self.data = []
        for cond_name, group in data_dict.items():
            label = condition_map[cond_name]
            self.data.append({
                'gen_data_1': group['gen_data_1'],
                'gen_data_2': group['gen_data_2'],
                'disc_data': group['disc_data'],
                'label': label
            })
        self.segment_length = time
        self.batch_size = batch_size
        self.repetition = repetition

    def __len__(self):  # number of samples to train per epoch
        return self.batch_size * self.repetition

    def __getitem__(self, idx):
        group = np.random.choice(self.data)
        label = group['label']

        A_full = group['gen_data_1'][np.random.randint(len(group['gen_data_1']))].T
        A_start = (A_full.shape[1] - self.segment_length) // 2
        A = A_full[:, A_start:A_start + self.segment_length]  # select only the central part data for training

        B_full = group['gen_data_2'][np.random.randint(len(group['gen_data_2']))].T
        B_start = (B_full.shape[1] - self.segment_length) // 2
        B = B_full[:, B_start:B_start + self.segment_length]

        C_full = group['disc_data'][np.random.randint(len(group['disc_data']))].T
        C_start = (C_full.shape[1] - self.segment_length) // 2
        C = C_full[:, C_start:C_start + self.segment_length]

        A = torch.tensor(A, dtype=torch.float32).unsqueeze(0)
        B = torch.tensor(B, dtype=torch.float32).unsqueeze(0)
        C = torch.tensor(C, dtype=torch.float32).unsqueeze(0)
        condition = torch.tensor(label, dtype=torch.long)
        return A, B, C, condition


# ------------------------------
# Example training loop setup
# ------------------------------
if __name__ == "__main__":
    condition_map = {"emg_LWSA": 0, "emg_LWSD": 1}
    data_dict = {
        "emg_LWSA": {
            "gen_data_1": [np.random.randn(2000, 65) for _ in range(10)],
            "gen_data_2": [np.random.randn(2000, 65) for _ in range(10)],
            "disc_data":  [np.random.randn(2000, 65) for _ in range(10)],
        },
        "emg_LWSD": {
            "gen_data_1": [np.random.randn(2000, 65) for _ in range(10)],
            "gen_data_2": [np.random.randn(2000, 65) for _ in range(10)],
            "disc_data":  [np.random.randn(2000, 65) for _ in range(10)],
        },
    }

    batch_size = 512
    segment_length = 1200

    dataset = EMGFusionDataset(data_dict, condition_map, segment_length=segment_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    gen = EMGFusionGenerator(num_channels=65, time=segment_length, num_conditions=len(condition_map)).to(device)
    disc = EMGFusionDiscriminator(num_channels=65, time=segment_length, num_conditions=len(condition_map)).to(device)

    summary(gen, input_size=(batch_size, 1, 28, 28))

    gen_opt = torch.optim.Adam(gen.parameters(), lr=1e-4)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()
    recon_loss = nn.L1Loss()

    for A, B, C, cond in loader:
        A, B, C, cond = A.to(device), B.to(device), C.to(device), cond.to(device)

        # Train Discriminator
        fake_C = gen(A, B, cond).detach()
        real_label = torch.ones(A.size(0), 1, device=device)
        fake_label = torch.zeros(A.size(0), 1, device=device)

        disc_real = disc(A, B, C, cond)
        disc_fake = disc(A, B, fake_C, cond)

        loss_real = loss_fn(disc_real, real_label)
        loss_fake = loss_fn(disc_fake, fake_label)
        disc_loss = (loss_real + loss_fake) / 2

        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()

        # Train Generator
        fake_C = gen(A, B, cond)
        pred_fake = disc(A, B, fake_C, cond)

        adv_loss = loss_fn(pred_fake, real_label)
        l1 = recon_loss(fake_C, C)
        gen_loss = adv_loss + 10 * l1  # Weighted sum

        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        print(f"Gen Loss: {gen_loss.item():.4f}, Disc Loss: {disc_loss.item():.4f}")
        break
