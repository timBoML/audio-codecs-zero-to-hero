import torch
import torch.nn as nn
import torchaudio


class ResidualUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
                nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x + self.conv(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.residual_unit = ResidualUnit(in_channels)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2 if stride % 2 == 0 else stride // 2 + 1,
        )

    def forward(self, x):
        x = self.residual_unit(x)
        x = self.conv1d(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            DownsampleBlock(in_channels=32, out_channels=64, stride=2),
            DownsampleBlock(in_channels=64, out_channels=128, stride=4),
            DownsampleBlock(in_channels=128, out_channels=256, stride=5),
            DownsampleBlock(in_channels=256, out_channels=512, stride=8),
        )
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
        self.proj = nn.Conv1d(512, 128, kernel_size=1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1d_transposed = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=stride // 2 if stride % 2 == 0 else stride // 2 + 1,
            output_padding=stride % 2
        )
        self.residual_unit = ResidualUnit(out_channels)

    def forward(self, x):
        x = self.conv1d_transposed(x)
        x = self.residual_unit(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_up = nn.ConvTranspose1d(in_channels=128, out_channels=512, kernel_size=1)
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
        self.layers = nn.Sequential(
            UpsampleBlock(in_channels=512, out_channels=256, stride=8),
            UpsampleBlock(in_channels=256, out_channels=128, stride=5),
            UpsampleBlock(in_channels=128, out_channels=64, stride=4),
            UpsampleBlock(in_channels=64, out_channels=32, stride=2),
            nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = self.proj_up(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.layers(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, dim):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings=n_codes, embedding_dim=dim)
        self.n_codes = n_codes
        self.dim = dim
        self.last_indices = None
        self.n_codes_tensor = torch.arange(n_codes)
        self.max_entropy = torch.log(torch.tensor(self.n_codes, dtype=torch.float32))

    def forward(self, z):
        z_flatten = torch.flatten(z, start_dim=0, end_dim=1)

        if self.last_indices is not None:
            with torch.no_grad():
                self.n_codes_tensor = self.n_codes_tensor.to(z.device)
                isin_tensor = torch.isin(self.n_codes_tensor, self.last_indices)
                not_used_indices = (isin_tensor != True).nonzero(as_tuple=True)[0]
                random_embeddings = z_flatten[torch.randint(0, z_flatten.shape[0], (len(not_used_indices),))]
                old_center = self.codebook.weight[not_used_indices]
                self.codebook.weight[not_used_indices] = old_center * 0.5 + random_embeddings * 0.5

        z_sum_squared = torch.sum(z_flatten ** 2, dim=1, keepdim=True)
        codebook_sum_squared = torch.sum(self.codebook.weight ** 2, dim=1)
        dot_product = z_flatten @ self.codebook.weight.T
        distances = z_sum_squared - 2 * dot_product + codebook_sum_squared
        indices = torch.argmin(distances, dim=1)
        self.last_indices = indices

        z_quantized = self.codebook(indices)
        z_quantized_st = z_flatten + (z_quantized - z_flatten).detach()
        commitment_loss = torch.mean((z_quantized.detach() - z_flatten) ** 2)
        z_quantized_st = z_quantized_st.reshape(z.shape)

        usage = torch.bincount(indices, minlength=self.n_codes).float()
        probs = usage / usage.sum()
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        entropy_loss = self.max_entropy - entropy
        return z_quantized_st, indices, commitment_loss, entropy_loss


class RVQ(nn.Module):
    def __init__(self, n_levels, n_codes, dim):
        super().__init__()
        self.quantizers = nn.ModuleList([VectorQuantizer(n_codes, dim) for _ in range(n_levels)])

    def forward(self, z):
        residual = z
        total_loss = torch.zeros(1, device=z.device, dtype=torch.float32).squeeze()
        total_entropy_loss = torch.zeros(1, device=z.device, dtype=torch.float32).squeeze()
        total_quantized = torch.zeros_like(z)
        all_indices = []
        for quantizer in self.quantizers:
            z_q, indices, loss, entropy_loss = quantizer(residual)
            residual = residual - z_q
            total_loss += loss
            total_entropy_loss += entropy_loss
            total_quantized += z_q
            all_indices.append(indices)
        return total_quantized, total_loss / len(self.quantizers), total_entropy_loss / len(self.quantizers), all_indices


class SimpleCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.rvq = RVQ(n_levels=8, n_codes=1024, dim=128)
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        z, commitment_loss, entropy_loss, indices = self.rvq(z)
        z = z.permute(0, 2, 1)
        x_hat = self.decoder(z)
        return x_hat, commitment_loss, entropy_loss


def frequency_loss_fn(x, x_hat):
    total_loss_l1 = torch.tensor(0.0).to(x.device)
    total_loss_l2 = torch.tensor(0.0).to(x.device)
    frequency_bins = 64
    scales = [512, 1024, 2048, 4096]

    for window_len in scales:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=window_len,
            hop_length=window_len // 4,
            n_mels=frequency_bins
        ).to(x.device)
        mel = torch.log(mel_transform(x) + 1e-7)
        mel_hat = torch.log(mel_transform(x_hat) + 1e-7)
        mel_delta = mel - mel_hat
        total_loss_l1 += torch.mean(torch.abs(mel_delta))
        total_loss_l2 += torch.mean(mel_delta ** 2)

    return (total_loss_l1 + total_loss_l2) / len(scales)
