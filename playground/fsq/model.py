import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from vocos import Vocos

class HubertEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.backbone = bundle.get_model()
        self.sample_rate = bundle.sample_rate

    def forward(self, waveform):
        # Expects 16kHz
        features, _ = self.backbone.extract_features(waveform)
        acoustic_features = torch.stack(features[0:3]).mean(dim=0) # Layers 0, 1, 2
        semantic_features = torch.stack(features[9:12]).mean(dim=0) # Layers 9, 10, 11
        return acoustic_features, semantic_features


class FusionDownProj(nn.Module):
    def __init__(self, n_levels):
        super().__init__()
        self.down_proj = nn.Linear(768 * 2, n_levels)
        self.n_levels = n_levels
    
    def forward(self, acoustic, semantic):
        # acoustic: B, T, dim
        # semantic: B, T, dim
        x = torch.cat([acoustic, semantic], dim=2) # B, T, dim * 2
        x = self.down_proj(x) # B * T, n_levels
        x = F.gelu(x)
        return x

class Quantizer(nn.Module):
    def __init__(self, n_levels):
        super().__init__()
        self.n_levels = n_levels
        self.register_buffer('levels', torch.linspace(-1, 1, n_levels))

    def forward(self, x):
        # x: [B, 1]
        x = torch.tanh(x)
        distances = torch.abs(self.levels - x)
        best_matching_index = distances.sort().indices[:, 0]
        x_quantized = self.levels[best_matching_index].unsqueeze(1)
        x_quantized_st = x + (x_quantized - x).detach() # gradient = 1 hack
        return x_quantized_st, best_matching_index.unsqueeze(1) # [B, 1]

class FSQ(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        multiplier = []
        for index in range(len(levels)):
            sub_levels = levels[:index]
            if sub_levels is None:
                multiplier.append(torch.tensor([0], dtype=torch.long))
            multiplier.append(torch.prod(torch.tensor(sub_levels, dtype=torch.long)))
        self.register_buffer('multiplier', torch.tensor(multiplier))

        self.quantizers = nn.ModuleList()
        for level in levels:
            self.quantizers.append(Quantizer(level))

    def forward(self, x):
        # x: B, T, n_levels
        x_flatten = torch.flatten(x, start_dim=0, end_dim=1) # B * T, n_levels

        y = []
        z = []
        for i, quantizer in enumerate(self.quantizers):
            x_quantized_st, code = quantizer(x_flatten[:, i].unsqueeze(1))
            y.append(code)
            z.append(x_quantized_st)

        y = torch.stack(y).squeeze() # n_levels, B * T
        y = y.permute(1, 0) # B * T, n_levels
        y = y * self.multiplier # B * T, n_levels
        code = torch.sum(y, dim=1) # B * T
        z = torch.stack(z).squeeze()
        return z.reshape(x.shape), code.reshape(x.shape[0], x.shape[1])


class DecoderTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=256, num_heads=8, num_layers=4, output_dim=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(model_dim, output_dim)
        self.vocos_tokens_per_second = 94
        self.hubert_tokens_per_second = 50

    def forward(self, x):
        # x: B, T_tokens, D
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.output_proj(x)
        x = x.transpose(1, 2)  # B, n_mels, T_tokens
        audio_length = x.shape[2] / self.hubert_tokens_per_second
        target_frames = audio_length * self.vocos_tokens_per_second
        x = F.interpolate(x, size=int(target_frames), mode='linear', align_corners=False)
        return x


class VocosHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    def forward(self, x):
        x = self.vocos.backbone(x)
        audio = self.vocos.head(x)
        return audio


class Codec(nn.Module):
    def __init__(self, first_stage=True):
        super().__init__()
        self.first_stage = first_stage
        self.encoder = HubertEncoder() 
        self.fusing_down_proj = FusionDownProj(n_levels=8)
        self.fsq = FSQ(levels=[8, 8, 8, 8, 4, 4, 4, 4])
        self.transformer = DecoderTransformer(8)
        self.vocoder = VocosHead()

        if first_stage:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.vocoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
            self.vocoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.first_stage:
            self.encoder.eval()
            self.vocoder.eval()
        return self

    def forward(self, waveform_batch, return_mel=False):
        acoustic, semantic = self.encoder(waveform_batch)      
        x = self.fusing_down_proj(acoustic, semantic)
        x, _ = self.fsq(x)
        mel = self.transformer(x)
        audio = self.vocoder(mel)
        if return_mel:
            return audio, mel
        return audio
