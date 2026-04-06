import argparse
import numpy as np
import torch
import torch.optim as optim
import torchaudio
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import Codec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--input-ckpt", type=str, default=None)
    return parser.parse_args()


def align_waveforms(x, x_hat):
    target_len = min(x.shape[-1], x_hat.shape[-1])
    return x[..., :target_len], x_hat[..., :target_len]


def frequency_loss_fn(x, x_hat, sample_rate=24000):
    x, x_hat = align_waveforms(x, x_hat)
    total_loss_l1 = torch.tensor(0.0, device=x.device)
    total_loss_l2 = torch.tensor(0.0, device=x.device)
    frequency_bins = 64
    scales = [512, 1024, 2048, 4096]

    for window_len in scales:
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
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


torch.set_float32_matmul_precision("high")
load_dotenv()
args = parse_args()

dataset = load_dataset("shb777/gemini-flash-2.0-speech")
dataset = dataset["en"].shuffle()
dataset = dataset.remove_columns(["puck", "phoneme_length", "text"])


class SpeechDataset(Dataset):
    def __init__(self, hf_dataset, clip_length_24khz=24000):
        self.data = hf_dataset
        self.clip_length_24khz = clip_length_24khz

    def __len__(self):
        return len(self.data)

    def random_crop(self, audio):
        if len(audio) < self.clip_length_24khz:
            return None
        if len(audio) == self.clip_length_24khz:
            return audio
        start = np.random.randint(0, len(audio) - self.clip_length_24khz)
        return audio[start:start + self.clip_length_24khz]

    def __getitem__(self, idx):
        audio_tensor = torch.tensor(self.data[idx]["kore"]["array"])
        target_24khz = self.random_crop(audio_tensor)
        if target_24khz is None:
            idx = np.random.randint(0, len(self.data))
            return self.__getitem__(idx)
        input_16khz = torchaudio.functional.resample(target_24khz.squeeze(), 24000, 16000)
        return input_16khz, target_24khz.squeeze()


def save_sample(codec, dataset, epoch, path="sample.wav"):
    codec.eval()
    with torch.no_grad():
        x_16khz, x_24khz = dataset[0]
        x_hat = codec(x_16khz.unsqueeze(0).cuda())
        torchaudio.save(f"{path}_{epoch}_reconstructed.wav", x_hat.squeeze(0).cpu(), 24000)
        torchaudio.save(f"{path}_{epoch}_original.wav", x_24khz.unsqueeze(0), 24000)
    codec.train()


def train_step(batch, codec, optimizer):
    x_16khz, x_target_24khz = [tensor.cuda() for tensor in batch]
    x_hat = codec(x_16khz)
    x_target_24khz, x_hat = align_waveforms(x_target_24khz, x_hat)
    loss_time = torch.mean(torch.abs(x_target_24khz - x_hat)) * 1.0
    frequency_loss = frequency_loss_fn(x_target_24khz, x_hat, sample_rate=24000) * 1.0
    total_loss = loss_time + frequency_loss
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss_time.item(), frequency_loss.item(), grad_norm


speech_dataset = SpeechDataset(dataset)
dataloader = DataLoader(speech_dataset, batch_size=16, shuffle=True)

codec = Codec(first_stage=args.stage == 1).cuda()
if args.input_ckpt is not None:
    state_dict = torch.load(args.input_ckpt, map_location="cpu")
    codec.load_state_dict(state_dict)
optimizer = optim.AdamW(codec.parameters(), lr=5e-5, betas=(0.5, 0.9), eps=1e-10)

for epoch in range(25):
    save_sample(codec, speech_dataset, 0)
    pbar = tqdm(dataloader)
    for batch in pbar:
        loss_time, loss_frequency, grad_norm = train_step(batch, codec, optimizer)
        pbar.set_postfix(
            loss_time=f"{loss_time:.8}",
            loss_frequency=f"{loss_frequency:.4f}",
            grad_norm=f"{grad_norm:.4f}"
        )
    save_sample(codec, speech_dataset, epoch)
    torch.save(codec.state_dict(), f"checkpoint_epoch_{epoch}.pt")
