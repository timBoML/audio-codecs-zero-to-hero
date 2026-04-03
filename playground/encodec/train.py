import numpy as np
import torch
import torch.optim as optim
import torchaudio
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import SimpleCodec, frequency_loss_fn

torch.set_float32_matmul_precision("high")
load_dotenv()

dataset = load_dataset("shb777/gemini-flash-2.0-speech")
dataset = dataset["en"].shuffle()
dataset = dataset.remove_columns(["puck", "phoneme_length", "text"])


class SpeechDataset(Dataset):
    def __init__(self, hf_dataset, clip_length=24000):
        self.data = hf_dataset
        self.clip_length = clip_length

    def __len__(self):
        return len(self.data)

    def random_crop(self, audio):
        if len(audio) < self.clip_length:
            return None
        if len(audio) == self.clip_length:
            return audio
        start = np.random.randint(0, len(audio) - self.clip_length)
        return audio[start:start + self.clip_length]

    def __getitem__(self, idx):
        audio_tensor = torch.tensor(self.data[idx]["kore"]["array"])
        audio_crop = self.random_crop(audio_tensor)
        if audio_crop is None:
            idx = np.random.randint(0, len(self.data))
            return self.__getitem__(idx)
        return audio_crop.unsqueeze(0)


def save_sample(codec, dataset, epoch, path="sample.wav"):
    codec.eval()
    with torch.no_grad():
        x = dataset[0].unsqueeze(0).cuda()
        x_hat, _, _ = codec(x)
        torchaudio.save(f"{path}_{epoch}_reconstructed.wav", x_hat.squeeze(0).cpu(), 24000)
        torchaudio.save(f"{path}_{epoch}_original.wav", x.squeeze(0).cpu(), 24000)
    codec.train()


def train_step(batch, codec, optimizer):
    x = batch.cuda()
    x_hat, commitment_loss, entropy_loss = codec(x)
    loss_time = torch.mean(torch.abs(x - x_hat)) * 1.0
    frequency_loss = frequency_loss_fn(x, x_hat) * 1.0
    commitment_loss = commitment_loss * 1.0
    entropy_loss = entropy_loss * 1.0
    total_loss = loss_time + frequency_loss + commitment_loss + entropy_loss
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), max_norm=10.0)
    optimizer.step()
    optimizer.zero_grad()
    return loss_time.item(), frequency_loss.item(), commitment_loss.item(), entropy_loss.item(), grad_norm


speech_dataset = SpeechDataset(dataset)
dataloader = DataLoader(speech_dataset, batch_size=16, shuffle=True)

codec = SimpleCodec().cuda()
optimizer = optim.AdamW(codec.parameters(), lr=3e-4, betas=(0.5, 0.9), eps=1e-10)

for epoch in range(25):
    save_sample(codec, speech_dataset, 0)
    pbar = tqdm(dataloader)
    for batch in pbar:
        loss_time, loss_frequency, loss_commitment, entropy_loss, grad_norm = train_step(batch, codec, optimizer)
        pbar.set_postfix(
            loss_time=f"{loss_time:.8}",
            loss_frequency=f"{loss_frequency:.4f}",
            loss_commitment=f"{loss_commitment:.4f}",
            entropy_loss=f"{entropy_loss:.4f}",
            grad_norm=f"{grad_norm:.4f}"
        )
    save_sample(codec, speech_dataset, epoch)
    torch.save(codec.state_dict(), f"checkpoint_epoch_{epoch}.pt")
