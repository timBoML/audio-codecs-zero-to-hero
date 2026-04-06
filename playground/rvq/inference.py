#!/usr/bin/env python3
"""
inference.py — reconstruct audio through a trained SimpleCodec checkpoint.

Usage:
    python inference.py --checkpoint checkpoint_epoch_24.pt --input audio.wav --output reconstructed.wav
"""

import argparse

import torch
import torchaudio

from model import SimpleCodec


def load_model(checkpoint_path: str, device: torch.device) -> SimpleCodec:
    codec = SimpleCodec().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    codec.load_state_dict(state)
    codec.eval()
    return codec


def reconstruct(codec: SimpleCodec, input_path: str, output_path: str, device: torch.device):
    waveform, sr = torchaudio.load(input_path)

    if sr != 24000:
        waveform = torchaudio.functional.resample(waveform, sr, 24000)

    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    x = waveform.unsqueeze(0).to(device)  # [1, 1, T]

    with torch.no_grad():
        x_hat, _, _ = codec(x)

    torchaudio.save(output_path, x_hat.squeeze(0).cpu(), 24000)
    print(f"Saved reconstructed audio to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SimpleCodec inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--input", required=True, help="Path to input .wav file")
    parser.add_argument("--output", default="reconstructed.wav", help="Path for output .wav file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    codec = load_model(args.checkpoint, device)
    reconstruct(codec, args.input, args.output, device)


if __name__ == "__main__":
    main()
