import numpy as np
import torch
from snntorch import spikegen

X = np.load("data/emg_signals.npy")

# Normalize safely
X = (X - X.min()) / (X.max() - X.min() + 1e-8)

# Rate coding
spikes = spikegen.rate(
    torch.tensor(X, dtype=torch.float32),
    num_steps=100
)

torch.save(spikes, "data/spikes.pt")

print("Spike tensor shape:", spikes.shape)
