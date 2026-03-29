import torch
import numpy as np
from snn_model import SNN

spikes = torch.load("data/spikes.pt")

model = SNN(spikes.shape[2])
model.load_state_dict(torch.load("snn_model.pth"))

np.save("fc1_w.npy", model.fc1.weight.detach().numpy())
np.save("fc2_w.npy", model.fc2.weight.detach().numpy())

print("Weights Exported")
