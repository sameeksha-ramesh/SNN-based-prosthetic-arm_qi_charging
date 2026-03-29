import torch
import numpy as np
from snn_model import SNN

spikes = torch.load("data/spikes.pt")
labels = np.load("data/labels.npy")

model = SNN(spikes.shape[2])
model.load_state_dict(torch.load("snn_model.pth"))
model.eval()

with torch.no_grad():
    spk_out = model(spikes)
    spk_sum = spk_out.sum(dim=0)
    pred = torch.argmax(spk_sum[0]).item()

print("Predicted:", pred)
print("Actual:", labels[0])
