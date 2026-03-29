import torch
import torch.nn as nn
import numpy as np
from snn_model import SNN

spikes = torch.load("data/spikes.pt")
labels = torch.tensor(np.load("data/labels.npy"), dtype=torch.long)

model = SNN(spikes.shape[2])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

epochs = 10

for e in range(epochs):

    optimizer.zero_grad()

    spk_out = model(spikes)
    spk_sum = spk_out.sum(dim=0)

    loss = loss_fn(spk_sum, labels)

    loss.backward()
    optimizer.step()

    print("Epoch:", e+1, "Loss:", loss.item())

torch.save(model.state_dict(), "snn_model.pth")

print("Training Complete")
