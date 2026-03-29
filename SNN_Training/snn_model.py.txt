import torch
import torch.nn as nn
import snntorch as snn

class SNN(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.lif1 = snn.Leaky(beta=0.95)

        self.fc2 = nn.Linear(32, 2)
        self.lif2 = snn.Leaky(beta=0.95)

    def forward(self, x):

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk_rec = []

        for t in range(x.size(0)):

            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk_rec.append(spk2)

        return torch.stack(spk_rec)
