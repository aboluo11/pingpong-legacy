import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, gru_sz):
        super().__init__()
        #input size: 80*80
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32*7*7, gru_sz),
            nn.ReLU()
        )
        self.policy = nn.Linear(gru_sz, 1)
        self.critic = nn.Linear(gru_sz, 1)
        self.gru = nn.GRUCell(gru_sz,gru_sz)

    def forward(self, x, state):
        x = (x-x.mean())/x.std()
        x = self.convs(x)
        x = state = self.gru(x, state)
        p = self.policy(x).view(-1)
        p = F.sigmoid(p)
        value = self.critic(x).view(-1)
        return p, value, state