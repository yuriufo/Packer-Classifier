import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class ODEfunc(nn.Module):
    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-3, atol=1e-3):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = rtol
        self.atol = atol

    def forward(self, x, t=None):
        if t is not None:
            self.integration_time = t
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(
            self.odefunc,
            x,
            self.integration_time,
            rtol=self.rtol,
            atol=self.atol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODEnet(nn.Module):
    def __init__(self, in_channels, state_channels, output_classes, tol=1e-3):
        super().__init__()

        self.downsampling_layers = nn.Sequential(
            nn.Conv2d(in_channels, state_channels, 3, 1), norm(state_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(state_channels, state_channels, 4, 2, 1),
            norm(state_channels), nn.ReLU(inplace=True),
            nn.Conv2d(state_channels, state_channels, 4, 2, 1))

        self.feature_layers = ODEBlock(
            ODEfunc(state_channels), rtol=tol, atol=tol)

        self.before_fc = nn.Sequential(
            norm(state_channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(state_channels, output_classes)

    def forward(self, x, apply_softmax=False):
        out = self.downsampling_layers(x)
        out = self.feature_layers(out)
        out = self.before_fc(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        if apply_softmax:
            out = F.softmax(out, dim=1)

        return out


if __name__ == '__main__':
    x = torch.Tensor(128, 3, 32, 32)
    print(x.shape)

    modle = ODEnet(3, 64, 10, 0.1)

    modle(x)