import torch
import torch.nn as nn
import torch.nn.functional as F


class ST(nn.Module):
    def __init__(self):
        super(ST, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True))

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32), nn.ReLU(True), nn.Linear(32, 3 * 2))

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


if __name__ == '__main__':
    x = torch.Tensor(128, 3, 32, 32)
    model = ST()

    y = model(x)
    print(y.shape)
