from torch import nn
from dpipe.layers.resblock import ResBlock2d
from dpipe.layers.conv import PreActivation2d
import torch


class Discriminator(nn.Module):
    def __init__(self, kernel_size=3, n_filters_init=8):
        super(Discriminator, self).__init__()
        n = n_filters_init
        self.discriminator = nn.Sequential(
            nn.Conv2d(n * 8, n * 16, stride=2, kernel_size=3),
            nn.Conv2d(n * 16, n * 32, stride=2, kernel_size=3),
            nn.Conv2d(n * 32, n * 64, stride=2, kernel_size=3),
        )

        self.flat_map_dim = self.dummy_forward()
        self.fully_conn = nn.Linear(self.flat_map_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def dummy_forward(self):
        dummy_x = torch.rand(1, 64, 36, 36)
        feature_map_discriminator = self.discriminator(dummy_x)
        flat_map_dim = feature_map_discriminator.view(1, -1).shape[-1]
        return flat_map_dim

    def forward(self, x):
        feature_map_discriminator = self.discriminator(x)
        batch_size = feature_map_discriminator.shape[0]
        flat_map = self.fully_conn(feature_map_discriminator.view(batch_size,-1))
        return self.sigmoid(flat_map)



# x = torch.rand(8, 64, 36, 36)
# model = Discriminator()
# print(model(x))