from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.LeakyReLU(negative_slope=0.2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid()

        self.main = nn.Sequential(
            self.conv1, self.relu1,
            self.conv2, self.bn2, self.relu2,
            self.conv3, self.relu3,
            self.conv4
        )
        self.output = None

    def forward(self, x):
        untransformed_output = self.main(x)
        self.output = self.sigmoid(untransformed_output)
        return self.output, untransformed_output

    def get_output(self):
        return self.output
