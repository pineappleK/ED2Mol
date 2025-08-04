from torch import nn
from torch.autograd import Variable


class ED_generator(nn.Module):

    def __init__(self, dim_tmp=128, drop_rate=0.2, stride=2, padding=2):
        super(ED_generator, self).__init__()
        self.dim_tmp = dim_tmp
        self.drop_rate = drop_rate
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, self.dim_tmp // 4, kernel_size=5, padding=self.padding),
            nn.LeakyReLU(),
            nn.Dropout3d(self.drop_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.dim_tmp // 4, self.dim_tmp // 2, kernel_size=4, stride=self.stride, padding=self.padding-1),
            nn.InstanceNorm3d(self.dim_tmp // 2),
            nn.LeakyReLU(),
            nn.Dropout3d(self.drop_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(self.dim_tmp // 2, self.dim_tmp, kernel_size=3, stride=self.stride, padding=self.padding-1),
            nn.InstanceNorm3d(self.dim_tmp),
            nn.LeakyReLU(),
            nn.Dropout3d(self.drop_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(self.dim_tmp, self.dim_tmp * 2, kernel_size=4),
            nn.InstanceNorm3d(self.dim_tmp * 2),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(self.drop_rate)
        )
        self.fc1 = nn.Linear(self.dim_tmp * 2 * 4 * 4 * 4, self.dim_tmp)
        self.fc2 = nn.Linear(self.dim_tmp * 2 * 4 * 4 * 4, self.dim_tmp)
        self.fc3 = nn.Linear(self.dim_tmp, self.dim_tmp * 3 * 3 * 3)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(self.dim_tmp, self.dim_tmp // 4, kernel_size=5, output_padding=self.padding-1, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(self.dim_tmp // 4, self.dim_tmp // 8, kernel_size=5, output_padding=self.padding-1, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(self.dim_tmp // 8, self.dim_tmp // 16, kernel_size=5, output_padding=self.padding-1, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose3d(self.dim_tmp // 16, 1, kernel_size=6, stride=self.stride, padding=self.padding),
            nn.LeakyReLU()
        )
    
    def encode(self, grid):
        e1 = self.conv1(grid)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        return self.fc1(e4.view(e4.size()[0], -1)), self.fc2(e4.view(e4.size()[0], -1))
        
    def sampling(self, z_mean, z_logvar):
        std = z_logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(z_mean)
    
    def decode(self, z):
        d1 = self.deconv1(self.fc3(z).view(z.size()[0], self.dim_tmp, 3, 3, 3))
        d2 = self.deconv2(d1)
        d3 = self.deconv3(d2)
        d4 = self.deconv4(d3)
        return d4
    
    def forward(self, grid):
        z_mean, z_logvar = self.encode(grid)
        z = self.sampling(z_mean, z_logvar)
        output = self.decode(z)
        return z_mean, z_logvar, output  