import torch as th
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_sort_pool
from models.EGNN_Block import EGNNlayer


class ResidualBlock(nn.Module):

    def __init__(self, channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.padding = padding
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class GrowthPointPredictionModel(nn.Module):

    def __init__(self, dim_in=10, dim_out=3, dim_edge_feat=7, num_layers=2):
        super(GrowthPointPredictionModel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_edge_feat = dim_edge_feat
        self.num_layers = num_layers
        self.EGNN = EGNNlayer(self.dim_in, self.dim_in, self.dim_in, edge_feat_size=self.dim_edge_feat)
        self.fc = nn.Linear(self.dim_in, self.dim_in)
        self.fc2 = nn.Linear(self.dim_in, self.dim_out)
        
    def _make_layer(self, channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, edge_index, x, coords, edge_feat):
        for i in range(self.num_layers):
            x_out, coords_out = self.EGNN(edge_index, x, coords, edge_feat)
            x_out = x_out.relu()
        x = self.fc(x_out)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


class TorsionAnglePredictionModel(nn.Module):

    def __init__(self, frag_dim_in=32, core_dim_in=35, egnn_dim_tmp=1024, egnn_dim_out=256, egnn_num_layers=7, dim_edge_feat=7,
                 cnn_dim_in=1, cnn_dim_tmp=512, stride=2, padding=3, dim_tmp=128, dim_out=36, num_layers=6):
        super(TorsionAnglePredictionModel, self).__init__()
        self.frag_dim_in = frag_dim_in
        self.core_dim_in = core_dim_in
        self.egnn_dim_tmp = egnn_dim_tmp
        self.egnn_dim_out = egnn_dim_out
        self.egnn_num_layers = egnn_num_layers
        self.dim_edge_feat = dim_edge_feat
        self.cnn_dim_in = cnn_dim_in
        self.cnn_dim_tmp = cnn_dim_tmp
        self.stride = stride
        self.padding = padding
        self.dim_tmp = dim_tmp
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.layer_frag = EGNNlayer(self.frag_dim_in, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        self.layer_core = EGNNlayer(self.core_dim_in, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        self.layer1 = EGNNlayer(self.egnn_dim_out, self.egnn_dim_tmp, self.egnn_dim_out, edge_feat_size=self.dim_edge_feat)
        self.fc = nn.Linear(self.egnn_dim_out * 2 + self.cnn_dim_tmp * 3, self.dim_tmp * 2)
        self.fc2 = nn.Linear(self.dim_tmp * 2, self.dim_tmp * 2)
        self.fc3 = nn.Linear(self.dim_tmp * 2, self.dim_tmp)
        self.fc1 = nn.Linear(self.dim_tmp, self.dim_out)
        self.bn1 = nn.BatchNorm1d(self.dim_tmp)
        self.bn3d = nn.BatchNorm3d(self.cnn_dim_in)
        self.conv3D1 = nn.Conv3d(self.cnn_dim_in, self.cnn_dim_tmp // 8, kernel_size=7, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D2 = nn.Conv3d(self.cnn_dim_tmp // 8, self.cnn_dim_tmp // 4, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D3 = nn.Conv3d(self.cnn_dim_tmp // 4, self.cnn_dim_tmp // 2, kernel_size=3, stride=self.stride, padding=self.padding, bias=False)
        self.conv3D4 = nn.Conv3d(self.cnn_dim_tmp // 2, self.cnn_dim_tmp, kernel_size=3, stride=self.stride, padding=self.padding, bias=False) 
        self.layer3D1 = self._make_layer(self.cnn_dim_tmp // 8)
        self.layer3D2 = self._make_layer(self.cnn_dim_tmp // 4)
        self.layer3D3 = self._make_layer(self.cnn_dim_tmp // 2)
        self.layer3D4 = self._make_layer(self.cnn_dim_tmp)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, channels, num_blocks=2):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return nn.Sequential(*layers)

    def forward(self, edge_index, x, coords, edge_feat, batch,edge_index1, x1, coords1, edge_feat1, batch1, grid):
        x_out, coords_out = self.layer_frag(edge_index, x, coords, edge_feat)
        x_out = x_out.relu()
        for i in range(self.egnn_num_layers):
            x_out, coords_out = self.layer1(edge_index, x_out, coords_out, edge_feat)
            x_out = x_out.relu()
        readout = global_sort_pool(x_out.squeeze(dim=0), batch, k=3)

        x_out1, coords_out1 = self.layer_core(edge_index1, x1, coords1, edge_feat1)
        x_out1 = x_out1.relu()
        for i in range(self.egnn_num_layers):
            x_out1, coords_out1 = self.layer1(edge_index1, x_out1, coords_out1, edge_feat1)
            x_out1 = x_out1.relu()
        readout1 = global_sort_pool(x_out1, batch1, k=3) 

        grid = self.bn3d(grid)
        grid = self.conv3D1(grid)
        grid = self.layer3D1(grid)
        grid = self.conv3D2(grid)
        grid = self.layer3D2(grid)
        grid = self.conv3D3(grid)
        grid = self.layer3D3(grid)   
        grid = self.conv3D4(grid)
        grid = self.layer3D4(grid)
        grid = self.avgpool(grid)
        grid = th.flatten(grid, 1)

        readout = th.cat((readout, readout1, grid), dim=1)
        readout = self.fc(readout)
        for i in range(self.num_layers):
            readout = self.fc2(readout).relu() + readout
        readout = self.fc3(readout).relu()
        readout = self.bn1(readout)
        readout = self.fc1(readout)
        readout = F.softmax(readout, dim=1)
        return readout