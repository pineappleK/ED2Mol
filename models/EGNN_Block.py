import torch as th
from torch import nn
from torch_geometric.nn import MessagePassing


class EGNNlayer(MessagePassing):

    def __init__(self, in_size, hid_size, out_size, edge_feat_size=0):
        super(EGNNlayer, self).__init__()
        
        self.phi_e = nn.Sequential(nn.Linear(2 * in_size + edge_feat_size + 1, hid_size),
                                  nn.ReLU(),
                                  nn.Linear(hid_size, out_size),
                                  nn.ReLU())
        
        self.phi_x = nn.Sequential(nn.Linear(out_size, hid_size),
                                  nn.ReLU(),
                                  nn.Linear(hid_size, 1),
                                  nn.ReLU())
        
        self.phi_h = nn.Sequential(nn.Linear(in_size + out_size, hid_size),
                                  nn.ReLU(),
                                  nn.Linear(hid_size, out_size),
                                  nn.ReLU())
    
    def forward(self, edge_index, x, coords, edge_feat):

        rela_diff = coords[edge_index[0]] - coords[edge_index[1]]
        dist = th.norm(coords[edge_index[0]] - coords[edge_index[0]], dim = 1, keepdim = True)
        
        edge_feat = th.cat([edge_feat, dist], 1)
        
        x_out, coords_out = self.propagate(edge_index, x = x, coords = coords, 
                                           edge_feat = edge_feat, dist = dist, rela_diff = rela_diff)                
        return x_out, coords_out
        
    def propagate(self, edge_index, size = None, **kwargs):
        
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size,
                                         kwargs)
        
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        msg = self.message(**msg_kwargs)
        
        w = self.phi_x(msg)
        rela_diff = kwargs["rela_diff"]
        aggr_w = self.aggregate(w * rela_diff, **aggr_kwargs)
        coords_out = kwargs["coords"] + aggr_w
        
        msg = self.aggregate(msg, **aggr_kwargs)
        
        x = kwargs["x"]
        x_out = self.update(x, msg)
        return x_out, coords_out
        
    def message(self, x_i, x_j, edge_feat):
        edge_feat = edge_feat.float()
        message = self.phi_e(th.cat([x_i, x_j, edge_feat], 1))
        return message

    def update(self, x, message):
        x_out = self.phi_h(th.cat([x, message], 1))
        return x_out