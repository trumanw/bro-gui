import dgl
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn as nn
from torch.nn import functional as F

class SFTModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, ):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.temp_linear = nn.Sequential(nn.Linear(1, hidden_dim, bias=False), nn.ReLU(inplace=True), )
        self.phy_linear = nn.Sequential(nn.Linear(9, hidden_dim, bias=False), nn.ReLU(inplace=True), )
        self.dp1 = nn.Dropout(0.08)
        self.dp2 = nn.Dropout(0.08)

    def forward(self, g, temp, phys):
        phys = self.phy_linear(torch.cat([phys, temp], dim=1))
        temp = self.temp_linear(temp)
        h = g.ndata['h'].float()
        h1 = F.relu(self.ln1(self.conv1(g, h)))
        h1 = F.relu(self.ln2(self.conv2(g, h1)))
        g.ndata['h'] = h1
        hg = dgl.mean_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg = torch.cat([F.normalize(hg, p=2, dim=1), F.normalize(hg_max, p=2, dim=1), temp, phys], dim=1)
        output = F.relu(self.ln3(self.classify1(hg)))
        output = self.dp1(output)
        output = F.relu(self.ln4(self.classify2(output)))
        output = self.dp2(output)
        output = self.classify3(output)
        return output
    
class CMCModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, ):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.phy_linear = nn.Sequential(nn.Linear(8, hidden_dim, bias=False), nn.ReLU(inplace=True), )

    def forward(self, g, phys):
        phys = self.phy_linear(phys)
        h = g.ndata['h'].float()
        h.requires_grad = True
        h1 = F.relu(self.ln1(self.conv1(g, h)))
        h1 = F.relu(self.ln2(self.conv2(g, h1)))
        g.ndata['h'] = h1
        hg = dgl.mean_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg = torch.cat([F.normalize(hg, p=2, dim=1), F.normalize(hg_max, p=2, dim=1), phys], dim=1)
        output = F.relu(self.ln3(self.classify1(hg)))
        output = F.relu(self.ln4(self.classify2(output)))
        output = self.classify3(output)
        return output
    
class KrafftModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, ):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim)
        self.phy_linear = nn.Sequential(nn.Linear(8, hidden_dim, bias=False), nn.ReLU(inplace=True), )

    def forward(self, g):
        h = g.ndata['h'].float()
        h.requires_grad = True
        h1 = F.relu(self.ln1(self.conv1(g, h)))
        h1 = F.relu(self.ln2(self.conv2(g, h1)))
        g.ndata['h'] = h1
        hg = dgl.mean_nodes(g, 'h')
        hg_max = dgl.max_nodes(g, 'h')
        hg = torch.cat([F.normalize(hg, p=2, dim=1), F.normalize(hg_max, p=2, dim=1)], dim=1)
        output = F.relu(self.ln3(self.classify1(hg)))
        output = F.relu(self.ln4(self.classify2(output)))
        output = self.classify3(output)
        return output
    