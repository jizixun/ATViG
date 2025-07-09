import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import SimplePatchifier, TwoLayerNN


class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1, sim_threshold=None):
        super().__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.sim_threshold = sim_threshold
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features*4)
        self.out_layer2 = TwoLayerNN(in_features, in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features*2, in_features, 1, 1, groups=head_num)

    def similarity_threshold_graph(self, x, threshold=0.5):
        # x: [B, N, C]
        B, N, C = x.shape
        x_norm = F.normalize(x, dim=2)  
        sim_matrix = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, N, N]
        adj = (sim_matrix > threshold).float()  # [B, N, N]
        return adj

    def forward(self, x):
        B, N, C = x.shape

        if self.sim_threshold is not None:
            adj = self.similarity_threshold_graph(x, self.sim_threshold)  # [B, N, N]
            # 取每个patch的邻居索引（只保留阈值内的邻居）
            graph = adj.nonzero(as_tuple=False).view(B, -1, 2)  # [?, 2]
            # 这里graph的处理方式与原KNN不同，后续聚合时需按实际需求调整
            # 为兼容原有代码，暂时取每个patch前num_edges个邻居
            sim = x @ x.transpose(-1, -2)
            graph = sim.topk(self.k, dim=-1).indices
        else:
            sim = x @ x.transpose(-1, -2)
            graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)

        # aggregation
        neibor_features = x[torch.arange(
            B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack(
            [x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(self.out_layer1(
            F.gelu(x).view(B * N, -1)).view(B, N, -1))
        x = x + shortcut

        x = self.droppath2(self.out_layer2(F.gelu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1)) + x

        return x


class VGNN(nn.Module):
    def __init__(self, in_features=3*16*16, out_feature=320, num_patches=196,
                 num_ViGBlocks=16, num_edges=9, head_num=1, sim_threshold=None):
        super().__init__()

        self.patchifier = SimplePatchifier()
        # self.patch_embedding = TwoLayerNN(in_features)
        self.patch_embedding = nn.Sequential(
            nn.Linear(in_features, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//8),
            nn.BatchNorm1d(out_feature//8),
            nn.GELU(),
            nn.Linear(out_feature//8, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature),
            nn.BatchNorm1d(out_feature)
        )
        self.pose_embedding = nn.Parameter(
            torch.rand(num_patches, out_feature))

        self.blocks = nn.Sequential(
            *[ViGBlock(out_feature, num_edges, head_num, sim_threshold)
              for _ in range(num_ViGBlocks)])

    def forward(self, x):
        x = self.patchifier(x)
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x


class Classifier(nn.Module):
    def __init__(self, in_features=3*16*16, out_feature=320,
                 num_patches=196, num_ViGBlocks=16, hidden_layer=1024,
                 num_edges=9, head_num=1, n_classes=10, sim_threshold=None):
        super().__init__()
        self.backbone = VGNN(in_features, out_feature,
                             num_patches, num_ViGBlocks,
                             num_edges, head_num, sim_threshold)

        self.predictor = nn.Sequential(
            nn.Linear(out_feature*num_patches, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Linear(hidden_layer, n_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        B, N, C = features.shape
        x = self.predictor(features.view(B, -1))
        return features, x
