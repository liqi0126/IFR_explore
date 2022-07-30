import torch
from torch import nn

from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=0.8,
                nsample=64,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=1,
                radius=3.2,
                nsample=64,
                mlp=[512, 512, 512, 512],
                use_xyz=True,
            )
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(512, self.hparams['feat_dim']),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        bottleneck_feats = l_features[-1].squeeze(-1)
        return l_xyz, l_features, self.fc_layer2(bottleneck_feats)


class BinaryRelationNetwork(nn.Module):
    def __init__(self, use_scene_attr=False, thres=0.2):
        super(BinaryRelationNetwork, self).__init__()
        self.feat_dim = 64
        self.use_scene_attr = use_scene_attr
        self.thres = thres
        self.encoder = PointNet2({'feat_dim': self.feat_dim})
        input_dim = 128 + 1
        if use_scene_attr:
            input_dim += 128
            self.scene_attr_net = nn.Sequential(
                nn.Linear(22, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 64),
            )

        self.relation_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(True),
            nn.Linear(4, 1),
        )

        self.relation_loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(5.))

    """
        Input: B x N x 3
        Output: B x N x 3, B x F
    """
    def forward(self, src_pcs, dst_pcs, src_pos, dst_pos, src_rot, dst_rot, src_size, dst_size, src_attr, dst_attr):
        src_l_xyz, src_l_features, src_feats = self.encoder(src_pcs.repeat(1, 1, 2))
        dst_l_xyz, dst_l_features, dst_feats = self.encoder(dst_pcs.repeat(1, 1, 2))
        feats = torch.cat((src_feats, dst_feats), -1)

        if self.use_scene_attr:
            src_scene_attr_embedding = self.scene_attr_net(torch.cat([src_pos, src_rot, src_size, src_attr], -1))
            dst_scene_attr_embedding = self.scene_attr_net(torch.cat([dst_pos, dst_rot, dst_size, dst_attr], -1))
            feats = torch.cat([feats, src_scene_attr_embedding, dst_scene_attr_embedding, (src_pos == dst_pos).all(-1).float()], -1)
        else:
            feats = torch.cat([feats, (src_pos == dst_pos).all(-1).float().unsqueeze(-1)], -1)

        relation_logits = self.relation_net(feats).squeeze()
        return relation_logits

    def get_loss(self, relation_logits, gt_relation):
        # gt_relation = ((src_gt.sum(-1) != 0) & (tgt_gt.sum(-1) != 0))
        relation_loss = self.relation_loss_fn(relation_logits, gt_relation.float())

        relation = relation_logits.sigmoid()
        accuracy = ((relation >= self.thres) == gt_relation).float().mean()
        tp = (gt_relation & (relation > self.thres)).sum()
        tp_fp = (relation > self.thres).sum()
        tp_fn = gt_relation.sum()

        return relation_loss.mean(), accuracy, tp, tp_fp, tp_fn

    def get_feat(self, pcs):
        with torch.no_grad():
            return self.encoder(pcs.repeat(1, 1, 2).cuda())

    def build_prior(self, pcs, pos, rot, size, attr):
        relations = []
        for src_pcs, src_pos, src_rot, src_size, src_attr in zip(pcs, pos, rot, size, attr):
            src_pcs_group = []
            dst_pcs_group = []
            src_pos_group = []
            dst_pos_group = []
            src_rot_group = []
            dst_rot_group = []
            src_size_group = []
            dst_size_group = []
            src_attr_group = []
            dst_attr_group = []
            for dst_pcs, dst_pos, dst_rot, dst_size, dst_attr in zip(pcs, pos, rot, size, attr):
                src_pos_group.append(torch.Tensor(src_pos))
                src_rot_group.append(torch.Tensor(src_rot))
                src_size_group.append(torch.Tensor(src_size))
                src_attr_group.append(torch.Tensor(src_attr))
                src_pcs_group.append(src_pcs)
                dst_pos_group.append(torch.Tensor(dst_pos))
                dst_rot_group.append(torch.Tensor(dst_rot))
                dst_size_group.append(torch.Tensor(dst_size))
                dst_attr_group.append(torch.Tensor(dst_attr))
                dst_pcs_group.append(dst_pcs)
            src_pcs = torch.stack(src_pcs_group).cuda()
            src_pos = torch.stack(src_pos_group).cuda()
            dst_pcs = torch.stack(dst_pcs_group).cuda()
            dst_pos = torch.stack(dst_pos_group).cuda()
            with torch.no_grad():
                relation = self.forward(src_pcs, dst_pcs, src_pos, dst_pos, src_rot, dst_rot, src_size, dst_size, src_attr, dst_attr)
            relations.append(relation)
        relations = torch.sigmoid(torch.stack(relations))
        return relations
