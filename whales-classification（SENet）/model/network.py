import torch.nn
from .SENet import *
from .loss_function import *
from torch.nn import init


class net(nn.Module):
    def __init__(self, num_classes=5005, inchannels=3):
        super().__init__()

        self.net = se_resnext50_32x4d(inchannels=inchannels, load_pretrain=True)

        planes = 2048
        local_planes = 512

        # for global features
        self.global_bn = nn.BatchNorm1d(planes)
        self.global_bn.bias.requires_grad_(False)
        self.fc = nn.Linear(planes, num_classes)
        init.normal_(self.fc.weight, std=0.001)
        init.constant_(self.fc.bias, 0)

        # for local features
        self.local_conv = nn.Conv2d(planes, local_planes, 1)
        self.local_bn = nn.BatchNorm2d(local_planes)
        self.local_bn.bias.requires_grad_(False)

    def forward(self, x, label=None):
        feature = self.net(x)

        # for global feature
        global_feature = F.avg_pool2d(feature, feature.size()[2:])
        global_feature = global_feature.view(global_feature.size(0), -1)
        global_feature = F.dropout(global_feature, p=0.2)
        global_feature = self.global_bn(global_feature)
        # l2_norm
        global_norm = torch.norm(global_feature, 2, dim=1, keepdim=True)
        global_feature = torch.div(global_feature, global_norm)

        # for local feature
        local_feature = torch.mean(feature, -1, keepdim=True)
        local_feature = self.local_conv(local_feature)
        local_feature = self.local_bn(local_feature)
        local_feature = local_feature.squeeze(-1).permute(0, 2, 1)
        # l2_norm
        local_norm = torch.norm(local_feature, 2, dim=1, keepdim=True)
        local_feature = torch.div(local_feature, local_norm)

        out = self.fc(global_feature) * 16
        return global_feature, local_feature, out

    def getLoss(self, global_feature, local_feature, results, labels):

        g_loss = global_loss(TripletLoss(margin=0.3), global_feature, labels)[0]
        l_loss = local_loss(TripletLoss(margin=0.3), local_feature, labels)[0]
        triple_loss = g_loss + l_loss

        s_loss = sigmoid_loss(results, labels, topk=30)

        self.loss = triple_loss + s_loss

    def load_checkpoint(self, checkpoint, skip):

        load_state_dict = torch.load(checkpoint)

        keys = [self.state_dict.keys()]
        for key in keys:
            if key in skip:
                continue
            try:
                self.state_dict()[key] = load_state_dict[key]
            except:
                print('%s loaded error...' % str(self.state_dict()[key]))

        self.load_state_dict(self.state_dict())
