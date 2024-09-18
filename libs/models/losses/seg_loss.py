import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module
class EnhancedCLRNetSegLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, num_classes=5, ignore_label=255, bg_weight=0.5, edge_weight=2.0, curve_weight=1.5):
        super(EnhancedCLRNetSegLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.ignore_label = ignore_label

        # 初始化权重
        weights = torch.ones(num_classes)
        weights[0] = bg_weight  # 背景权重
        self.register_buffer('weights', weights)

        self.edge_weight = edge_weight
        self.curve_weight = curve_weight

    def compute_edge_weights(self, targets):
        # 计算边缘权重
        edge_kernel = torch.tensor([[-1, -1, -1],
                                    [-1,  8, -1],
                                    [-1, -1, -1]], dtype=torch.float32, device=targets.device)
        edge_kernel = edge_kernel.view(
            1, 1, 3, 3).repeat(self.num_classes, 1, 1, 1)
        one_hot_targets = F.one_hot(
            targets, self.num_classes).permute(0, 3, 1, 2).float()
        edges = F.conv2d(one_hot_targets, edge_kernel, padding=1)
        edge_weights = torch.exp(
            self.edge_weight * edges.abs().sum(dim=1, keepdim=True))
        return edge_weights

    def compute_curve_weights(self, targets):
        # 计算曲率权重
        dx = targets[:, :, 2:] - targets[:, :, :-2]
        d2x = targets[:, :, 2:] - 2 * targets[:, :, 1:-1] + targets[:, :, :-2]
        curvature = torch.abs(d2x) / (1 + dx.pow(2))**1.5
        curve_weights = torch.exp(
            self.curve_weight * F.pad(curvature, (1, 1), mode='replicate'))
        return curve_weights

    def forward(self, preds, targets):
        # 计算边缘权重和曲率权重
        edge_weights = self.compute_edge_weights(targets)
        curve_weights = self.compute_curve_weights(targets)

        # 组合权重
        combined_weights = edge_weights * curve_weights

        # 应用 log_softmax
        log_probs = F.log_softmax(preds, dim=1)

        # 计算加权交叉熵损失
        loss = F.nll_loss(log_probs, targets.long(), weight=self.weights,
                          ignore_index=self.ignore_label, reduction='none')
        weighted_loss = loss * combined_weights.squeeze(1)

        return weighted_loss.mean() * self.loss_weight


@LOSSES.register_module
class CLRNetSegLoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, num_classes=5, ignore_label=255, bg_weight=0.5):
        super(CLRNetSegLoss, self).__init__()
        self.loss_weight = loss_weight
        weights = torch.ones(num_classes)
        weights[0] = bg_weight
        self.criterion = torch.nn.NLLLoss(
            ignore_index=ignore_label, weight=weights)

    def forward(self, preds, targets):
        loss = self.criterion(F.log_softmax(preds, dim=1), targets.long())
        return loss * self.loss_weight
