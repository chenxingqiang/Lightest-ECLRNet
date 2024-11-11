import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmdet.core import build_assigner, build_prior_generator

class LightROIGather(nn.Module):
    def __init__(self, dim, num_priors):
        super().__init__()
        self.light_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        # Progressive feature fusion with shared parameters
        self.shared_fusion = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, dim, 1),
            nn.BatchNorm2d(dim)
        )

        # Group convolution for parameter sharing
        self.group_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=8)

    def forward(self, features, global_context):
        # Light attention mechanism
        attention = self.light_attention(global_context)
        features = features * attention.unsqueeze(-1).unsqueeze(-1)

        # Progressive fusion with shared parameters
        fused = self.shared_fusion(features)

        # Group convolution
        output = self.group_conv(fused)
        return output

@HEADS.register_module()
class LightestECLRNetHead(nn.Module):
    def __init__(self,
                 anchor_generator,
                 in_channels,
                 num_classes=2,
                 num_priors=192,
                 refine_layers=3,
                 loss_cls=None,
                 loss_bbox=None,
                 loss_iou=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        self.anchor_generator = build_prior_generator(anchor_generator)
        self.num_priors = num_priors
        self.refine_layers = refine_layers

        # Lightweight ROI gather
        self.roi_gather = LightROIGather(in_channels, num_priors)

        # Shared classification and regression heads
        self.shared_fcs = nn.ModuleList([
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels // 2)
        ])

        # Classification head with progressive channel reduction
        self.cls_fcs = nn.ModuleList([
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, num_classes)
        ])

        # Regression head with progressive channel reduction
        self.reg_fcs = nn.ModuleList([
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, 4)  # bbox regression
        ])

        # Optimized losses
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        # Dynamic loss weights
        self.register_parameter('loss_weights',
                              nn.Parameter(torch.zeros(3)))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, feats):
        batch_size = feats[0].shape[0]

        # Generate priors
        priors = self.anchor_generator.grid_anchors(
            feats[0].shape[-2:], device=feats[0].device)

        # Progressive refinement
        cls_scores = []
        bbox_preds = []

        for i in range(self.refine_layers):
            feat = feats[i]

            # ROI gather with lightweight attention
            roi_feat = self.roi_gather(feat, priors)

            # Shared feature extraction
            x = roi_feat.view(batch_size * self.num_priors, -1)
            for fc in self.shared_fcs:
                x = fc(x)

            # Classification branch
            cls_feat = x
            for fc in self.cls_fcs:
                cls_feat = fc(cls_feat)
            cls_score = cls_feat.view(batch_size, self.num_priors, -1)

            # Regression branch
            reg_feat = x
            for fc in self.reg_fcs:
                reg_feat = fc(reg_feat)
            bbox_pred = reg_feat.view(batch_size, self.num_priors, -1)

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)

        return cls_scores, bbox_preds

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels):
        # Dynamic loss weights
        loss_weights = F.softmax(self.loss_weights, dim=0)

        # Classification loss with adaptive focal loss
        cls_loss = sum(self.loss_cls(score, gt_labels) * w
                      for score, w in zip(cls_scores, loss_weights))

        # Regression loss with soft IoU
        reg_loss = sum(self.loss_bbox(pred, gt_bboxes) * w
                      for pred, w in zip(bbox_preds, loss_weights))

        # IoU loss with boundary awareness
        iou_loss = sum(self.loss_iou(pred, gt_bboxes) * w
                      for pred, w in zip(bbox_preds, loss_weights))

        return {
            'loss_cls': cls_loss,
            'loss_bbox': reg_loss,
            'loss_iou': iou_loss
        }