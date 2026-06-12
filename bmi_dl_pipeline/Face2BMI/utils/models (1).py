
import torch
import torch.nn as nn
import timm
import copy


class BMIModel(nn.Module):
    def __init__(self, arch, dropout=0.3, pretrained=True):
        super(BMIModel, self).__init__()
        self.backbone = timm.create_model(
            arch, pretrained=pretrained, num_classes=0
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, 1)
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)


def get_model(arch, dropout=0.3, pretrained=True):
    return BMIModel(arch, dropout, pretrained)


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema   = copy.deepcopy(model)
        self.decay = decay
        self.ema.eval()

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(),
                                model.parameters()):
                ema_p.data.mul_(self.decay).add_(
                    p.data, alpha=1 - self.decay
                )

    def __call__(self, x):
        return self.ema(x)
