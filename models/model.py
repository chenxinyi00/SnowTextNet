from addict import Dict
from torch import nn
import torch.nn.functional as F

from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head
from DGSNet import SnowFormer  # 确保 SnowFormer 可以被正确导入


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        带有 SnowFormer 预处理的 PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')

        self.snowformer = SnowFormer(in_channels=3, out_channels=3)  # 添加 SnowFormer 作为预处理网络
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, snow_img, img):
        _, _, H, W = snow_img.size()
        cleaned_img = self.snowformer(snow_img, img)  # 先去雪

        backbone_out = self.backbone(cleaned_img, img)  # 送入文本检测网络
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y


if __name__ == '__main__':
    import torch

    device = torch.device('cpu')
    snow_img = torch.zeros(2, 3, 640, 640).to(device)
    img = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'backbone': {'type': 'resnet18', 'pretrained': True, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},
        'head': {'type': 'DBHead', 'out_channels': 2, 'k': 50},
    }
    model = Model(model_config=model_config).to(device)

    y = model(snow_img, img)
    print(y.shape)
    print(model.name)
    print(model)
