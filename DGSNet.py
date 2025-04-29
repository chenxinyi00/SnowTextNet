# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # 计算模型的参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())
#
#
# # 轻量级 Guidance Fusion Block
# class GuidanceFusionBlock(nn.Module):
#     def __init__(self, g_in_channel, out_channel):
#         super(GuidanceFusionBlock, self).__init__()
#         self.g_conv = nn.Conv2d(g_in_channel, out_channel, kernel_size=1)
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x, g):
#         g1 = self.g_conv(g)
#         if x.shape != g1.shape:
#             g1 = F.interpolate(g1, size=x.size()[2:], mode='bilinear', align_corners=False)
#         out = x + g1
#         out = self.out_conv(out)
#         return x * out
#
#
# # 轻量级 Transformer 替换模块（使用简单的卷积）
# class LightweightTransformer(nn.Module):
#     def __init__(self, dim):
#         super(LightweightTransformer, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
#         self.norm = nn.BatchNorm2d(dim)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         return self.relu(self.norm(self.conv2(self.conv1(x))))
#
#
# # 轻量级 SnowFormer
# class SnowFormer(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, dim=[16, 32, 64]):
#         super(SnowFormer, self).__init__()
#         self.embed = nn.Conv2d(in_channels, dim[0], kernel_size=3, padding=1)
#         self.guidance_fusion = GuidanceFusionBlock(g_in_channel=in_channels, out_channel=dim[0])
#
#         self.down1 = nn.Conv2d(dim[0], dim[1], kernel_size=3, stride=2, padding=1)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(dim[1], dim[1], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         # 替换 Transformer 层
#         self.light_transformer = LightweightTransformer(dim[1])
#
#         self.up1 = nn.ConvTranspose2d(dim[1], dim[0], kernel_size=4, stride=2, padding=1)
#         self.decoder = nn.Sequential(
#             nn.Conv2d(dim[0], dim[0], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         self.out_conv = nn.Conv2d(dim[0], out_channels, kernel_size=3, padding=1)
#
#     def forward(self, x, guidance):
#         x0 = self.embed(x)
#         x0 = self.guidance_fusion(x0, guidance)
#
#         x1 = self.down1(x0)
#         x1 = self.encoder(x1)
#         x1 = self.light_transformer(x1)  # 轻量级 Transformer 处理
#
#         x2 = self.up1(x1)
#         x2 = self.decoder(x2)
#
#         out = self.out_conv(x2) + x
#         return out
#
#
# # 计算参数量
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters())
#
#
# # Example usage
# if __name__ == '__main__':
#     model = SnowFormer()
#     print(f'Total number of parameters: {count_parameters(model)}')


# Guidance Fusion Block
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from timm.models.layers import trunc_normal_
# import math
#
# # Guidance Fusion Block
# class GuidanceFusionBlock(nn.Module):
#     def __init__(self, g_in_channel, out_channel):
#         super(GuidanceFusionBlock, self).__init__()
#         self.g_conv1 = nn.Sequential(
#             nn.Conv2d(g_in_channel, out_channel, kernel_size=1),
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel, padding_mode='reflect'),
#             nn.ReLU(inplace=True),
#         )
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=1),
#             nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel, padding_mode='reflect'),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x, g):
#         g1 = self.g_conv1(g)
#         if x.shape != g1.shape:
#             g1 = F.interpolate(g1, size=x.size()[2:], mode='bilinear', align_corners=False)
#         out = x + g1
#         out = self.out_conv(out)
#         out = x + out
#         return out
#
# # Guidance Attention Block
# class GuidanceAttentionBlock(nn.Module):
#     def __init__(self, x_in_channel, g_in_channel, out_channel):
#         super(GuidanceAttentionBlock, self).__init__()
#         self.W_x = nn.Sequential(
#             nn.Conv2d(x_in_channel, out_channel, kernel_size=1, stride=1, padding=0),
#         )
#         self.W_g = nn.Sequential(
#             nn.Conv2d(g_in_channel, out_channel, kernel_size=1, stride=1, padding=0),
#         )
#         self.ga = nn.Sequential(
#             nn.Conv2d(out_channel, out_channel, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x, g):
#         x1 = self.W_x(x)
#         g1 = self.W_g(g)
#         ga = self.ga(x1 + g1)
#         return x * ga
#
# # Transformer Encoder Layer
# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, dim, heads, dropout=0.1):
#         super(TransformerEncoderLayer, self).__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Linear(dim * 4, dim)
#         )
#         self.layernorm1 = nn.LayerNorm(dim)
#         self.layernorm2 = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         attn_output, _ = self.attn(x, x, x)
#         x = self.layernorm1(x + self.dropout(attn_output))
#         ffn_output = self.ffn(x)
#         x = self.layernorm2(x + self.dropout(ffn_output))
#         return x
#
# # SnowFormer with Guidance Integration and Transformer Encoder
# class SnowFormer(nn.Module):
#     def __init__(self,
#                  in_channels=3,
#                  out_channels=3,
#                  dim=[16, 32, 64, 128, 256],
#                  transformer_blocks=8,
#                  head=[1, 2, 4, 8, 16],
#                  global_atten=True):
#         super(SnowFormer, self).__init__()
#         self.embed = nn.Conv2d(in_channels, dim[0], kernel_size=3, padding=1)
#         self.guidance_fusion_layer0 = GuidanceFusionBlock(g_in_channel=in_channels, out_channel=dim[0])
#         self.guidance_fusion_layer1 = GuidanceFusionBlock(g_in_channel=in_channels, out_channel=dim[1])
#
#         self.down0 = nn.Conv2d(dim[0], dim[1], kernel_size=3, stride=2, padding=1)
#         self.encoder_level0 = nn.Sequential(
#             nn.Conv2d(dim[0], dim[0], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.encoder_level1 = nn.Sequential(
#             nn.Conv2d(dim[1], dim[1], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         self.down1 = nn.Conv2d(dim[1], dim[2], kernel_size=3, stride=2, padding=1)
#         self.encoder_level2 = nn.Sequential(
#             nn.Conv2d(dim[2], dim[2], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#
#         # Add Transformer Encoder Layer
#         self.transformer_layers = nn.ModuleList(
#             [TransformerEncoderLayer(dim=dim[2], heads=head[2]) for _ in range(transformer_blocks)]
#         )
#
#         self.up0 = nn.ConvTranspose2d(dim[1], dim[0], kernel_size=4, stride=2, padding=1)
#         self.decoder_level0 = nn.Sequential(
#             nn.Conv2d(dim[0], dim[0], kernel_size=3, padding=1),
#             nn.ReLU(inplace=True)
#         )
#         self.guidance_attention_layer = GuidanceAttentionBlock(x_in_channel=dim[0], g_in_channel=in_channels, out_channel=dim[0])
#
#         self.out_conv = nn.Conv2d(dim[0], out_channels, kernel_size=3, padding=1)
#
#     def forward(self, x, guidance):
#         # Encoding with guidance fusion
#         inp_enc_level0 = self.embed(x)
#         inp_enc_level0 = self.guidance_fusion_layer0(inp_enc_level0, guidance)
#         inp_enc_level0 = self.encoder_level0(inp_enc_level0)
#
#         inp_enc_level1 = self.down0(inp_enc_level0)
#         inp_enc_level1 = self.guidance_fusion_layer1(inp_enc_level1, guidance)
#         inp_enc_level1 = self.encoder_level1(inp_enc_level1)
#
#         inp_enc_level2 = self.down1(inp_enc_level1)
#         inp_enc_level2 = self.encoder_level2(inp_enc_level2)
#
#         # Transformer Encoding Layer
#         inp_enc_level2 = inp_enc_level2.flatten(2).permute(2, 0, 1)  # Flatten and permute for multi-head attention
#         for transformer in self.transformer_layers:
#             inp_enc_level2 = transformer(inp_enc_level2)
#         inp_enc_level2 = inp_enc_level2.permute(1, 2, 0).reshape_as(inp_enc_level2)  # Reshape back
#
#         # Decoding with guidance attention
#         inp_dec_level0 = self.up0(inp_enc_level1)
#         inp_dec_level0 = self.decoder_level0(inp_dec_level0)
#         inp_dec_level0 = self.guidance_attention_layer(inp_dec_level0, guidance)
#
#         # Output
#         out = self.out_conv(inp_dec_level0) + x
#         return out
#
# # Example usage
# if __name__ == '__main__':
#     model = SnowFormer()
#     input_image = torch.randn(1, 3, 256, 256)
#     guidance_image = torch.randn(1, 3, 256, 256)
#     output = model(input_image, guidance_image)
#     print(output.shape)




import torch
import torch.nn as nn
import torch.nn.functional as F

# Guidance Fusion Block
class GuidanceFusionBlock(nn.Module):
    def __init__(self, g_in_channel, out_channel):
        super(GuidanceFusionBlock, self).__init__()
        self.g_conv1 = nn.Conv2d(g_in_channel, out_channel, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, groups=out_channel),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        g1 = self.g_conv1(g)
        if x.shape != g1.shape:
            g1 = F.interpolate(g1, size=x.size()[2:], mode='bilinear', align_corners=False)
        out = x + g1
        out = self.out_conv(out)
        return x + out

# Guidance Attention Block
class GuidanceAttentionBlock(nn.Module):
    def __init__(self, x_in_channel, g_in_channel, out_channel):
        super(GuidanceAttentionBlock, self).__init__()
        self.W_x = nn.Conv2d(x_in_channel, out_channel, kernel_size=1)
        self.W_g = nn.Conv2d(g_in_channel, out_channel, kernel_size=1)
        self.ga = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, g):
        ga = self.ga(self.W_x(x) + self.W_g(g))
        return x * ga

# SnowFormer with Guidance Integration
class SnowFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=[16, 32, 64]):
        super(SnowFormer, self).__init__()
        self.embed = nn.Conv2d(in_channels, dim[0], kernel_size=3, padding=1)
        self.guidance_fusion_layer0 = GuidanceFusionBlock(g_in_channel=in_channels, out_channel=dim[0])
        self.guidance_fusion_layer1 = GuidanceFusionBlock(g_in_channel=in_channels, out_channel=dim[1])

        self.down0 = nn.Conv2d(dim[0], dim[1], kernel_size=3, stride=2, padding=1)
        self.encoder_level1 = nn.Conv2d(dim[1], dim[1], kernel_size=3, padding=1)

        self.down1 = nn.Conv2d(dim[1], dim[2], kernel_size=3, stride=2, padding=1)
        self.encoder_level2 = nn.Conv2d(dim[2], dim[2], kernel_size=3, padding=1)

        self.up0 = nn.ConvTranspose2d(dim[1], dim[0], kernel_size=4, stride=2, padding=1)
        self.decoder_level0 = nn.Conv2d(dim[0], dim[0], kernel_size=3, padding=1)
        self.guidance_attention_layer = GuidanceAttentionBlock(x_in_channel=dim[0], g_in_channel=in_channels, out_channel=dim[0])

        self.out_conv = nn.Conv2d(dim[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, guidance):
        inp_enc_level0 = self.guidance_fusion_layer0(self.embed(x), guidance)
        inp_enc_level1 = self.guidance_fusion_layer1(self.down0(inp_enc_level0), guidance)
        inp_enc_level2 = self.encoder_level2(self.down1(inp_enc_level1))
        inp_dec_level0 = self.guidance_attention_layer(self.decoder_level0(self.up0(inp_enc_level1)), guidance)
        out = self.out_conv(inp_dec_level0) + x
        return out

# Example usage
if __name__ == '__main__':
    model = SnowFormer()
    input_image = torch.randn(1, 3, 256, 256)
    guidance_image = torch.randn(1, 3, 256, 256)
    output = model(input_image, guidance_image)
    print(output.shape)

# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         se = self.avg_pool(x)
#         se = self.fc(se)
#         return x * se
#
#
# class MultiScaleConvBlockV2(nn.Module):
#     def __init__(self, in_channels, out_channels, use_residual=True):
#         super(MultiScaleConvBlockV2, self).__init__()
#         self.use_residual = use_residual
#
#         self.branch3x3 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.branch5x5 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, groups=2),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.branch7x7 = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 1x1 reduce
#             nn.Conv2d(out_channels, out_channels, kernel_size=7, padding=3, groups=out_channels),  # depthwise
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.fuse = nn.Sequential(
#             nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         self.se = SEBlock(out_channels)
#
#     def forward(self, x):
#         out3 = self.branch3x3(x)
#         out5 = self.branch5x5(x)
#         out7 = self.branch7x7(x)
#
#         out = torch.cat([out3, out5, out7], dim=1)
#         out = self.fuse(out)
#         out = self.se(out)
#
#         if self.use_residual and x.shape == out.shape:
#             return out + x
#         else:
#             return out
