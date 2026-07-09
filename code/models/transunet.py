import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import einops

from .base import CNNBlock, ViTBlock, OutputBlock, MultiCNNBlock


class CNNBackbone(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.cnn_blocks = nn.ModuleList()

        for _ in range(size):
            self.cnn_blocks.append(CNNBlock(in_channels, out_channels, kernel_size=3, padding=1))
            self.cnn_blocks.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels
            out_channels *= 2

    def forward(self, x):
        route_connection = []

        for layer in self.cnn_blocks:
            if isinstance(layer, CNNBlock):
                x = layer(x)
                route_connection.append(x)
            else:
                x = layer(x)
        
        return x, route_connection


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_class, padding, size=4):
        super().__init__()
        self.decoder_layers = nn.ModuleList()
        self.padding = padding

        for _ in range(size):
            # Use conv transpose for upsampling
            self.decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            self.decoder_layers.append(MultiCNNBlock(in_channels, out_channels, padding, n_conv=2))
            in_channels //= 2
            out_channels //= 2

        # Use normal conv to remove bn and activation function
        self.decoder_layers.append(OutputBlock(out_channels*2, num_class))

    def forward(self, x, routes_connection):
        for layer in self.decoder_layers:
            if isinstance(layer, MultiCNNBlock):
                # concat channels
                x = torch.cat([x, routes_connection.pop(-1)], dim=1)
                x = layer(x)
                # print('Dec Conv', x.size()) 
            else:
                x = layer(x)
                # print('Up sample', x.size())

        return x

class TransUNet(nn.Module):
    def __init__(self, in_channels, start_out_channels, size, num_class, padding, encoder_cfg):
        super().__init__()

        enc_in_channels = start_out_channels*(2**(size-1))
        self.cnn_backbone = CNNBackbone(in_channels=in_channels, out_channels=start_out_channels, size=size)
        self.trans_enc = ViTBlock(encoder_cfg, enc_in_channels)
        self.bottle_neck = CNNBlock(in_channels=encoder_cfg['projection_dim'], out_channels=enc_in_channels*2, padding=1)
        self.decoder = Decoder(
            start_out_channels*(2**size), start_out_channels*(2**(size-1)),
            num_class, padding=padding, size=size
        )

    def forward(self, x):
        x, routes = self.cnn_backbone(x)
        x = self. trans_enc(x)

        # print('Routes:')
        # for r in routes:
        #     print(r.size())

        # print('encoder out:', x.size())
        h = w = int(x.size(1) ** 0.5)
        x = einops.rearrange(x, 'b (h w) D -> b D h w', h=h, w=w)
        # print('encoder reshaped out:', x.size())
        
        x = self.bottle_neck(x)
        # print('bottle neck out: ', x.size())
        x = self.decoder(x, routes)

        return x
    

if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    img_size = x.size(-1)
    batch_size = x.size(0)
    size = 3

    encoder_cfg = dict(
        patch_size=16,
        n_trans=12,
        projection_dim=32,
        mlp_head_units=[1024, 512],
        num_heads=4,
        batch_size=batch_size
    )
    encoder_cfg['num_patches'] = ((img_size//(2**size)) // encoder_cfg['patch_size']) ** 2
    encoder_cfg['feed_forward_dim'] = encoder_cfg['projection_dim'] * 2

    model = TransUNet(3, 64, size, num_classes=1, padding=1, encoder_cfg=encoder_cfg)
    summary(model, input_data=x, col_width=20, depth=5, row_settings=["depth", "var_names"], col_names=["input_size", "kernel_size", "output_size", "params_percent"])