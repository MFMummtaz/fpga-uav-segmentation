import torch
import torch.nn as nn
from torchinfo import summary

class DPU_AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation=1):
        super().__init__()
        h, w = mixer_kernel
        if max(h, w) >= 7:
            # Stacked 3x3s depthwise for 7x7 Receptive Field
            self.dw = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
            )
        else:
            self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=dilation, 
                                groups=dim, dilation=dilation, bias=False)

    def forward(self, x):
        return x + self.dw(x)

class VarianceMatchedSkip(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.match = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.match(x)

class FauxChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        gc = dim // 2
        self.squeeze = nn.Conv2d(dim, gc, kernel_size=1, bias=False)
        self.squeeze_act = nn.ReLU6(inplace=True)
        
        self.mix_global = nn.Conv2d(gc, gc, kernel_size=1, groups=1, bias=False)
        self.mix_semi = nn.Conv2d(gc, gc, kernel_size=1, groups=2, bias=False)
        self.mix_local = nn.Conv2d(gc, gc, kernel_size=1, groups=4, bias=False)
        
        self.excite = nn.Conv2d(gc * 3, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x_sq = self.squeeze_act(self.squeeze(x))
        
        b1 = self.mix_global(x_sq)
        b2 = self.mix_semi(x_sq)
        b3 = self.mix_local(x_sq)
        
        x_fused = torch.cat([b1, b2, b3], dim=1)

        attention_out = self.bn(self.excite(x_fused))
        return self.act(x + attention_out)

class FauxAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        gc = dim // 4
        
        # Squeeze channels
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1, bias=False)
        
        # Multi-scale Context Branches (Simulating Attention Heads)
        # Branch 1: Local context (Dilation 1)
        self.dw1 = nn.Conv2d(gc, gc, kernel_size=3, padding=1, dilation=1, groups=gc, bias=False)
        # Branch 2: Mid context (Dilation 2)
        self.dw2 = nn.Conv2d(gc, gc, kernel_size=3, padding=2, dilation=2, groups=gc, bias=False)
        # Branch 3: Global context (Dilation 4 - Massive Receptive Field)
        self.dw3 = nn.Conv2d(gc, gc, kernel_size=3, padding=4, dilation=4, groups=gc, bias=False)

        self.bn = nn.BatchNorm2d(4 * gc)
        self.pw2 = nn.Conv2d(4 * gc, dim, kernel_size=1, bias=False)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x_red = self.pw1(x)
        # Concatenate multi-scale contexts (The "Attention" Fusion)
        x_fused = torch.cat([x_red, self.dw1(x_red), 
                             self.dw2(x_red), self.dw3(x_red)], dim=1)
        return self.act(self.pw2(self.bn(x_fused)))

class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.dw = DPU_AxialDW(in_c, mixer_kernel=(7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False) 
        self.down = nn.MaxPool2d((2, 2))
        self.act = nn.ReLU6(inplace=True)
        self.channel_attn = FauxChannelAttention(in_c)

    def forward(self, x):
        feat = self.bn(self.dw(x))
        attended_feat = self.channel_attn(feat)
        skip = attended_feat
        x = self.act(self.down(self.pw(attended_feat)))
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest') 
        
        self.skip_matcher = VarianceMatchedSkip(skip_c) 
        
        self.pw = nn.Conv2d(in_c + skip_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = DPU_AxialDW(out_c, mixer_kernel=(7, 7))
        self.act = nn.ReLU6(inplace=True)
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1, bias=False)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.skip_matcher(skip) 
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))
        return x


class Novelty_ULite(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding=3, bias=False)
        self.bn_in = nn.BatchNorm2d(16)
        self.act_in = nn.ReLU6(inplace=True)

        self.e1 = EncoderBlock(in_c=16, out_c=32)
        self.e2 = EncoderBlock(in_c=32, out_c=32)
        self.e3 = EncoderBlock(in_c=32, out_c=64)
        self.e4 = EncoderBlock(in_c=64, out_c=64)
        self.e5 = EncoderBlock(in_c=64, out_c=128)

        self.b5 = FauxAttentionBlock(dim=128)

        self.d5 = DecoderBlock(in_c=128, skip_c=64, out_c=64)
        self.d4 = DecoderBlock(in_c=64, skip_c=64, out_c=64)
        self.d3 = DecoderBlock(in_c=64, skip_c=32, out_c=32)
        self.d2 = DecoderBlock(in_c=32, skip_c=32, out_c=32)
        self.d1 = DecoderBlock(in_c=32, skip_c=16, out_c=16)
        
        self.conv_out = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.act_in(self.bn_in(self.conv_in(x)))
        
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        x = self.b5(x)

        x = self.d5(x, skip5)
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        
        x = self.conv_out(x)
        return x

if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = Novelty_ULite().to(device)
    summary(model, (1, 3, 512, 1024))