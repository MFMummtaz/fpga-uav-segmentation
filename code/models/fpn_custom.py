
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.ops as ops 
from .attn import PAConv

# class DCNAlignmentBlock(nn.Module):
#     def __init__(self, channels=256):
#         super(DCNAlignmentBlock, self).__init__()
#         # Predicts x,y offsets for 3x3 grid (18 channels).
#         # Input is channels*2 because we concatenate high-res and low-res features.
#         # self.offset_conv = nn.Conv2d(channels * 2, 18, kernel_size=3, padding=1)
#         self.offset_mask_conv = nn.Conv2d(channels * 2, 27, kernel_size=3, padding=1)
        
#         # The Deformable Convolution
#         self.dcn = ops.DeformConv2d(channels, channels, kernel_size=3, padding=1)

#     def forward(self, high_res, low_res):
#         # 1. Upsample the low-resolution semantic features
#         _, _, H, W = high_res.shape
#         upsampled_low_res = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=True)
        
#         # 2. Concatenate to let the network "see" the misalignment
#         concat_feat = torch.cat([high_res, upsampled_low_res], dim=1)
        
#         # 3. Predict offsets and warp the upsampled features
#         # offsets = self.offset_conv(concat_feat)
#         # aligned_low_res = self.dcn(upsampled_low_res, offsets)

#         offset_mask = self.offset_mask_conv(concat_feat)
#         offsets = offset_mask[:, :18, :, :]
#         mask = torch.sigmoid(offset_mask[:, 18:, :, :])

#         aligned_low_res = self.dcn(upsampled_low_res, offsets, mask=mask)
        
#         # 4. Fuse the aligned features with the high-res features
#         return high_res + aligned_low_res

class Bottleneck_Custom(nn.Module):
    def __init__(self, in_ch, out_ch, last_stride, dilation, drop_rate=0.1, attention=True):
        super(Bottleneck_Custom, self).__init__()

        self.attention = attention

        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, bias=False)
        self.BN1 = nn.BatchNorm2d(in_ch)

        #adding singleconv depthwise
        self.conv20 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        self.conv21 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=last_stride, padding=dilation, dilation=dilation)
        self.BN2 = nn.BatchNorm2d(in_ch)
        
        self.downsample = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=last_stride, padding=0, dilation=dilation)
        self.BN_downsample = nn.BatchNorm2d(out_ch)
        
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.BN3 = nn.BatchNorm2d(out_ch)

        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)

        self.PAN_Attn = PAConv(nf=out_ch)


    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.BN1(out)
        out = self.act(out)
        
        out = self.conv20(out)
        out = self.conv21(out)
        out = self.BN2(out)
        out = self.act(out)
        
        out = self.conv3(out)
        out = self.BN3(out)

        if self.attention:
            out = self.PAN_Attn(out)
        
        identity = self.downsample(identity)
        identity = self.BN_downsample(identity)
        
        out += identity
        out = self.act(out)
        out = self.dropout(out)
        
        return out

class FPN(nn.Module):

    def __init__(self, num_blocks, num_classes):
        super(FPN, self).__init__()
        self.in_planes = 32
        self.num_classes = num_classes

        self.conv_in_maxpool = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_planes,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Bottom-up layers
        self.layer1 = self._make_layer(Bottleneck_Custom,  64, num_blocks[0], stride=1, dilation=2)
        self.layer2 = self._make_layer(Bottleneck_Custom, 128, num_blocks[1], stride=2, dilation=2)
        self.layer3 = self._make_layer(Bottleneck_Custom, 256, num_blocks[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(Bottleneck_Custom, 512, num_blocks[3], stride=2, dilation=2)

        # Top layer
        self.toplayer = nn.Conv2d(self.in_planes, 64, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # Deform conv
        # self.dcn_align_p3 = DCNAlignmentBlock(channels=64)
        # self.dcn_align_p2 = DCNAlignmentBlock(channels=64)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, self.num_classes, kernel_size=1, stride=1, padding=0)

        # num_groups, num_channels
        # self.gn1 = nn.GroupNorm(32, 32) 
        # self.gn2 = nn.GroupNorm(64, 64)
        self.bn1 = nn.BatchNorm2d(32) 
        self.bn2 = nn.BatchNorm2d(64)
        
        self.act_func = nn.ReLU(inplace=True)

    # def _upsample(self, x, h, w):
    #     return F.interpolate(x, size=(int(h), int(w)), mode='bilinear', align_corners=False)
    
    def _upsample(self, x, scale):
        # DPU-friendly upsampling with no dynamic shapes
        return F.interpolate(x, scale_factor=scale, mode='nearest')


    # def _make_layer(self, Bottleneck, planes, num_blocks, stride):
    def _make_layer(self, Bottleneck, output, num_blocks, stride, dilation):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_planes, output, stride, dilation, drop_rate=0.1, attention=True))
            self.in_planes = output

        return nn.Sequential(*layers)


    # def _upsample_add(self, x, y):
    #     '''Upsample and add two feature maps.
    #     Args:
    #       x: (Variable) top feature map to be upsampled.
    #       y: (Variable) lateral feature map.
    #     Returns:
    #       (Variable) added feature map.
    #     Note in PyTorch, when input size is odd, the upsampled feature map
    #     with `F.upsample(..., scale_factor=2, mode='nearest')`
    #     maybe not equal to the lateral feature map size.
    #     e.g.
    #     original input size: [N,_,15,15] ->
    #     conv2d feature map size: [N,_,8,8] ->
    #     upsampled feature map size: [N,_,16,16]
    #     So we choose bilinear upsample which supports arbitrary output sizes.
    #     '''
    #     _,_,H,W = y.size()
    #     return F.interpolate(x, size=(int(H),int(W)), mode='bilinear', align_corners=False) + y
    
    def _upsample_add(self, x, y):
        # Fully static interpolation. DPU compliant.
        upsampled_x = F.interpolate(x, scale_factor=2, mode='nearest')
        return upsampled_x + y


    def forward(self, x):
        
        # Bottom-up
        c1 = self.conv_in_maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)


        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        # # p3 aligns latlayer2(c3) with upsampled p4
        # p3 = self.dcn_align_p3(high_res=self.latlayer2(c3), low_res=p4)
        # # p2 aligns latlayer3(c2) with upsampled p3
        # p2 = self.dcn_align_p2(high_res=self.latlayer3(c2), low_res=p3)
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        # # Semantic
        # _, _, h, w = p2.size()
        # # 256->256
        # s5 = self._upsample(self.act_func(self.bn2(self.conv2(p5))), h, w)
        # # 256->256
        # s5 = self._upsample(self.act_func(self.bn2(self.conv2(s5))), h, w)
        # # 256->128
        # s5 = self._upsample(self.act_func(self.bn1(self.semantic_branch(s5))), h, w)

        # # 256->256
        # s4 = self._upsample(self.act_func(self.bn2(self.conv2(p4))), h, w)
        # # 256->128
        # s4 = self._upsample(self.act_func(self.bn1(self.semantic_branch(s4))), h, w)

        # # 256->128
        # s3 = self._upsample(self.act_func(self.bn1(self.semantic_branch(p3))), h, w)

        # s2 = self.act_func(self.bn1(self.semantic_branch(p2)))

        # return self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)

        # Semantic Branch - NO DYNAMIC SHAPES!
        # p5 needs 8x scale to match p2
        s5 = self.act_func(self.bn2(self.conv2(p5)))
        s5 = self._upsample(s5, scale=8)
        s5 = self.act_func(self.bn2(self.conv2(s5)))
        s5 = self.act_func(self.bn1(self.semantic_branch(s5)))

        # p4 needs 4x scale to match p2
        s4 = self.act_func(self.bn2(self.conv2(p4)))
        s4 = self._upsample(s4, scale=4)
        s4 = self.act_func(self.bn1(self.semantic_branch(s4)))

        # p3 needs 2x scale to match p2
        s3 = self.act_func(self.bn1(self.semantic_branch(p3)))
        s3 = self._upsample(s3, scale=2)

        # p2 is already at target scale
        s2 = self.act_func(self.bn1(self.semantic_branch(p2)))

        return self._upsample(self.conv3(s2 + s3 + s4 + s5), scale=4)

    
if __name__ == "__main__":
    DEVICE = "cuda:1"
    model = FPN([2,4,8,16], 32).to(device=DEVICE)
    input = torch.rand(1,3,512,1024).to(device=DEVICE)
    output = model(input)
    print(output.size())