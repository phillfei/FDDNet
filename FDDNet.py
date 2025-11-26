
from turtle import forward

import torch 
import torch.nn as nn
import torch.nn.functional as F
import sys
# from models.swintransformer import *
import math

def conv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def dsconv_3x3(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, groups=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

def conv_1x1(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )

class SimpleMLP(nn.Module):
    def __init__(self, input_channels, output_channels=None, expansion_ratio=2):
        super().__init__()
        if output_channels is None:
            output_channels = input_channels
            
        hidden_dim = input_channels * expansion_ratio
        self.mlp = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels)
        )
        self.activation = nn.ReLU(inplace=True)
        
        if input_channels != output_channels:
            self.residual_proj = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        else:
            self.residual_proj = None
        
    def forward(self, x):
        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
            
        out = self.mlp(x)
        out = out + residual
        out = self.activation(out)
        return out

class MAD(nn.Module):
    """Magnitude-based Adaptive Difference Enhancement
    
    Core idea:
    1. Compute difference magnitude (L2 norm)
    2. Use nonlinear function (tanh + exp) to adaptively amplify significant differences
    3. Larger differences are amplified more, helping backbone focus on real change regions
    """
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(0.5))
        
        self.diff_transform = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, featA, featB):
        B, C, H, W = featA.shape
        
        diff = featB - featA
        diff_magnitude = torch.sqrt((diff ** 2).sum(dim=1, keepdim=True) + 1e-6)
        magnitude_mean = diff_magnitude.mean(dim=[2, 3], keepdim=True)
        magnitude_normalized = diff_magnitude / (magnitude_mean + 1e-6)
        importance = torch.tanh(magnitude_normalized)
        amplification_factor = 1.0 + self.gamma * (torch.exp(importance) - 1.0)
        amplified_diff = diff * amplification_factor
        amplified_diff = self.diff_transform(amplified_diff)
        
        return amplified_diff


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes//16, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class SelfAttentionBlock(nn.Module):
    """
    query_feats: (B, C, h, w)
    key_feats: (B, C, h, w)
    value_feats: (B, C, h, w)

    output: (B, C, h, w)
    """
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels,
                 key_query_num_convs, value_out_num_convs):
        super(SelfAttentionBlock, self).__init__()
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
        )
        self.query_project = self.buildproject(
            in_channels=query_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs
        )
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=value_out_num_convs
        )
        self.out_project = self.buildproject(
            in_channels=transform_channels,
            out_channels=out_channels,
            num_convs=value_out_num_convs
        )
        self.transform_channels = transform_channels

    def forward(self, query_feats, key_feats, value_feats):
        batch_size = query_feats.size(0)

        query = self.query_project(query_feats)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous() #(B, h*w, C)

        key = self.key_project(key_feats)
        key = key.reshape(*key.shape[:2], -1) # (B, C, h*w)

        value = self.value_project(value_feats)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous() # (B, h*w, C)

        sim_map = torch.matmul(query, key)
       
        sim_map = (self.transform_channels ** -0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1) #(B, h*w, K)
        
        context = torch.matmul(sim_map, value) #(B, h*w, C)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:]) #(B, C, h, w)

        context = self.out_project(context) #(B, C, h, w)
        return context
    def buildproject(self, in_channels, out_channels, num_convs):
        convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for _ in range(num_convs-1):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        if len(convs) > 1:
            return nn.Sequential(*convs)
        return convs[0]

class TFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TFF, self).__init__()
        self.catconvA = dsconv_3x3(in_channel * 2, in_channel)
        self.catconvB = dsconv_3x3(in_channel * 2, in_channel)
        self.catconv = dsconv_3x3(in_channel * 2, out_channel)
        self.convA = nn.Conv2d(in_channel, 1, 1)
        self.convB = nn.Conv2d(in_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, xA, xB):
        x_diff = xA - xB

        x_diffA = self.catconvA(torch.cat([x_diff, xA], dim=1))
        x_diffB = self.catconvB(torch.cat([x_diff, xB], dim=1))

        A_weight = self.sigmoid(self.convA(x_diffA))
        B_weight = self.sigmoid(self.convB(x_diffB))

        xA = A_weight * xA
        xB = B_weight * xB

        x = self.catconv(torch.cat([xA, xB], dim=1))

        return x

class SFF(nn.Module):
    def __init__(self, in_channel):
        super(SFF, self).__init__()
        self.conv_small = conv_1x1(in_channel, in_channel)
        self.conv_big = conv_1x1(in_channel, in_channel)
        self.catconv = conv_3x3(in_channel*2, in_channel)
        self.attention = SelfAttentionBlock(
            key_in_channels=in_channel,
            query_in_channels = in_channel,
            transform_channels = in_channel // 2,
            out_channels = in_channel,
            key_query_num_convs=2,
            value_out_num_convs=1
        )
    
    def forward(self, x_small, x_big):
        img_size  =x_big.size(2), x_big.size(3)
        x_small = F.interpolate(x_small, img_size, mode="bilinear", align_corners=False)
        x = self.conv_small(x_small) + self.conv_big(x_big)
        new_x = self.attention(x, x, x_big)

        out = self.catconv(torch.cat([new_x, x_big], dim=1))
        return out

class PixelAdaptiveDistanceSSFF(nn.Module):
    def __init__(self, channels):
        super(PixelAdaptiveDistanceSSFF, self).__init__()
        
        self.proj_big = nn.Conv2d(channels, channels, 1)
        self.proj_small = nn.Conv2d(channels, channels, 1)
        
        self.pixel_weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 2, 1),
            nn.Sigmoid()
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.local_weight = nn.Parameter(torch.tensor(0.7))
        self.global_weight = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, x_big, x_small):
        B, C, H, W = x_small.shape
        
        big_proj = self.proj_big(x_big)
        small_proj = self.proj_small(x_small)
        
        big_proj_up = F.interpolate(big_proj, (H, W), 
                                     mode='bilinear', align_corners=False)
        
        concat_feat = torch.cat([big_proj_up, small_proj], dim=1)
        
        pixel_weights = self.pixel_weight_net(concat_feat)
        local_alpha = pixel_weights[:, 0:1, :, :]
        local_beta = pixel_weights[:, 1:2, :, :]
        
        global_weights = self.global_context(concat_feat)
        global_alpha = global_weights[:, 0:1, :, :]
        global_beta = global_weights[:, 1:2, :, :]
        
        weight_sum = torch.abs(self.local_weight) + torch.abs(self.global_weight)
        norm_local = torch.abs(self.local_weight) / weight_sum
        norm_global = torch.abs(self.global_weight) / weight_sum
        
        final_alpha = local_alpha * norm_local + global_alpha * norm_global
        final_beta = local_beta * norm_local + global_beta * norm_global
        
        x_big_up = F.interpolate(x_big, (H, W), 
                                 mode='bilinear', align_corners=False)
        
        fused = final_alpha * x_big_up + final_beta * x_small
        
        output = self.refine(fused)
        
        return output

BidirectionalSSFF = PixelAdaptiveDistanceSSFF
SSFF = PixelAdaptiveDistanceSSFF


class GaussianDiffusionStep(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, noisy_feat, guide_feat):
        concat = torch.cat([noisy_feat, guide_feat], dim=1)
        estimated_detail = self.noise_estimator(concat)
        denoised = noisy_feat + self.residual_weight * estimated_detail
        fused = self.fusion(torch.cat([denoised, guide_feat], dim=1))
        
        return fused


class ProgressiveGaussianDecoder(nn.Module):
    def __init__(self, in_channel, num_class, diffusion_steps=3):
        super().__init__()
        
        self.diffusion_steps = diffusion_steps
        
        self.diffusion_blocks = nn.ModuleList([
            GaussianDiffusionStep(in_channel) 
            for _ in range(diffusion_steps)
        ])
        
        self.scale_enhance = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True)
            )
            for _ in range(diffusion_steps)
        ])
        
        self.gaussian_smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, in_channel, 5, padding=2, groups=in_channel, bias=False),
                nn.Conv2d(in_channel, in_channel, 1, bias=False),
                nn.BatchNorm2d(in_channel)
            )
            for _ in range(diffusion_steps)
        ])
        
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, num_class, 1)
        )
        
    def forward(self, x1, x2, x3, x4):
        current = x4
        
        current = F.interpolate(current, size=x3.shape[2:], 
                               mode='bilinear', align_corners=False)
        current = self.gaussian_smooth[0](current)
        guide_x3 = self.scale_enhance[0](x3)
        current = self.diffusion_blocks[0](current, guide_x3)
        
        current = F.interpolate(current, size=x2.shape[2:], 
                               mode='bilinear', align_corners=False)
        current = self.gaussian_smooth[1](current)
        guide_x2 = self.scale_enhance[1](x2)
        current = self.diffusion_blocks[1](current, guide_x2)
        
        current = F.interpolate(current, size=x1.shape[2:], 
                               mode='bilinear', align_corners=False)
        current = self.gaussian_smooth[2](current)
        guide_x1 = self.scale_enhance[2](x1)
        current = self.diffusion_blocks[2](current, guide_x1)
        
        output = self.classifier(current)
        
        return output


class LightDecoder(nn.Module):
    def __init__(self, in_channel, num_class):
        super(LightDecoder, self).__init__()
        self.catconv = conv_3x3(in_channel*4, in_channel)
        self.decoder = nn.Conv2d(in_channel, num_class, 1)
    
    def forward(self, x1, x2, x3, x4):
        x2 = F.interpolate(x2, scale_factor=2, mode="bilinear")
        x3 = F.interpolate(x3, scale_factor=4, mode="bilinear")
        x4 = F.interpolate(x4, scale_factor=8, mode="bilinear")

        out = self.decoder(self.catconv(torch.cat([x1, x2, x3, x4], dim=1)))
        return out



# fca
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    # MultiSpectralAttentionLayer(planes * 4, c2wh[planes], c2wh[planes],  reduction=reduction, freq_sel_method = 'top16')
    # c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
    # planes * 4 -> channel, c2wh[planes] -> dct_h, c2wh[planes] -> dct_w
    # (64*4,56,56)
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape       # (4,256,64,64)
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    # MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter


class FDDNet(nn.Module):
    def __init__(self, num_class, channel_list=[64, 128, 256, 512], transform_feat=128):
        super(FDDNet, self).__init__()


        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])


        # Add MAD modules for each level to amplify differences
        self.mad1 = MAD(channel_list[0])
        self.mad2 = MAD(channel_list[1])
        self.mad3 = MAD(channel_list[2])
        self.mad4 = MAD(channel_list[3])
        
        self.mlp1 = SimpleMLP(channel_list[0], output_channels=transform_feat)
        self.mlp2 = SimpleMLP(channel_list[1], output_channels=transform_feat)
        self.mlp3 = SimpleMLP(channel_list[2], output_channels=transform_feat)
        self.mlp4 = SimpleMLP(channel_list[3], output_channels=transform_feat)

        # self.sff1 = SFF(transform_feat)
        # self.sff2 = SFF(transform_feat)
        # self.sff3 = SFF(transform_feat)

        self.ssff1 = PixelAdaptiveDistanceSSFF(transform_feat)
        self.ssff2 = PixelAdaptiveDistanceSSFF(transform_feat)
        self.ssff3 = PixelAdaptiveDistanceSSFF(transform_feat)

        self.lightdecoder = ProgressiveGaussianDecoder(transform_feat, num_class, diffusion_steps=3)

        self.catconv = conv_3x3(transform_feat*4, transform_feat)
    
    def forward(self, x):
        featuresA, featuresB = x
        xA1, xA2, xA3, xA4 = featuresA
        xB1, xB2, xB3, xB4 = featuresB

        # Use MAD to amplify difference regions
        diff1_enhanced = self.mad1(xA1, xB1)  # (B, C, H, W)
        diff2_enhanced = self.mad2(xA2, xB2)
        diff3_enhanced = self.mad3(xA3, xB3)
        diff4_enhanced = self.mad4(xA4, xB4)

        enhanced_xA1 = xA1 + diff1_enhanced
        enhanced_xA2 = xA2 + diff2_enhanced
        enhanced_xA3 = xA3 + diff3_enhanced
        enhanced_xA4 = xA4 + diff4_enhanced
        
        x111 = self.mlp1(enhanced_xA1)
        x222 = self.mlp2(enhanced_xA2)
        x333 = self.mlp3(enhanced_xA3)
        x444 = self.mlp4(enhanced_xA4)

        x1_new = self.ssff1(x444, x111)
        x2_new = self.ssff2(x444, x222)
        x3_new = self.ssff3(x444, x333)

        # print(x1_new.shape)
        # print(x444.shape)
        # print(x111.shape)

        # print(x2_new.shape)
        # print(x444.shape)
        # print(x222.shape)
        # x4_new = self.catconv(torch.cat([x444, x1_new, x2_new, x3_new], dim=1))
        # print(x4_new.shape)
        out = self.lightdecoder(x1_new, x2_new, x3_new, x444)
        # print(out.shape)
        out = F.interpolate(out, scale_factor=4, mode="bilinear")
        # print(out.shape)
        #return out
        return out


if __name__ == "__main__":
    print("FDDNet Parameter Calculation")
    print("=" * 50)
    
    net = FDDNet(num_class=2, channel_list=[64, 128, 256, 512], transform_feat=128).cuda()
    
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    print("\nTesting forward pass...")
    featuresA = [
        torch.randn(1, 64, 64, 64).cuda(),
        torch.randn(1, 128, 32, 32).cuda(),
        torch.randn(1, 256, 16, 16).cuda(),
        torch.randn(1, 512, 8, 8).cuda()
    ]
    featuresB = [
        torch.randn(1, 64, 64, 64).cuda(),
        torch.randn(1, 128, 32, 32).cuda(),
        torch.randn(1, 256, 16, 16).cuda(),
        torch.randn(1, 512, 8, 8).cuda()
    ]
    
    input_data = (featuresA, featuresB)
    out = net(input_data)
    
    print(f"Output shape: {out.shape}")
    
    print("\nCalculating FLOPs...")
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
        
        net.eval()
        flops_analysis = FlopCountAnalysis(net, (input_data,))
        total_flops = flops_analysis.total()
        
        print(f"Total FLOPs: {total_flops:,}")
        print(f"Total GFLOPs: {total_flops / 1e9:.4f}")
        
        print("\nDetailed FLOPs analysis:")
        print(flop_count_table(flops_analysis, max_depth=2))
        
    except ImportError:
        print("Warning: fvcore not installed, cannot calculate FLOPs")
        print("Please run: pip install fvcore")
    except Exception as e:
        print(f"FLOPs calculation error: {e}")
    
    print("\nCalculating FPS...")
    try:
        import time
        
        net.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = net(input_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        num_runs = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = net(input_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = num_runs / total_time
        avg_time_per_frame = total_time / num_runs * 1000
        
        print(f"Number of runs: {num_runs}")
        print(f"Total time: {total_time:.4f} seconds")
        print(f"Average time per frame: {avg_time_per_frame:.4f} ms")
        print(f"FPS: {fps:.2f}")
        
    except Exception as e:
        print(f"FPS calculation error: {e}")
    
    print("\n FDDNet parameter, FLOPs and FPS calculation completed!")

