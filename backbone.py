"""
Backbone networks for change detection
基于timm库的常见CNN模型backbone
支持VGG11、ResNet18、DenseNet121
每个backbone输出四个层级的特征 [x1, x2, x3, x4]
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional, Union


class CommonCNNBackbone(nn.Module):
    """
    基于timm库的常见CNN模型backbone
    支持VGG11、DenseNet121、ResNet18等常见模型
    """
    
    def __init__(self, 
                 model_name: str = 'resnet18',
                 pretrained: bool = True,
                 out_stride: int = 32,
                 in_channels: int = 3,
                 **kwargs):
        """
        初始化常见CNN backbone
        
        Args:
            model_name: timm模型名称
            pretrained: 是否使用预训练权重
            out_stride: 输出步长，支持8, 16, 32
            in_channels: 输入通道数
            **kwargs: 其他参数传递给timm模型
        """
        super(CommonCNNBackbone, self).__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_stride = out_stride
        
        # 创建timm模型，使用features_only=True获取多层特征
        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            features_only=True,  # 获取多层特征
            **kwargs
        )
        
        # 获取特征层信息（如果存在）
        if hasattr(self.backbone, 'feature_info'):
            self.feature_info = self.backbone.feature_info
        else:
            self.feature_info = None
    
    def _get_feature_info(self):
        """获取特征层信息"""
        # 获取模型的特征层信息
        feature_info = []
        
        # 这里需要根据具体模型获取特征层信息
        # 不同模型的特征层结构不同，需要具体实现
        
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播，返回四个层级的特征 [x1, x2, x3, x4]
        使用features_only=True获取多层特征
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            特征张量列表，包含四个层级的特征 [x1, x2, x3, x4]
        """
        # 使用features_only=True获取多层特征
        features = self.backbone(x)
        
        # 确保返回四个特征
        if len(features) < 4:
            # 如果特征数量不足，重复最后一个特征
            while len(features) < 4:
                features.append(features[-1])
        elif len(features) > 4:
            # 如果特征数量过多，只取前四个
            features = features[:4]
        
        return features


class VGG11Backbone(CommonCNNBackbone):
    """VGG11 backbone (最轻量化)"""
    
    def __init__(self, 
                 model_name: str = 'vgg11',
                 pretrained: bool = True,
                 out_stride: int = 32,
                 **kwargs):
        # timm中VGG11的正确名称是'vgg11_bn'
        if model_name == 'vgg11':
            model_name = 'vgg11'
        super(VGG11Backbone, self).__init__(
            model_name=model_name,
            pretrained=pretrained,
            out_stride=out_stride,
            **kwargs
        )
    
    def _get_feature_info(self):
        """获取VGG11特征层信息"""
        # VGG11的特征层信息
        feature_info = []
        
        # 这里需要根据具体模型获取特征层信息
        # VGG11通常有多个特征层
        
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        VGG11前向传播，返回四个层级的特征 [x1, x2, x3, x4]
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            特征张量列表，包含四个层级的特征 [x1, x2, x3, x4]
        """
        features = self.backbone(x)
        num_features = len(features)
        
        if num_features >= 4:
            # VGG11: 选择[2,3,4,5]获得56,28,14,7尺寸
            x1 = features[2] if num_features > 2 else features[0]  # [B, 256, 56, 56] - 56×56
            x2 = features[3] if num_features > 3 else features[-1]  # [B, 512, 28, 28] - 28×28
            x3 = features[4] if num_features > 4 else features[-1]  # [B, 512, 14, 14] - 14×14
            x4 = features[5] if num_features > 5 else features[-1]  # [B, 512, 7, 7] - 7×7
        else:
            # 如果特征层不足4个，使用自适应池化创建多尺度特征
            if num_features > 0:
                last_feat = features[-1]
                B, C, H, W = last_feat.shape
                
                x1 = last_feat  # 原始尺寸
                x2 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//2, W//2)) if H >= 2 and W >= 2 else last_feat
                x3 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//4, W//4)) if H >= 4 and W >= 4 else last_feat
                x4 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (1, 1))
            else:
                # 如果没有特征，使用输入创建多尺度特征
                B, C, H, W = x.shape
                x1 = x
                x2 = torch.nn.functional.adaptive_avg_pool2d(x, (H//2, W//2)) if H >= 2 and W >= 2 else x
                x3 = torch.nn.functional.adaptive_avg_pool2d(x, (H//4, W//4)) if H >= 4 and W >= 4 else x
                x4 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        return [x1, x2, x3, x4]


class ResNet18Backbone(CommonCNNBackbone):
    """ResNet18 backbone"""
    
    def __init__(self, 
                 model_name: str = 'resnet18',
                 pretrained: bool = True,
                 out_stride: int = 32,
                 **kwargs):
        super(ResNet18Backbone, self).__init__(
            model_name=model_name,
            pretrained=pretrained,
            out_stride=out_stride,
            **kwargs
        )
    
    def _get_feature_info(self):
        """获取ResNet18特征层信息"""
        # ResNet18的特征层信息
        feature_info = []
        
        # 这里需要根据具体模型获取特征层信息
        # ResNet18通常有多个特征层
        
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        ResNet18前向传播，返回四个层级的特征 [x1, x2, x3, x4]
        提取ResNet18不同层的特征
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            特征张量列表，包含四个层级的特征 [x1, x2, x3, x4]
        """
        # 获取ResNet18的多层特征
        features = self.backbone(x)
        num_features = len(features)
        
        if num_features >= 4:
            # ResNet18: 选择[0,1,2,3]获得不同尺寸的特征
            # 通常对应layer1, layer2, layer3, layer4的输出
            x1 = features[0]  # [B, 64, H/4, W/4] - 对应layer1输出
            x2 = features[1]  # [B, 128, H/8, W/8] - 对应layer2输出
            x3 = features[2]  # [B, 256, H/16, W/16] - 对应layer3输出
            x4 = features[3]  # [B, 512, H/32, W/32] - 对应layer4输出
        else:
            # 如果特征层不足4个，使用自适应池化创建多尺度特征
            if num_features > 0:
                last_feat = features[-1]
                B, C, H, W = last_feat.shape
                
                x1 = last_feat  # 原始尺寸
                x2 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//2, W//2)) if H >= 2 and W >= 2 else last_feat
                x3 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//4, W//4)) if H >= 4 and W >= 4 else last_feat
                x4 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (1, 1))
            else:
                # 如果没有特征，使用输入创建多尺度特征
                B, C, H, W = x.shape
                x1 = x
                x2 = torch.nn.functional.adaptive_avg_pool2d(x, (H//2, W//2)) if H >= 2 and W >= 2 else x
                x3 = torch.nn.functional.adaptive_avg_pool2d(x, (H//4, W//4)) if H >= 4 and W >= 4 else x
                x4 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        return [x1, x2, x3, x4]


class DenseNet121Backbone(CommonCNNBackbone):
    """DenseNet121 backbone (最轻量化)"""
    
    def __init__(self, 
                 model_name: str = 'densenet121',
                 pretrained: bool = True,
                 out_stride: int = 32,
                 **kwargs):
        super(DenseNet121Backbone, self).__init__(
            model_name=model_name,
            pretrained=pretrained,
            out_stride=out_stride,
            **kwargs
        )
    
    def _get_feature_info(self):
        """获取DenseNet121特征层信息"""
        # DenseNet121的特征层信息
        feature_info = []
        
        # 这里需要根据具体模型获取特征层信息
        # DenseNet121通常有多个特征层
        
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        DenseNet121前向传播，返回四个层级的特征 [x1, x2, x3, x4]
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            特征张量列表，包含四个层级的特征 [x1, x2, x3, x4]
        """
        features = self.backbone(x)
        num_features = len(features)
        
        if num_features >= 4:
            # DenseNet121: 选择[1,2,3,4]获得56,28,14,7尺寸
            x1 = features[1]  # [B, 256, 56, 56] - 56×56
            x2 = features[2]  # [B, 512, 28, 28] - 28×28
            x3 = features[3]  # [B, 1024, 14, 14] - 14×14
            x4 = features[4] if num_features > 4 else features[-1]  # [B, 1024, 7, 7] - 7×7
        else:
            # 如果特征层不足4个，使用自适应池化创建多尺度特征
            if num_features > 0:
                last_feat = features[-1]
                B, C, H, W = last_feat.shape
                
                x1 = last_feat  # 原始尺寸
                x2 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//2, W//2)) if H >= 2 and W >= 2 else last_feat
                x3 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//4, W//4)) if H >= 4 and W >= 4 else last_feat
                x4 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (1, 1))
            else:
                # 如果没有特征，使用输入创建多尺度特征
                B, C, H, W = x.shape
                x1 = x
                x2 = torch.nn.functional.adaptive_avg_pool2d(x, (H//2, W//2)) if H >= 2 and W >= 2 else x
                x3 = torch.nn.functional.adaptive_avg_pool2d(x, (H//4, W//4)) if H >= 4 and W >= 4 else x
                x4 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        return [x1, x2, x3, x4]


class ChangeDetectionBackbone(nn.Module):
    """
    Wrapper class that processes two images (A and B) through the backbone
    Returns features in the format expected by FDDNet: (featuresA, featuresB)
    """
    def __init__(self, backbone_name='resnet18', pretrained=True):
        """
        Args:
            backbone_name: 'vgg11', 'resnet18', or 'densenet121'
            pretrained: Whether to use pretrained weights
        """
        super(ChangeDetectionBackbone, self).__init__()
        
        if backbone_name.lower() == 'vgg11':
            self.backbone = VGG11Backbone(pretrained=pretrained)
        elif backbone_name.lower() == 'resnet18':
            self.backbone = ResNet18Backbone(pretrained=pretrained)
        elif backbone_name.lower() == 'densenet121':
            self.backbone = DenseNet121Backbone(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from 'vgg11', 'resnet18', 'densenet121'")
        
    def forward(self, xA, xB):
        """
        Args:
            xA: First image tensor (B, C, H, W)
            xB: Second image tensor (B, C, H, W)
        Returns:
            List of [featuresA, featuresB], where each is a list of 4 feature maps
        """
        featuresA = self.backbone(xA)
        featuresB = self.backbone(xB)
        return [featuresA, featuresB]


# 便捷函数
def get_vgg11(pretrained=True, out_stride=32):
    """获取VGG11模型 (最轻量化)"""
    return VGG11Backbone(
        model_name='vgg11_bn',  # timm中VGG11的正确名称
        pretrained=pretrained,
        out_stride=out_stride
    )


def get_resnet18(pretrained=True, out_stride=32):
    """获取ResNet18模型"""
    return ResNet18Backbone(
        model_name='resnet18',
        pretrained=pretrained,
        out_stride=out_stride
    )


def get_densenet121(pretrained=True, out_stride=32):
    """获取DenseNet121模型 (最轻量化)"""
    return DenseNet121Backbone(
        model_name='densenet121',
        pretrained=pretrained,
        out_stride=out_stride
    )


def create_backbone(model_name: str, pretrained: bool = True, out_stride: int = 32, **kwargs):
    """
    创建backbone的便捷函数
    
    Args:
        model_name: 模型名称 ('vgg11', 'resnet18', 'densenet121')
        pretrained: 是否使用预训练权重
        out_stride: 输出步长
        **kwargs: 其他参数
        
    Returns:
        backbone实例
    """
    if model_name.lower() == 'vgg11':
        return VGG11Backbone(pretrained=pretrained, out_stride=out_stride, **kwargs)
    elif model_name.lower() == 'resnet18':
        return ResNet18Backbone(pretrained=pretrained, out_stride=out_stride, **kwargs)
    elif model_name.lower() == 'densenet121':
        return DenseNet121Backbone(pretrained=pretrained, out_stride=out_stride, **kwargs)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}. 支持: 'vgg11', 'resnet18', 'densenet121'")


if __name__ == "__main__":
    # 测试backbones
    print("="*80)
    print("测试Backbone模型")
    print("="*80)
    
    # 测试输入
    imgA = torch.randn(2, 3, 256, 256)
    imgB = torch.randn(2, 3, 256, 256)
    
    # 测试VGG11
    print("\n1. 测试VGG11 backbone...")
    try:
        vgg_backbone = ChangeDetectionBackbone('vgg11', pretrained=False)
        featuresA, featuresB = vgg_backbone(imgA, imgB)
        print(f"   FeaturesA: {[f.shape for f in featuresA]}")
        print(f"   FeaturesB: {[f.shape for f in featuresB]}")
        print("   ✅ VGG11测试通过")
    except Exception as e:
        print(f"   ❌ VGG11测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试ResNet18
    print("\n2. 测试ResNet18 backbone...")
    try:
        resnet_backbone = ChangeDetectionBackbone('resnet18', pretrained=False)
        featuresA, featuresB = resnet_backbone(imgA, imgB)
        print(f"   FeaturesA: {[f.shape for f in featuresA]}")
        print(f"   FeaturesB: {[f.shape for f in featuresB]}")
        print("   ✅ ResNet18测试通过")
    except Exception as e:
        print(f"   ❌ ResNet18测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试DenseNet121
    print("\n3. 测试DenseNet121 backbone...")
    try:
        densenet_backbone = ChangeDetectionBackbone('densenet121', pretrained=False)
        featuresA, featuresB = densenet_backbone(imgA, imgB)
        print(f"   FeaturesA: {[f.shape for f in featuresA]}")
        print(f"   FeaturesB: {[f.shape for f in featuresB]}")
        print("   ✅ DenseNet121测试通过")
    except Exception as e:
        print(f"   ❌ DenseNet121测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 详细调试特征层
    print("\n" + "="*80)
    print("详细特征层调试")
    print("="*80)
    
    debug_models = ['vgg11', 'resnet18', 'densenet121']
    
    for model_name in debug_models:
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        try:
            backbone = create_backbone(model_name, pretrained=False)
            
            # 创建测试输入
            test_input = torch.randn(1, 3, 224, 224)
            print(f"输入尺寸: {test_input.shape}")
            
            # 获取所有特征层
            all_features = backbone.backbone(test_input)
            print(f"\n{model_name}总共有 {len(all_features)} 个特征层:")
            
            for i, feat in enumerate(all_features):
                print(f"features[{i}]: {feat.shape}")
            
            # 获取我们代码中使用的四个特征
            final_features = backbone(test_input)
            print(f"\n我们代码返回的四个特征:")
            for i, feat in enumerate(final_features):
                print(f"x{i+1}: {feat.shape}")
                
        except Exception as e:
            print(f"{model_name}调试失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ 所有测试完成!")
    print("="*80)
