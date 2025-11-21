"""
Model builder for change detection
Combines backbone and FDDNet to create a complete change detection model
"""

import torch
import torch.nn as nn
from backbone import ChangeDetectionBackbone
from FDDNet import FDDNet


def build_from_cfg(cfg):
    """
    从配置构建模型组件
    
    Args:
        cfg: 配置字典，包含 'type' 和 'kwargs'
    
    Returns:
        构建的模型组件
    """
    if isinstance(cfg, dict):
        component_type = cfg.get('type')
        kwargs = cfg.get('kwargs', {})
        
        if component_type == 'ChangeDetectionBackbone':
            return ChangeDetectionBackbone(**kwargs)
        elif component_type == 'FDDNet':
            return FDDNet(**kwargs)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
    else:
        # 如果cfg不是字典，直接返回（向后兼容）
        return cfg


class myModel(nn.Module):
    """
    完整的变更检测模型
    结合backbone和decoderhead
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: 配置对象或字典，包含backbone和decoderhead的配置
        """
        super(myModel, self).__init__()
        self.backbone = build_from_cfg(cfg.backbone)
        self.decoderhead = build_from_cfg(cfg.decoderhead)
    
    def forward(self, x1, x2, gtmask=None):
        """
        前向传播
        
        Args:
            x1: 第一张图像 (B, C, H, W)
            x2: 第二张图像 (B, C, H, W)
            gtmask: 可选的ground truth mask，用于训练时使用（目前FDDNet不支持，会被忽略）
        
        Returns:
            变更检测结果
        """
        backbone_outputs = self.backbone(x1, x2)
        if gtmask == None:
            x_list = self.decoderhead(backbone_outputs)
        else:
            # FDDNet目前不支持gtmask参数，暂时忽略
            # 如果需要支持gtmask，需要修改FDDNet的forward方法
            x_list = self.decoderhead(backbone_outputs)
        return x_list


class ModelConfig:
    """
    模型配置类，用于存储backbone和decoderhead的配置
    """
    def __init__(self, backbone_cfg, decoderhead_cfg):
        self.backbone = backbone_cfg
        self.decoderhead = decoderhead_cfg


def get_channel_list_by_backbone(backbone_name):
    """
    根据backbone名称获取对应的channel_list
    
    Args:
        backbone_name: Backbone名称 ('vgg11', 'resnet18', or 'densenet121')
    
    Returns:
        对应的channel_list
    """
    backbone_name = backbone_name.lower()
    channel_map = {
        'vgg11': [256, 512, 512, 512],
        'resnet18': [64, 128, 256, 512],
        'densenet121': [256, 512, 1024, 1024]
    }
    
    if backbone_name not in channel_map:
        raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from {list(channel_map.keys())}")
    
    return channel_map[backbone_name]


def build_model(backbone_name='resnet18', num_class=2, pretrained=True, 
                channel_list=None, transform_feat=128):
    """
    Build a complete change detection model by combining backbone and FDDNet
    
    Args:
        backbone_name: Name of the backbone ('vgg11', 'resnet18', or 'densenet121')
        num_class: Number of output classes (default: 2 for binary change detection)
        pretrained: Whether to use pretrained backbone weights (default: True)
        channel_list: List of channel numbers for each feature level. 
                     If None, will be automatically selected based on backbone_name:
                     - vgg11: [256, 512, 512, 512]
                     - resnet18: [64, 128, 256, 512]
                     - densenet121: [256, 512, 1024, 1024]
        transform_feat: Number of channels after feature transformation (default: 128)
    
    Returns:
        Complete model that takes (imgA, imgB) as input and outputs change map
    
    Example:
        >>> model = build_model('resnet18', num_class=2, pretrained=True)
        >>> imgA = torch.randn(1, 3, 256, 256)
        >>> imgB = torch.randn(1, 3, 256, 256)
        >>> output = model(imgA, imgB)
        >>> print(output.shape)  # (1, 2, 256, 256)
    """
    # 如果没有提供channel_list，根据backbone自动选择
    if channel_list is None:
        channel_list = get_channel_list_by_backbone(backbone_name)
        print(f"Using default channel_list for {backbone_name}: {channel_list}")
    
    # 创建配置
    backbone_cfg = {
        'type': 'ChangeDetectionBackbone',
        'kwargs': {
            'backbone_name': backbone_name,
            'pretrained': pretrained
        }
    }
    
    decoderhead_cfg = {
        'type': 'FDDNet',
        'kwargs': {
            'num_class': num_class,
            'channel_list': channel_list,
            'transform_feat': transform_feat
        }
    }
    
    cfg = ModelConfig(backbone_cfg, decoderhead_cfg)
    
    # 创建模型
    model = myModel(cfg)
    return model


if __name__ == "__main__":
    # Test build_model function
    print("Testing build_model function...")
    
    # Test input
    imgA = torch.randn(1, 3, 256, 256)
    imgB = torch.randn(1, 3, 256, 256)
    
    # Test with ResNet18
    print("\n1. Testing ResNet18 model...")
    model = build_model('resnet18', num_class=2, pretrained=False)
    output = model(imgA, imgB)
    print(f"   Output shape: {output.shape}")
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Test with VGG11
    print("\n2. Testing VGG11 model...")
    model_vgg = build_model('vgg11', num_class=2, pretrained=False)
    output_vgg = model_vgg(imgA, imgB)
    print(f"   Output shape: {output_vgg.shape}")
    
    # Test with DenseNet121
    print("\n3. Testing DenseNet121 model...")
    model_dense = build_model('densenet121', num_class=2, pretrained=False)
    output_dense = model_dense(imgA, imgB)
    print(f"   Output shape: {output_dense.shape}")
    
    # Test direct myModel usage
    print("\n4. Testing direct myModel usage...")
    backbone_cfg = {
        'type': 'ChangeDetectionBackbone',
        'kwargs': {'backbone_name': 'resnet18', 'pretrained': False}
    }
    decoderhead_cfg = {
        'type': 'FDDNet',
        'kwargs': {'num_class': 2, 'channel_list': [64, 128, 256, 512], 'transform_feat': 128}
    }
    cfg = ModelConfig(backbone_cfg, decoderhead_cfg)
    model_direct = myModel(cfg)
    output_direct = model_direct(imgA, imgB)
    print(f"   Output shape: {output_direct.shape}")
    
    print("\n✅ All tests passed!")
