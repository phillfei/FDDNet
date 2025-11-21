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
    Build model component from config
    
    Args:
        cfg: Config dict containing 'type' and 'kwargs'
    
    Returns:
        Built model component
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
        return cfg


class myModel(nn.Module):
    """
    Complete change detection model
    Combines backbone and decoderhead
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: Config object or dict containing backbone and decoderhead config
        """
        super(myModel, self).__init__()
        self.backbone = build_from_cfg(cfg.backbone)
        self.decoderhead = build_from_cfg(cfg.decoderhead)
    
    def forward(self, x1, x2, gtmask=None):
        """
        Forward pass
        
        Args:
            x1: First image (B, C, H, W)
            x2: Second image (B, C, H, W)
            gtmask: Optional ground truth mask for training (currently not supported by FDDNet, will be ignored)
        
        Returns:
            Change detection result
        """
        backbone_outputs = self.backbone(x1, x2)
        if gtmask == None:
            x_list = self.decoderhead(backbone_outputs)
        else:
            x_list = self.decoderhead(backbone_outputs)
        return x_list


class ModelConfig:
    """
    Model config class for storing backbone and decoderhead config
    """
    def __init__(self, backbone_cfg, decoderhead_cfg):
        self.backbone = backbone_cfg
        self.decoderhead = decoderhead_cfg


def get_channel_list_by_backbone(backbone_name):
    """
    Get channel_list based on backbone name
    
    Args:
        backbone_name: Backbone name ('vgg11', 'resnet18', or 'densenet121')
    
    Returns:
        Corresponding channel_list
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
    if channel_list is None:
        channel_list = get_channel_list_by_backbone(backbone_name)
        print(f"Using default channel_list for {backbone_name}: {channel_list}")
    
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
