"""
Backbone networks for change detection
Common CNN model backbones based on timm library
Supports VGG11, ResNet18, DenseNet121
Each backbone outputs four levels of features [x1, x2, x3, x4]
"""

import torch
import torch.nn as nn
import timm
from typing import List, Optional, Union


class CommonCNNBackbone(nn.Module):
    """
    Common CNN model backbone based on timm library
    Supports VGG11, DenseNet121, ResNet18 and other common models
    """
    
    def __init__(self, 
                 model_name: str = 'resnet18',
                 pretrained: bool = True,
                 out_stride: int = 32,
                 in_channels: int = 3,
                 **kwargs):
        """
        Initialize common CNN backbone
        
        Args:
            model_name: timm model name
            pretrained: Whether to use pretrained weights
            out_stride: Output stride, supports 8, 16, 32
            in_channels: Input channels
            **kwargs: Other parameters passed to timm model
        """
        super(CommonCNNBackbone, self).__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.out_stride = out_stride
        
        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            features_only=True,
            **kwargs
        )
        
        if hasattr(self.backbone, 'feature_info'):
            self.feature_info = self.backbone.feature_info
        else:
            self.feature_info = None
    
    def _get_feature_info(self):
        """Get feature layer information"""
        feature_info = []
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass, returns four levels of features [x1, x2, x3, x4]
        Uses features_only=True to get multi-layer features
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors containing four levels [x1, x2, x3, x4]
        """
        features = self.backbone(x)
        
        if len(features) < 4:
            while len(features) < 4:
                features.append(features[-1])
        elif len(features) > 4:
            features = features[:4]
        
        return features


class VGG11Backbone(CommonCNNBackbone):
    """VGG11 backbone"""
    
    def __init__(self, 
                 model_name: str = 'vgg11',
                 pretrained: bool = True,
                 out_stride: int = 32,
                 **kwargs):
        if model_name == 'vgg11':
            model_name = 'vgg11'
        super(VGG11Backbone, self).__init__(
            model_name=model_name,
            pretrained=pretrained,
            out_stride=out_stride,
            **kwargs
        )
    
    def _get_feature_info(self):
        """Get VGG11 feature layer information"""
        feature_info = []
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        VGG11 forward pass, returns four levels of features [x1, x2, x3, x4]
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors containing four levels [x1, x2, x3, x4]
        """
        features = self.backbone(x)
        num_features = len(features)
        
        if num_features >= 4:
            x1 = features[2] if num_features > 2 else features[0]
            x2 = features[3] if num_features > 3 else features[-1]
            x3 = features[4] if num_features > 4 else features[-1]
            x4 = features[5] if num_features > 5 else features[-1]
        else:
            if num_features > 0:
                last_feat = features[-1]
                B, C, H, W = last_feat.shape
                
                x1 = last_feat
                x2 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//2, W//2)) if H >= 2 and W >= 2 else last_feat
                x3 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//4, W//4)) if H >= 4 and W >= 4 else last_feat
                x4 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (1, 1))
            else:
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
        """Get ResNet18 feature layer information"""
        feature_info = []
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        ResNet18 forward pass, returns four levels of features [x1, x2, x3, x4]
        Extracts features from different layers of ResNet18
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors containing four levels [x1, x2, x3, x4]
        """
        features = self.backbone(x)
        num_features = len(features)
        
        if num_features >= 4:
            x1 = features[0]
            x2 = features[1]
            x3 = features[2]
            x4 = features[3]
        else:
            if num_features > 0:
                last_feat = features[-1]
                B, C, H, W = last_feat.shape
                
                x1 = last_feat
                x2 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//2, W//2)) if H >= 2 and W >= 2 else last_feat
                x3 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//4, W//4)) if H >= 4 and W >= 4 else last_feat
                x4 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (1, 1))
            else:
                B, C, H, W = x.shape
                x1 = x
                x2 = torch.nn.functional.adaptive_avg_pool2d(x, (H//2, W//2)) if H >= 2 and W >= 2 else x
                x3 = torch.nn.functional.adaptive_avg_pool2d(x, (H//4, W//4)) if H >= 4 and W >= 4 else x
                x4 = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        
        return [x1, x2, x3, x4]


class DenseNet121Backbone(CommonCNNBackbone):
    """DenseNet121 backbone"""
    
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
        """Get DenseNet121 feature layer information"""
        feature_info = []
        return feature_info
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        DenseNet121 forward pass, returns four levels of features [x1, x2, x3, x4]
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            List of feature tensors containing four levels [x1, x2, x3, x4]
        """
        features = self.backbone(x)
        num_features = len(features)
        
        if num_features >= 4:
            x1 = features[1]
            x2 = features[2]
            x3 = features[3]
            x4 = features[4] if num_features > 4 else features[-1]
        else:
            if num_features > 0:
                last_feat = features[-1]
                B, C, H, W = last_feat.shape
                
                x1 = last_feat
                x2 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//2, W//2)) if H >= 2 and W >= 2 else last_feat
                x3 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (H//4, W//4)) if H >= 4 and W >= 4 else last_feat
                x4 = torch.nn.functional.adaptive_avg_pool2d(last_feat, (1, 1))
            else:
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


def get_vgg11(pretrained=True, out_stride=32):
    """Get VGG11 model"""
    return VGG11Backbone(
        model_name='vgg11_bn',
        pretrained=pretrained,
        out_stride=out_stride
    )


def get_resnet18(pretrained=True, out_stride=32):
    """Get ResNet18 model"""
    return ResNet18Backbone(
        model_name='resnet18',
        pretrained=pretrained,
        out_stride=out_stride
    )


def get_densenet121(pretrained=True, out_stride=32):
    """Get DenseNet121 model"""
    return DenseNet121Backbone(
        model_name='densenet121',
        pretrained=pretrained,
        out_stride=out_stride
    )


def create_backbone(model_name: str, pretrained: bool = True, out_stride: int = 32, **kwargs):
    """
    Convenience function to create backbone
    
    Args:
        model_name: Model name ('vgg11', 'resnet18', 'densenet121')
        pretrained: Whether to use pretrained weights
        out_stride: Output stride
        **kwargs: Other parameters
        
    Returns:
        Backbone instance
    """
    if model_name.lower() == 'vgg11':
        return VGG11Backbone(pretrained=pretrained, out_stride=out_stride, **kwargs)
    elif model_name.lower() == 'resnet18':
        return ResNet18Backbone(pretrained=pretrained, out_stride=out_stride, **kwargs)
    elif model_name.lower() == 'densenet121':
        return DenseNet121Backbone(pretrained=pretrained, out_stride=out_stride, **kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}. Supported: 'vgg11', 'resnet18', 'densenet121'")


if __name__ == "__main__":
    print("="*80)
    print("Testing Backbone Models")
    print("="*80)
    
    imgA = torch.randn(2, 3, 256, 256)
    imgB = torch.randn(2, 3, 256, 256)
    
    print("\n1. Testing VGG11 backbone...")
    try:
        vgg_backbone = ChangeDetectionBackbone('vgg11', pretrained=False)
        featuresA, featuresB = vgg_backbone(imgA, imgB)
        print(f"   FeaturesA: {[f.shape for f in featuresA]}")
        print(f"   FeaturesB: {[f.shape for f in featuresB]}")
        print("   ✅ VGG11 test passed")
    except Exception as e:
        print(f"   ❌ VGG11 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing ResNet18 backbone...")
    try:
        resnet_backbone = ChangeDetectionBackbone('resnet18', pretrained=False)
        featuresA, featuresB = resnet_backbone(imgA, imgB)
        print(f"   FeaturesA: {[f.shape for f in featuresA]}")
        print(f"   FeaturesB: {[f.shape for f in featuresB]}")
        print("   ✅ ResNet18 test passed")
    except Exception as e:
        print(f"   ❌ ResNet18 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing DenseNet121 backbone...")
    try:
        densenet_backbone = ChangeDetectionBackbone('densenet121', pretrained=False)
        featuresA, featuresB = densenet_backbone(imgA, imgB)
        print(f"   FeaturesA: {[f.shape for f in featuresA]}")
        print(f"   FeaturesB: {[f.shape for f in featuresB]}")
        print("   ✅ DenseNet121 test passed")
    except Exception as e:
        print(f"   ❌ DenseNet121 test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Detailed Feature Layer Debugging")
    print("="*80)
    
    debug_models = ['vgg11', 'resnet18', 'densenet121']
    
    for model_name in debug_models:
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        try:
            backbone = create_backbone(model_name, pretrained=False)
            
            test_input = torch.randn(1, 3, 224, 224)
            print(f"Input size: {test_input.shape}")
            
            all_features = backbone.backbone(test_input)
            print(f"\n{model_name} has {len(all_features)} feature layers:")
            
            for i, feat in enumerate(all_features):
                print(f"features[{i}]: {feat.shape}")
            
            final_features = backbone(test_input)
            print(f"\nFour features returned by our code:")
            for i, feat in enumerate(final_features):
                print(f"x{i+1}: {feat.shape}")
                
        except Exception as e:
            print(f"{model_name} debugging failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(" All tests completed!")
    print("="*80)
