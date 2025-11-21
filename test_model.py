"""
Test script for change detection model
Supports selecting backbone, weights, and dataset
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from build_model import build_model, myModel
from change_detection_dataset import create_change_detection_dataloader, get_default_transform, get_label_transform


def create_dummy_dataset(num_samples=10, img_size=256, batch_size=2):
    """
    Create a dummy dataset for testing
    
    Args:
        num_samples: Number of samples in the dataset
        img_size: Size of input images (assumed square)
        batch_size: Batch size for data loader
    
    Returns:
        Data loader
    """
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, img_size):
            self.num_samples = num_samples
            self.img_size = img_size
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            imgA = torch.randn(3, self.img_size, self.img_size)
            imgB = torch.randn(3, self.img_size, self.img_size)
            # Dummy label (binary change map)
            label = torch.randint(0, 2, (self.img_size, self.img_size)).float()
            return imgA, imgB, label
    
    dataset = DummyDataset(num_samples, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    return dataloader


def load_checkpoint(model, checkpoint_path, device='cpu', strict=True):
    """
    Load model weights from checkpoint file
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        strict: Whether to strictly enforce that the keys of state_dict match
    
    Returns:
        Model with loaded weights
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    # Map checkpoint keys to model keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        
        # Remove 'module.' prefix if present
        if name.startswith('module.'):
            name = name[7:]
        
        # Map old checkpoint format to new model format
        # 处理 net.backbone.* 格式
        if name.startswith('net.backbone.backbone.backbone.'):
            # net.backbone.backbone.backbone.* -> backbone.backbone.backbone.*
            name = name.replace('net.backbone.backbone.backbone.', 'backbone.backbone.backbone.')
        elif name.startswith('net.backbone.backbone.'):
            # net.backbone.backbone.* -> backbone.backbone.backbone.* (添加一层backbone)
            name = name.replace('net.backbone.backbone.', 'backbone.backbone.backbone.')
        elif name.startswith('net.backbone.'):
            # net.backbone.* -> backbone.backbone.*
            name = name.replace('net.backbone.', 'backbone.backbone.')
        
        # 处理直接的 backbone.backbone.* 格式（checkpoint中可能是这种格式）
        # 需要映射到 backbone.backbone.backbone.* (因为模型结构是 myModel -> ChangeDetectionBackbone -> VGG11Backbone -> timm模型)
        if name.startswith('backbone.backbone.') and not name.startswith('backbone.backbone.backbone.'):
            # 检查是否是timm模型的特征层（如features_*, layer*, blocks*, stages*等）
            # 这些需要添加一层backbone，因为timm模型在VGG11Backbone内部
            if any(keyword in name for keyword in ['features_', 'layer', 'blocks', 'stages', 'conv', 'bn']):
                name = name.replace('backbone.backbone.', 'backbone.backbone.backbone.', 1)
        
        # net.decoderhead.* -> decoderhead.*
        if name.startswith('net.decoderhead.'):
            name = name.replace('net.decoderhead.', 'decoderhead.')
        
        # Map contrast_attn to mad (MAD module naming)
        if 'contrast_attn' in name:
            name = name.replace('contrast_attn', 'mad')
        
        # Remove num_batches_tracked keys (not needed for inference)
        if 'num_batches_tracked' in name:
            continue
        
        new_state_dict[name] = v
    
    # Load weights with strict matching
    try:
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
        if strict:
            if missing_keys:
                raise RuntimeError(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected_keys}")
            print("✅ Checkpoint loaded successfully (strict=True)")
        else:
            if missing_keys:
                print(f"⚠️  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {unexpected_keys}")
            print("✅ Checkpoint loaded successfully (strict=False)")
    except Exception as e:
        if strict:
            print(f"❌ Error loading checkpoint with strict=True: {e}")
            raise e
        else:
            print(f"⚠️  Warning: {e}")
            print("Trying to load with partial matching...")
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"⚠️  Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {unexpected_keys}")
    
    return model


def test_model(model, dataloader, device='cpu', num_batches=None, has_labels=True):
    """
    Test model with dataset and calculate F1 score
    
    Args:
        model: Model instance
        dataloader: Data loader
        device: Device to run on
        num_batches: Number of batches to test (None for all)
        has_labels: Whether the dataset has ground truth labels (default: True)
    
    Returns:
        Dictionary with test results including F1, Precision, Recall
    """
    model.eval()
    model = model.to(device)
    
    total_time = 0
    total_samples = 0
    
    # For F1 calculation: accumulate all predictions and labels
    all_predictions = []
    all_labels = []
    
    total_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
    print(f"Processing {total_batches} batches (total samples: {len(dataloader.dataset)})...")
    if not has_labels:
        print("  Note: No ground truth labels available, skipping metric calculation")
    
    with torch.no_grad():
        for batch_idx, (imgA, imgB, label) in enumerate(dataloader):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            imgA = imgA.to(device)
            imgB = imgB.to(device)
            label = label.to(device)
            
            # Forward pass
            start_time = time.time()
            output = model(imgA, imgB)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            total_samples += imgA.size(0)
            
            # Get predictions (for binary classification)
            if output.size(1) == 2:
                pred = torch.argmax(output, dim=1)  # (B, H, W)
                
                # Only collect labels if we have ground truth
                if has_labels:
                    label_long = label.long()  # (B, H, W)
                    
                    # Flatten and collect predictions and labels
                    pred_flat = pred.cpu().numpy().flatten()
                    label_flat = label_long.cpu().numpy().flatten()
                    
                    all_predictions.append(pred_flat)
                    all_labels.append(label_flat)
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Processed {batch_idx + 1}/{total_batches} batches ({total_samples} samples), "
                      f"Time: {elapsed_time*1000:.2f}ms/batch")
    
    # Calculate metrics (only if we have labels)
    if has_labels and len(all_predictions) > 0:
        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)
        
        # Calculate metrics
        pixel_accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='binary', zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate IoU (Intersection over Union) for change pixels (class 1)
        # IoU = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
    else:
        pixel_accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        tn, fp, fn, tp = 0, 0, 0, 0
        iou = 0.0
    
    results = {
        'total_samples': total_samples,
        'total_time': total_time,
        'avg_time_per_sample': total_time / total_samples if total_samples > 0 else 0,
        'fps': total_samples / total_time if total_time > 0 else 0,
        'pixel_accuracy': pixel_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }
    
    return results


def print_model_info(model, device='cpu'):
    """
    Print model information
    
    Args:
        model: Model instance
        device: Device
    """
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size
    param_size = total_params * 4 / (1024 * 1024)  # MB (assuming float32)
    
    print("\n" + "="*60)
    print("Model Information")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {param_size:.2f} MB")
    print(f"Device: {device}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test change detection model')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='vgg11',
                        choices=['vgg11', 'resnet18', 'densenet121'],
                        help='Backbone network (default: resnet18)')
    parser.add_argument('--num_class', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Use pretrained backbone weights')
    
    # Weight loading
    parser.add_argument('--checkpoint', type=str, default='test_change_f1=0.7759-epoch=54.ckpt',
                        help='Path to checkpoint file to load')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='folder',
                        choices=['dummy', 'custom', 'folder'],
                        help='Dataset type: dummy (synthetic), custom (real dataset), or folder (test all images in folder) (default: dummy)')
    parser.add_argument('--data_root', type=str, default='F:/CLCD/CLCD/test/',
                        help='Root directory of the dataset (required for custom/folder dataset)')
    parser.add_argument('--imgA_dir', type=str, default='image1',
                        help='Subdirectory name for time A images (default: image1)')
    parser.add_argument('--imgB_dir', type=str, default='image2',
                        help='Subdirectory name for time B images (default: image2)')
    parser.add_argument('--label_dir', type=str, default='label',
                        help='Subdirectory name for labels (default: label, optional for folder mode)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples in dataset for dummy (default: 10)')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Input image size. If None, keep original image size (default: None - keep original)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Number of batches to test (default: all - processes entire dataset)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("Change Detection Model Test")
    print("="*60)
    print(f"Backbone: {args.backbone}")
    print(f"Number of classes: {args.num_class}")
    print(f"Pretrained backbone: {args.pretrained}")
    print(f"Checkpoint: {args.checkpoint if args.checkpoint else 'None'}")
    print(f"Dataset: {args.dataset}")
    if args.dataset == 'custom' or args.dataset == 'folder':
        print(f"Data root: {args.data_root}")
        print(f"Image A dir: {args.imgA_dir}")
        print(f"Image B dir: {args.imgB_dir}")
        if args.dataset == 'custom':
            print(f"Label dir: {args.label_dir}")
        else:
            print(f"Label dir: None (folder mode - testing all images without labels)")
    else:
        print(f"Number of samples: {args.num_samples}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")
    print("="*60)
    
    # Build model
    print("\nBuilding model...")
    # channel_list会根据backbone自动选择，不需要手动指定
    model = build_model(
        backbone_name=args.backbone,
        num_class=args.num_class,
        pretrained=args.pretrained,
        channel_list=None,  # None表示根据backbone自动选择
        transform_feat=128
    )
    
    # 验证模型类型
    if not isinstance(model, myModel):
        raise TypeError(f"Expected myModel instance, got {type(model)}")
    
    # Load checkpoint if provided
    if args.checkpoint:
        model = load_checkpoint(model, args.checkpoint, device=device)
    
    # Print model info
    print_model_info(model, device=device)
    
    # Create dataset
    print(f"\nCreating {args.dataset} dataset...")
    if args.dataset == 'dummy':
        dataloader = create_dummy_dataset(
            num_samples=args.num_samples,
            img_size=args.img_size,
            batch_size=args.batch_size
        )
    elif args.dataset == 'custom' or args.dataset == 'folder':
        if args.data_root is None:
            raise ValueError("--data_root must be specified for custom/folder dataset")
        if not os.path.exists(args.data_root):
            raise ValueError(f"Data root directory does not exist: {args.data_root}")
        
        # folder模式：测试文件夹中所有图片，标签可选
        # custom模式：必须有标签
        if args.dataset == 'custom':
            label_dir = args.label_dir
            has_labels = True
        elif args.dataset == 'folder':
            # folder模式：如果提供了label_dir且存在，则使用标签
            if args.label_dir:
                label_path = os.path.join(args.data_root, args.label_dir)
                if os.path.exists(label_path):
                    label_dir = args.label_dir
                    has_labels = True
                else:
                    label_dir = None
                    has_labels = False
                    print(f"  - Warning: Label directory not found: {label_path}, will skip metric calculation")
            else:
                label_dir = None
                has_labels = False
        
        dataloader = create_change_detection_dataloader(
            root_dir=args.data_root,
            imgA_dir=args.imgA_dir,
            imgB_dir=args.imgB_dir,
            label_dir=label_dir,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            img_size=args.img_size,
            is_training=False
        )
        print(f"Loaded {len(dataloader.dataset)} image pairs from {args.data_root}")
        if args.dataset == 'folder':
            print(f"  - Image A directory: {os.path.join(args.data_root, args.imgA_dir)}")
            print(f"  - Image B directory: {os.path.join(args.data_root, args.imgB_dir)}")
            if has_labels:
                print(f"  - Label directory: {os.path.join(args.data_root, args.label_dir)}")
                print(f"  - Testing all {len(dataloader.dataset)} image pairs with labels")
            else:
                print(f"  - Testing all {len(dataloader.dataset)} image pairs (no labels)")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Test model
    print("\nTesting model...")
    # has_labels已经在上面确定了
    results = test_model(model, dataloader, device=device, num_batches=args.num_batches, has_labels=has_labels)
    
    # Print results
    print("\n" + "="*60)
    print("Test Results")
    print("="*60)
    print(f"Total samples tested: {results['total_samples']}")
    print(f"Total time: {results['total_time']:.4f} seconds")
    print(f"Average time per sample: {results['avg_time_per_sample']*1000:.2f} ms")
    print(f"FPS (Frames Per Second): {results['fps']:.2f}")
    
    if has_labels:
        print("\n" + "-"*60)
        print("Evaluation Metrics")
        print("-"*60)
        print(f"Pixel Accuracy: {results['pixel_accuracy']*100:.4f}%")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"IoU (Intersection over Union): {results['iou']:.4f}")
        print("\n" + "-"*60)
        print("Confusion Matrix")
        print("-"*60)
        cm = results['confusion_matrix']
        print(f"True Negatives (TN):  {cm['tn']:,}")
        print(f"False Positives (FP): {cm['fp']:,}")
        print(f"False Negatives (FN): {cm['fn']:,}")
        print(f"True Positives (TP):  {cm['tp']:,}")
        print(f"\nConfusion Matrix:")
        print(f"        Predicted")
        print(f"        0      1")
        print(f"Actual 0 [{cm['tn']:6d} {cm['fp']:6d}]")
        print(f"       1 [{cm['fn']:6d} {cm['tp']:6d}]")
    else:
        print("\n" + "-"*60)
        print("Note: No ground truth labels available")
        print("      All images have been processed successfully")
        print("      Metrics cannot be calculated without labels")
    
    print("="*60)
    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()

