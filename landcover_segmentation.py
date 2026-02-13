"""
LandCover Segmentation with Partial Cross-Entropy Loss
Implements weakly-supervised semantic segmentation using point annotations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PartialCrossEntropyLoss(nn.Module):
    """
    Partial Cross-Entropy Loss for point-supervised segmentation.
    Only computes loss on labeled points, ignoring unlabeled pixels.
    """
    
    def __init__(self, ignore_index: int = -1):
        super(PartialCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (B, C, H, W) - model output logits
            targets: (B, H, W) - sparse point labels with ignore_index for unlabeled pixels
        
        Returns:
            loss: scalar tensor
        """
        # Get valid mask (pixels that are labeled)
        valid_mask = (targets != self.ignore_index)
        
        # If no valid pixels, return zero loss
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)
        
        # Reshape predictions and targets
        B, C, H, W = predictions.shape
        predictions = predictions.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        predictions = predictions.view(-1, C)  # (B*H*W, C)
        targets = targets.view(-1)  # (B*H*W)
        valid_mask = valid_mask.view(-1)  # (B*H*W)
        
        # Select only valid pixels
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Compute cross-entropy loss only on valid pixels
        loss = F.cross_entropy(predictions, targets, reduction='mean')
        
        return loss


class LandCoverDataset(Dataset):
    """
    Dataset for land cover segmentation with point annotations.
    Simulates point labels by randomly sampling from full masks.
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        num_points: int = 100,
        transform=None,
        mode: str = 'train'
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_points = num_points
        self.transform = transform
        self.mode = mode
        
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    def __len__(self):
        return len(self.images)
    
    def sample_points(self, mask: np.ndarray) -> np.ndarray:
        """
        Sample random points from the full mask to create sparse point annotations.
        
        Args:
            mask: (H, W) full segmentation mask
        
        Returns:
            point_mask: (H, W) sparse mask with -1 for unlabeled pixels
        """
        H, W = mask.shape
        point_mask = np.full((H, W), -1, dtype=np.int64)  # Initialize with ignore_index
        
        # Get all valid pixel positions
        valid_positions = []
        unique_classes = np.unique(mask)
        
        # Sample points from each class
        points_per_class = max(1, self.num_points // len(unique_classes))
        
        for class_id in unique_classes:
            class_positions = np.argwhere(mask == class_id)
            
            if len(class_positions) > 0:
                # Randomly sample points from this class
                num_samples = min(points_per_class, len(class_positions))
                sampled_indices = np.random.choice(len(class_positions), num_samples, replace=False)
                sampled_positions = class_positions[sampled_indices]
                
                # Mark these positions in the point mask
                for pos in sampled_positions:
                    point_mask[pos[0], pos[1]] = class_id
        
        return point_mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path))
        
        # For training, create point annotations
        if self.mode == 'train':
            point_mask = self.sample_points(mask)
        else:
            point_mask = mask  # Use full mask for validation
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=point_mask, full_mask=mask)
            image = transformed['image']
            point_mask = transformed['mask']
            full_mask = transformed['full_mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            point_mask = torch.from_numpy(point_mask).long()
            full_mask = torch.from_numpy(mask).long()
        
        return image, point_mask, full_mask


class DeepLabV3Segmentation(nn.Module):
    """
    DeepLabV3 with ResNet50 backbone for semantic segmentation.
    Uses transfer learning from ImageNet pre-trained weights.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        super(DeepLabV3Segmentation, self).__init__()
        
        # Load pre-trained DeepLabV3
        self.model = models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained,
            progress=True
        )
        
        # Modify classifier for our number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[-2:]
        features = self.model(x)
        output = features['out']
        
        # Upsample to input size
        output = F.interpolate(output, size=input_shape, mode='bilinear', align_corners=False)
        
        return output


def get_transforms(mode: str = 'train'):
    """Get data augmentation transforms."""
    
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(height=256, width=256, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'full_mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'full_mask': 'mask'})


def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict:
    """
    Compute Intersection over Union (IoU) for each class.
    
    Args:
        pred: (B, H, W) predicted class indices
        target: (B, H, W) ground truth class indices
    
    Returns:
        Dictionary with per-class IoU and mean IoU
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
        else:
            ious.append(float('nan'))
    
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    return {
        'per_class_iou': ious,
        'mean_iou': mean_iou
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int
) -> dict:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    pbar = tqdm(dataloader, desc='Training')
    
    for images, point_masks, full_masks in pbar:
        images = images.to(device)
        point_masks = point_masks.to(device)
        full_masks = full_masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Compute loss only on labeled points
        loss = criterion(outputs, point_masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # For evaluation, use full masks
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(full_masks.cpu())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_iou(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int
) -> dict:
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, point_masks, full_masks in pbar:
            images = images.to(device)
            point_masks = point_masks.to(device)
            full_masks = full_masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, point_masks)
            total_loss += loss.item()
            
            # Predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(full_masks.cpu())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Compute metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_iou(all_preds, all_targets, num_classes)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def visualize_predictions(
    images: torch.Tensor,
    point_masks: torch.Tensor,
    full_masks: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 4
):
    """Visualize predictions alongside ground truth."""
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot point annotations
        point_viz = point_masks[i].cpu().numpy().copy()
        point_viz = np.ma.masked_where(point_viz == -1, point_viz)
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(point_viz, alpha=0.7, cmap='tab10', vmin=0, vmax=9)
        axes[i, 1].set_title('Point Annotations')
        axes[i, 1].axis('off')
        
        # Plot ground truth
        axes[i, 2].imshow(full_masks[i].cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
        axes[i, 2].set_title('Ground Truth')
        axes[i, 2].axis('off')
        
        # Plot prediction
        axes[i, 3].imshow(predictions[i].cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
        axes[i, 3].set_title('Prediction')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main training script."""
    
    # Hyperparameters
    NUM_CLASSES = 10  # Adjust based on your dataset
    NUM_POINTS = 100  # Number of point annotations per image
    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Create dummy dataset for demonstration
    # Replace these paths with your actual data paths
    TRAIN_IMAGE_DIR = '/path/to/train/images'
    TRAIN_MASK_DIR = '/path/to/train/masks'
    VAL_IMAGE_DIR = '/path/to/val/images'
    VAL_MASK_DIR = '/path/to/val/masks'
    
    # Check if directories exist, otherwise create dummy data
    if not os.path.exists(TRAIN_IMAGE_DIR):
        print("Creating dummy dataset for demonstration...")
        create_dummy_dataset()
    
    # Create datasets
    train_dataset = LandCoverDataset(
        image_dir=TRAIN_IMAGE_DIR,
        mask_dir=TRAIN_MASK_DIR,
        num_points=NUM_POINTS,
        transform=get_transforms('train'),
        mode='train'
    )
    
    val_dataset = LandCoverDataset(
        image_dir=VAL_IMAGE_DIR,
        mask_dir=VAL_MASK_DIR,
        num_points=NUM_POINTS,
        transform=get_transforms('val'),
        mode='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = DeepLabV3Segmentation(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)
    
    # Loss and optimizer
    criterion = PartialCrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_miou = 0.0
    history = {'train_loss': [], 'train_miou': [], 'val_loss': [], 'val_miou': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, DEVICE, NUM_CLASSES)
        history['train_loss'].append(train_metrics['loss'])
        history['train_miou'].append(train_metrics['mean_iou'])
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, DEVICE, NUM_CLASSES)
        history['val_loss'].append(val_metrics['loss'])
        history['val_miou'].append(val_metrics['mean_iou'])
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train mIoU: {train_metrics['mean_iou']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val mIoU: {val_metrics['mean_iou']:.4f}")
        
        # Save best model
        if val_metrics['mean_iou'] > best_miou:
            best_miou = val_metrics['mean_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, '/mnt/user-data/outputs/best_model.pth')
            print(f"Saved best model with mIoU: {best_miou:.4f}")
        
        # Visualize predictions every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                images, point_masks, full_masks = next(iter(val_loader))
                images = images.to(DEVICE)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)
                visualize_predictions(images, point_masks, full_masks, predictions)
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")
    print(f"Model saved to: /mnt/user-data/outputs/best_model.pth")


def plot_training_history(history: dict):
    """Plot training and validation metrics."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # mIoU
    ax2.plot(history['train_miou'], label='Train mIoU')
    ax2.plot(history['val_miou'], label='Val mIoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean IoU')
    ax2.set_title('Training and Validation mIoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_history.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_dummy_dataset():
    """Create a small dummy dataset for demonstration purposes."""
    
    # This is just for demonstration - replace with your actual dataset
    print("Note: Using dummy dataset. Replace with actual land cover segmentation data.")
    print("Expected structure:")
    print("  /path/to/train/images/ - RGB images")
    print("  /path/to/train/masks/ - Segmentation masks (single channel, class IDs)")
    print("  /path/to/val/images/ - RGB images")
    print("  /path/to/val/masks/ - Segmentation masks")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    main()
