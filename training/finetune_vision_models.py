"""
Fine-tune vision models on CIFAR-10 dataset.

This script trains MobileNetV2 and ResNet18 models on CIFAR-10.
Use this if you want full control over training hyperparameters or need
to train on a different dataset.

By default, experiments use ImageNet pre-trained models, which work well
for CIFAR-10 evaluation without fine-tuning.

Output: training/checkpoints/{model_name}/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from src.utils.logger import get_logger

LOGGER = get_logger("training.finetune_vision")

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# Data augmentation
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def load_cifar10_dataset():
    """Load CIFAR-10 dataset."""
    LOGGER.info("Loading CIFAR-10 dataset...")
    
    train_dataset = CIFAR10(
        root='data/cifar10',
        train=True,
        download=True,
        transform=TRAIN_TRANSFORM
    )
    
    test_dataset = CIFAR10(
        root='data/cifar10',
        train=False,
        download=True,
        transform=TEST_TRANSFORM
    )
    
    LOGGER.info(f"Train: {len(train_dataset)} examples")
    LOGGER.info(f"Test: {len(test_dataset)} examples")
    
    return train_dataset, test_dataset


def create_model(model_name):
    """Create model with modified classifier for CIFAR-10."""
    if model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=True)
        # Modify classifier for CIFAR-10 (10 classes)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # Modify final layer for CIFAR-10
        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model_name):
    """Train a single model."""
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"Training {model_name}")
    LOGGER.info(f"{'='*60}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"Using device: {device}")
    
    # Load dataset
    train_dataset, test_dataset = load_cifar10_dataset()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    LOGGER.info(f"Creating {model_name} model...")
    model = create_model(model_name)
    model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_test_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        LOGGER.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        LOGGER.info(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}")
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        LOGGER.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            output_dir = Path(f"training/checkpoints/{model_name}_cifar10")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_acc,
            }, output_dir / 'best_model.pth')
            
            LOGGER.info(f"Saved best model to {output_dir} (test_acc={test_acc:.4f})")
    
    LOGGER.info(f"\nTraining complete. Best test accuracy: {best_test_accuracy:.4f}")
    
    return best_test_accuracy


def main():
    LOGGER.info("Starting vision model fine-tuning on CIFAR-10...")
    LOGGER.info(f"Batch size: {BATCH_SIZE}")
    LOGGER.info(f"Learning rate: {LEARNING_RATE}")
    LOGGER.info(f"Epochs: {NUM_EPOCHS}")
    
    models_to_train = ['mobilenetv2', 'resnet18']
    results = {}
    
    for model_name in models_to_train:
        try:
            best_acc = train_model(model_name)
            results[model_name] = best_acc
        except Exception as e:
            LOGGER.error(f"Failed to train {model_name}: {e}")
            results[model_name] = None
    
    # Summary
    LOGGER.info("\n" + "="*60)
    LOGGER.info("Training Summary")
    LOGGER.info("="*60)
    for model_name, best_acc in results.items():
        if best_acc is not None:
            LOGGER.info(f"{model_name}: {best_acc:.4f}")
        else:
            LOGGER.info(f"{model_name}: FAILED")
    
    LOGGER.info("\nCheckpoints saved to training/checkpoints/")
    LOGGER.info("To use these checkpoints, update src/models/model_zoo.py")


if __name__ == "__main__":
    main()

