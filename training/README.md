# Model Training Scripts

This directory contains scripts for fine-tuning models from scratch on SST-2 (text) and CIFAR-10 (vision) datasets.

## When to Use These Scripts

### Default Approach (Recommended)

**Use pre-trained checkpoints** from HuggingFace and PyTorch:

- **Text models:** Already fine-tuned on SST-2
  - `distilbert-base-uncased-finetuned-sst-2-english`
  - `philschmid/MiniLM-L6-H384-uncased-sst2`
  
- **Vision models:** ImageNet pre-trained (works well for CIFAR-10)
  - `torchvision.models.mobilenet_v2(pretrained=True)`
  - `torchvision.models.resnet18(pretrained=True)`

**Advantages:**
- Fast to run (no training time)
- Reproducible (fixed checkpoints)
- Standard practice in research
- Validated performance

**When to use:**
- Quick evaluation and prototyping
- Standard benchmarking
- When training resources are limited
- When reproducibility is critical

### Custom Training (Optional)

**Train models from scratch** using scripts in this directory:

- `finetune_text_models.py` - Train DistilBERT and MiniLM on SST-2
- `finetune_vision_models.py` - Train MobileNetV2 and ResNet18 on CIFAR-10

**Advantages:**
- Full control over hyperparameters
- Can train on different datasets
- Can modify model architecture
- Useful for ablation studies

**When to use:**
- Need to train on custom datasets
- Want to experiment with hyperparameters
- Conducting ablation studies on training
- Need to verify training process

## Usage

### Training Text Models

```bash
# Train DistilBERT and MiniLM on SST-2
python training/finetune_text_models.py
```

**Requirements:**
- GPU recommended (training takes ~2-3 hours on GPU, ~12-15 hours on CPU)
- ~2GB disk space for checkpoints
- ~4GB GPU memory

**Output:**
- `training/checkpoints/distilbert_sst2/` - Fine-tuned DistilBERT
- `training/checkpoints/minilm_sst2/` - Fine-tuned MiniLM

**Expected performance:**
- DistilBERT: ~91-92% validation accuracy
- MiniLM: ~89-90% validation accuracy

### Training Vision Models

```bash
# Train MobileNetV2 and ResNet18 on CIFAR-10
python training/finetune_vision_models.py
```

**Requirements:**
- GPU recommended (training takes ~3-4 hours on GPU, ~20-24 hours on CPU)
- ~1GB disk space for checkpoints
- ~6GB GPU memory

**Output:**
- `training/checkpoints/mobilenetv2_cifar10/` - Fine-tuned MobileNetV2
- `training/checkpoints/resnet18_cifar10/` - Fine-tuned ResNet18

**Expected performance:**
- MobileNetV2: ~92-93% test accuracy
- ResNet18: ~94-95% test accuracy

## Using Custom Checkpoints

After training, update `src/models/model_zoo.py` to use your custom checkpoints:

```python
# For text models
if model_name == 'distilbert':
    model = AutoModelForSequenceClassification.from_pretrained(
        'training/checkpoints/distilbert_sst2'  # Your custom checkpoint
    )

# For vision models
if model_name == 'mobilenetv2':
    model = models.mobilenet_v2(pretrained=False)
    checkpoint = torch.load('training/checkpoints/mobilenetv2_cifar10/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
```

## Hyperparameter Tuning

### Text Models

Edit `finetune_text_models.py`:

```python
BATCH_SIZE = 16          # Increase for faster training (requires more memory)
LEARNING_RATE = 2e-5     # Decrease if training is unstable
NUM_EPOCHS = 3           # Increase for better convergence
MAX_LENGTH = 128         # Increase for longer sequences
WARMUP_STEPS = 500       # Adjust based on dataset size
```

### Vision Models

Edit `finetune_vision_models.py`:

```python
BATCH_SIZE = 64          # Increase for faster training (requires more memory)
LEARNING_RATE = 0.001    # Decrease if training is unstable
NUM_EPOCHS = 20          # Increase for better convergence
WEIGHT_DECAY = 1e-4      # Regularization strength
MOMENTUM = 0.9           # SGD momentum
```

## Validation

After training, validate your models:

```bash
# Run accuracy profiling with custom checkpoints
python experiments/02_profile_accuracy.py

# Check that accuracy matches expected values
cat results/accuracy_profiles.csv
```

## ETH Zurich Thesis Standards

For an ETH Zurich Master's thesis:

1. **Default approach:** Use pre-trained checkpoints
   - Document the checkpoint versions used
   - Report their validation performance
   - Cite the original papers

2. **Optional:** Provide training scripts
   - Shows you understand the full pipeline
   - Allows reviewers to reproduce from scratch
   - Useful for ablation studies

3. **Documentation:** Clearly state which approach was used
   - In README: "Experiments use pre-trained checkpoints"
   - In paper: "We use DistilBERT fine-tuned on SST-2 (Wolf et al., 2020)"
   - In appendix: "Training scripts provided for reproducibility"

## Troubleshooting

### Out of Memory

Reduce batch size:
```python
BATCH_SIZE = 8  # or 4 for very limited memory
```

### Training Too Slow

Use GPU or reduce epochs:
```python
NUM_EPOCHS = 1  # Quick test
```

### Poor Accuracy

Check:
1. Dataset loaded correctly
2. Learning rate not too high
3. Sufficient training epochs
4. Data augmentation appropriate

## References

- DistilBERT: Sanh et al., "DistilBERT, a distilled version of BERT", 2019
- MobileNetV2: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", 2018
- ResNet: He et al., "Deep Residual Learning for Image Recognition", 2016
- SST-2: Socher et al., "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank", 2013
- CIFAR-10: Krizhevsky, "Learning Multiple Layers of Features from Tiny Images", 2009

