# Training Scripts

Fine-tuning entry points for the models used by the planner. Skip unless you need
custom checkpoints — the rest of the pipeline uses public pre-trained models from
Hugging Face and torchvision.

## Scripts

- `finetune_text_models.py` — fine-tune DistilBERT and MiniLM on SST-2.
- `finetune_vision_models.py` — fine-tune MobileNetV2 and ResNet18 on CIFAR-10.

## Usage

```bash
python training/finetune_text_models.py
python training/finetune_vision_models.py
```

Checkpoints are written under `training/checkpoints/`. To consume them, point
`src/anytime_serving/models/model_zoo.py` at the checkpoint directories instead of the public
Hugging Face IDs.

## Hyperparameters

Edit the constants at the top of each script (`BATCH_SIZE`, `LEARNING_RATE`,
`NUM_EPOCHS`). The defaults are starting points and are not tuned; measure any
resulting checkpoint with `scripts/profile_variants.py` before serving it.
