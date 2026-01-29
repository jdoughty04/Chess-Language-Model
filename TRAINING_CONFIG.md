# Training Configuration Guide

The training system uses a YAML-based configuration file. This document details all available configuration options.

## Usage

To start training, provide the path to your configuration file:

```bash
python src/training/train.py --config configs/my_config.yaml
```

## Top-Level Configuration

The configuration file is structured into two main sections: root-level training settings and a nested `model` section.

### General Settings

These settings control the training loop, data loading, and logging. They can be placed at the root level of the YAML file or under a `training` section (which is flattened during loading).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | str | `"default_run"` | Name of the experiment, used for logging and checkpoint naming. |
| `output_dir` | str | `"checkpoints"` | Directory where checkpoints and logs will be saved. |
| `samples_dir` | str | `"data/preprocessed/samples"` | Directory containing the preprocessed dataset. |
| `num_epochs` | int | `10` | Total number of training epochs. |
| `batch_size` | int | `4` | Batch size per GPU. |
| `gradient_accumulation_steps` | int | `8` | Number of steps to accumulate gradients before updating weights. |
| `learning_rate` | float | `2e-4` | Peak learning rate for the scheduler. |
| `warmup_ratio` | float | `0.1` | Fraction of steps to use for learning rate warmup. |
| `max_length` | int | `512` | Maximum sequence length for the model. |
| `val_split` | float | `0.1` | Fraction of data to use for validation. |
| `gradient_clip_val` | float/null | `30.0` | Maximum gradient norm for clipping. Set to `null` to disable. |
| `save_steps` | int | `100` | Number of steps between saving checkpoints. |
| `logging_steps` | int | `1` | Number of steps between logging metrics. |
| `fp16` | bool | `False` | Enable mixed precision training with FP16. |
| `bf16` | bool | `True` | Enable mixed precision training with BF16 (recommended for Ampere+ GPUs). |
| `gradient_checkpointing` | bool | `False` | Enable gradient checkpointing to save memory at the cost of speed. |

### WandB Logging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_wandb` | bool | `True` | Enable Weights & Biases logging. |
| `wandb_project` | str | `"chess-commentary"` | WandB project name. |
| `wandb_run_name` | str/null | `None` | Optional custom name for the WandB run. |

---

## Model Configuration

All model-related settings are nested under the `model` key.

### Core Model Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | str | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` | HuggingFace model ID to use as the base. |
| `mode` | str | `"hybrid"` | Architecture mode: `"hybrid"`, `"engineered"`, or `"perceiver"`. |
| `load_in_8bit` | bool | `True` | Load base model in 8-bit quantization. |
| `use_flash_attention` | bool | `True` | Use Flash Attention 2 if available. |
| `use_torch_compile` | bool | `True` | Use `torch.compile` for faster inference/training. |
| `use_fen_tokens` | bool | `False` | Use explicit FEN tokens in the input. |

### Feature Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engineered_features_type` | str | `"simplified"` | Type of engineered features: `"simplified"` or `"main"`. "main" uses the full 204-dim vector. |
| `lc0_dim` | int | `768` | Dimension of LC0 hidden states (e.g., 768 for BT3, 1024 for BT4). Used only in hybrid mode. |

### LoRA Configuration (`model.lora`)

Settings for Low-Rank Adaptation (LoRA) fine-tuning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | `16` | Rank of the LoRA matrices. |
| `alpha` | int | `32` | Scaling factor for LoRA. |
| `dropout` | float | `0.05` | Dropout probability for LoRA layers. |
| `unfreeze_epoch` | int | `2` | Epoch to start training the base model (if using progressive unfreezing). |
| `progressive_merge` | bool | `False` | Whether to merge LoRA weights back into base model progressively. |
| `target_modules` | list | `["q_proj", ...]` | List of module names to apply LoRA to. |

### Hybrid Mode Configuration (`model.hybrid`)

Specific settings for `mode: "hybrid"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lc0_proj_dim` | int | `128` | Dimension to project LC0 states to before concatenation. |

> [!NOTE]
> Hybrid mode uses the **simplified** engineered feature set by default to save memory and compute. To use the full feature vector, set `model.engineered_features_type: "main"`.

### Perceiver Mode Configuration (`model.perceiver`)

Specific settings for `mode: "perceiver"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d_model` | int | `256` | Hidden dimension of the Perceiver model. |
| `n_layers_encoder` | int | `12` | Number of encoder layers. |
| `n_heads` | int | `8` | Number of attention heads. |
| `n_latents` | int | `64` | Number of latent vectors. |
| `n_layers_pooling` | int | `4` | Number of layers in the pooling stage. |
| `mlp_expansion` | int | `4` | Expansion factor for MLP layers. |

---

## Example Configurations

### Engineered Features (Fast & Efficient)

```yaml
experiment_name: "engineered_v1"
output_dir: "checkpoints/engineered_v1"

model:
  mode: "engineered"
  engineered_features_type: "main"
  load_in_8bit: false

training:
  batch_size: 32
  learning_rate: 7e-4
  num_epochs: 10
  gradient_clip_val: null
```

### Hybrid Mode (Experimental)

```yaml
experiment_name: "hybrid_v1"
output_dir: "checkpoints/hybrid_v1"

model:
  mode: "hybrid"
  base_model: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  hybrid:
    lc0_proj_dim: 256
  lora:
    r: 64
    alpha: 128

training:
  learning_rate: 1e-4
  batch_size: 4
  gradient_accumulation_steps: 8
```
