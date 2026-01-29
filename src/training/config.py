from dataclasses import dataclass, field
from typing import Optional, Literal
from pathlib import Path
import yaml

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    unfreeze_epoch: int = 2
    progressive_merge: bool = False
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

@dataclass
class HybridConfig:
    lc0_proj_dim: int = 128

@dataclass
class PerceiverConfig:
    d_model: int = 256
    n_layers_encoder: int = 12
    n_heads: int = 8
    n_latents: int = 64
    n_layers_pooling: int = 4
    mlp_expansion: int = 4
    
@dataclass
class ModelConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    mode: Literal["hybrid", "engineered", "perceiver"] = "hybrid"
    
    # Universal settings
    load_in_8bit: bool = True
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    use_fen_tokens: bool = False
    
    # Feature settings
    engineered_features_type: Literal["simplified", "main"] = "simplified"
    
    # LC0 settings (only used for hybrid mode)
    lc0_dim: int = 768
    
    # Nested configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)

@dataclass
class TrainingConfig:
    experiment_name: str = "default_run"
    output_dir: str = "checkpoints"
    samples_dir: str = "data/preprocessed/samples"
    
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    max_length: int = 512
    val_split: float = 0.1
    gradient_clip_val: Optional[float] = 30.0
    
    save_steps: int = 100
    logging_steps: int = 1
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "chess-commentary"
    wandb_run_name: Optional[str] = None
    
    # Optimization
    gradient_checkpointing: bool = False
    
    # Nested configs
    model: ModelConfig = field(default_factory=ModelConfig)
    
    def __post_init__(self):
        # Allow nested dicts to be converted to dataclasses
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.model.lora, dict):
            self.model.lora = LoRAConfig(**self.model.lora)
        if isinstance(self.model.hybrid, dict):
            self.model.hybrid = HybridConfig(**self.model.hybrid)
            
        # Ensure float types for numeric fields that might be parsed as strings (e.g. from YAML)
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)
        if isinstance(self.warmup_ratio, str):
            self.warmup_ratio = float(self.warmup_ratio)
        if isinstance(self.val_split, str):
            self.val_split = float(self.val_split)

def load_config(path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        
    # Handle nested dataclass initialization
    model_data = data.get("model", {})
    
    # Handle misplaced gradient_clip_val (it belongs in TrainingConfig)
    if "gradient_clip_val" in model_data:
        val = model_data.pop("gradient_clip_val")
        if "gradient_clip_val" not in data:
            data["gradient_clip_val"] = val
    if "lora" in model_data and isinstance(model_data["lora"], dict):
        model_data["lora"] = LoRAConfig(**model_data["lora"])
    if "hybrid" in model_data and isinstance(model_data["hybrid"], dict):
        model_data["hybrid"] = HybridConfig(**model_data["hybrid"])
    if "perceiver" in model_data and isinstance(model_data["perceiver"], dict):
        model_data["perceiver"] = PerceiverConfig(**model_data["perceiver"])
    
    if model_data:
        data["model"] = ModelConfig(**model_data)

    # Flatten 'training' section if present (fixes compatibility with nested config structure)
    if "training" in data and isinstance(data["training"], dict):
        training_config = data.pop("training")
        data.update(training_config)
        
    return TrainingConfig(**data)
