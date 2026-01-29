"""
Chess Commentary Training Script with LoRA

Trains TinyLlama with LoRA fine-tuning to generate chess commentary
using LC0 transformer hidden states as position context.

Architecture:
- ChessPositionAdapter: LC0 hidden states -> 64 prefix embeddings
- TinyLlama 1.1B with LoRA (q_proj, v_proj, k_proj, o_proj)
- 8-bit quantization for memory efficiency (fits in 6GB VRAM)
"""

import os
import sys
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optional wandb import for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)

from training.chess_adapter import ChessPositionAdapter, EngineeredPositionAdapter, HybridPositionAdapter, extract_engineered_features
try:
    from training.perceiver_adapter import PerceiverChessAdapter, extract_perceiver_features
except ImportError:
    PerceiverChessAdapter = None
    extract_perceiver_features = None
from training.config import TrainingConfig, ModelConfig, load_config


# Default configuration
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROJECTION_DIM = 128
DEFAULT_LC0_DIM = 768
DEFAULT_NUM_LAYERS = 4


class ChessCommentaryModel(nn.Module):
    """
    Combines ChessPositionAdapter with TinyLlama + LoRA.
    
    Architecture:
    1. ChessPositionAdapter projects LC0 hidden states to 64 prefix embeddings
    2. Prefix embeddings are prepended to text token embeddings
    3. Combined sequence is processed by TinyLlama with LoRA
    """
    
    
    def __init__(self, config: ModelConfig, torch_dtype: torch.dtype = torch.float16):
        super().__init__()
        
        self.config = config
        self.base_model_name = config.base_model
        self.torch_dtype = torch_dtype
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for 8-bit
        if config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None
        
        # Check Flash Attention 2 availability
        attn_impl = None
        if config.use_flash_attention:
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print(f"Flash Attention 2 enabled")
            except Exception as e:
                print(f"Flash Attention 2 not available: {e}")
                print("Using default SDPA attention")
        
        # Load base model
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "torch_dtype": self.torch_dtype,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )
        
        # Prepare for k-bit training (enables gradients for 8-bit)
        if config.load_in_8bit:
            self.llm = prepare_model_for_kbit_training(self.llm)
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_config)

        # Ensure input embeddings require grads - CRITICAL for Adapter training
        # This ensures gradients backpropagate through the entire frozen LLM to reach the adapter
        # regardless of gradient checkpointing settings.
        self.llm.enable_input_require_grads()
        if config.use_flash_attention:
             # Flash Attention 2 specific robustness: ensure we don't accidentally detach
             pass
        
        # Get LLM hidden dimension
        llm_dim = self.llm.config.hidden_size
        
        # Chess position adapter
        if config.mode == "hybrid":
            # Use hybrid adapter (engineered features + LC0 embeddings)
            self.adapter = HybridPositionAdapter(
                lc0_dim=config.lc0_dim,
                lc0_proj_dim=config.hybrid.lc0_proj_dim,
                llm_dim=llm_dim,
                num_layers=4, # Hardcoded or add to config if really needed, but simplification was requested
            )
            print(f"Using HybridPositionAdapter (Engineered + LC0, lc0_proj_dim={config.hybrid.lc0_proj_dim})")
        elif config.mode == "engineered":
            self.adapter = EngineeredPositionAdapter(
                llm_dim=llm_dim,
            )
            print("Using EngineeredPositionAdapter (FEN-based, 204-dim features)")
        elif config.mode == "perceiver":
            if PerceiverChessAdapter is None:
                raise ImportError("Perceiver adapter requested but training.perceiver_adapter module not found.")
            self.adapter = PerceiverChessAdapter(config, d_model=config.perceiver.d_model)
            print("Using PerceiverChessAdapter")
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
        
        # Move adapter to same device as LLM embeddings
        self._sync_devices()
        
        # 1 side token + 64 square tokens = 65 prefix tokens
        self.num_prefix_tokens = self.adapter.get_num_prefix_tokens()

        # FEN tokens feature
        self.use_fen_tokens = getattr(config, "use_fen_tokens", False)
        
        # Apply torch.compile for faster forward/backward passes (PyTorch 2.0+)
        if config.use_torch_compile and hasattr(torch, 'compile'):
            try:
                self.llm = torch.compile(self.llm, mode="reduce-overhead")
                print("torch.compile enabled (reduce-overhead mode)")
            except Exception as e:
                print(f"torch.compile failed, using eager mode: {e}")
    
    def _sync_devices(self):
        """Ensure adapter is on the same device as LLM embeddings."""
        embed_device = next(self.llm.get_input_embeddings().parameters()).device
        self.adapter = self.adapter.to(embed_device)
    
    def forward(
        self, 
        lc0_hidden_states: Optional[dict[str, torch.Tensor]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        side_to_move: Optional[torch.Tensor] = None,
        fen: Optional[list[str]] = None,
        engineered_features: Optional[torch.Tensor] = None,
        perceiver_features: Optional[tuple] = None,
    ):
        """
        Forward pass combining chess embeddings with text.
        
        Args:
            lc0_hidden_states: Dict with layer tensors, each (B, 64, 768). 
                               Ignored when use_engineered_features=True.
            input_ids: Text token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            labels: Target labels for loss (B, seq_len)
            side_to_move: Boolean tensor (B,) - True=White, False=Black
            fen: List of FEN strings (for LC0 adapter piece encoding)
            engineered_features: Pre-computed features (B, 64, 204)
            perceiver_features: Tuple of (sq_features, glob_features)
        
        Returns:
            Model outputs with loss if labels provided
        """
        batch_size = input_ids.shape[0]
        
        # Get adapter device (where we need all tensors to be)
        adapter_device = next(self.adapter.parameters()).device
        
        # 1. Get position embeddings from adapter (1 side token + 64 square embeddings)
        if self.config.mode == "hybrid":
            # Hybrid adapter takes both LC0 hidden states and engineered features
            if engineered_features is None:
                raise ValueError("engineered_features must be provided when mode='hybrid'")
            lc0_states = {k: v.to(adapter_device) for k, v in lc0_hidden_states.items()}
            position_embeds = self.adapter(lc0_states, engineered_features, side_to_move=side_to_move)  # (B, 65, hidden_size)
        elif self.config.mode == "engineered":
            # Engineered adapter takes pre-computed feature tensors
            if engineered_features is None:
                raise ValueError("engineered_features must be provided when mode='engineered'")
            position_embeds = self.adapter(engineered_features, side_to_move=side_to_move)  # (B, 65, hidden_size)
        elif self.config.mode == "perceiver":
            if perceiver_features is None:
                 raise ValueError("perceiver_features must be provided when mode='perceiver'")
            # perceiver_features is tuple (sq, gl)
            # Move to adapter device
            sq, gl = perceiver_features
            perceiver_features = (sq.to(adapter_device), gl.to(adapter_device))
            position_embeds = self.adapter(perceiver_features, side_to_move=side_to_move)
        else:
             raise ValueError(f"Unknown mode: {self.config.mode}")
        
        # Move input tensors to same device as LLM
        device = next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        
        # 2. Get token embeddings from LLM
        token_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq, hidden)
        
        # 3. Cast adapter output to match LLM embedding dtype (float16)
        position_embeds = position_embeds.to(dtype=token_embeds.dtype)
        
        # 4. Optionally add FEN tokens right after position embeddings
        if self.use_fen_tokens and fen is not None:
            # Tokenize FEN strings for batch
            fen_encodings = self.tokenizer(
                fen,
                padding=True,
                truncation=True,
                max_length=64,  # FEN is ~70 chars max, leaves some buffer
                return_tensors="pt",
            )
            fen_ids = fen_encodings["input_ids"].to(device)
            fen_mask = fen_encodings["attention_mask"].to(device)
            fen_embeds = self.llm.get_input_embeddings()(fen_ids)  # (B, fen_len, hidden)
            self._fen_len = fen_ids.shape[1]  # Store for mask/label extension
            self._fen_mask = fen_mask
        else:
            fen_embeds = None
            self._fen_len = 0
            self._fen_mask = None
        
        # 5. Prepend position embeddings (side token + squares) as prefix
        if fen_embeds is not None:
            combined_embeds = torch.cat([position_embeds, fen_embeds, token_embeds], dim=1)
        else:
            combined_embeds = torch.cat([position_embeds, token_embeds], dim=1)
        
        # 6. Extend attention mask for prefix tokens (and optional FEN tokens)
        attention_mask = attention_mask.to(device)
        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=device
        )
        if self._fen_len > 0 and self._fen_mask is not None:
            combined_mask = torch.cat([prefix_mask, self._fen_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # 7. Extend labels if provided (use -100 for prefix and FEN - no loss)
        if labels is not None:
            labels = labels.to(device)
            prefix_len = self.num_prefix_tokens + self._fen_len
            prefix_labels = torch.full(
                (batch_size, prefix_len),
                -100,
                dtype=labels.dtype,
                device=device
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            combined_labels = None
        
        # 6. Forward through LLM
        # Explicitly generate position_ids to ensure Flash Attention handles inputs_embeds correctly
        # Combined mask is (B, Total_Len) with 1s for valid, 0s for pad
        # Standard Llama position_ids: cumsum of mask, minus 1, with pads set to 1
        position_ids = combined_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(combined_mask == 0, 1)
        
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            position_ids=position_ids,
            labels=combined_labels,
        )
        
        return outputs
    
    def generate(
        self,
        lc0_hidden_states: dict[str, torch.Tensor],
        side_to_move: bool = True,
        prompt: str = "Provide commentary.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        fen: Optional[str] = None,
    ) -> str:
        """
        Generate commentary for a chess position.
        
        Args:
            lc0_hidden_states: LC0 hidden states for the position
            side_to_move: True = White to move, False = Black to move
            prompt: Text prompt (minimal by design)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated commentary string
        """
        self.eval()
        
        # Prepare prompt using chat template
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        )
        
        device = next(self.llm.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Get embeddings
        batch_size = 1
        # Get embeddings
        batch_size = 1
        if self.config.mode == "hybrid":
            # Hybrid adapter takes both LC0 states and engineered features
            if fen is None:
                raise ValueError("FEN is required for hybrid features mode")
            features = extract_engineered_features(fen).unsqueeze(0).to(device)  # (1, 64, 204)
            lc0_states = {k: v.unsqueeze(0).to(device) if v.dim() == 2 else v.to(device) 
                          for k, v in lc0_hidden_states.items()}
            side_tensor = torch.tensor([side_to_move], dtype=torch.bool, device=device)
            position_embeds = self.adapter(lc0_states, features, side_to_move=side_tensor)
        elif self.config.mode == "engineered":
            # Engineered adapter takes feature tensor - extract on-the-fly for inference
            if fen is None:
                raise ValueError("FEN is required for engineered features mode")
            features = extract_engineered_features(fen).unsqueeze(0).to(device)  # (1, 64, 204)
            side_tensor = torch.tensor([side_to_move], dtype=torch.bool, device=device)
            position_embeds = self.adapter(features, side_to_move=side_tensor)
        else:
             raise ValueError(f"Unknown mode: {self.config.mode}")
        token_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Cast adapter output to match LLM embedding dtype
        position_embeds = position_embeds.to(dtype=token_embeds.dtype)
        
        # Optionally add FEN tokens right after position embeddings
        if self.use_fen_tokens and fen is not None:
            fen_encodings = self.tokenizer(
                [fen],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            fen_ids = fen_encodings["input_ids"].to(device)
            fen_mask = fen_encodings["attention_mask"].to(device)
            fen_embeds = self.llm.get_input_embeddings()(fen_ids)
            fen_len = fen_ids.shape[1]
        else:
            fen_embeds = None
            fen_mask = None
            fen_len = 0
        
        if fen_embeds is not None:
            combined_embeds = torch.cat([position_embeds, fen_embeds, token_embeds], dim=1)
        else:
            combined_embeds = torch.cat([position_embeds, token_embeds], dim=1)
        
        # Extend attention mask
        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=device
        )
        if fen_len > 0 and fen_mask is not None:
            combined_mask = torch.cat([prefix_mask, fen_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode (skip prefix tokens not needed as generate with inputs_embeds returns only new tokens)
        generated_ids = outputs[0]
        commentary = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return commentary
    
    def save_pretrained(self, output_dir: str):
        """Save adapter and LoRA weights."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights
        self.llm.save_pretrained(output_path / "lora")
        
        # Save adapter weights
        torch.save(self.adapter.state_dict(), output_path / "adapter.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path / "tokenizer")
        
        print(f"Saved model to {output_dir}")
    
    @classmethod
    def load_pretrained(
        cls, 
        checkpoint_dir: str, 
        config: Optional[ModelConfig] = None,
    ):
        """Load saved adapter and LoRA weights."""
        checkpoint_path = Path(checkpoint_dir)
        
        # If config not provided, try to load from checkpoint (TODO: save config with checkpoint)
        # For now, we assume config is passed or we default to something safe (but hybrid/engineered mode matters)
        if config is None:
            # Try to infer mode from adapter weights? Or just default to hybrid
            # Better to require config if we can't infer
            # For this refactor, let's assume config is passed in train() loop
             raise ValueError("config object required for load_pretrained")

        # Create model (initializes with random LoRA weights)
        model = cls(config)
        
        # Load adapter weights
        adapter_path = checkpoint_path / "adapter.pt"
        if adapter_path.exists():
            print(f"Loading adapter weights from {adapter_path}")
            model.adapter.load_state_dict(torch.load(adapter_path, weights_only=False))
        
        # Load LoRA weights
        lora_path = checkpoint_path / "lora"
        if lora_path.exists():
            print(f"Loading LoRA weights from {lora_path}")
            # We need to load the trained weights into the PEFT model
            # Since model.llm is already a PeftModel, we can use load_adapter
            try:
                model.llm.load_adapter(str(lora_path), adapter_name="default")
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                print("Attempting alternative loading method...")
                from peft import PeftModel
                # Reload base model and wrap with loaded LoRA
                # Note: This is heavier but safer fallback
                # Access the underlying base model (original transformer)
                base = model.llm.base_model.model
                model.llm = PeftModel.from_pretrained(base, str(lora_path))
        
        return model
    
    def print_trainable_parameters(self):
        """Print trainable parameter counts."""
        # Adapter parameters
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        
        # LoRA parameters  
        lora_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        
        # Total LLM parameters
        total_llm = sum(p.numel() for p in self.llm.parameters())
        
        print(f"\nTrainable Parameters:")
        print(f"  Adapter: {adapter_params:,}")
        print(f"  LoRA: {lora_params:,}")
        print(f"  Total trainable: {adapter_params + lora_params:,}")
        print(f"  Total LLM (frozen): {total_llm:,}")
        print(f"  Trainable %: {100 * (adapter_params + lora_params) / total_llm:.2f}%")
    
    def freeze_lora(self):
        """Freeze LoRA parameters (train only adapter)."""
        for name, param in self.llm.named_parameters():
            if 'lora_' in name:
                param.requires_grad = False
        print("LoRA parameters frozen - training adapter only")
    
    def unfreeze_lora(self):
        """Unfreeze LoRA parameters (train both adapter and LoRA)."""
        for name, param in self.llm.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        print("LoRA parameters unfrozen - training adapter + LoRA")
    
    def is_lora_frozen(self) -> bool:
        """Check if LoRA parameters are frozen."""
        for name, param in self.llm.named_parameters():
            if 'lora_' in name:
                return not param.requires_grad
        return True  # No LoRA params found
    
    def merge_and_reinit_lora(self):
        """
        Merge current LoRA weights into base model and reinitialize fresh LoRA.
        
        This implements "progressive LoRA" where each epoch's LoRA learns on top
        of previously merged weights, allowing cumulative learning.
        """
        import torch.nn.init as init
        
        # Merge LoRA into base weights
        self.llm.merge_adapter()
        print("  Merged LoRA weights into base model")
        
        # Reinitialize LoRA weights (lora_A: Kaiming, lora_B: zeros)
        lora_a_count = 0
        lora_b_count = 0
        for name, param in self.llm.named_parameters():
            if 'lora_A' in name:
                # Kaiming uniform initialization (default PEFT behavior)
                init.kaiming_uniform_(param, a=5**0.5)
                lora_a_count += 1
            elif 'lora_B' in name:
                # Zero initialization (so LoRA starts as identity)
                init.zeros_(param)
                lora_b_count += 1
        
        # Unmerge so we can train new LoRA (keeps merged weights, LoRA now identity)
        self.llm.unmerge_adapter()
        
        print(f"  Reinitialized {lora_a_count} lora_A and {lora_b_count} lora_B matrices")


class ChessCommentaryTrainingDataset(Dataset):
    """
    Training dataset that tokenizes commentary on the fly.
    """
    
    def __init__(
        self,
        samples_dir: str,
        tokenizer,
        max_length: int = 512,
        prompt: str = "Provide commentary on this chess position.",
        use_engineered_features: bool = False,
        use_hybrid_features: bool = False,
        use_perceiver_features: bool = False,
        feature_mode: str = "simplified",
    ):
        self.samples_dir = Path(samples_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        self.use_engineered_features = use_engineered_features
        self.use_hybrid_features = use_hybrid_features
        self.use_perceiver_features = use_perceiver_features
        self.feature_mode = feature_mode
        
        # Find all sample files
        self.sample_files = sorted(self.samples_dir.glob("*.pt"))
        print(f"Found {len(self.sample_files)} training samples")
        if use_engineered_features:
            print("  [Engineered features] Pre-computing features during data loading")
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        # Load preprocessed sample
        sample = torch.load(self.sample_files[idx], weights_only=False)
        
        # Convert hidden states to float32 and validate shapes (only if needed)
        lc0_hidden_states = {}
        if self.use_hybrid_features:
            raw_states = sample.get("lc0_hidden_states", {})
            if not raw_states and self.use_hybrid_features:
                 # Warn or error? Error is safer for hybrid mode
                 raise ValueError(f"Sample {self.sample_files[idx]} missing LC0 states for hybrid mode")
                 
            for k, v in raw_states.items():
                tensor = v.float()
                # Expected shape: (64, 768) - 64 squares, 768 embedding dim
                if tensor.dim() != 2 or tensor.shape[0] != 64 or tensor.shape[1] != 768:
                    raise ValueError(
                        f"Malformed lc0_hidden_states in {self.sample_files[idx]}: "
                        f"layer {k} has shape {tensor.shape}, expected (64, 768)"
                    )
                lc0_hidden_states[k] = tensor
        elif not self.use_engineered_features:
            # Legacy/Pure LC0 mode also needs states
             raw_states = sample.get("lc0_hidden_states", {})
             for k, v in raw_states.items():
                lc0_hidden_states[k] = v.float()
        
        # Extract side-to-move from FEN
        # FEN format: "pieces turn castling en_passant halfmove fullmove"
        # e.g., "rnbqkb1r/... w KQkq - 0 1" -> 'w' = White to move
        fen = sample.get("fen", "")
        side_to_move = True  # Default to White
        if fen:
            fen_parts = fen.split()
            if len(fen_parts) >= 2:
                side_to_move = (fen_parts[1] == 'w')
        
        # Pre-compute engineered features if enabled (moves CPU work out of forward pass)
        # Used by both engineered-only mode and hybrid mode
        engineered_features = None
        if (self.use_engineered_features or self.use_hybrid_features) and fen:
            engineered_features = extract_engineered_features(fen, mode=self.feature_mode)  # (64, 204)
            
        # Online extraction for Perceiver
        perceiver_sq_feats = None
        perceiver_glob_feats = None
        if self.use_perceiver_features and fen:
            if extract_perceiver_features is None:
                raise ImportError("Perceiver features requested but training.perceiver_adapter module not found.")
            sq, glob = extract_perceiver_features(fen)
            perceiver_sq_feats = sq
            perceiver_glob_feats = glob
        
        # Create prompt using chat template
        messages = [{"role": "user", "content": self.prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Full text = prompt + commentary
        full_text = prompt_text + sample["commentary"]
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Create labels - mask prompt tokens with -100
        prompt_encoding = self.tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        prompt_len = prompt_encoding["input_ids"].shape[1]
        
        labels = encoding["input_ids"].clone()
        labels[:, :prompt_len] = -100  # Don't compute loss on prompt
        
        result = {
            "lc0_hidden_states": lc0_hidden_states,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "side_to_move": side_to_move,
            "fen": fen,
        }
        
        if engineered_features is not None:
            result["engineered_features"] = engineered_features
            
        if perceiver_sq_feats is not None:
            result["perceiver_sq_features"] = perceiver_sq_feats
            result["perceiver_glob_features"] = perceiver_glob_feats
        
        return result


def collate_fn(batch):
    """Custom collate function to handle dict of LC0 hidden states and engineered features."""
    # Stack LC0 hidden states
    lc0_keys = batch[0]["lc0_hidden_states"].keys()
    lc0_hidden_states = {
        key: torch.stack([item["lc0_hidden_states"][key] for item in batch])
        for key in lc0_keys
    }
    
    result = {
        "lc0_hidden_states": lc0_hidden_states,
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "side_to_move": torch.tensor([item["side_to_move"] for item in batch], dtype=torch.bool),
        "fen": [item["fen"] for item in batch],  # List of FEN strings
    }
    
    # Stack engineered features if present
    if "engineered_features" in batch[0]:
        result["engineered_features"] = torch.stack([item["engineered_features"] for item in batch])

    if "perceiver_sq_features" in batch[0]:
        sq = torch.stack([item["perceiver_sq_features"] for item in batch])
        glob = torch.stack([item["perceiver_glob_features"] for item in batch])
        result["perceiver_features"] = (sq, glob)
    
    return result


def save_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    plateau_scheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    lora_unfrozen: bool,
):
    """Save training state for resume capability."""
    state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "plateau_scheduler_state_dict": plateau_scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "lora_unfrozen": lora_unfrozen,
    }
    torch.save(state, checkpoint_dir / "training_state.pt")


def load_training_state(checkpoint_dir: Path):
    """Load training state for resume."""
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        return torch.load(state_path, weights_only=False)
    return None


def train(config: TrainingConfig):
    """
    Main training function using configuration object.
    
    Args:
        config: TrainingConfig object containing all settings
    """
    print("=" * 60)
    print("Chess Commentary Training with LoRA")
    print("=" * 60)
    
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if config.use_wandb:
        # Convert config to dict for logging
        from dataclasses import asdict
        wandb_config = asdict(config)
        
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=wandb_config,
        )
        print(f"\nWandb initialized: {wandb.run.name}")
    
    # Initialize model
    # Note: We rely on the model config to determine mode
    print(f"\n[DEBUG] Config use_torch_compile: {config.model.use_torch_compile}")
    
    print(f"\nLoading model: {config.model.base_model}")
    
    # Determine model dtype based on training precision
    model_dtype = torch.float16
    if config.bf16:
        model_dtype = torch.bfloat16
        print("Initializing model in bfloat16")
    elif config.fp16:
        model_dtype = torch.float16
        print("Initializing model in float16")
        
    model = ChessCommentaryModel(config.model, torch_dtype=model_dtype)
    
    model.print_trainable_parameters()
    
    # Enable Gradient Checkpointing (Critical for VRAM with large batches)
    if hasattr(config, "gradient_checkpointing") and config.gradient_checkpointing:
        print("\n[Memory] Enabling Gradient Checkpointing")
        model.llm.gradient_checkpointing_enable()
        # Required for LoRA/Adapter training with frozen base
        model.llm.enable_input_require_grads()
        # Disable cache as it's incompatible with checkpointing
        model.llm.config.use_cache = False
    
    # Progressive LoRA settings
    lora_unfreeze_epoch = config.model.lora.unfreeze_epoch
    progressive_lora_merge = config.model.lora.progressive_merge
    
    if lora_unfreeze_epoch == 1:
        print(f"\n[Progressive LoRA] Training with LoRA from epoch 1")
        lora_unfrozen = True
    elif lora_unfreeze_epoch == 0:
        print(f"\n[Progressive LoRA] LoRA will remain frozen (adapter-only training)")
        model.freeze_lora()
        lora_unfrozen = False
    else:
        print(f"\n[Progressive LoRA] Starting with adapter-only, will unfreeze LoRA at epoch {lora_unfreeze_epoch}")
        model.freeze_lora()
        lora_unfrozen = False
    
    # Determine learning rate
    learning_rate = config.learning_rate
    
    def build_optimizer():
        """Build optimizer with current trainable parameters."""
        trainable = list(model.adapter.parameters())
        if not model.is_lora_frozen():
            trainable += [p for p in model.llm.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=learning_rate)
    
    # Create dataset with train-val split
    print(f"\nLoading training data from: {config.samples_dir}")
    full_dataset = ChessCommentaryTrainingDataset(
        samples_dir=config.samples_dir,
        tokenizer=model.tokenizer,
        max_length=config.max_length,
        use_engineered_features=(config.model.mode == "engineered"),
        use_hybrid_features=(config.model.mode == "hybrid"),
        use_perceiver_features=(config.model.mode == "perceiver"),
        feature_mode=getattr(config.model, 'engineered_features_type', 'sparse'),
    )
    
    # Split into train and validation
    from torch.utils.data import random_split
    total_size = len(full_dataset)
    val_size = int(total_size * config.val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    print(f"  Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Determine number of workers based on platform
    import platform
    is_linux = platform.system() == "Linux"
    # Allow config to override default logic if num_workers is exposed? 
    # For now, default to 4 on Linux, 0 on Windows
    actual_num_workers = 4 if is_linux else 0
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=actual_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=2 if actual_num_workers > 0 else None,
        persistent_workers=True if actual_num_workers > 0 else False,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=actual_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    if actual_num_workers > 0:
        print(f"  DataLoader: {actual_num_workers} workers, pin_memory=True, prefetch_factor=2")
    else:
        print(f"  DataLoader: single-threaded (Windows mode)")
    
    # Calculate training steps
    total_steps = len(train_dataloader) * config.num_epochs
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    
    print(f"\nTraining configuration:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {effective_batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gradient clip: {config.gradient_clip_val}")
    
    # Initial optimizer (adapter only)
    optimizer = build_optimizer()
    
    # Learning rate scheduler (warmup + linear decay)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # ReduceLROnPlateau scheduler for when val loss gets worse
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=0,  # Reduce immediately when val loss increases
    )
    best_val_loss = float('inf')
    last_lr = learning_rate  # Track LR for manual logging
    
    # Mixed precision setup
    use_amp = (config.fp16 or config.bf16) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler() if config.fp16 and torch.cuda.is_available() else None
    
    if config.bf16:
        print(f"  Using bf16 mixed precision (no GradScaler needed)")
    elif config.fp16:
        print(f"  Using fp16 mixed precision with GradScaler")
    
    # Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config.model.load_in_8bit:
        model.to(device)
    
    model.train()
    global_step = 0
    running_loss = 0.0
    last_grad_norm = 0.0
    start_epoch = 0
    
    # Profiling stats
    import time
    profile_stats = {
        "data_time": 0.0,
        "forward_time": 0.0,
        "backward_time": 0.0,
        "optimizer_time": 0.0,
        "step_time": 0.0,
    }
    last_log_time = time.time()
    batch_start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_loss = 0.0
        num_train_batches = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Measure data loading time (time since last loop end)
            current_time = time.time()
            profile_stats["data_time"] += current_time - batch_start_time
            
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            t0 = time.time()
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    outputs = model(
                        lc0_hidden_states=batch["lc0_hidden_states"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        side_to_move=batch["side_to_move"],
                        fen=batch["fen"],
                        engineered_features=batch.get("engineered_features"),
                        perceiver_features=batch.get("perceiver_features"),
                    )
                    loss = outputs.loss / config.gradient_accumulation_steps
                
                profile_stats["forward_time"] += time.time() - t0
                
                # Backward pass
                t0 = time.time()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                
                # Calculate gradient norm
                trainable_params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                if config.gradient_clip_val is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.gradient_clip_val)
                else:
                    # Calculate norm without clipping for logging
                    parameters = [p for p in trainable_params]
                    if len(parameters) == 0:
                        total_norm = torch.tensor(0.0)
                    else:
                        device = parameters[0].grad.device
                        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
                last_grad_norm = total_norm.item()
                
                profile_stats["backward_time"] += time.time() - t0
            else:
                outputs = model(
                    lc0_hidden_states=batch["lc0_hidden_states"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    side_to_move=batch["side_to_move"],
                    fen=batch["fen"],
                    engineered_features=batch.get("engineered_features"),
                    perceiver_features=batch.get("perceiver_features"),
                )
                loss = outputs.loss / config.gradient_accumulation_steps
                
                profile_stats["forward_time"] += time.time() - t0
                
                # Backward pass
                t0 = time.time()
                loss.backward()
                
                # Calculate gradient norm
                trainable_params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                if config.gradient_clip_val is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.gradient_clip_val)
                else:
                    # Calculate norm without clipping for logging
                    parameters = [p for p in trainable_params]
                    if len(parameters) == 0:
                        total_norm = torch.tensor(0.0)
                    else:
                        device = parameters[0].grad.device
                        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(device) for p in parameters]), 2.0)
                last_grad_norm = total_norm.item()
                
                profile_stats["backward_time"] += time.time() - t0
            
            running_loss += loss.item()
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_train_batches += 1
            
            # Gradient accumulation
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # Optimizer step
                t0 = time.time()
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                scheduler.step()
                profile_stats["optimizer_time"] += time.time() - t0
                
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0:
                    avg_loss = running_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    
                    # Calculate timing metrics
                    log_interval_duration = time.time() - last_log_time
                    steps_in_interval = config.logging_steps
                    
                    # Prepare logs
                    logs = {
                        "train/loss": avg_loss,
                        "train/grad_norm": last_grad_norm,
                        "train/learning_rate": lr,
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                        # Performance metrics
                        "perf/data_loading_sec": profile_stats["data_time"] / steps_in_interval,
                        "perf/forward_pass_sec": profile_stats["forward_time"] / steps_in_interval,
                        "perf/backward_pass_sec": profile_stats["backward_time"] / steps_in_interval,
                        "perf/optimizer_sec": profile_stats["optimizer_time"] / steps_in_interval,
                        "perf/step_total_sec": log_interval_duration / steps_in_interval,
                        "perf/samples_per_sec": (steps_in_interval * effective_batch_size) / log_interval_duration,
                    }
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "t/step": f"{logs['perf/step_total_sec']:.2f}s"
                    })
                    
                    # Log to wandb
                    if config.use_wandb:
                        wandb.log(logs, step=global_step)
                    
                    running_loss = 0.0
                    
                    # Reset stats
                    for k in profile_stats:
                        profile_stats[k] = 0.0
                    last_log_time = time.time()
                
                # Save checkpoint (keep only latest to save disk space)
                if global_step % config.save_steps == 0:
                    checkpoint_path = output_path / f"checkpoint-{global_step}"
                    model.save_pretrained(str(checkpoint_path))
                    save_training_state(
                        checkpoint_path, optimizer, scheduler, plateau_scheduler,
                        epoch, global_step, best_val_loss, lora_unfrozen
                    )
                    
                    # Delete previous step checkpoint
                    if not hasattr(train, '_last_step_checkpoint'):
                        train._last_step_checkpoint = None
                    if train._last_step_checkpoint and train._last_step_checkpoint.exists():
                        import shutil
                        shutil.rmtree(train._last_step_checkpoint)
                        print(f"  Deleted old checkpoint: {train._last_step_checkpoint.name}")
                    train._last_step_checkpoint = checkpoint_path
            
            # Reset batch start time for next iteration
            batch_start_time = time.time()
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_train_batches
        print(f"\nEpoch {epoch+1} complete. Train loss: {avg_epoch_loss:.4f}")
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, desc="Validation", leave=False):
                val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        val_outputs = model(
                            lc0_hidden_states=val_batch["lc0_hidden_states"],
                            input_ids=val_batch["input_ids"],
                            attention_mask=val_batch["attention_mask"],
                            labels=val_batch["labels"],
                            side_to_move=val_batch["side_to_move"],
                            fen=val_batch["fen"],
                            engineered_features=val_batch.get("engineered_features"),
                            perceiver_features=val_batch.get("perceiver_features"),
                        )
                else:
                    val_outputs = model(
                        lc0_hidden_states=val_batch["lc0_hidden_states"],
                        input_ids=val_batch["input_ids"],
                        attention_mask=val_batch["attention_mask"],
                        labels=val_batch["labels"],
                        side_to_move=val_batch["side_to_move"],
                        fen=val_batch["fen"],
                        engineered_features=val_batch.get("engineered_features"),
                        perceiver_features=val_batch.get("perceiver_features"),
                    )
                
                val_loss += val_outputs.loss.item()
                num_val_batches += 1
        
        model.train()
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
        
        print(f"  Validation loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        
        # Step plateau scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        plateau_scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  [ReduceLROnPlateau] LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Track best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  New best validation loss!")
        
        # Progressive LoRA: Unfreeze at specified epoch
        if not lora_unfrozen and lora_unfreeze_epoch > 0 and (epoch + 1) >= lora_unfreeze_epoch:
            print(f"\n[Progressive LoRA] Unfreezing LoRA at epoch {epoch + 1}")
            model.unfreeze_lora()
            lora_unfrozen = True
            
            # Rebuild optimizer with LoRA parameters
            optimizer = build_optimizer()
            # Reset scheduler
            remaining_steps = (config.num_epochs - epoch - 1) * len(train_dataloader)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=remaining_steps,
            )
            model.print_trainable_parameters()
            
            if config.use_wandb:
                wandb.log({"progressive_lora/stage": 2, "progressive_lora/unfreeze_epoch": epoch + 1}, step=global_step)
        
        # Log epoch metrics to wandb
        lora_stage = epoch + 1 if progressive_lora_merge else (2 if lora_unfrozen else 1)
        if config.use_wandb:
            wandb.log({
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": avg_val_loss,
                "epoch/val_perplexity": val_perplexity,
                "epoch/epoch": epoch + 1,
                "progressive_lora/lora_active": lora_unfrozen or progressive_lora_merge,
                "progressive_lora/stage": lora_stage,
            }, step=global_step)
        
        # Progressive LoRA Merge
        if progressive_lora_merge and epoch < config.num_epochs - 1:
            print(f"\n[Progressive LoRA] Merging and reinitializing LoRA for stage {epoch + 2}")
            model.merge_and_reinit_lora()
            
            optimizer = build_optimizer()
            remaining_steps = (config.num_epochs - epoch - 1) * len(train_dataloader)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=remaining_steps,
            )
            
            if config.use_wandb:
                wandb.log({"progressive_lora/merged_at_epoch": epoch + 1}, step=global_step)
        
        # Save epoch checkpoint
        epoch_checkpoint_path = output_path / f"epoch-{epoch+1}"
        model.save_pretrained(str(epoch_checkpoint_path))
        save_training_state(
            epoch_checkpoint_path, optimizer, scheduler, plateau_scheduler,
            epoch, global_step, best_val_loss, lora_unfrozen
        )
        print(f"Saved epoch {epoch+1} checkpoint to: {epoch_checkpoint_path}")
        
        # Delete previous epoch checkpoint
        if not hasattr(train, '_last_epoch_checkpoint'):
            train._last_epoch_checkpoint = None
        if train._last_epoch_checkpoint and train._last_epoch_checkpoint.exists():
            import shutil
            shutil.rmtree(train._last_epoch_checkpoint)
            print(f"  Deleted old epoch checkpoint: {train._last_epoch_checkpoint.name}")
        train._last_epoch_checkpoint = epoch_checkpoint_path
        
        # Upload epoch checkpoint to wandb
        if config.use_wandb:
            if not hasattr(train, '_epoch_checkpoints'):
                train._epoch_checkpoints = []
            
            artifact_name = f"epoch-{epoch+1}-checkpoint"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Epoch {epoch+1} checkpoint (loss: {avg_epoch_loss:.4f})",
                metadata={
                    "epoch": epoch + 1,
                    "loss": avg_epoch_loss,
                    "global_step": global_step,
                }
            )
            artifact.add_dir(str(epoch_checkpoint_path))
            wandb.log_artifact(artifact)
            print(f"Uploaded epoch {epoch+1} checkpoint to wandb")
            
            train._epoch_checkpoints.append(artifact_name)
            
            if len(train._epoch_checkpoints) > 5:
                old_artifact_name = train._epoch_checkpoints.pop(0)
                try:
                    api = wandb.Api()
                    old_artifact = api.artifact(f"{config.wandb_project}/{old_artifact_name}:latest")
                    old_artifact.delete()
                    print(f"Deleted old artifact: {old_artifact_name}")
                except Exception as e:
                    print(f"Note: Could not delete old artifact {old_artifact_name}: {e}")
    
    # Save final model
    final_path = output_path / "final"
    model.save_pretrained(str(final_path))
    
    # Finish wandb run
    if config.use_wandb:
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final model saved to: {final_path}")
    print("=" * 60)
    
    return model


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train chess commentary model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        train(config)
    except Exception as e:
        print(f"Error starting training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
