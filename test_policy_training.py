
import sys
import torch
import torch.nn as nn
from pathlib import Path
import os
import shutil

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from training.config import ModelConfig, PerceiverConfig
from training.policy_model import PerceiverPolicyModel
from data_collection.policy_dataset import ChessPolicyDataset

def create_dummy_pgn():
    content = """[Event "Test Game"]
[Site "Prediction Test"]
[Date "2024.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7 1-0
"""
    with open("test_policy.pgn", "w") as f:
        f.write(content)
    return "test_policy.pgn"

def test_pipeline():
    print("--- Test Policy Pipeline ---")
    pgn_file = create_dummy_pgn()
    
    # 1. Test Dataset
    print("\n[1] Testing Dataset...")
    dataset = ChessPolicyDataset(pgn_file, start_index=0, end_index=1, infinite=False)
    
    item = None
    count = 0
    for features, label in dataset:
        item = (features, label)
        count += 1
        if count >= 3: break
        
    print(f"Extracted {count} samples from dummy game.")
    
    if count == 0:
        print("FAIL: No samples extracted.")
        return
    
    sq, glob = item[0]
    print(f"Features: Sq={sq.shape}, Glob={glob.shape}, Label={item[1]}")
    
    # 2. Test Model
    print("\n[2] Testing Model...")
    config = ModelConfig(mode="perceiver")
    config.perceiver = PerceiverConfig(
        d_model=64, # Small for test
        n_layers_encoder=2,
        n_heads=4,
        n_latents=16
    )
    
    model = PerceiverPolicyModel(config)
    
    # Forward
    logits = model((sq.unsqueeze(0), glob.unsqueeze(0))) # Batch=1
    print(f"Logits shape: {logits.shape}")
    
    assert logits.shape == (1, 4096)
    
    # 3. Test Backward
    print("\n[3] Testing Backward...")
    criterion = nn.CrossEntropyLoss()
    target = torch.tensor([item[1]], dtype=torch.long)
    
    loss = criterion(logits, target)
    print(f"Loss: {loss.item()}")
    
    loss.backward()
    print("Backward pass successful.")
    
    # Cleanup
    if os.path.exists(pgn_file):
        os.remove(pgn_file)
        
    print("\nPASS: All checks passed.")

if __name__ == "__main__":
    test_pipeline()
