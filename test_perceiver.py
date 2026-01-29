
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from training.perceiver_adapter import PerceiverChessAdapter, extract_perceiver_features
from training.config import ModelConfig, PerceiverConfig

def test_feature_extraction():
    print("\n[Test] Feature Extraction")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # Start pos
    sq, glob = extract_perceiver_features(fen)
    
    print(f"Square Features Shape: {sq.shape}") # Should be (64, 83)
    print(f"Global Features Shape: {glob.shape}") # Should be (4, 16)
    
    assert sq.shape == (64, 83)
    assert glob.shape == (4, 16)
    
    # Check specifics
    # A1 (index 0) is White Rook
    # Type: Rook (index 3 in 0-5) -> 3+1 = 4? 
    # Logic: 0=Empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K.
    # A1 Rook: sq[0, 4] should be 1.0. 
    # Color: White -> sq[0, 7] = 1.0
    
    print(f"A1 Piece One-Hot: {sq[0, 1:7]}")
    # Verify A1 is Rook (index 3 -> 4th item in slice?)
    # Slice 1:7 is indices 1,2,3,4,5,6 (P,N,B,R,Q,K)
    # A1 is Rook (R). R is index 3 in P...K list? 
    # P=0, N=1, B=2, R=3. So index 3 relative to start.
    # index 1+3 = 4. 
    if sq[0, 4] == 1.0:
        print("PASS: A1 is correctly identified as Rook")
    else:
        print("FAIL: A1 is not identified as Rook")
        
    # Check Ray Distances
    # A1 Rook blocked by A2 Pawn (North) and B1 Knight (East).
    # Rays: 75-82.
    # Directions: N, NE, E, SE, S, SW, W, NW
    # A1 (0,0). North is (1,0). A2 is (1,0). Distance should be 0? 
    # "encode how far it can go before reaching a piece".
    # Logic: loop `dist+=1` while not piece.
    # At A1, North is A2. Piece at A2. 
    # While loop: curr=A2. Board has piece. Break. dist=0?
    # Or dist=1?
    # My code: `curr_r += dr`. `dist += 1`. `if piece: break`.
    # A1 -> curr A2. dist=1. Piece exists. Break. Result 1.
    # 1/7.0 = 0.1428
    
    print(f"A1 Rays: {sq[0, 75:]}")
    if sq[0, 75] > 0:
        print(f"PASS: A1 North Ray > 0 ({sq[0, 75]})")
    else:
        print("FAIL: A1 North Ray is 0")
        
    print("Feature Extraction Test Complete")
    return sq, glob

def test_model_forward(sq, glob):
    print("\n[Test] Model Forward Pass")
    config = ModelConfig(mode="perceiver")
    config.perceiver = PerceiverConfig(
        d_model=128, # Smaller for test
        n_layers_encoder=2,
        n_heads=4,
        n_latents=32
    )
    
    model = PerceiverChessAdapter(config=config)
    print(f"Model instantiated. Latents: {model.n_latents}")
    
    # Create batch
    sq_batch = sq.unsqueeze(0).expand(2, -1, -1) # B=2
    glob_batch = glob.unsqueeze(0).expand(2, -1, -1)
    
    side_to_move = torch.tensor([True, False])
    
    perceiver_features = (sq_batch, glob_batch)
    
    output = model(perceiver_features, side_to_move=side_to_move)
    print(f"Output Shape: {output.shape}")
    
    # Expected: (B, K+1, 2048) -> (2, 33, 2048)
    expected_shape = (2, 33, 2048)
    if output.shape == expected_shape:
        print("PASS: Output shape matches expected.")
    else:
        print(f"FAIL: Expected {expected_shape}, got {output.shape}")

if __name__ == "__main__":
    sq, glob = test_feature_extraction()
    test_model_forward(sq, glob)
