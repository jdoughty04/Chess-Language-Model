"""
Data Preprocessing Pipeline for Chess Commentary Training

This module converts raw JSONL training data into preprocessed format
with cached LC0 hidden states.

Data flow:
1. Load dataset (contains game_id, ply, fen, commentary)
2. Look up original PGN from local PGN directory
3. Replay game to target ply, reconstructing board history
4. Extract LC0 hidden states with full 8-position history
5. Save preprocessed samples as .pt files
"""

import json
from pathlib import Path
from typing import Optional
import torch
import chess
import chess.pgn
from io import StringIO
from tqdm import tqdm

# Import LC0 extractor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.lc0_extractor import LC0HiddenStateExtractor


# Default location for local PGN files
DEFAULT_PGN_DIR = Path(__file__).parent.parent.parent / "data" / "games"


def extract_game_id(lichess_url: str) -> str:
    """Extract game ID from Lichess URL."""
    # URL format: https://lichess.org/XXXXXXXX
    parts = lichess_url.rstrip('/').split('/')
    return parts[-1][:8]  # Game IDs are 8 characters


class LocalPGNIndex:
    """
    Index for looking up games from local PGN files.
    
    Loads all PGN files from a directory and builds an in-memory index
    mapping game IDs to their PGN text for fast lookup.
    """
    
    def __init__(self, pgn_dir: Path = None):
        """
        Initialize the PGN index.
        
        Args:
            pgn_dir: Directory containing .pgn files
        """
        self.pgn_dir = pgn_dir or DEFAULT_PGN_DIR
        self.games: dict[str, str] = {}  # game_id -> pgn_text
        self._load_all_pgns()
    
    def _load_all_pgns(self):
        """Load all PGN files and index games by ID."""
        if not self.pgn_dir.exists():
            print(f"Warning: PGN directory not found: {self.pgn_dir}")
            return
        
        pgn_files = list(self.pgn_dir.glob("*.pgn"))
        if not pgn_files:
            print(f"Warning: No PGN files found in {self.pgn_dir}")
            return
        
        print(f"Loading PGN files from {self.pgn_dir}...")
        
        for pgn_file in pgn_files:
            self._index_pgn_file(pgn_file)
        
        print(f"Indexed {len(self.games)} games from {len(pgn_files)} PGN file(s)")
    
    def _index_pgn_file(self, pgn_file: Path):
        """Parse a PGN file and add all games to the index."""
        try:
            with open(pgn_file, 'r', encoding='utf-8') as f:
                pgn_text = f.read()
            
            # Split into individual games
            # Each game starts with [Event
            current_game_lines = []
            current_game_id = None
            
            for line in pgn_text.split('\n'):
                if line.startswith('[Event ') and current_game_lines:
                    # Save previous game
                    if current_game_id:
                        self.games[current_game_id] = '\n'.join(current_game_lines)
                    current_game_lines = []
                    current_game_id = None
                
                current_game_lines.append(line)
                
                # Extract GameId from header
                if line.startswith('[GameId "'):
                    current_game_id = line.split('"')[1]
                elif line.startswith('[Site "https://lichess.org/'):
                    # Fallback: extract from Site URL if GameId not present
                    if current_game_id is None:
                        url = line.split('"')[1]
                        current_game_id = extract_game_id(url)
            
            # Don't forget the last game
            if current_game_lines and current_game_id:
                self.games[current_game_id] = '\n'.join(current_game_lines)
                
        except Exception as e:
            print(f"Error loading {pgn_file}: {e}")
    
    def get_pgn(self, game_id: str) -> Optional[str]:
        """
        Look up PGN text for a game ID.
        
        Args:
            game_id: 8-character Lichess game ID
            
        Returns:
            PGN string or None if not found
        """
        return self.games.get(game_id)


def replay_to_ply(pgn_text: str, target_ply: int) -> Optional[chess.Board]:
    """
    Replay a PGN game to a specific ply (half-move).
    
    Args:
        pgn_text: PGN string
        target_ply: Target ply number (0-indexed from start)
        
    Returns:
        chess.Board at target position with full move history
    """
    try:
        game = chess.pgn.read_game(StringIO(pgn_text))
        if game is None:
            return None
        
        board = game.board()
        moves = list(game.mainline_moves())
        
        # Check we have enough moves
        if target_ply > len(moves):
            print(f"Warning: target_ply {target_ply} > total moves {len(moves)}")
            target_ply = len(moves)
        
        # Replay to target ply
        for i, move in enumerate(moves[:target_ply]):
            board.push(move)
        
        return board
        
    except Exception as e:
        print(f"Error replaying PGN: {e}")
        return None




def prepare_sample(
    sample: dict,
    pgn_index: LocalPGNIndex
) -> Optional[dict]:
    """
    Prepare a sample for LC0 extraction (look up PGN, replay game).
    
    This does all the work EXCEPT LC0 extraction, which will be batched.
    
    Args:
        sample: Raw sample from output.jsonl
        pgn_index: Local PGN index for game lookup
        
    Returns:
        Dict with preparation data, or None if failed
    """
    game_id = extract_game_id(sample["game_id"])
    ply = sample["ply"]
    
    # Look up PGN from local index
    pgn_text = pgn_index.get_pgn(game_id)
    if pgn_text is None:
        print(f"Warning: Game {game_id} not found in local PGN index")
        return None
    
    # Replay to target ply
    board = replay_to_ply(pgn_text, ply)
    if board is None:
        return None
    
    # Extract commentary from the sample
    try:
        commentary = sample["generated_commentary"]["samples"][0]["commentary"]
    except (KeyError, TypeError, IndexError):
        # Sample might be malformed or missing commentary
        return None
    
    # Get last 8 UCI moves for reference
    move_history = []
    for move in board.move_stack[-8:]:
        move_history.append(move.uci())
    
    return {
        "board": board,
        "fen": board.fen(),
        "move_history": move_history,
        "commentary": commentary,
        "game_id": sample["game_id"],
        "ply": ply,
        "sample_id": f"{game_id}_{ply}"
    }


def preprocess_dataset(
    input_jsonl: str,
    output_dir: str,
    network_path: str = None,
    device: str = "cpu",  # Use CPU for preprocessing to avoid VRAM contention
    batch_size: int = 32,  # Batch size for LC0 inference
    pgn_dir: str = None,   # Directory containing local PGN files
    skip_lc0: bool = False # Skip LC0 extraction
):
    """
    Preprocess entire dataset with batched LC0 extraction.
    
    Args:
        input_jsonl: Path to input JSONL file
        output_dir: Directory to save preprocessed .pt files
        network_path: Path to LC0 network (BT3/BT4)
        device: Device for LC0 inference
        batch_size: Number of samples to batch for LC0 inference
        pgn_dir: Directory containing local PGN files (defaults to data/games)
        skip_lc0: If True, skip LC0 extraction and save samples without hidden states
    """
    input_path = Path(input_jsonl)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    samples_dir = output_path / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Load local PGN index
    pgn_path = Path(pgn_dir) if pgn_dir else None
    pgn_index = LocalPGNIndex(pgn_path)
    
    if not pgn_index.games:
        print("Error: No games found in PGN index. Cannot proceed.")
        return
    
    # Initialize LC0 extractor (unless skipped)
    extractor = None
    if not skip_lc0:
        print("Initializing LC0 extractor...")
        extractor = LC0HiddenStateExtractor(
            network_path=network_path,
            device=device
        )
    else:
        print("Skipping LC0 extraction (engineered features only mode)")
    
    # Load raw samples
    print(f"Loading samples from {input_jsonl}...")
    with open(input_path, 'r') as f:
        raw_samples = [json.loads(line) for line in f]
    
    print(f"Found {len(raw_samples)} samples")
    print(f"Using batch size: {batch_size}")
    
    # Filter out already-processed samples
    samples_to_process = []
    skipped_count = 0
    
    for sample in raw_samples:
        sample_id = f"{extract_game_id(sample['game_id'])}_{sample['ply']}"
        output_file = samples_dir / f"{sample_id}.pt"
        if output_file.exists():
            skipped_count += 1
        else:
            samples_to_process.append(sample)
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} already-processed samples")
    
    if not samples_to_process:
        print("All samples already processed!")
        return
    
    # Process in batches with parallel preparation
    processed_count = 0
    failed_count = 0
    
    # Use ThreadPoolExecutor for parallel sample preparation
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    # Number of preparation workers
    num_workers = 8
    
    print(f"Using {num_workers} workers for parallel sample preparation")
    
    # Create progress bar for remaining samples
    pbar = tqdm(total=len(samples_to_process), desc="Preprocessing")
    
    # Process samples in chunks for better batching
    chunk_size = batch_size * 2  # Prepare 2x batch_size at a time for pipelining
    
    for chunk_start in range(0, len(samples_to_process), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(samples_to_process))
        chunk_samples = samples_to_process[chunk_start:chunk_end]
        
        # Parallel preparation of this chunk
        batch_prepared = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_sample = {
                executor.submit(prepare_sample, sample, pgn_index): sample 
                for sample in chunk_samples
            }
            
            for future in as_completed(future_to_sample):
                result = future.result()
                if result is not None:
                    batch_prepared.append(result)
                else:
                    failed_count += 1
                    pbar.update(1)
        
        # Now process prepared samples in GPU batches
        # If skipping LC0, just save directly
        if skip_lc0:
            for idx, prepared_sample in enumerate(batch_prepared):
                result = {
                    "fen": prepared_sample["fen"],
                    "move_history": prepared_sample["move_history"],
                    "lc0_hidden_states": {},  # Empty dict indicates no LC0 data
                    "commentary": prepared_sample["commentary"],
                    "game_id": prepared_sample["game_id"],
                    "ply": prepared_sample["ply"]
                }
                
                output_file = samples_dir / f"{prepared_sample['sample_id']}.pt"
                torch.save(result, output_file)
                processed_count += 1
                pbar.update(1)
            continue
            
        # Otherwise run LC0 inference
        for i in range(0, len(batch_prepared), batch_size):
            batch = batch_prepared[i:i + batch_size]
            boards = [p["board"] for p in batch]
            
            try:
                batch_hidden_states = extractor.extract_batch(boards)
                
                # Save each sample
                for idx, prepared_sample in enumerate(batch):
                    # Extract this sample's hidden states from batch
                    lc0_states = {}
                    for layer_name, batch_states in batch_hidden_states.items():
                        lc0_states[layer_name] = torch.from_numpy(
                            batch_states[idx]
                        ).half()  # (64, 768) float16
                    
                    result = {
                        "fen": prepared_sample["fen"],
                        "move_history": prepared_sample["move_history"],
                        "lc0_hidden_states": lc0_states,
                        "commentary": prepared_sample["commentary"],
                        "game_id": prepared_sample["game_id"],
                        "ply": prepared_sample["ply"]
                    }
                    
                    output_file = samples_dir / f"{prepared_sample['sample_id']}.pt"
                    torch.save(result, output_file)
                    processed_count += 1
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\nLC0 batch extraction failed: {e}")
                failed_count += len(batch)
                pbar.update(len(batch))
    
    pbar.close()
    
    print(f"\nPreprocessing complete!")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output dir: {output_path}")


def main():
    """Main entry point for data preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess chess commentary data")
    parser.add_argument(
        "--input", "-i",
        default="output.jsonl",
        help="Input JSONL file with raw samples"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/preprocessed",
        help="Output directory for preprocessed data"
    )
    parser.add_argument(
        "--network", "-n",
        default=None,
        help="Path to LC0 network file (defaults to BT3 if available)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda",
        help="Device for LC0 inference (cpu or cuda)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=512,
        help="Batch size for LC0 inference (default: 32)"
    )
    parser.add_argument(
        "--pgn-dir",
        default=None,
        help="Directory containing local PGN files (defaults to data/games)"
    )
    parser.add_argument(
        "--skip-lc0",
        action="store_true",
        help="Skip LC0 extraction (much faster, for engineered features only)"
    )
    
    args = parser.parse_args()
    
    # Try to find network if not specified (only if we need it)
    if args.network is None and not args.skip_lc0:
        # Look for common locations
        # Look for common locations
        candidates = [
            "BT3-768x15x24h-swa-2790000.pb.gz",
            "src/training/lc0_cache/BT3-768x15x24h-swa-2790000.pb.gz",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                args.network = candidate
                break
    
    if args.network is None and not args.skip_lc0:
        print("Error: No LC0 network found. Please specify with --network")
        return 1
    
    preprocess_dataset(
        input_jsonl=args.input,
        output_dir=args.output,
        network_path=args.network,
        device=args.device,
        batch_size=args.batch_size,
        pgn_dir=args.pgn_dir,
        skip_lc0=args.skip_lc0
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
