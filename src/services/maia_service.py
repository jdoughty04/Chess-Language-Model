"""Slim Maia service for humanlike move generation.

This is a minimal replacement for CARA's MaiaService. The heavy lifting
is done directly via maia2 imports in generator.py - this module provides
a clean interface and the convert_uci_pv_to_san helper function.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import chess


@dataclass
class HumanlikeMove:
    """Represents a humanlike move with metadata."""
    elo: int
    move_san: str
    move_uci: str
    probability: float = 0.0
    continuation_san: List[str] = field(default_factory=list)
    continuation_uci: List[str] = field(default_factory=list)
    formatted_line: str = ""


class MaiaService:
    """Minimal Maia service wrapper around maia2.
    
    Provides humanlike move generation at various Elo levels.
    Falls back gracefully if maia2 is not available.
    """
    
    ELO_LEVELS = [800, 1200, 1800]
    
    def __init__(self, engine_path: Optional[str] = None):
        self.engine_path = engine_path
        self._model = None
        self._prepared = None
        self._available = False
    
    def initialize(self) -> bool:
        """Initialize maia2 model. Returns True if successful."""
        try:
            from maia2 import model, inference
            self._model = model.from_pretrained(type="rapid", device="cpu")
            self._prepared = inference.prepare()
            self._available = True
        except ImportError:
            self._available = False
        except Exception as e:
            print(f"Warning: Maia initialization failed: {e}")
            self._available = False
        return self._available
    
    def is_available(self) -> bool:
        """Check if Maia is available."""
        return self._available
    
    def get_moves(self, fen: str, elo: int = 800, k: int = 5) -> List[Tuple[str, float]]:
        """Get top k Maia moves at specified Elo.
        
        Args:
            fen: FEN position string
            elo: Maia Elo level (800, 1200, 1800, etc.)
            k: Number of top moves to return
            
        Returns:
            List of (move_uci, probability) tuples, sorted by probability
        """
        if not self._available:
            return []
        try:
            from maia2 import inference
            move_probs, _ = inference.inference_each(
                self._model, self._prepared, fen, elo, elo
            )
            if move_probs:
                return sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:k]
        except Exception as e:
            print(f"Maia move generation failed: {e}")
        return []
    
    def cleanup(self):
        """Cleanup resources."""
        self._model = None
        self._prepared = None
        self._available = False


def convert_uci_pv_to_san(fen: str, uci_pv: str, start_move: int) -> str:
    """Convert UCI PV string to formatted SAN with move numbers.
    
    Args:
        fen: Starting FEN position
        uci_pv: Space-separated UCI moves (e.g., "g4f6 f8f6 g6f6")
        start_move: Starting move number
        
    Returns:
        Formatted SAN string (e.g., "22. Nf6+ Rxf6 23. Qxf6")
    """
    if not uci_pv:
        return ""
    
    board = chess.Board(fen)
    white_to_move = board.turn == chess.WHITE
    moves = uci_pv.strip().split()
    result = []
    current_move = start_move
    is_white = white_to_move
    
    for i, uci in enumerate(moves):
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            san = board.san(move)
            
            if is_white:
                result.append(f"{current_move}. {san}")
            else:
                if i == 0:
                    result.append(f"{current_move}... {san}")
                else:
                    result.append(san)
                current_move += 1
            
            board.push(move)
            is_white = not is_white
        except (ValueError, chess.InvalidMoveError):
            break
    
    return " ".join(result)
