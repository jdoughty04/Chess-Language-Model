"""
Standalone wrapper for CARA rule engine to analyze lines.
"""

import sys
import os
import chess
from typing import List, Dict, Any, Optional

# Ensure project root is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# CRITICAL: Mock PyQt6 for headless execution
# CARA models import Qt classes but we don't need UI logic
try:
    import PyQt6
    from PyQt6.QtCore import QAbstractTableModel, Qt
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["PyQt6"] = MagicMock()
    sys.modules["PyQt6.QtCore"] = MagicMock()
    sys.modules["PyQt6.QtGui"] = MagicMock()

    # Mock specific Qt classes used in models
    class MockQAbstractTableModel: pass
    sys.modules["PyQt6.QtCore"].QAbstractTableModel = MockQAbstractTableModel
    sys.modules["PyQt6.QtCore"].Qt.ItemDataRole.DisplayRole = 0

# Add CARA directory to path
cara_root = os.path.join(project_root, "CARA")
if cara_root not in sys.path:
    sys.path.append(cara_root)
    
from app.services.game_highlights.rule_registry import RuleRegistry
from app.services.game_highlights.base_rule import RuleContext, GameHighlight
from app.models.moveslist_model import MoveData

class CARAAnalyzer:
    """
    Bridge between raw chess data and CARA's rule engine.
    Synthesizes required context (MoveData, RuleContext) to run rules on lines.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {'rules': {}} 
        self.registry = RuleRegistry(config)
        self.rules = self.registry.get_all_rules()
        
    def analyze_line(self, start_fen: str, moves_san: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a sequence of moves from a starting position.
        
        Args:
            start_fen: FEN of the position before the first move in moves_san
            moves_san: List of SAN moves to play and analyze
            
        Returns:
            List of detected highlights (dicts)
        """
        board = chess.Board(start_fen)
        game_moves = [] # List of MoveData
        
        start_move_num = board.fullmove_number
        
        # Determine the "previous" index relative to the first move we will make
        # If we are at Move 3 (White), board.fullmove_number is 3. 
        # We need game_moves[1] (Move 2) to hold the start_fen as "fen_black" (after black moved)
        # If we are at Move 3 (Black), board.fullmove_number is 3.
        # We need game_moves[2] (Move 3) to hold the start_fen as "fen_white" (after white moved)
        
        # Pre-fill potentially large list if starting late in game
        # Just ensure we have capacity up to start_move_num
        for i in range(start_move_num + 5): # buffer
            game_moves.append(MoveData(i + 1))
            
        # Initialize the "state before the line starts"
        prev_idx = start_move_num - 2  # default for White to move (e.g. 3 -> idx 1)
        if board.turn == chess.BLACK:
            prev_idx = start_move_num - 1 # (e.g. 3 -> idx 2)
            
        if prev_idx >= 0:
            md = game_moves[prev_idx]
            if board.turn == chess.WHITE:
                md.fen_black = start_fen
            else:
                md.fen_white = start_fen
            self._set_material_counts(md, board)

        all_highlights = []
        
        for i, move_san in enumerate(moves_san):
            try:
                move = board.parse_san(move_san)
                
                # Determine move number and color
                move_num = board.fullmove_number
                color_moved = board.turn
                
                # Create or update MoveData
                # MoveData structure in CARA stores White and Black moves in the same row
                # Row index (0-based) corresponds to (move_number - 1)
                
                row_idx = move_num - 1
                
                # Ensure game_moves list is large enough
                while len(game_moves) <= row_idx:
                    game_moves.append(MoveData(len(game_moves) + 1))
                    
                current_move_data = game_moves[row_idx]
                
                if color_moved == chess.WHITE:
                    current_move_data.white_move = move_san
                    # Capture check (simplified)
                    if board.is_capture(move):
                        current_move_data.white_capture = "x" 
                    
                else:
                    current_move_data.black_move = move_san
                    if board.is_capture(move):
                        current_move_data.black_capture = "x"
                
                # Make move
                board.push(move)
                
                # Update FEN and Material AFTER move
                if color_moved == chess.WHITE:
                    current_move_data.fen_white = board.fen()
                else:
                    current_move_data.fen_black = board.fen()
                    
                self._set_material_counts(current_move_data, board, color_moved)

                # 2. Run Rules for this move
                # We need to construct context looking back at previous row
                prev_move_data = game_moves[row_idx-1] if row_idx > 0 else None
                
                context = self._build_context(
                    row_idx, 
                    game_moves, 
                    prev_move_data,
                    current_move_data
                )
                
                # Run applicable rules
                for rule in self.rules:
                    if not rule.is_enabled():
                        continue
                        
                    # Filter rules? For now run all.
                    # Note: Many rules rely on CPL/Eval which we likely don't have yet 
                    # unless provided. For now we assume no engine data.
                    # Rules like WeakSquare, Fork, Pin should work with just board/moves.
                    
                    try:
                        rule_highlights = rule.evaluate(current_move_data, context)
                        for h in rule_highlights:
                            # Filter out duplicates or irrelevant ones?
                            # Add some metadata about which move in the line caused it
                            highlight_dict = {
                                "rule": h.rule_type,
                                "description": h.description,
                                "move_san": move_san,
                                "move_ply": i + 1, # relative to start of line
                                "fen": board.fen()
                            }
                            all_highlights.append(highlight_dict)
                    except Exception:
                        pass # Rule failed (likely missing data), skip
                        
            except ValueError:
                print(f"Invalid move {move_san} for FEN {board.fen()}")
                break
                
        return all_highlights

    def _set_material_counts(self, move_data: MoveData, board: chess.Board, side_just_moved: chess.Color = None):
        """Populate material counts in MoveData based on board state."""
        # This is a bit tricky because MoveData expects counts *after* the move
        # but stored in the corresponding fields
        
        # Helper to count
        def count(pt, color): return len(board.pieces(pt, color))
        
        # We update both checks regardless? 
        # CARA MoveData has separate fields for material counts implicitly? 
        # No, checking MoveData init: white_queens, black_queens etc. 
        # It seems these are 'current' counts associated with that row.
        
        move_data.white_queens = count(chess.QUEEN, chess.WHITE)
        move_data.white_rooks = count(chess.ROOK, chess.WHITE)
        move_data.white_bishops = count(chess.BISHOP, chess.WHITE)
        move_data.white_knights = count(chess.KNIGHT, chess.WHITE)
        move_data.white_pawns = count(chess.PAWN, chess.WHITE)
        
        move_data.black_queens = count(chess.QUEEN, chess.BLACK)
        move_data.black_rooks = count(chess.ROOK, chess.BLACK)
        move_data.black_bishops = count(chess.BISHOP, chess.BLACK)
        move_data.black_knights = count(chess.KNIGHT, chess.BLACK)
        move_data.black_pawns = count(chess.PAWN, chess.BLACK)
        
        # Material values
        piece_vals = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300, chess.ROOK: 500, chess.QUEEN: 900}
        
        w_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_vals.items())
        b_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_vals.items())
        
        move_data.white_material = w_mat
        move_data.black_material = b_mat

    def _build_context(self, index: int, moves: List[MoveData], prev: MoveData, curr: MoveData) -> RuleContext:
        """Construct RuleContext for rule evaluation."""
        
        # Use previous move data for 'prev_' fields if available
        p_w_b = prev.white_bishops if prev else 0
        p_b_b = prev.black_bishops if prev else 0
        p_w_n = prev.white_knights if prev else 0
        p_b_n = prev.black_knights if prev else 0
        p_w_q = prev.white_queens if prev else 0
        p_b_q = prev.black_queens if prev else 0
        p_w_r = prev.white_rooks if prev else 0
        p_b_r = prev.black_rooks if prev else 0
        p_w_p = prev.white_pawns if prev else 0
        p_b_p = prev.black_pawns if prev else 0
        p_w_mat = prev.white_material if prev else 0
        p_b_mat = prev.black_material if prev else 0
        
        return RuleContext(
            move_index=index,
            total_moves=len(moves),
            opening_end=10, # default
            middlegame_end=30, # default
            prev_move=prev,
            next_move=None, # Future lookahead not handled here
            prev_white_bishops=p_w_b,
            prev_black_bishops=p_b_b,
            prev_white_knights=p_w_n,
            prev_black_knights=p_b_n,
            prev_white_queens=p_w_q,
            prev_black_queens=p_b_q,
            prev_white_rooks=p_w_r,
            prev_black_rooks=p_b_r,
            prev_white_pawns=p_w_p,
            prev_black_pawns=p_b_p,
            prev_white_material=p_w_mat,
            prev_black_material=p_b_mat,
            last_book_move_number=0,
            theory_departed=False,
            good_move_max_cpl=50,
            inaccuracy_max_cpl=100,
            mistake_max_cpl=300,
            shared_state={},
            moves=moves
        )
