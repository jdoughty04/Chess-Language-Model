"""
Tactical detection module for chess positions.

Provides detection of tactical patterns:
- Forks: A piece attacks two or more valuable enemy pieces simultaneously
- Pins: A piece pins an enemy piece to a more valuable piece (usually the king)
- Material changes: Detected changes in material between positions
- Skewers: A sliding piece attacks a valuable piece that must move, exposing a piece behind

Adapted from CARA's game highlight rules for standalone use.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import chess


# Piece values in centipawns
PIECE_VALUES = {
    "q": 900, "Q": 900,
    "r": 500, "R": 500,
    "b": 300, "B": 300,
    "n": 300, "N": 300,
    "p": 100, "P": 100,
    "k": 0, "K": 0  # King has no material value but is important for tactics
}

PIECE_TYPE_VALUES = {
    chess.QUEEN: 900,
    chess.ROOK: 500,
    chess.BISHOP: 300,
    chess.KNIGHT: 300,
    chess.PAWN: 100,
    chess.KING: 0
}


@dataclass
class TacticalFeature:
    """Represents a detected tactical feature."""
    tactic_type: str  # "fork", "pin", "skewer", "material_gain", "material_loss"
    description: str
    attacking_piece: Optional[str] = None  # e.g., "White knight on c7"
    attacking_square: Optional[str] = None
    target_pieces: Optional[List[str]] = None  # e.g., ["Black king on e8", "Black queen on a8"]
    target_squares: Optional[List[str]] = None
    value: Optional[int] = None  # Material value involved (centipawns)
    
    def __str__(self) -> str:
        return f"{self.tactic_type.upper()}: {self.description}"


def get_piece_value(piece: chess.Piece) -> int:
    """Get centipawn value of a piece."""
    return PIECE_TYPE_VALUES.get(piece.piece_type, 0)


def get_piece_name(piece: chess.Piece) -> str:
    """Get human-readable piece name."""
    names = {
        chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"
    }
    color = "White" if piece.color == chess.WHITE else "Black"
    return f"{color} {names.get(piece.piece_type, 'piece')}"


def detect_forks(board: chess.Board) -> List[TacticalFeature]:
    """
    Detect fork opportunities in the current position.
    
    A fork is when a single piece attacks two or more enemy pieces simultaneously,
    where at least one target is undefended or the fork includes the king.
    
    Args:
        board: Current board position
        
    Returns:
        List of TacticalFeature objects describing detected forks
    """
    forks = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        color = piece.color
        opponent_color = not color
        
        # Get all squares this piece attacks
        attacked_squares = board.attacks(square)
        
        # Find enemy pieces on attacked squares
        enemy_pieces_attacked = []
        attacks_king = False
        undefended_valuable = 0
        
        for target_sq in attacked_squares:
            target_piece = board.piece_at(target_sq)
            if target_piece and target_piece.color == opponent_color:
                piece_value = get_piece_value(target_piece)
                is_king = target_piece.piece_type == chess.KING
                is_undefended = not board.is_attacked_by(opponent_color, target_sq)
                
                # Track valuable pieces (>=300 cp: knight, bishop, rook, queen) or king
                if piece_value >= 300 or is_king:
                    enemy_pieces_attacked.append({
                        'square': target_sq,
                        'piece': target_piece,
                        'value': piece_value,
                        'is_king': is_king,
                        'is_undefended': is_undefended
                    })
                    if is_king:
                        attacks_king = True
                    if is_undefended and piece_value >= 300:
                        undefended_valuable += 1
        
        # Fork requires attacking at least 2 valuable enemy pieces
        if len(enemy_pieces_attacked) < 2:
            continue
            
        # Check if the forking piece can be captured by equal or lesser value piece
        # Important: we must check if the capture is LEGAL, not just if the square is attacked.
        # A piece that is pinned cannot legally capture the forking piece.
        attacker_value = get_piece_value(piece)
        can_be_safely_captured = False
        
        # Check each potential attacker to see if it can LEGALLY capture
        for attacker_sq in board.attackers(opponent_color, square):
            attacker = board.piece_at(attacker_sq)
            if attacker and get_piece_value(attacker) <= attacker_value:
                # Verify the capture is actually a legal move (handles pins)
                capture_move = chess.Move(attacker_sq, square)
                if capture_move in board.legal_moves:
                    can_be_safely_captured = True
                    break
        
        if can_be_safely_captured:
            continue
            
        # Valid fork conditions:
        # 1. Fork includes king + at least one valuable piece (fork with check)
        # 2. At least 2 valuable undefended pieces
        is_valid_fork = False
        if attacks_king and len(enemy_pieces_attacked) >= 2:
            is_valid_fork = True
        elif undefended_valuable >= 2:
            is_valid_fork = True
            
        if is_valid_fork:
            piece_name = get_piece_name(piece)
            sq_name = chess.square_name(square)
            target_descs = [
                f"{get_piece_name(t['piece'])} on {chess.square_name(t['square'])}"
                for t in enemy_pieces_attacked
            ]
            target_squares = [chess.square_name(t['square']) for t in enemy_pieces_attacked]
            
            forks.append(TacticalFeature(
                tactic_type="fork",
                description=f"{piece_name} on {sq_name} forks {', '.join(target_descs)}",
                attacking_piece=piece_name,
                attacking_square=sq_name,
                target_pieces=target_descs,
                target_squares=target_squares,
                value=sum(t['value'] for t in enemy_pieces_attacked if t['is_undefended'])
            ))
    
    return forks


def detect_pins(board: chess.Board) -> List[TacticalFeature]:
    """
    Detect pins in the current position.
    
    A pin is when a sliding piece (bishop, rook, queen) attacks an enemy piece
    that cannot move because doing so would expose a more valuable piece behind it.
    
    Refined logic:
    - Reports absolute pins (to King)
    - Reports relative pins ONLY if capturing the screened piece is a positive trade 
      (value(screened) > value(attacker)) or if the screened piece is undefended.
    
    Args:
        board: Current board position
        
    Returns:
        List of TacticalFeature objects describing detected pins
    """
    pins = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        # Only sliding pieces can create pins
        if piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            continue
            
        color = piece.color
        opponent_color = not color
        attacker_value = get_piece_value(piece)
        
        # Check each direction for potential pins
        directions = []
        if piece.piece_type in [chess.ROOK, chess.QUEEN]:
            directions.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])  # Ranks and files
        if piece.piece_type in [chess.BISHOP, chess.QUEEN]:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])  # Diagonals
            
        for df, dr in directions:
            pinned_piece = None
            pinned_square = None
            
            # Walk along the ray
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            for dist in range(1, 8):
                new_file = file + df * dist
                new_rank = rank + dr * dist
                
                if not (0 <= new_file <= 7 and 0 <= new_rank <= 7):
                    break
                    
                target_sq = chess.square(new_file, new_rank)
                target_piece = board.piece_at(target_sq)
                
                if target_piece is None:
                    continue
                    
                if target_piece.color == color:
                    # Own piece blocks the ray
                    break
                    
                # Enemy piece found
                if pinned_piece is None:
                    # First enemy piece - potential pinned piece
                    pinned_piece = target_piece
                    pinned_square = target_sq
                else:
                    # Second enemy piece - the piece behind (target of pin)
                    behind_value = get_piece_value(target_piece)
                    is_king_behind = target_piece.piece_type == chess.KING
                    
                    # Logic Refinement:
                    # A pin is "meaningful" if:
                    # 1. We are pinning against the King (Absolute Pin)
                    # 2. We are pinning against a piece we can profitably take
                    #    - Either it's undefended
                    #    - Or Value(Behind) > Value(Attacker) (Positive Trade)
                    
                    is_meaningful = False
                    
                    if is_king_behind:
                        is_meaningful = True
                    else:
                        is_undefended = not board.is_attacked_by(opponent_color, target_sq)
                        if is_undefended:
                             # If it's undefended, we usually want it unless it's just a pawn and we are a queen?
                             # Let's say any undefended piece > pawn is interesting, or valid trade.
                             # If I am a Queen pinning a Pawn to an undefended Knight(300)... actually 
                             # strict definition: can we take it? Yes.
                             is_meaningful = True
                        elif behind_value > attacker_value:
                             # Positive trade: e.g. Bishop(300) pins Knight to Rook(500)
                             is_meaningful = True
                    
                    if is_meaningful:
                        piece_name = get_piece_name(piece)
                        sq_name = chess.square_name(square)
                        pinned_name = get_piece_name(pinned_piece)
                        pinned_sq_name = chess.square_name(pinned_square)
                        behind_name = get_piece_name(target_piece)
                        behind_sq_name = chess.square_name(target_sq)
                        
                        pins.append(TacticalFeature(
                            tactic_type="pin",
                            description=f"{piece_name} on {sq_name} pins {pinned_name} on {pinned_sq_name} to {behind_name} on {behind_sq_name}",
                            attacking_piece=piece_name,
                            attacking_square=sq_name,
                            target_pieces=[f"{pinned_name} on {pinned_sq_name}"],
                            target_squares=[pinned_sq_name, behind_sq_name],
                            value=get_piece_value(pinned_piece) # Value of the pinned piece is typically the "cost" to move it? Or just relevance.
                            ))
                    break  # Stop after finding second piece
    
    return pins


def detect_discovered_attacks(board: chess.Board) -> List[TacticalFeature]:
    """
    Detect discovered attack opportunities.
    
    A discovered attack occurs when moving a piece unblocks an attack 
    from a sliding piece behind it (bishop, rook, queen).
    
    Args:
        board: Current board position
        
    Returns:
        List of TacticalFeature objects
    """
    discovered_attacks = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        # Only sliding pieces can create discovered attacks (as the "battery" rear piece)
        if piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            continue
            
        color = piece.color
        opponent_color = not color
        
        # Check rays
        directions = []
        if piece.piece_type in [chess.ROOK, chess.QUEEN]:
            directions.extend([(0, 1), (0, -1), (1, 0), (-1, 0)]) 
        if piece.piece_type in [chess.BISHOP, chess.QUEEN]:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            
        for df, dr in directions:
            blocker_piece = None
            blocker_square = None
            
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            for dist in range(1, 8):
                new_file = file + df * dist
                new_rank = rank + dr * dist
                
                if not (0 <= new_file <= 7 and 0 <= new_rank <= 7):
                    break
                    
                target_sq = chess.square(new_file, new_rank)
                target_piece = board.piece_at(target_sq)
                
                if target_piece is None:
                    continue
                
                if blocker_piece is None:
                    # Found the first piece. 
                    if target_piece.color == color:
                        # It is OUR piece. It effectively blocks our slider.
                        # This piece could move to create a discovered attack.
                        blocker_piece = target_piece
                        blocker_square = target_sq
                    else:
                        # Enemy piece blocks immediately - no discovered attack possible through here
                        break
                else:
                    # We already have a blocker (our piece).
                    # Now we found a second piece.
                    if target_piece.color == opponent_color:
                        # It is an ENEMY piece.
                        # So: [MySlider] ... [MyBlocker] ... [EnemyTarget]
                        # If [MyBlocker] moves, [MySlider] attacks [EnemyTarget].
                        
                        target_value = get_piece_value(target_piece)
                        is_valuable = target_value > 100 or target_piece.piece_type == chess.KING
                        
                        # We report this if the target is valuable enough to care about
                        if is_valuable:
                            slider_name = get_piece_name(piece)
                            slider_sq_name = chess.square_name(square)
                            blocker_name = get_piece_name(blocker_piece)
                            blocker_sq_name = chess.square_name(blocker_square)
                            target_name = get_piece_name(target_piece)
                            target_sq_name = chess.square_name(target_sq)
                            
                            discovered_attacks.append(TacticalFeature(
                                tactic_type="discovered_attack",
                                description=f"Moving {blocker_name} on {blocker_sq_name} discovers attack by {slider_name} on {target_name} ({target_sq_name})",
                                attacking_piece=slider_name, # The piece doing the discovered attack
                                attacking_square=slider_sq_name,
                                target_pieces=[f"{target_name} on {target_sq_name}"],
                                target_squares=[target_sq_name],
                                value=target_value
                            ))
                    break # Stop after finding the target (or second friendly piece)

    return discovered_attacks


def detect_material_change(board_before: chess.Board, board_after: chess.Board) -> List[TacticalFeature]:
    """
    Detect material changes between two positions.
    
    Args:
        board_before: Board position before the move
        board_after: Board position after the move
        
    Returns:
        List of TacticalFeature objects describing material changes
    """
    changes = []
    
    def count_material(board: chess.Board, color: chess.Color) -> int:
        total = 0
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            count = len(board.pieces(piece_type, color))
            total += count * PIECE_TYPE_VALUES[piece_type]
        return total
    
    white_before = count_material(board_before, chess.WHITE)
    black_before = count_material(board_before, chess.BLACK)
    white_after = count_material(board_after, chess.WHITE)
    black_after = count_material(board_after, chess.BLACK)
    
    white_change = white_after - white_before
    black_change = black_after - black_before
    
    # Detect significant material changes (>=100 cp, i.e., at least a pawn)
    if white_change < -100:
        changes.append(TacticalFeature(
            tactic_type="material_loss",
            description=f"White lost {abs(white_change)} centipawns of material",
            value=abs(white_change)
        ))
    elif white_change > 100:
        # This shouldn't normally happen unless it's a promotion
        if white_change >= 800:  # Likely promotion
            changes.append(TacticalFeature(
                tactic_type="promotion",
                description=f"White gained material (likely promotion): +{white_change} cp",
                value=white_change
            ))
    
    if black_change < -100:
        changes.append(TacticalFeature(
            tactic_type="material_loss",
            description=f"Black lost {abs(black_change)} centipawns of material",
            value=abs(black_change)
        ))
    elif black_change > 100:
        if black_change >= 800:
            changes.append(TacticalFeature(
                tactic_type="promotion",
                description=f"Black gained material (likely promotion): +{black_change} cp",
                value=black_change
            ))
    
    return changes


def detect_skewers(board: chess.Board) -> List[TacticalFeature]:
    """
    Detect meaningful skewers in the current position.
    
    A skewer is when a sliding piece attacks a valuable piece that must move,
    exposing a less valuable piece behind it. Only reports skewers where
    the behind piece is actually vulnerable (undefended or insufficiently defended).
    
    Args:
        board: Current board position
        
    Returns:
        List of TacticalFeature objects describing detected skewers
    """
    skewers = []
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
            
        # Only sliding pieces can create skewers
        if piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            continue
            
        color = piece.color
        opponent_color = not color
        attacker_value = get_piece_value(piece)
        
        # Check each direction
        directions = []
        if piece.piece_type in [chess.ROOK, chess.QUEEN]:
            directions.extend([(0, 1), (0, -1), (1, 0), (-1, 0)])
        if piece.piece_type in [chess.BISHOP, chess.QUEEN]:
            directions.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            
        for df, dr in directions:
            front_piece = None
            front_square = None
            
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            for dist in range(1, 8):
                new_file = file + df * dist
                new_rank = rank + dr * dist
                
                if not (0 <= new_file <= 7 and 0 <= new_rank <= 7):
                    break
                    
                target_sq = chess.square(new_file, new_rank)
                target_piece = board.piece_at(target_sq)
                
                if target_piece is None:
                    continue
                    
                if target_piece.color == color:
                    break
                    
                # Enemy piece found
                if front_piece is None:
                    # First enemy piece - the more valuable piece in front
                    front_piece = target_piece
                    front_square = target_sq
                else:
                    # Second enemy piece - less valuable piece behind
                    front_value = get_piece_value(front_piece)
                    behind_value = get_piece_value(target_piece)
                    is_king_front = front_piece.piece_type == chess.KING
                    
                    # Skewer: front piece is more valuable than behind (or is king)
                    # The front piece must move, exposing the behind piece
                    if is_king_front or front_value > behind_value + 100:
                        # Only meaningful skewers: behind piece must be capturable profitably
                        # Check 1: Is the behind piece undefended?
                        behind_is_defended = board.is_attacked_by(opponent_color, target_sq)
                        
                        # Check 2: Calculate attack/defense balance for the behind piece
                        # Count attackers (including the skewering piece which would attack after front moves)
                        # For simplicity, we check if capturing the behind piece would be profitable
                        
                        if not behind_is_defended:
                            # Undefended piece - clear skewer
                            is_meaningful = behind_value >= 100  # At least a pawn
                        else:
                            # Defended piece - check if we can win it profitably
                            # The skewer is meaningful if:
                            # 1. Behind piece value > attacker value (we can trade up)
                            # 2. Or it's defended only by the front piece (which must move)
                            
                            # Count defenders
                            defenders = list(board.attackers(opponent_color, target_sq))
                            num_defenders = len(defenders)
                            
                            # If there's only one defender and it's the front piece, 
                            # the behind piece becomes undefended after front moves
                            front_is_only_defender = (num_defenders == 1 and front_square in defenders)
                            
                            # Calculate if capture is profitable
                            # After front piece moves, we can capture behind piece
                            # The opponent can recapture with remaining defenders
                            if front_is_only_defender:
                                is_meaningful = behind_value >= 100
                            elif behind_value > attacker_value:
                                # We can trade up - meaningful even if defended
                                is_meaningful = True
                            else:
                                # Defended by multiple pieces and we can't trade up - not meaningful
                                is_meaningful = False
                        
                        if is_meaningful and behind_value >= 100:
                            piece_name = get_piece_name(piece)
                            sq_name = chess.square_name(square)
                            front_name = get_piece_name(front_piece)
                            front_sq_name = chess.square_name(front_square)
                            behind_name = get_piece_name(target_piece)
                            behind_sq_name = chess.square_name(target_sq)
                            
                            defense_note = " (undefended)" if not behind_is_defended else ""
                            
                            skewers.append(TacticalFeature(
                                tactic_type="skewer",
                                description=f"{piece_name} on {sq_name} skewers {front_name} on {front_sq_name}, exposing {behind_name} on {behind_sq_name}{defense_note}",
                                attacking_piece=piece_name,
                                attacking_square=sq_name,
                                target_pieces=[f"{front_name} on {front_sq_name}", f"{behind_name} on {behind_sq_name}"],
                                target_squares=[front_sq_name, behind_sq_name],
                                value=behind_value
                            ))
                    break
    
    return skewers




def detect_executed_discovered_attacks(board: chess.Board, board_before: chess.Board) -> List[TacticalFeature]:
    """
    Detect discovered attacks that were just executed by the last move.
    
    Compares attacks of static sliders before and after the move.
    If a slider (that didn't move) gains an attack on a valuable piece,
    it counts as a discovered attack.
    
    Args:
        board: Board after the move
        board_before: Board before the move
        
    Returns:
        List of TacticalFeature
    """
    discovered = []
    
    if not board.move_stack:
        return []
        
    last_move = board.peek()
    moved_piece_sq = last_move.to_square
    moved_piece = board.piece_at(moved_piece_sq)
    
    if not moved_piece:
        return []
        
    color = moved_piece.color
    opponent_color = not color
    
    # Iterate over all friendly pieces to find sliders
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color:
            continue
            
        # Skip the piece that just moved (that would be a direct attack, not discovered)
        if sq == moved_piece_sq:
            continue
            
        # Only sliders can discover attacks
        if piece.piece_type not in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
            continue
            
        # Check attacks before and after
        # Note: We need to use the board_before state for the slider's previous attacks
        # But the slider is at 'sq' in board (and should be at 'sq' in board_before if it didn't move)
        # Verify piece didn't change (e.g. capture/promotion weirdness, though we filtered moved_piece)
        piece_before = board_before.piece_at(sq)
        if piece_before != piece:
            continue
            
        attacks_after = board.attacks(sq)
        attacks_before = board_before.attacks(sq)
        
        # Find new attacks
        new_attacks = attacks_after - attacks_before
        
        for target_sq in new_attacks:
            target_piece = board.piece_at(target_sq)
            if target_piece and target_piece.color == opponent_color:
                 # Valuable target?
                 target_val = get_piece_value(target_piece)
                 if target_val >= 100: # Pawn or better
                     slider_name = get_piece_name(piece)
                     slider_sq_name = chess.square_name(sq)
                     target_name = get_piece_name(target_piece)
                     target_sq_name = chess.square_name(target_sq)
                     
                     discovered.append(TacticalFeature(
                        tactic_type="discovered_attack",
                        description=f"{slider_name} on {slider_sq_name} attacks {target_name} on {target_sq_name} (revealed)",
                        attacking_piece=slider_name,
                        attacking_square=slider_sq_name,
                        target_pieces=[f"{target_name} on {target_sq_name}"],
                        target_squares=[target_sq_name],
                        value=target_val
                     ))
                     
    return discovered

def detect_all_tactics(board: chess.Board, board_before: Optional[chess.Board] = None) -> Dict[str, List[TacticalFeature]]:
    """
    Detect all tactical features in a position.
    
    Args:
        board: Current board position
        board_before: Optional previous position for material change detection
        
    Returns:
        Dictionary with tactic types as keys and lists of TacticalFeature as values
    """
    first_tactics = {
        "forks": detect_forks(board),
        "pins": detect_pins(board),
        "skewers": detect_skewers(board),
        "battery_attacks": detect_discovered_attacks(board), # Renamed from "discovered_attacks" to separate potential ones
        "discovered_attacks": [], # Will hold ONLY executed ones
        "material_changes": []
    }
    
    if board_before is not None:
        first_tactics["material_changes"] = detect_material_change(board_before, board)
        
        # Add executed discovered attacks
        executed = detect_executed_discovered_attacks(board, board_before)
        if executed:
            first_tactics["discovered_attacks"].extend(executed)
    
    return first_tactics


def format_tactics_summary(tactics: Dict[str, List[TacticalFeature]]) -> str:
    """
    Format tactical features into a readable summary.
    
    Args:
        tactics: Dictionary from detect_all_tactics
        
    Returns:
        Formatted string summary
    """
    lines = []
    
    total_count = sum(len(v) for v in tactics.values())
    if total_count == 0:
        return "No significant tactics detected."
    
    for tactic_type, features in tactics.items():
        if features:
            lines.append(f"\n{tactic_type.upper().replace('_', ' ')}:")
            for feature in features:
                lines.append(f"  • {feature.description}")
    
    return "\n".join(lines)


# Convenience function for quick tactical check
def has_tactics(board: chess.Board) -> bool:
    """Quick check if any tactics exist in position."""
    tactics = detect_all_tactics(board)
    return any(len(v) > 0 for v in tactics.values())


if __name__ == "__main__":
    # Quick test
    test_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"  # Scholar's mate setup
    board = chess.Board(test_fen)
    print(f"Testing position: {test_fen}")
    print(f"Board:\n{board}\n")
    
    tactics = detect_all_tactics(board)
    print(format_tactics_summary(tactics))
