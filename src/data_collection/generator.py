"""
Standalone generator adapted from CARA TrainingDataController.
Removes PyQt dependencies and allows headless operation.
"""

import json
from typing import Dict, Any, Optional, List, Tuple
import chess
import chess.pgn
import io
import sys
import os

# Setup path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")

if src_dir not in sys.path:
    sys.path.append(src_dir)

from services.ai_service import AIService
from services.maia_service import MaiaService, convert_uci_pv_to_san

# Import tactical detector for humanlike line analysis
try:
    from tactical_detector import detect_all_tactics, TacticalFeature
    TACTICAL_DETECTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: tactical_detector not available: {e}")
    TACTICAL_DETECTOR_AVAILABLE = False



class CommentaryGenerator:
    """Controller for generating training data samples from chess positions.
    Adapted for standalone usage.
    """
    
    def __init__(self, config: Dict[str, Any], app_controller, quiet: bool = False) -> None:
        self.config = config
        self.app_controller = app_controller
        self.ai_service = AIService(config)
        self.quiet = quiet
        # accept maia_queues from config or separate arg?
        # Generator is init in gatherer.py via worker_init. 
        # worker_init calls CommentaryGenerator(settings, app_controller)
        # We should put queues in settings['maia_queues']?
        # Yes, that's the easiest injection point.
        self.maia_queues = config.get('maia_queues') # (input_queue, result_queue)
        
        # Load engine overrides from settings if present
        overrides = config.get('engine_overrides', {})
        self.engine_depth = overrides.get('depth', 15)
        self.engine_multipv = overrides.get('multipv', 3)
        if not self.quiet and (self.engine_depth != 15 or self.engine_multipv != 3):
            print(f"Using Engine Overrides: depth={self.engine_depth}, multipv={self.engine_multipv}")

        # self.maia_service logic removed in favor of remote service
        self.maia_service = None # or used for fallback? No, fully decoupled.
        self._maia_enabled = True 
        self._engine = None
        # self._maia_model = None
        # self._maia_prepared = None

    def _get_engine(self):
        """Get or initialize the chess engine."""
        if self._engine is not None:
            try:
                # Test if engine is still alive
                self._engine.ping()
                return self._engine
            except Exception:
                self._engine = None
        
        engines = self.config.get('engines', [])
        engine_path = engines[0].get('path', '') if engines else ''
        
        if engine_path and os.path.exists(engine_path):
            import chess.engine
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                return self._engine
            except Exception as e:
                if not self.quiet: print(f"Error starting engine: {e}")
        return None

    def close(self):
        """Close resources."""
        if hasattr(self, '_engine') and self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
        
        if hasattr(self, 'maia_service') and self.maia_service:
            try:
                self.maia_service.cleanup()
            except Exception:
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

    # _get_maia_model REMOVED

    def get_context_for_current_position(self) -> Dict[str, Any]:
        """Gather all relevant context for the current position."""
        game_controller = self.app_controller.game_controller
        game_model = game_controller.get_game_model()
        active_game = game_model.active_game
        ply_index = game_model.get_active_move_ply()
        
        if not active_game:
            return {}

        move_assessment = "Unknown"
        engine_eval = "Unknown"
        best_move = "Unknown"
        played_move = ""
        current_pv = ""
        best_move_2 = ""
        best_move_3 = ""
        cpl_2 = ""
        cpl_3 = ""

        # 1. Basic Position Info
        if hasattr(active_game, 'game_object') and active_game.game_object:
            chess_game = active_game.game_object
        else:
            pgn_io = io.StringIO(active_game.pgn)
            chess_game = chess.pgn.read_game(pgn_io)
        board = chess_game.board() 
        
        # Navigate to current ply
        node = chess_game
        move_history = []
        for i in range(ply_index):
            if node.variations:
                next_node = node.variation(0)
                move_history.append(board.san(next_node.move))
                board.push(next_node.move)
                node = next_node
            else:
                break
                
        fen = self.app_controller.get_current_fen()
        # Fix move number calculation: ply // 2 + 1 corresponds to standard move numbering
        # ply 0 (start) -> 1. ply 1 (after 1.e4) -> 1 (Black to move). ply 2 -> 2.
        move_number = ply_index // 2 + 1
        side_to_move = "White" if " w " in fen else "Black"
        
        # Determine played move from history if available
        if not played_move and move_history:
            last_san = move_history[-1]
            if side_to_move == "White":
                 # Current side is White, meaning Black just moved
                 played_move = f"{move_number-1}... {last_san}" if move_number > 1 else f"... {last_san}" # Edge case if ply=1
            else:
                 # Current side is Black, meaning White just moved
                 played_move = f"{move_number}. {last_san}"
        
        if not played_move and move_history:
            last_san = move_history[-1]
            if ply_index % 2 == 1:
                played_move = f"{move_number}. {last_san}"
            else:
                played_move = f"{move_number-1}... {last_san}"
        
        # Fallback for assessment using NAGs from the node (if PGN has them)
        if move_assessment == "Unknown" and node:
            nags = node.nags
            if nags:
                first_nag = list(nags)[0]
                if 1 in nags: move_assessment = "Good move (!)"
                elif 2 in nags: move_assessment = "Mistake (?)"
                elif 3 in nags: move_assessment = "Brilliant move (!!)"
                elif 4 in nags: move_assessment = "Blunder (??)"
                elif 5 in nags: move_assessment = "Speculative move (!?)"
                elif 6 in nags: move_assessment = "Dubious move (?!)"
                elif 9 in nags: move_assessment = "Forced move"
                elif 10 in nags: move_assessment = "Drawish"
                elif 14 in nags: move_assessment = "White has a slight advantage"
                elif 15 in nags: move_assessment = "Black has a slight advantage"
                elif 16 in nags: move_assessment = "White has a moderate advantage"
                elif 17 in nags: move_assessment = "Black has a moderate advantage"
                elif 18 in nags: move_assessment = "White has a decisive advantage"
                elif 19 in nags: move_assessment = "Black has a decisive advantage"
                else: move_assessment = f"Annotated (NAG {first_nag})"

        positional_features = self._get_positional_features(board)
        
        # 2b. Attacks/Defenses
        attacks_defenses = self._get_attacks_and_defenses(board)
        
        # 3. Game Highlights & Analysis
        # Note: Pre-analyzed game data from CARA is not supported in standalone mode.
        # Chess_transformer generates its own context via wrappers.py and the positional analyzer.

        maia_context = ""
        # Maia disabled by default for now to reduce complexity

        # Convert engine PV
        formatted_pv = ""
        if current_pv:
            formatted_pv = convert_uci_pv_to_san(fen, current_pv, move_number)
        
        # Calculate full multi-PV lines
        # In this standalone version, we might want to optionally skip this if no engine path
        multipv_lines, analysis_cache = self._calculate_multipv_lines(fen, move_number, num_lines=self.engine_multipv, depth=self.engine_depth)

        # Fallback: derive assessment from engine eval if still Unknown
        if move_assessment == "Unknown" and multipv_lines:
            eval_str = multipv_lines[0].get('eval', '')
            if eval_str.startswith('M'):
                # Mate score
                mate_in = int(eval_str[1:]) if eval_str[1:].lstrip('-').isdigit() else 0
                if mate_in > 0:
                    move_assessment = f"White has mate in {mate_in}"
                elif mate_in < 0:
                    move_assessment = f"Black has mate in {abs(mate_in)}"
            elif eval_str:
                try:
                    eval_cp = float(eval_str) * 100  # Convert to centipawns
                    if eval_cp > 500:
                        move_assessment = "White has a decisive advantage"
                    elif eval_cp > 150:
                        move_assessment = "White has a moderate advantage"
                    elif eval_cp > 50:
                        move_assessment = "White has a slight advantage"
                    elif eval_cp >= -50:
                        move_assessment = "Position is roughly equal"
                    elif eval_cp >= -150:
                        move_assessment = "Black has a slight advantage"
                    elif eval_cp >= -500:
                        move_assessment = "Black has a moderate advantage"
                    else:
                        move_assessment = "Black has a decisive advantage"
                except ValueError:
                    pass

        # Also update engine_eval from multipv lines if still Unknown
        if engine_eval == "Unknown" and multipv_lines:
            engine_eval = multipv_lines[0].get('eval', 'Unknown')

        # 5. Humanlike Tactical Lines (Maia moves from multiple ELO levels)
        # Get top 3 lines from 3 different ELO levels for diverse human perspective
        humanlike_tactical_lines = self._get_batch_humanlike_lines(
            fen, move_number, elo_levels=[1100, 1500, 1900], k=3, min_probability=0.05,
            analysis_cache=analysis_cache
        )

        # 4. Enhanced Semantic Features
        semantic_features = self._get_enhanced_semantic_features(board)
        
        # 6. Detect tactics in current position
        current_tactics = []
        if TACTICAL_DETECTOR_AVAILABLE:
            try:
                tactics = detect_all_tactics(board)
                for fork in tactics.get("forks", []):
                    current_tactics.append({"type": "fork", "description": fork.description})
                for pin in tactics.get("pins", []):
                    current_tactics.append({"type": "pin", "description": pin.description})
                for skewer in tactics.get("skewers", []):
                    current_tactics.append({"type": "skewer", "description": skewer.description})
            except Exception as e:
                if not self.quiet: print(f"Error detecting current position tactics: {e}")
        
        return {
            "fen": fen,
            "pgn_snippet": "\n".join(move_history[-10:]),
            "side_to_move": side_to_move,
            "move_number": move_number,
            "positional_features": positional_features,
            "semantic_features": semantic_features,
            "played_move": played_move,
            "move_assessment": move_assessment,
            "engine_eval": engine_eval,
            "best_move": best_move,
            "best_move_2": best_move_2,
            "best_move_3": best_move_3,
            "cpl_2": cpl_2,
            "cpl_3": cpl_3,
            "current_pv": formatted_pv,
            "multipv_lines": multipv_lines,
            "attacks_defenses": attacks_defenses,
            "maia_context": maia_context,
            "humanlike_tactical_lines": humanlike_tactical_lines,
            "current_tactics": current_tactics
        }
    
    def _get_positional_features(self, board: chess.Board) -> Dict[str, List[str]]:
        white_features = []
        black_features = []
        try:
            heatmap_controller = self.app_controller.get_positional_heatmap_controller()
            if heatmap_controller and heatmap_controller.analyzer:
                for perspective in [chess.WHITE, chess.BLACK]:
                    details = heatmap_controller.analyzer.get_detailed_evaluation(board, perspective)
                    
                    if details and 'pieces' in details:
                        seen_features = set()
                        for square, info in details['pieces'].items():
                            piece_color = info['color']
                            square_name = info['square']
                            for rule in info.get('rules', []):
                                if abs(rule['score']) > 0.01:
                                    if rule['name'] == "Piece Activity":
                                        continue
                                        
                                    feature = f"{piece_color.title()} {rule['name']} on {square_name}"
                                    if feature not in seen_features:
                                        if piece_color.lower() == 'white':
                                            if feature not in white_features:
                                                white_features.append(feature)
                                        else:
                                            if feature not in black_features:
                                                black_features.append(feature)
                                        seen_features.add(feature)
        except Exception as e:
            print(f"Error extracting positional features: {e}")
            
        return {"white": white_features, "black": black_features}
    
    def _get_attacks_and_defenses(self, board: chess.Board) -> Dict[str, Dict[str, List[str]]]:
        PIECE_NAMES = {
            chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
            chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"
        }
        white_attacks = []
        black_attacks = []
        white_defends = []
        black_defends = []
        
        try:
            # First, identify which pieces are actually under attack (by the opponent)
            # A piece is under attack if board.attackers(opponent_color, square) is not empty
            pieces_under_attack = set()
            for sq in chess.SQUARES:
                p = board.piece_at(sq)
                if p:
                    opponent_color = not p.color
                    if board.attackers(opponent_color, sq):
                        pieces_under_attack.add(sq)

            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is None:
                    continue
                    
                piece_color = "White" if piece.color == chess.WHITE else "Black"
                piece_name = PIECE_NAMES.get(piece.piece_type, "piece")
                square_name = chess.square_name(square)
                
                attacks = board.attacks(square)
                for target_sq in attacks:
                    target_piece = board.piece_at(target_sq)
                    if target_piece:
                        target_color = "White" if target_piece.color == chess.WHITE else "Black"
                        target_name = PIECE_NAMES.get(target_piece.piece_type, "piece")
                        target_sq_name = chess.square_name(target_sq)
                        
                        if piece.color != target_piece.color:
                            attack_str = f"{piece_color} {piece_name} on {square_name} attacks {target_color}'s {target_name} on {target_sq_name}"
                            if piece.color == chess.WHITE:
                                white_attacks.append(attack_str)
                            else:
                                black_attacks.append(attack_str)
                        else:
                            # Only list defense if the piece is actually under threat
                            if target_sq in pieces_under_attack:
                                defend_str = f"{piece_color} {piece_name} on {square_name} defends {target_name} on {target_sq_name}"
                                if piece.color == chess.WHITE:
                                    white_defends.append(defend_str)
                                else:
                                    black_defends.append(defend_str)
        except Exception as e:
            print(f"Error analyzing attacks/defenses: {e}")
        
        return {
            "white_attacks": white_attacks,
            "black_attacks": black_attacks,
            "white_defends": white_defends,
            "black_defends": black_defends
        }
    
    def _calculate_multipv_lines(self, fen: str, move_number: int, num_lines: int = 3, depth: int = 15) -> Tuple[List[Dict[str, Any]], Dict[str, Dict]]:
        lines = []
        cache = {}
        try:
            import chess.engine
            
            board = chess.Board(fen)
            engine = self._get_engine()
            if not engine:
                return lines, cache
            
            try:
                result = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=num_lines
                )
                
                for i, info in enumerate(result):
                    pv = info.get("pv", [])
                    score = info.get("score")
                    
                    if pv and score:
                        first_move_san = board.san(pv[0])
                        pov = score.white()
                        if pov.is_mate():
                            eval_str = f"M{pov.mate()}"
                        else:
                            eval_str = f"{pov.score()/100:+.2f}"
                        
                        pv_uci = " ".join(m.uci() for m in pv[:8])
                        pv_san = convert_uci_pv_to_san(fen, pv_uci, move_number)
                        
                        # Detect tactics in this line
                        line_tactics = []
                        if TACTICAL_DETECTOR_AVAILABLE:
                            try:
                                check_board = board.copy()
                                prev_check = board.copy()
                                
                                # Initial tactics (to dedup against)
                                root_tactics = detect_all_tactics(check_board)
                                seen_descriptions = set()
                                for t_list in root_tactics.values():
                                    for t in t_list:
                                        seen_descriptions.add(t.description)

                                for move in pv[:6]:
                                    if move in check_board.legal_moves:
                                        move_san = check_board.san(move)
                                        prev_check = check_board.copy()
                                        check_board.push(move)
                                        
                                        tactics = detect_all_tactics(check_board, prev_check)
                                        
                                        # Separate static (persistent) and dynamic (transient/executed) tactics
                                        # Static: forks, pins, skewers, battery_attacks
                                        # Dynamic: discovered_attacks (executed), material_changes
                                        
                                        current_static_descriptions = set()
                                        
                                        # Process Static Tactics
                                        for key in ["forks", "pins", "skewers", "battery_attacks"]:
                                            for feat in tactics.get(key, []):
                                                current_static_descriptions.add(feat.description)
                                                if feat.description not in seen_descriptions:
                                                    line_tactics.append({
                                                        "type": key[:-1] if key.endswith('s') else key, # singularize roughly
                                                        "after_move": move_san,
                                                        "description": feat.description
                                                    })
                                        
                                        # Process Dynamic Tactics (Always show if detected, as they are events)
                                        # Executed Discovered Attacks
                                        for da in tactics.get("discovered_attacks", []):
                                            line_tactics.append({
                                                "type": "discovered_attack",
                                                "after_move": move_san,
                                                "description": da.description
                                            })
                                            
                                        # Update seen set for next ply
                                        # We replace seen with current (if it disappeared, it can reappear as "new")
                                        # Or do we want strict "once shown, never again"?
                                        # User said: "only report tactics when a move is made that spawns their detection."
                                        # If it disappears and comes back, it is respawned. So `seen = current`.
                                        seen_descriptions = current_static_descriptions
                                        
                                    else:
                                        break
                            except Exception:
                                pass
                        
                        line_data = {
                            "move": first_move_san,
                            "eval": eval_str,
                            "pv_line": pv_san,
                            "tactics": line_tactics
                        }
                        lines.append(line_data)
                        
                        # Add to cache (key by UCI)
                        if pv:
                            move_uci = pv[0].uci()
                            cache[move_uci] = line_data.copy()
            except Exception:
                pass
                
        except Exception as e:
            if not self.quiet: print(f"Error calculating multi-PV lines: {e}")
        
        return lines, cache

    def _get_batch_humanlike_lines(self, fen: str, move_number: int, 
                                   elo_levels: List[int], k: int = 3, min_probability: float = 0.05,
                                   analysis_cache: Dict[str, Dict] = None) -> Dict[int, List[Dict]]:
        """
        Get humanlike lines for multiple ELOs, deduplicating the expensive analysis.
        """
        results = {}
        
        # 1. Collect raw moves for all ELOs
        all_candidates = {} # elo -> list of (uci, prob)
        unique_moves = set()
        
        for elo in elo_levels:
            raw_moves = self._get_raw_maia_moves(fen, elo, k, min_probability)
            all_candidates[elo] = raw_moves
            for uci, _ in raw_moves:
                unique_moves.add(uci)
                
        if not unique_moves:
            return {}
            
        # 2. Analyze unique moves
        analyzed_moves = {} # uci -> analysis_dict
        
        # Initialize engine once
        import chess.engine
        engine = self._get_engine()
        if not engine:
            return {}
            
        board = chess.Board(fen)
        white_to_move = board.turn == chess.WHITE
        
        for uci in unique_moves:
            # Check cache first
            if analysis_cache and uci in analysis_cache:
                analyzed_moves[uci] = analysis_cache[uci]
                continue
                
            analysis = self._analyze_single_move(engine, board, uci, move_number, white_to_move)
            if analysis:
                analyzed_moves[uci] = analysis
                
        # 3. Distribute results
        for elo, candidates in all_candidates.items():
            elo_lines = []
            for uci, prob in candidates:
                if uci in analyzed_moves:
                    # Create a copy and add probability
                    line = analyzed_moves[uci].copy()
                    line["probability"] = prob
                    elo_lines.append(line)
            if elo_lines:
                results[elo] = elo_lines
                
        return results

    def _get_raw_maia_moves(self, fen: str, elo: int, k: int, min_probability: float) -> List[Tuple[str, float]]:
        if not self.maia_queues:
            return []
            
        input_q, result_dict = self.maia_queues
        import uuid
        import time
        
        req_id = str(uuid.uuid4())
        # Request: (req_id, fen, elo, k, min_probability)
        try:
            input_q.put((req_id, fen, elo, k, min_probability))
            
            # Wait for result with simple polling
            # Using shared Dict from Manager
            # Model loading can take 10-20s on first run, so allow a generous timeout
            timeout = 30.0 
            start = time.time()
            
            while time.time() - start < timeout:
                if req_id in result_dict:
                    result = result_dict.pop(req_id)
                    return result
                time.sleep(0.01) # fast poll
            
            if not self.quiet: print(f"Batch inference timed out for {req_id}")
            
        except Exception as e:
            if not self.quiet: print(f"Error in batch inference request: {e}")
            
        return []

    def _analyze_single_move(self, engine, board, move_uci, move_number, white_to_move) -> Optional[Dict]:
        try:
            import chess.engine
            first_move = chess.Move.from_uci(move_uci)
            if first_move not in board.legal_moves:
                return None
            
            first_move_san = board.san(first_move)
            test_board = board.copy()
            test_board.push(first_move)
            
            # Get Stockfish's PV
            info = engine.analyse(test_board, chess.engine.Limit(depth=12))
            pv = info.get("pv", [])
            
            # Build full line
            full_pv_san = [first_move_san]
            analysis_board = board.copy()
            analysis_board.push(first_move)
            
            for move in pv[:6]:
                if move in analysis_board.legal_moves:
                    full_pv_san.append(analysis_board.san(move))
                    analysis_board.push(move)
                else:
                    break
            
            # Tactics
            tactics_found = []
            if TACTICAL_DETECTOR_AVAILABLE:
                check_board = board.copy()
                prev_check = board.copy()
                
                # Initial tactics
                root_tactics = detect_all_tactics(check_board)
                seen_descriptions = set()
                for t_list in root_tactics.values():
                    for t in t_list:
                        seen_descriptions.add(t.description)
                
                for move_san in full_pv_san:
                    try:
                        move = check_board.parse_san(move_san)
                        prev_check = check_board.copy()
                        check_board.push(move)
                        
                        tactics = detect_all_tactics(check_board, prev_check)
                        # Get eval at this position if tactics found
                        position_eval = None
                        
                        current_static_descriptions = set()
                        
                        # Process Static
                        for key in ["forks", "pins", "skewers", "battery_attacks"]:
                            for feat in tactics.get(key, []):
                                current_static_descriptions.add(feat.description)
                                if feat.description not in seen_descriptions:
                                    tactics_found.append({
                                        "type": key[:-1] if key.endswith('s') else key,
                                        "after_move": move_san,
                                        "description": feat.description,
                                        "eval": position_eval
                                    })
                        
                        # Process Executed
                        for da in tactics.get("discovered_attacks", []):
                            tactics_found.append({
                                "type": "discovered_attack", "after_move": move_san,
                                "description": da.description, "eval": position_eval
                            })
                            
                        seen_descriptions = current_static_descriptions
                            
                    except ValueError:
                        break
            
            pv_formatted = self._format_pv_with_numbers(full_pv_san, move_number, white_to_move)
            
            return {
                "move": first_move_san,
                "pv_line": pv_formatted,
                "tactics": tactics_found
            }
        except Exception as e:
            if not self.quiet: print(f"Error analyzing move {move_uci}: {e}")
            return None
    
    def _format_pv_with_numbers(self, moves: List[str], start_move: int, white_starts: bool) -> str:
        """Format PV moves with proper move numbers."""
        if not moves:
            return ""
        
        result = []
        current_move = start_move
        is_white = white_starts
        
        for i, m in enumerate(moves):
            if is_white:
                result.append(f"{current_move}. {m}")
            else:
                if i == 0:
                    result.append(f"{current_move}... {m}")
                else:
                    result.append(m)
                current_move += 1
            is_white = not is_white
        
        return " ".join(result)


    def _get_enhanced_semantic_features(self, board: chess.Board) -> Dict[str, Any]:
        """Extract deeper semantic features for ground-truth context."""
        features = {}
        
        # 1. Material Imbalance
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
        white_mat = sum(len(board.pieces(pt, chess.WHITE)) * val for pt, val in piece_values.items())
        black_mat = sum(len(board.pieces(pt, chess.BLACK)) * val for pt, val in piece_values.items())
        features['material_balance'] = white_mat - black_mat
        features['material_count'] = {
            'white': {pt: len(board.pieces(pt, chess.WHITE)) for pt in piece_values},
            'black': {pt: len(board.pieces(pt, chess.BLACK)) for pt in piece_values}
        }
        
        # 2. Pawn Structure
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        features['pawn_structure'] = {
            'white': self._analyze_pawns(white_pawns, chess.WHITE),
            'black': self._analyze_pawns(black_pawns, chess.BLACK)
        }
        
        # 3. King Safety
        features['king_safety'] = {
            'white': self._analyze_king_safety(board, chess.WHITE),
            'black': self._analyze_king_safety(board, chess.BLACK)
        }
        
        # 4. Mobility / Space (Simplified)
        features['mobility'] = {
            'white': board.legal_moves.count() if board.turn == chess.WHITE else 0, # Approximation if not turn
            'black': board.legal_moves.count() if board.turn == chess.BLACK else 0
        }
        
        # Center Control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        white_center = sum(1 for sq in center_squares if board.is_attacked_by(chess.WHITE, sq))
        black_center = sum(1 for sq in center_squares if board.is_attacked_by(chess.BLACK, sq))
        features['center_control'] = {'white': white_center, 'black': black_center}
        
        return features

    def _analyze_pawns(self, pawns, color):
        files = [chess.square_file(sq) for sq in pawns]
        file_counts = {f: files.count(f) for f in range(8)}
        doubled = sum(1 for c in file_counts.values() if c > 1)
        isolated = 0
        for sq in pawns:
            f = chess.square_file(sq)
            if file_counts.get(f-1, 0) == 0 and file_counts.get(f+1, 0) == 0:
                isolated += 1
        return {'count': len(pawns), 'doubled': doubled, 'isolated': isolated}

    def _analyze_king_safety(self, board, color):
        king_sq = board.king(color)
        if king_sq is None: return "King missing"
        
        # Pawn shield
        rank = chess.square_rank(king_sq)
        file = chess.square_file(king_sq)
        shield_pawns = 0
        # Check pawns in front of king (direction depends on color)
        front_rank = rank + 1 if color == chess.WHITE else rank - 1
        if 0 <= front_rank <= 7:
            for f_offset in [-1, 0, 1]:
                check_file = file + f_offset
                if 0 <= check_file <= 7:
                    if board.piece_at(chess.square(check_file, front_rank)) == chess.Piece(chess.PAWN, color):
                        shield_pawns += 1
                        
        # Open lines
        is_castled = (file > 4 or file < 3) # Roughly
        safety_score = shield_pawns
        return f"Shield: {shield_pawns} pawns, Castled: {is_castled}"

    def construct_prompt(self, context: Dict[str, Any]) -> str:
        """Public alias for _construct_prompt"""
        return self._construct_prompt(context)

    def _construct_prompt(self, context: Dict[str, Any]) -> str:
        # Copied logic from TrainingDataController
        played_move = context.get('played_move', '')
        best_move = context.get('best_move', '')
        
        extra = context.get('semantic_features', {})
        mat = extra.get('material_balance', 0)
        pawn_structure = extra.get('pawn_structure', {})
        king_safety = extra.get('king_safety', {})
        center = extra.get('center_control', {})
        
        semantic_section = f"""
DETAILED SEMANTIC DATA:
[Material Balance]: {mat:+d} (Positive = White advantage)
[Center Control]: White controls {center.get('white', '?')} squares, Black controls {center.get('black', '?')} squares.
[Pawn Structure]:
  - White: {pawn_structure.get('white', {})}
  - Black: {pawn_structure.get('black', {})}
[King Safety]:
  - White: {king_safety.get('white', '?')}
  - Black: {king_safety.get('black', '?')}
"""
        
        "" # highlights removed
        
        maia_context = context.get('maia_context', '')
        maia_section = ""
        if maia_context and maia_context not in ["", "(Maia analysis not available)", "(Maia analysis failed)"]:
            maia_section = f"""
HUMANLIKE MOVE ANALYSIS (how players of different levels would approach this):
{maia_context}
"""
        
        multipv_lines = context.get('multipv_lines', [])
        engine_eval = context.get('engine_eval', 'Unknown')
        
        # If engine_eval is Unknown but we have multipv_lines, use the first line's eval
        if engine_eval == 'Unknown' and multipv_lines:
            engine_eval = multipv_lines[0].get('eval', 'Unknown')
        
        engine_section = ""
        if multipv_lines:
            lines_str = ""
            for i, line in enumerate(multipv_lines, 1):
                lines_str += f"  {i}. {line['move']} [Eval: {line['eval']}]\n"
                lines_str += f"     Line: {line['pv_line']}\n"
                # Show tactics if any were found
                line_tactics = line.get('tactics', [])
                if line_tactics:
                    for tactic in line_tactics:
                        lines_str += f"     → {tactic['type'].upper()} after {tactic['after_move']}: {tactic['description']}\n"
            
            engine_section = f"""
ENGINE ANALYSIS (top 3 continuations from this position):
{lines_str}
"""
        elif context.get('current_pv') or context.get('best_move'):
            current_pv = context.get('current_pv', '')
            best_move = context.get('best_move', '')
            engine_section = f"""
ENGINE ANALYSIS (best continuation from this position):
[Evaluation]: {engine_eval}
[Best Move]: {best_move}
[Best Line (PV)]: {current_pv if current_pv else 'N/A'}
"""
        
        
        # Format humanlike lines section (grouped by ELO level)
        humanlike_tactical = context.get('humanlike_tactical_lines', {})
        humanlike_section = ""
        if humanlike_tactical:
            lines_str = ""
            for elo_level in sorted(humanlike_tactical.keys()):
                lines_str += f"  [{elo_level} ELO player moves]:\n"
                for line in humanlike_tactical[elo_level]:
                    prob_pct = line['probability'] * 100
                    lines_str += f"    {line['move']} ({prob_pct:.0f}% likely): {line['pv_line']}\n"
                    if line.get('tactics'):
                        for tactic in line['tactics']:
                            lines_str += f"      -> {tactic['type'].upper()} after {tactic['after_move']}: {tactic['description']}\n"
                lines_str += "\n"
            
            humanlike_section = f"""
HUMANLIKE LINES (typical moves by players at different skill levels):
{lines_str}"""

        # Format current position tactics
        current_tactics = context.get('current_tactics', [])
        current_tactics_section = ""
        if current_tactics:
            current_tactics_str = "\n".join(f"  - {t['type'].upper()}: {t['description']}" for t in current_tactics)
            current_tactics_section = f"""
CURRENT POSITION TACTICS:
{current_tactics_str}
"""
        
        base_prompt = f"""
GENERATE CHESS POSITION COMMENTARY

POSITION CONTEXT:
[FEN]: {context.get('fen')}
[Side to Move]: {context.get('side_to_move')}
[Move Number]: {context.get('move_number')}
{current_tactics_section}
{semantic_section}
"""
        atk_def = context.get('attacks_defenses', {})
        white_attacks = atk_def.get('white_attacks', [])
        black_attacks = atk_def.get('black_attacks', [])
        white_defends = atk_def.get('white_defends', [])
        black_defends = atk_def.get('black_defends', [])
        
        white_atk_str = "\n".join(f"  - {a}" for a in white_attacks[:50]) if white_attacks else "  (none)"
        black_atk_str = "\n".join(f"  - {a}" for a in black_attacks[:50]) if black_attacks else "  (none)"
        white_def_str = "\n".join(f"  - {d}" for d in white_defends[:50]) if white_defends else "  (none)"
        black_def_str = "\n".join(f"  - {d}" for d in black_defends[:50]) if black_defends else "  (none)"
        
        attacks_section = f"""
BOARD TENSIONS:
[White Attacks]:
{white_atk_str}

[Black Attacks]:
{black_atk_str}

[White Defends]:
{white_def_str}

[Black Defends]:
{black_def_str}
"""
        
        return base_prompt + attacks_section + engine_section + humanlike_section + maia_section + f"""

TASK:
Generate chess position commentary:
1. Commentary: Overall positional assessment - key factors, imbalances, and evaluation

GUIDELINES:
- Your purpose is to generate intuitive descriptions of chess positions as curation of a chess-language model training dataset
- The context given should be used to deduce information about the position. Avoid baseless inferences and elaborations. 
- You should give humanlike commentary. Depending on your ability to produce commentary on the particular position, it may reflect observations of a player at any skill level. But should have a subjective feel, and should not directly reference or imply access to any of these artifacts or engine evaluations. 
- The "Humanlike Tactical Lines" section shows moves that typical players might play and the resulting tactical threats (forks, pins, discovered attacks). Use this along with the engine analysis to understand the most impactful tactics in the position that are likely to occur.
- Focus on intuitive understanding: what are the key pieces, active/potential threats, and imbalances in the position? In sharp positions, prioritize the most imminent tactics that occur in the sequences.


OUTPUT FORMAT (Strict JSON):
{{
  "fen": "{context.get('fen')}",
  "samples": [
    {{
      "commentary": "..."
    }},
  ]
}}
"""
    
    def generate_response(self, model_name: str, context: Dict[str, Any]) -> Any:
        prompt = self._construct_prompt(context)
        system_prompt = """You are a specialized Data Generation Assistant for Chess.
Your goal is to create high-quality training data for a Chess Language Model.
Output strictly valid JSON. Do not include markdown formatting (like ```json)."""
        
        # Directly call AI service, synchronous for standalone tool
        provider, model = AIService.parse_model_string(model_name)
        
        user_settings = self.app_controller.user_settings_service.get_settings()
        ai_settings = user_settings.get("ai_models", {})
        
        api_key = ""
        if provider == "openai":
            api_key = ai_settings.get("openai", {}).get("api_key", "")
        elif provider == "anthropic":
            api_key = ai_settings.get("anthropic", {}).get("api_key", "")
        elif provider == "google":
            api_key = ai_settings.get("google", {}).get("api_key", "")
            
        if not api_key:
             return {"error": f"No API Key configured for {provider}"}

        messages = [{"role": "user", "content": prompt}]
        
        success, response = self.ai_service.send_message(
            provider, model, api_key, messages, system_prompt
        )
        
        if success:
            try:
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                return json.loads(clean_response)
            except json.JSONDecodeError:
                return {"error": "JSON Parse Error", "raw": response}
        else:
            return {"error": f"AI Request Failed: {response}"}
