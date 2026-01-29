from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import chess

# Mocks for AppController components
class MockUserSettings:
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
    
    def get_settings(self):
        # Return structure expected by TrainingDataController and AIService
        return {
            "ai_models": {
                "openai": {"api_key": self.api_keys.get("openai", ""), "models": ["gpt-4o"]},
                "anthropic": {"api_key": self.api_keys.get("anthropic", ""), "models": ["claude-3-5-sonnet"]},
                "google": {"api_key": self.api_keys.get("google", ""), "models": ["gemini-1.5-pro"]}
            }
        }

@dataclass
class MockActiveGame:
    pgn: str

class MockGameModel:
    def __init__(self):
        self.active_game = None
        self.ply_index = 0
        self.is_game_analyzed = False 

    def get_active_move_ply(self):
        return self.ply_index

class MockGameController:
    def __init__(self):
        self.game_model = MockGameModel()
    
    def get_game_model(self):
        return self.game_model

class MockHeatmapController:
    def __init__(self, analyzer):
        self.analyzer = analyzer

class MockAppController:
    def __init__(self, api_keys=None):
        self.user_settings_service = MockUserSettings(api_keys)
        self.game_controller = MockGameController()
        self.heatmap_controller = None # Set later
        self.fen = chess.STARTING_FEN
    
    def get_current_fen(self):
        return self.fen

    def get_positional_heatmap_controller(self):
        return self.heatmap_controller
