"""Minimal AI service stub for model string parsing.

This is a slim replacement for CARA's AIService. Only implements
the functionality actually used by Chess_transformer scripts.
"""

from typing import Dict, Any, Optional, Tuple


class AIService:
    """Minimal AIService - primarily for parse_model_string().
    
    The actual API calls are handled directly by google-genai SDK
    in gatherer.py and generator.py.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @staticmethod
    def parse_model_string(model_string: str) -> Tuple[str, str]:
        """Parse a model string with provider prefix.
        
        Args:
            model_string: Model string like "Google: gemini-2.5-flash-lite"
            
        Returns:
            Tuple of (provider, model_name)
            
        Examples:
            "Google: gemini-2.5-flash" -> ("google", "gemini-2.5-flash")
            "OpenAI: gpt-4o" -> ("openai", "gpt-4o")
            "Anthropic: claude-3-5-sonnet" -> ("anthropic", "claude-3-5-sonnet")
        """
        if ":" in model_string:
            parts = model_string.split(":", 1)
            provider = parts[0].strip().lower()
            model = parts[1].strip()
            return provider, model
        # Default to google if no provider specified
        return "google", model_string.strip()
