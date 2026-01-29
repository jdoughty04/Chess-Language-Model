import sys
import os
import json
import argparse
import chess
import chess.pgn
import io

# Mock PyQt6 if not available, to allow importing app modules
# Mock PyQt6 if not available, to allow importing app modules
try:
    import PyQt6
except ImportError:

    import sys
    from unittest.mock import MagicMock
    import types
    
    # Create fake PyQt6 package
    pyqt6 = types.ModuleType("PyQt6")
    sys.modules["PyQt6"] = pyqt6
    
    # Mock PyQt6.QtCore
    qtcore = MagicMock()
    pyqt6.QtCore = qtcore
    sys.modules["PyQt6.QtCore"] = qtcore
    
    # Mock QObject
    class MockQObject:
        def __init__(self, *args, **kwargs):
            pass
            
    qtcore.QObject = MockQObject
    
    # Mock pyqtSignal
    class MockSignal:
        def __init__(self, *args, **kwargs):
            pass
        def emit(self, *args, **kwargs):
            pass
            
    qtcore.pyqtSignal = MockSignal
    qtcore.Qt = MagicMock()

    # Mock PyQt6.QtGui
    qtgui = MagicMock()
    pyqt6.QtGui = qtgui
    sys.modules["PyQt6.QtGui"] = qtgui
    
    # Mock specific classes used in GameData
    qtgui.QIcon = MagicMock
    qtgui.QPixmap = MagicMock
    qtgui.QPainter = MagicMock
    qtgui.QBrush = MagicMock
    qtgui.QColor = MagicMock
    
    # Mock PyQt6.QtWidgets (used in experiment.py later)
    qtwidgets = MagicMock()
    pyqt6.QtWidgets = qtwidgets
    sys.modules["PyQt6.QtWidgets"] = qtwidgets


# Mock asteval if not available
try:
    import asteval
except ImportError:
    import sys
    from unittest.mock import MagicMock
    
    asteval = MagicMock()
    sys.modules["asteval"] = asteval
    
    class MockInterpreter:
        def __init__(self, *args, **kwargs):
            self.symtable = {}
        def __call__(self, *args, **kwargs):
            return 0
            
    asteval.Interpreter = MockInterpreter


# Setup path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.append(src_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from wrappers import MockAppController, MockHeatmapController
from generator import CommentaryGenerator
# Import RichGenerator - use dynamic import inside main or try/except if file might not exist yet?
# Since we just created it, we can import it.
try:
    from rich_generator import RichCommentaryGenerator
except ImportError as e:
    print(f"Debug Import Error 1: {e}")
    # If running from different CWD, try src relative
    try:
        from src.data_collection.rich_generator import RichCommentaryGenerator
    except ImportError as e2:
        print(f"Debug Import Error 2: {e2}")
        RichCommentaryGenerator = None
        
from services.positional_heatmap.positional_analyzer import PositionalAnalyzer
from services.positional_heatmap.rule_registry import RuleRegistry

def load_settings():
    """Load settings from local user_settings.json in project root"""
    settings_path = os.path.join(project_root, "user_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            return json.load(f)
    print(f"Warning: user_settings.json not found at {settings_path}")
    return {}


def generate_with_thinking(prompt: str, model_name: str, api_key: str) -> dict:
    """Generate response using Gemini with thinking mode enabled for better reasoning."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai package not installed.")
        print("Install with: pip install google-genai")
        return {"error": "google-genai not installed"}
    
    # System instruction for chess commentary
    system_instruction = """You are a specialized Data Generation Assistant for Chess.
Your goal is to create high-quality training data for a Chess Language Model.
Output strictly valid JSON. Do not include markdown formatting (like ```json)."""
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Generate with thinking mode enabled (dynamic budget)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(thinking_budget=-1)  # Dynamic thinking
            )
        )
        
        response_text = response.text if response.text else ""
        
        # Parse the JSON response
        clean_response = response_text.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            return {"error": "JSON Parse Error", "raw": response_text}
            
    except Exception as e:
        return {"error": f"API Error: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="Generate chess commentary sample")
    parser.add_argument("--pgn", help="PGN string or file path", required=True)
    parser.add_argument("--ply", type=int, help="Ply number (half-move) to analyze", required=True)
    parser.add_argument("--model", help="AI Model to use (e.g. 'OpenAI: gpt-4o')", default="Google: gemini-flash-lite-latest")
    
    parser.add_argument("--engine", help="Path to Stockfish engine (overrides settings)", required=False)
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    
    parser.add_argument("--rich", action="store_true", help="Use Rich Commentary Generator with CARA analysis")
    parser.add_argument("--prompt-only", action="store_true", help="Generate and show prompt only, skip API call")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish engine analysis depth (default: 15)")
    
    args = parser.parse_args()
    
    # Load settings
    settings = load_settings()
    
    # Fix engine path
    engine_path = None
    if args.engine:
        engine_path = args.engine
    else:
        # Check settings
        engines = settings.get("engines", [])
        if engines:
            path_in_settings = engines[0].get("path", "")
            if os.path.exists(path_in_settings):
                engine_path = path_in_settings
            else:
                # Try relative to project directory
                relative_guess = os.path.join(current_dir, "stockfish", "stockfish-windows-x86-64-avx2.exe")
                if os.path.exists(relative_guess):
                    print(f"Auto-detected engine at: {relative_guess}")
                    engine_path = relative_guess
                    
    if engine_path:
        print(f"Using Engine: {engine_path}")
        # Update settings for CommentaryGenerator
        if "engines" not in settings: settings["engines"] = [{}]
        if not settings["engines"]: settings["engines"].append({})
        settings["engines"][0]["path"] = engine_path
    else:
        print("Warning: No valid engine path found. Analysis will be limited.")

    # Inject engine overrides from CLI
    if "engine_overrides" not in settings: settings["engine_overrides"] = {}
    settings["engine_overrides"]["depth"] = args.depth
    
    # Setup Maia Service (MultiProcessing)
    from multiprocessing import Manager, Process, Queue
    from services.maia_inference_service import service_entry_point
    import atexit
    
    manager = Manager()
    input_queue = Queue()
    result_dict = manager.dict()
    
    settings['maia_queues'] = (input_queue, result_dict)
    
    print("Starting Maia GPU Inference Service...")
    service_process = Process(target=service_entry_point, args=(input_queue, result_dict, settings))
    service_process.start()
    
    def cleanup_maia():
        if service_process.is_alive():
            print("Stopping Maia Service...")
            input_queue.put("STOP")
            service_process.join(timeout=2)
            if service_process.is_alive():
                service_process.terminate()
                
    atexit.register(cleanup_maia)

    # Extract API keys
    ai_models = settings.get("ai_models", {})
    api_keys = {
        "openai": ai_models.get("openai", {}).get("api_key", ""),
        "anthropic": ai_models.get("anthropic", {}).get("api_key", ""),
        "google": ai_models.get("google", {}).get("api_key", "")
    }
    
    # Setup App Controller Mock
    app_controller = MockAppController(api_keys=api_keys)
    
    # Setup Positional Heatmap (Required for feature extraction)
    # We need to configure the analyzer
    # Typically this comes from heatmap_config.json or default
    ui_config = settings.get('ui', {}) # Settings might not have full config
    # We might need to load CARA's default config or just pass empty
    # PositionalAnalyzer expects 'config' and 'rule_registry'
    
    # Load default logic config from rules
    # We can try to load from CARA/app/services/positional_heatmap if needed, 
    # but RuleRegistry has default rules if we pass the dict properly.
    # Ideally we'd load the rules.json
    
    # For now, simplistic approach:
    rule_config = {} # Empty might use defaults or fail?
    # Let's try to locate the analyzer rules.
    # In CARA they are often hardcoded or in json. 
    # RuleRegistry.__init__ takes a config dict.
    
    # Let's just try initializing with empty and see if it works
    rule_registry = RuleRegistry({'rules': {}})
    analyzer_config = {'cache_enabled': True}
    analyzer = PositionalAnalyzer(analyzer_config, rule_registry)
    
    heatmap_controller = MockHeatmapController(analyzer)
    app_controller.heatmap_controller = heatmap_controller
    
    # Setup Generator
    if args.rich:
        if RichCommentaryGenerator:
            print("Using RichCommentaryGenerator (CARA rules enabled)")
            generator = RichCommentaryGenerator(settings, app_controller)
        else:
            print("Error: RichCommentaryGenerator could not be imported. Falling back to standard.")
            generator = CommentaryGenerator(settings, app_controller)
    else:
        print("Using Standard CommentaryGenerator")
        generator = CommentaryGenerator(settings, app_controller)
    
    # Load Game
    pgn_path = args.pgn
    pgn_content = args.pgn
    is_file = False
    
    if os.path.exists(pgn_path):
        is_file = True
        with open(pgn_path, 'r') as f:
            pgn_content = f.read()
    elif pgn_path.endswith('.pgn'):
        # User likely intended a file
        print(f"Error: PGN file not found: {pgn_path}")
        # Try checking src directory as fallback
        src_path = os.path.join(src_dir, pgn_path)
        if os.path.exists(src_path):
             print(f"Found in src directory: {src_path}")
             with open(src_path, 'r') as f:
                pgn_content = f.read()
        else:
             return
    
    pgn_io = io.StringIO(pgn_content)
    game = chess.pgn.read_game(pgn_io)
    
    if not game:
        print("Error: Could not read PGN")
        return
        
    # Check if game has moves
    if not game.variations:
        print("Warning: PGN parsed but has no moves. Content might be invalid or headerless.")
        print(f"PGN Header Sample: {dict(game.headers)}")
        # Check if maybe it's a raw move list?
        # If read_game failed to see moves, maybe we need to be more lenient? 
        # Actually chess.pgn.read_game handles "1. e4" fine.
        # But if the file content was "test.pgn" (the string), it produces no moves.
        if len(pgn_content) < 50:
            print(f"Content loaded: '{pgn_content}'")
        return

    # Setup Board to target ply
    board = game.board()
    node = game
    for _ in range(args.ply):
        if node.variations:
            node = node.variation(0)
            board.push(node.move)
        else:
            print(f"Warning: Reached end of game at ply {_}")
            break
            
    fen = board.fen()
    
    # Update Mock State
    app_controller.fen = fen
    app_controller.game_controller.game_model.ply_index = args.ply
    app_controller.game_controller.game_model.active_game = type('obj', (object,), {'pgn': pgn_content})()
    
    print(f"Generating commentary for position: {fen}")
    print(f"Model: {args.model}")
    
    context = generator.get_context_for_current_position()
    prompt = generator.construct_prompt(context)
    
    result_text = "{}"
    
    if args.prompt_only:
        print("\n--- GENERATED PROMPT ---\n")
        try:
            print(prompt)
        except UnicodeEncodeError:
            # Fallback: force ascii
            print(prompt.encode('ascii', errors='replace').decode('ascii'))
            
        print("\n--- PROMPT END ---\n")
        print("Skipping AI generation as --prompt-only is set.")
    else:
        # Parse model to get provider
        from services.ai_service import AIService
        provider, model_name = AIService.parse_model_string(args.model)
        
        # Generate using thinking mode for Google models
        if provider.lower() == "google":
            ai_models = settings.get("ai_models", {})
            api_key = ai_models.get("google", {}).get("api_key", "")
            if api_key:
                print("Using Gemini with thinking mode enabled...")
                result = generate_with_thinking(prompt, model_name, api_key)
            else:
                print("Warning: No Google API key found, falling back to standard generation")
                result = generator.generate_response(args.model, context)
        else:
            # Fallback to standard generation for non-Google models
            result = generator.generate_response(args.model, context)
        
        result_text = json.dumps(result, indent=2)
        
        print("\n--- GENERATED RESULT ---\n")
        print(result_text)
 
    # Close resources before potential GUI blocking or early exit
    generator.close()
 
    # Show GUI
    if not args.no_gui:
        try:
            from PyQt6.QtWidgets import QApplication, QTextEdit, QWidget, QVBoxLayout, QLabel
            from PyQt6.QtCore import Qt
            
            # Check if QApplication already exists (unlikely in this script but good practice)
            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            
            # Window 1: Prompt
            w1 = QWidget()
            w1.setWindowTitle("Generated Prompt")
            l1 = QVBoxLayout()
            t1 = QTextEdit()
            t1.setPlainText(prompt)
            t1.setReadOnly(True)
            l1.addWidget(QLabel("Prompt sent to Model:"))
            l1.addWidget(t1)
            w1.setLayout(l1)
            w1.resize(600, 800)
            w1.show()
            
            # Window 2: Response (Only show if not prompt-only)
            if not args.prompt_only:
                w2 = QWidget()
                w2.setWindowTitle("Model Response")
                l2 = QVBoxLayout()
                t2 = QTextEdit()
                t2.setPlainText(result_text)
                t2.setReadOnly(True)
                l2.addWidget(QLabel("Response from Model:"))
                l2.addWidget(t2)
                w2.setLayout(l2)
                w2.resize(600, 800)
                w2.move(650, 100) # Move to side
                w2.show()
            
            sys.exit(app.exec())
        except ImportError:
            print("PyQt6 not found. Cannot show GUI windows.")

if __name__ == "__main__":
    main()
