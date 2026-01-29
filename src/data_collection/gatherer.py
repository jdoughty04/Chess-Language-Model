import sys
import os
import json
import argparse
import random
import chess
import chess.pgn
import time
import asyncio
import io
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple

# Setup path to import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(project_root, "src")

if src_dir not in sys.path:
    sys.path.append(src_dir)

from wrappers import MockAppController, MockHeatmapController
from generator import CommentaryGenerator
from services.positional_heatmap.positional_analyzer import PositionalAnalyzer
from services.positional_heatmap.rule_registry import RuleRegistry
from services.positional_heatmap.rule_registry import RuleRegistry
from services.ai_service import AIService
from services.maia_inference_service import service_entry_point
from multiprocessing import Manager, Process, Queue


class TokenTracker:
    """Track token usage across API calls."""
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0
    
    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += int(input_tokens)
        self.output_tokens += int(output_tokens)
        self.request_count += 1
    
    def print_summary(self):
        print(f"\n{'='*50}")
        print("TOKEN USAGE STATISTICS")
        print(f"{'='*50}")
        print(f"  Total Requests:    {self.request_count:,}")
        print(f"  Input Tokens:      {self.input_tokens:,}")
        print(f"  Output Tokens:     {self.output_tokens:,}")
        print(f"  Total Tokens:      {self.input_tokens + self.output_tokens:,}")
        print(f"{'='*50}")


def load_settings():
    """Load settings from local user_settings.json"""
    settings_path = os.path.join(project_root, "user_settings.json")
    settings = {}
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            
    # Resolve relative engine paths
    engines = settings.get("engines", [])
    if engines:
        path = engines[0].get("path", "")
        if path and not os.path.isabs(path):
            # Try project root
            abs_path = os.path.join(project_root, path)
            if os.path.exists(abs_path):
                engines[0]["path"] = abs_path
            # Fallback: Check if it exists relative to current dir
            elif os.path.exists(path):
                engines[0]["path"] = os.path.abspath(path)
                
    return settings


# Global generator for workers
_generator = None

def worker_init(settings, quiet=False):
    """Initialize generator in each worker process."""
    global _generator
    ai_models = settings.get("ai_models", {})
    google_api_key = ai_models.get("google", {}).get("api_key", "")
    api_keys = {
        "openai": ai_models.get("openai", {}).get("api_key", ""),
        "anthropic": ai_models.get("anthropic", {}).get("api_key", ""),
        "google": google_api_key
    }
    app_controller = MockAppController(api_keys=api_keys)
    rule_registry = RuleRegistry({'rules': {}})
    analyzer = PositionalAnalyzer({'cache_enabled': True}, rule_registry)
    app_controller.heatmap_controller = MockHeatmapController(analyzer)
    _generator = CommentaryGenerator(settings, app_controller, quiet=quiet)

def process_single_game(game_pgn: str, game_index: int, samples_per_game: int) -> List[Tuple[Dict, str]]:
    """Worker function to process items from a single game."""
    global _generator
    if _generator is None:
        return []
        
    items = []
    try:
        if isinstance(game_pgn, str):
            game_obj = chess.pgn.read_game(io.StringIO(game_pgn))
        else:
            game_obj = game_pgn
            
        if not game_obj:
            return []
            
        moves = list(game_obj.mainline_moves())
        if len(moves) < 20:
            return []
            
        start_ply = 20
        end_ply = len(moves)
        if end_ply <= start_ply:
            return []
            
        population = range(start_ply, end_ply)
        k = min(samples_per_game, len(population))
        selected_plies = set(random.sample(population, k))
        
        board = game_obj.board()
        current_ply = 0
        
        for move in moves:
            move_san = board.san(move)
            move_number = (current_ply // 2) + 1
            is_white_move = (current_ply % 2 == 0)
            
            if is_white_move:
                move_notation = f"{move_number}. {move_san}"
            else:
                move_notation = f"{move_number}...{move_san}"
            
            board.push(move)
            current_ply += 1
            
            if current_ply in selected_plies:
                _generator.app_controller.fen = board.fen()
                _generator.app_controller.game_controller.game_model.ply_index = current_ply
                _generator.app_controller.game_controller.game_model.active_game = type('obj', (object,), {'pgn': "", 'game_object': game_obj})()
                
                context = _generator.get_context_for_current_position()
                prompt = _generator.construct_prompt(context)
                
                metadata = {
                    "game_index": game_index,
                    "game_id": game_obj.headers.get("Site", "unknown"),
                    "ply": current_ply,
                    "move": move_notation,
                    "fen": board.fen()
                }
                items.append((metadata, prompt))
    except Exception as e:
        print(f"  [Worker] Error processing game {game_index}: {e}")
        
    finally:
        if _generator:
            _generator.close()
            
    return items


async def call_gemini_async(client, model_name, system_instruction, thinking_budget, prompt, metadata, token_tracker):
    """Asynchronous call to Gemini."""
    from google.genai import types
    try:
        config_params = {
            "system_instruction": system_instruction,
        }
        if thinking_budget != 0:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
            
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**config_params)
        )
        
        response_text = response.text if response.text else ""
        
        # Parse response
        clean_response = response_text.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        
        try:
            parsed_response = json.loads(clean_response)
        except json.JSONDecodeError:
            parsed_response = {"error": "JSON Parse Error", "raw": response_text}
            
        # Token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        token_tracker.add(input_tokens, output_tokens)
        
        return {
            **metadata,
            "prompt": prompt,
            "generated_commentary": parsed_response
        }
    except Exception as e:
        return {
            **metadata,
            "prompt": prompt,
            "generated_commentary": {"error": str(e)}
        }


async def process_batch_async(client, model_name, system_instruction, thinking_budget, batch_items, token_tracker, quiet):
    """Process a list of items asynchronously."""
    tasks = []
    for metadata, prompt in batch_items:
        tasks.append(call_gemini_async(client, model_name, system_instruction, thinking_budget, prompt, metadata, token_tracker))
    
    return await asyncio.gather(*tasks)


def process_games_batch(input_pgn, output_file, model, samples_per_game=1, max_games=None, api_batch_size=20, thinking_budget=-1, quiet=False, num_workers=1, engine_depth=15, engine_multipv=3, device=None):
    """Optimized batch processing using multiprocessing and asyncio."""
    
    settings = load_settings()
    # Inject CLI args into settings for worker access
    if 'engine_config' not in settings: settings['engine_config'] = {}
    
    ai_models = settings.get("ai_models", {})
    google_api_key = ai_models.get("google", {}).get("api_key", "")
    
    # Inject engine config overrides
    settings['engine_overrides'] = {
        'depth': engine_depth,
        'multipv': engine_multipv,
        'device': device
    }
    
    # Initialize Multiprocessing Manager for shared state
    manager = Manager()
    input_queue = Queue() 
    result_dict = manager.dict()
    
    # Inject maia queues into settings
    settings['maia_queues'] = (input_queue, result_dict)
    
    # Start the GPU Service Process
    if not quiet: print("Starting Maia GPU Inference Service...")
    service_process = Process(target=service_entry_point, args=(input_queue, result_dict, settings))
    service_process.start()
    
    try:
        if not google_api_key:
            print("Warning: Google API key not found. Phase 2 will be skipped.")
            
        provider, model_name = AIService.parse_model_string(model)
        if provider.lower() != "google":
            print(f"Error: Batch mode currently only supports Google models.")
            return

        # Phase 1: Collect Prompts (Multiprocessing)
        if not quiet: print(f"--- Phase 1: Collecting prompts using multiprocessing ---")
        
        all_games = []
        if not quiet: print("  Reading PGN file...")
        start_read = time.time()
        with open(input_pgn, 'r', encoding='utf-8') as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None or (max_games and len(all_games) >= max_games):
                    break
                all_games.append((str(game), len(all_games)))
                
                if not quiet and len(all_games) % 1000 == 0:
                    print(f"    Read {len(all_games)} games...", end='\r')
        
        if not quiet: print(f"  Read {len(all_games)} games in {time.time()-start_read:.1f}s.")

        batch_items = []
        if num_workers > 1:
            if not quiet: print(f"  Processing {len(all_games)} games using {num_workers} workers...")
            with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init, initargs=(settings, quiet)) as executor:
                futures = {executor.submit(process_single_game, g_pgn, idx, samples_per_game): idx for g_pgn, idx in all_games}
                for i, future in enumerate(futures):
                    try:
                        res = future.result()
                        batch_items.extend(res)
                    except Exception as e:
                        if not quiet: print(f"Error in worker result: {e}")
                    if not quiet and (i + 1) % 5 == 0:
                        print(f"  Progress: {i+1}/{len(all_games)} games, {len(batch_items)} samples collected.")
        else:
            if not quiet: print(f"  Processing {len(all_games)} games sequentially...")
            worker_init(settings, quiet=quiet)
            for g_pgn, idx in all_games:
                res = process_single_game(g_pgn, idx, samples_per_game)
                batch_items.extend(res)
                if not quiet: print(f"  Processed game {idx + 1}/{len(all_games)} ({len(res)} samples)")

        if not quiet: print(f"Collected {len(batch_items)} prompts.")
        if not batch_items: return

        # Phase 2: Async GenAI
        if not quiet: print(f"\n--- Phase 2: Sending {len(batch_items)} requests asynchronously (batch size {api_batch_size}) ---")
        
        # Initialize output file (overwrite if starting fresh)
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
        
        try:
            from google import genai
        except ImportError:
            print("Error: google-genai package not installed. Skipping Phase 2.")
            return

        client = genai.Client(api_key=google_api_key, http_options={'api_version': 'v1alpha'})
        token_tracker = TokenTracker()
        system_instruction = "You are a specialized Data Generation Assistant for Chess. Output strictly valid JSON."
        
        results = []
        
        async def run_all_batches():
            nonlocal results
            for i in range(0, len(batch_items), api_batch_size):
                end = min(i + api_batch_size, len(batch_items))
                current_batch = batch_items[i:end]
                if not quiet: print(f"  Sending requests {i+1}-{end}...")
                
                batch_results = await process_batch_async(client, model_name, system_instruction, thinking_budget, current_batch, token_tracker, quiet)
                results.extend(batch_results)
                
                # Robust Incremental Save: Append only the new batch results
                with open(output_file, 'a', encoding='utf-8') as outfile:
                    for entry in batch_results:
                        outfile.write(json.dumps(entry) + "\n")
                
                if end < len(batch_items):
                    await asyncio.sleep(1) # Rate limit safety

        asyncio.run(run_all_batches())
        
        if not quiet:
            print(f"\n--- Phase 3: Complete ---")
            print(f"Processed {len(all_games)} games, generated {len(results)} samples.")
            token_tracker.print_summary()
            
    finally:
        if _generator:
            try:
                _generator.close()
            except: pass
            
        # Clean up service
        if 'service_process' in locals() and service_process.is_alive():
            if not quiet: print("Stopping Maia Service...")
            input_queue.put("STOP")
            service_process.join(timeout=5)
            if service_process.is_alive():
                service_process.terminate()


def process_games(input_pgn, output_file, model, samples_per_game=1, max_games=None, use_batch=False, 
                  append_mode=False, api_batch_size=20, thinking_budget=-1, quiet=False, num_workers=1, profile=False,
                  engine_depth=15, engine_multipv=3, device=None):
    """Main entry point."""
    if profile:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        
    # Hack: To pass engine args to workers without changing every signature stack, 
    # we can inject them into the loaded settings inside process_games_batch, 
    # but process_games_batch calls load_settings() itself.
    # Better: process_games_batch should accept an override dict or we wrap it.
    # Actually, process_games_batch loads settings. Let's pass args to it.
    
    if use_batch:
        process_games_batch(input_pgn, output_file, model, samples_per_game, max_games, api_batch_size, thinking_budget, quiet, num_workers, engine_depth, engine_multipv, device)
    else:
        # Non-batch mode still available but deprecated for performance
        print("Warning: Non-batch mode is slower. Use --batch for high-performance generation.")
        process_games_batch(input_pgn, output_file, model, samples_per_game, max_games, 1, thinking_budget, quiet, num_workers, engine_depth, engine_multipv, device)
        
    if profile:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(30)
        stats.dump_stats("gatherer_phase1.prof")
        print("Profile data saved to gatherer_phase1.prof")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PGN file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--model", default="Google: gemini-2.5-flash-lite", help="Model string")
    parser.add_argument("--samples", type=int, default=1, help="Position samples per game")
    parser.add_argument("--max-games", type=int, default=None, help="Maximum number of games to process")
    parser.add_argument("--batch", action="store_true", help="Use async batch processing")
    parser.add_argument("--api-batch-size", type=int, default=15, help="Number of concurrent API calls")
    parser.add_argument("--thinking", type=int, default=-1, help="Thinking budget (0=off, -1=dynamic, >0=fixed)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for Phase 1 (1=sequential)")
    parser.add_argument("--quiet", action="store_true", help="Minimize output")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile")
    parser.add_argument("--depth", type=int, default=12, help="Stockfish engine analysis depth (default: 15)")
    parser.add_argument("--multipv", type=int, default=3, help="Stockfish MultiPV lines (default: 3)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Maia model (cuda/cpu)")
    
    args = parser.parse_args()
    process_games(
        args.input, args.output, args.model, args.samples, args.max_games, 
        args.batch, False, args.api_batch_size, args.thinking, args.quiet, args.workers, args.profile,
        args.depth, args.multipv, args.device
    )
