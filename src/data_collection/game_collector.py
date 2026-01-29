"""
Chess Game Collector - Downloads games from Lichess API
Collects games with Elo rating filter
"""

import requests
import os
import time
import json
import random
from datetime import datetime
from pathlib import Path


class LichessGameCollector:
    """Collects chess games from Lichess API and exports."""
    
    BASE_URL = "https://lichess.org/api"
    
    # Popular Lichess player pools by rating range (1000-2500)
    # These are example usernames - the script will dynamically find active players
    SEED_USERNAMES = [
        "DrNykterstein",  # Magnus Carlsen
        "Fins", 
        "penguingm1",
        "nihalsarin2004",
        "lance5500",
    ]
    
    def __init__(self, output_dir: str = "data/games", min_elo: int = 1000, max_elo: int = 3000,
                 exclude_pgn_files: list = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/x-ndjson",
            "User-Agent": "ChessCommentaryProject/1.0"
        })
        self.collected_games = 0
        self.collected_usernames = set()
        
        # Load existing game IDs to exclude (deduplication)
        self.excluded_game_ids = set()
        if exclude_pgn_files:
            # Handle potential comma-separated strings
            files_to_process = []
            for path_arg in exclude_pgn_files:
                files_to_process.extend([p.strip() for p in path_arg.split(',') if p.strip()])
            
            self.excluded_game_ids = self._load_existing_game_ids(files_to_process)
            print(f"📋 Loaded {len(self.excluded_game_ids)} existing game IDs to exclude")
        
        # Track newly collected game IDs for manifest
        self.new_game_ids = set()
    
    def _load_existing_game_ids(self, pgn_files: list) -> set:
        """
        Extract game IDs from existing PGN files for deduplication.
        
        Args:
            pgn_files: List of paths to existing PGN files
            
        Returns:
            Set of Lichess game IDs (8-character strings)
        """
        import re
        game_ids = set()
        
        for pgn_path in pgn_files:
            pgn_path = Path(pgn_path)
            if not pgn_path.exists():
                print(f"  Warning: PGN file not found: {pgn_path}")
                continue
            
            print(f"  Loading game IDs from: {pgn_path.name}")
            try:
                with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Extract game IDs from [Site "https://lichess.org/XXXXXXXX"] headers
                # Lichess game IDs are 8 alphanumeric characters
                pattern = r'\[Site "https://lichess\.org/([a-zA-Z0-9]{8})"\]'
                matches = re.findall(pattern, content)
                game_ids.update(matches)
                print(f"    Found {len(matches)} game IDs")
                
            except Exception as e:
                print(f"  Error reading {pgn_path}: {e}")
        
        return game_ids
    
    def _extract_game_id_from_json(self, game_data: dict) -> str:
        """
        Extract Lichess game ID from API response.
        
        Args:
            game_data: JSON response from Lichess API
            
        Returns:
            8-character game ID or empty string
        """
        game_id = game_data.get('id', '')
        return game_id[:8] if game_id else ''
        
    def get_user_games(self, username: str, max_games: int = 100, 
                       perf_type: str = "rapid,blitz,classical") -> list:
        """
        Fetch games for a specific user from Lichess API.
        
        Args:
            username: Lichess username
            max_games: Maximum number of games to fetch
            perf_type: Game types to include (rapid, blitz, classical, bullet)
            
        Returns:
            List of game PGN strings
        """
        url = f"{self.BASE_URL}/games/user/{username}"
        params = {
            "max": max_games,
            "perfType": perf_type,
            "rated": "true",
            "pgnInJson": "true",
            "clocks": "false",
            "evals": "false",
            "opening": "true"
        }
        
        games = []
        try:
            response = self.session.get(url, params=params, stream=True, timeout=60)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    game_data = json.loads(line.decode('utf-8'))
                    
                    # Filter by Elo rating
                    if self._check_elo_filter(game_data):
                        # Extract and check game ID for deduplication
                        game_id = self._extract_game_id_from_json(game_data)
                        if game_id and game_id in self.excluded_game_ids:
                            continue  # Skip already-collected game
                        
                        pgn = game_data.get('pgn', '')
                        if pgn and self._has_sufficient_moves(pgn):
                            games.append(pgn)
                            
                            # Track this game ID for manifest
                            if game_id:
                                self.new_game_ids.add(game_id)
                            
                            # Also collect opponent username for crawling
                            self._collect_opponent(game_data, username)
                            
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching games for {username}: {e}")
        except json.JSONDecodeError as e:
            print(f"  JSON decode error for {username}: {e}")
            
        return games
    
    def _check_elo_filter(self, game_data: dict) -> bool:
        """Check if both players meet the Elo requirement."""
        try:
            players = game_data.get('players', {})
            white_rating = players.get('white', {}).get('rating', 0)
            black_rating = players.get('black', {}).get('rating', 0)
            
            # Both players must be within the Elo range
            return (self.min_elo <= white_rating <= self.max_elo and 
                    self.min_elo <= black_rating <= self.max_elo)
        except (KeyError, TypeError):
            return False
    
    def _has_sufficient_moves(self, pgn: str, min_moves: int = 15) -> bool:
        """Check if the game has enough moves (not a quick resignation/abort)."""
        try:
            # Simple heuristic: count move numbers in PGN
            move_count = pgn.count('.')
            return move_count >= min_moves
        except:
            return False
    
    def _collect_opponent(self, game_data: dict, current_user: str):
        """Collect opponent usernames for crawling more games."""
        try:
            players = game_data.get('players', {})
            for color in ['white', 'black']:
                user_info = players.get(color, {}).get('user', {})
                username = user_info.get('name', '')
                if username and username.lower() != current_user.lower():
                    self.collected_usernames.add(username)
        except (KeyError, TypeError):
            pass
    
    def get_tv_games(self, count: int = 100) -> list:
        """
        Fetch recent TV games (featured high-quality games).
        """
        url = f"{self.BASE_URL}/tv/channel/classical/feed"  # or rapid, blitz
        games = []
        
        try:
            response = self.session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line and len(games) < count:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'd' in data and 'fen' in data['d']:
                            # TV feed provides live positions, not full games
                            # This is more useful for getting active player names
                            players = data['d'].get('players', [])
                            for player in players:
                                username = player.get('user', {}).get('name')
                                if username:
                                    self.collected_usernames.add(username)
                    except:
                        continue
        except:
            pass
            
        return games
    
    def get_leaderboard_players(self, perf_type: str = "rapid", count: int = 100) -> list:
        """Get top players from leaderboard to seed the collection."""
        url = f"{self.BASE_URL}/player/top/{count}/{perf_type}"
        usernames = []
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for user in data.get('users', []):
                username = user.get('username', '')
                if username:
                    usernames.append(username)
                    self.collected_usernames.add(username)
                    
        except Exception as e:
            print(f"Error fetching leaderboard: {e}")
            
        return usernames
    
    def get_active_players_from_pool(self, rating_range: tuple = (1000, 1500), 
                                      count: int = 50) -> list:
        """
        Get active players in a specific rating range by checking recent puzzles/activity.
        Since Lichess doesn't have direct rating-filtered player search,
        we use the puzzle API to find active players, then filter.
        """
        # Alternative: use the streamer endpoint or tournament participants
        url = f"{self.BASE_URL}/streamer/live"
        usernames = []
        
        try:
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                streamers = response.json()
                for streamer in streamers[:count]:
                    username = streamer.get('name', '')
                    if username:
                        usernames.append(username)
                        self.collected_usernames.add(username)
        except:
            pass
            
        return usernames
    
    def collect_games(self, target_count: int = 10000, 
                     games_per_user: int = 50,
                     perf_types: str = "rapid,blitz,classical") -> str:
        """
        Main collection loop - crawl games until target is reached.
        
        Args:
            target_count: Number of games to collect
            games_per_user: Maximum games to fetch per user
            perf_types: Game types to include
            
        Returns:
            Path to the output PGN file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"lichess_games_{self.min_elo}+_{timestamp}.pgn"
        
        all_games = []
        processed_users = set()
        
        print(f"🎯 Target: {target_count} games with {self.min_elo}+ Elo")
        print(f"📁 Output: {output_file}")
        print("-" * 50)
        
        # Phase 1: Seed with leaderboard players 
        print("\n📋 Phase 1: Fetching leaderboard players...")
        for perf in ["rapid", "blitz", "classical"]:
            self.get_leaderboard_players(perf, count=50)
            time.sleep(1)  # Rate limiting
        
        print(f"  Found {len(self.collected_usernames)} seed players")
        
        # Phase 2: Get games from active streamers/players
        print("\n🎮 Phase 2: Fetching active players...")
        self.get_active_players_from_pool()
        
        # Phase 3: Crawl games
        print("\n🔄 Phase 3: Collecting games...")
        
        # Start with seed usernames
        usernames_to_process = list(self.SEED_USERNAMES) + list(self.collected_usernames)
        random.shuffle(usernames_to_process)
        
        while len(all_games) < target_count and usernames_to_process:
            username = usernames_to_process.pop(0)
            
            if username.lower() in processed_users:
                continue
                
            processed_users.add(username.lower())
            
            print(f"  [{len(all_games):,}/{target_count:,}] Fetching games from: {username}...", end=" ")
            
            games = self.get_user_games(username, max_games=games_per_user, perf_type=perf_types)
            
            if games:
                all_games.extend(games)
                print(f"✓ Got {len(games)} games")
                
                # Every 500 games, save progress
                if len(all_games) % 500 < len(games):
                    self._save_progress(all_games, output_file)
                    
                # Add new usernames discovered from opponents
                new_users = [u for u in self.collected_usernames 
                            if u.lower() not in processed_users]
                random.shuffle(new_users)
                usernames_to_process.extend(new_users[:100])  # Limit queue growth
                self.collected_usernames -= set(new_users[:100])
            else:
                print("✗ No valid games")
            
            # Rate limiting - Lichess recommends 1 request per second
            time.sleep(1.5)
            
            # If we run out of users, try to find more
            if len(usernames_to_process) < 10 and len(all_games) < target_count:
                print("\n  🔍 Finding more players...")
                for perf in ["bullet", "ultraBullet"]:
                    self.get_leaderboard_players(perf, count=50)
                new_users = [u for u in self.collected_usernames 
                            if u.lower() not in processed_users]
                usernames_to_process.extend(new_users)
        
        # Final save
        self._save_games(all_games[:target_count], output_file)
        
        # Save manifest of collected game IDs
        self._save_manifest(output_file)
        
        print("\n" + "=" * 50)
        print(f"✅ Collection complete!")
        print(f"📊 Total games collected: {min(len(all_games), target_count):,}")
        print(f"📁 Saved to: {output_file}")
        
        return str(output_file)
    
    def _save_progress(self, games: list, output_file: Path):
        """Save current progress to file."""
        progress_file = output_file.with_suffix('.progress.pgn')
        self._save_games(games, progress_file)
        print(f"  💾 Progress saved: {len(games):,} games")
    
    def _save_games(self, games: list, output_file: Path):
        """Save games to PGN file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for pgn in games:
                f.write(pgn.strip())
                f.write("\n\n")
    
    def _save_manifest(self, output_file: Path):
        """Save manifest of collected game IDs for future deduplication."""
        manifest_file = output_file.with_suffix('.manifest.json')
        manifest = {
            "created": datetime.now().isoformat(),
            "game_count": len(self.new_game_ids),
            "game_ids": sorted(list(self.new_game_ids))
        }
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        print(f"📋 Manifest saved: {manifest_file.name} ({len(self.new_game_ids)} game IDs)")
        
    def download_database_sample(self, year: int = 2024, month: int = 12, 
                                  sample_size: int = 10000) -> str:
        """
        Alternative: Download from Lichess database exports.
        These are compressed monthly archives - much faster for bulk collection.
        
        Note: This downloads the full monthly database and samples from it.
        The files can be 10+ GB, so this is better for large collections.
        
        Args:
            year: Year of database
            month: Month of database
            sample_size: Number of games to extract
            
        Returns:
            Path to output file
        """
        # Lichess database URL format
        db_url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
        
        print(f"📥 Database URL: {db_url}")
        print("⚠️  Note: Monthly databases are large (10+ GB compressed)")
        print("    For faster collection, use collect_games() method instead.")
        print("    Or download manually and use filter_pgn_file() to extract games.")
        
        return db_url


def filter_pgn_file(input_file: str, output_file: str, 
                    min_elo: int = 1000, max_games: int = 10000) -> int:
    """
    Filter an existing PGN file to extract games meeting Elo criteria.
    Useful for processing Lichess database downloads.
    
    Args:
        input_file: Path to input PGN file
        output_file: Path to output PGN file
        min_elo: Minimum Elo for both players
        max_games: Maximum games to extract
        
    Returns:
        Number of games extracted
    """
    import chess.pgn
    
    count = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            while count < max_games:
                try:
                    game = chess.pgn.read_game(f_in)
                    if game is None:
                        break
                    
                    # Check Elo ratings
                    white_elo = game.headers.get('WhiteElo', '0')
                    black_elo = game.headers.get('BlackElo', '0')
                    
                    try:
                        white_elo = int(white_elo) if white_elo != '?' else 0
                        black_elo = int(black_elo) if black_elo != '?' else 0
                    except ValueError:
                        continue
                    
                    if white_elo >= min_elo and black_elo >= min_elo:
                        # Check game length (at least 15 moves)
                        moves = list(game.mainline_moves())
                        if len(moves) >= 30:  # 15 full moves = 30 half-moves
                            f_out.write(str(game))
                            f_out.write("\n\n")
                            count += 1
                            
                            if count % 1000 == 0:
                                print(f"  Extracted {count:,} games...")
                                
                except Exception as e:
                    continue
    
    print(f"✅ Extracted {count:,} games to {output_file}")
    return count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect chess games from Lichess")
    parser.add_argument("--count", type=int, default=100, 
                       help="Number of games to collect (default: 10000)")
    parser.add_argument("--min-elo", type=int, default=1000,
                       help="Minimum Elo rating (default: 1000)")
    parser.add_argument("--max-elo", type=int, default=3000,
                       help="Maximum Elo rating (default: 3000)")
    parser.add_argument("--output-dir", type=str, default="data/games2",
                       help="Output directory (default: data/games2)")
    parser.add_argument("--perf-types", type=str, default="rapid,blitz,classical",
                       help="Game types to collect (default: rapid,blitz,classical)")
    parser.add_argument("--filter-file", type=str, default=None,
                       help="Path to existing PGN file to filter for elo instead of downloading")
    parser.add_argument("--exclude-pgn", type=str, nargs="*", default=None,
                       help="Path(s) to existing PGN file(s) whose games should be excluded (deduplication)")
    
    args = parser.parse_args()
    
    if args.filter_file:
        # Filter an existing PGN file
        output = args.filter_file.replace('.pgn', f'_filtered_{args.min_elo}+.pgn')
        filter_pgn_file(args.filter_file, output, args.min_elo, args.count)
    else:
        # Collect from Lichess API
        collector = LichessGameCollector(
            output_dir=args.output_dir,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
            exclude_pgn_files=args.exclude_pgn
        )
        collector.collect_games(
            target_count=args.count,
            perf_types=args.perf_types
        )

