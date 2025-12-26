"""Optimized Q-Learning implementation for Blokus with performance improvements."""

import pickle
import random
import hashlib
from typing import Dict, List, Tuple, Any, Set
import numpy as np

from blokus_env.blokus_env import BlokusEnv


class QLearningAgentOptimized:
    """Optimized Q-Learning agent for Blokus game with performance improvements."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.995,
        min_exploration: float = 0.01,
    ):
        """Initialize optimized Q-Learning agent.

        Args:
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            exploration_rate: Initial exploration rate (epsilon)
            exploration_decay: Decay rate for exploration
            min_exploration: Minimum exploration rate
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Training statistics
        self.episodes = 0
        self.wins = 0
        self.losses = 0
        self.ties = 0

        # Q-table: state_hash -> action -> q_value
        self.q_table: Dict[str, Dict[Tuple[int, int, int, int], float]] = {}

        # Performance optimization: caching and precomputation
        self._state_cache: Dict[str, str] = {}  # state_tuple -> state_hash
        self._legal_actions_cache: Dict[str, List[Tuple[int, int, int, int]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Statistics
        self._total_state_hashes = 0
        self._total_legal_action_calls = 0

    def _get_state_hash_fast(self, state: np.ndarray, player: int) -> str:
        """Fast state hashing using MD5 hash of board state.

        Args:
            state: Current game state (20x20x4 board)
            player: Current player (0 or 1)

        Returns:
            Hash string for the state
        """
        self._total_state_hashes += 1

        # Create a unique key for caching
        cache_key = (state.data.tobytes(), player)

        # Check cache first
        if cache_key in self._state_cache:
            self._cache_hits += 1
            return self._state_cache[cache_key]

        self._cache_misses += 1

        # Create MD5 hash of the board state
        state_bytes = state.tobytes()
        state_hash = hashlib.md5(state_bytes).hexdigest()
        
        # Include player in the hash
        full_hash = f"{state_hash}_{player}"

        # Cache the result
        self._state_cache[cache_key] = full_hash

        return full_hash

    def _get_candidate_positions_optimized(self, env: BlokusEnv, player: int) -> Set[Tuple[int, int]]:
        """Optimized candidate position generation with reduced search space.

        Args:
            env: Blokus environment
            player: Current player

        Returns:
            Set of candidate positions to check for piece placement
        """
        candidate_positions = set()

        if not env.first_piece_placed[player]:
            # First piece: check positions around the player's starting corner
            player_corner = {
                0: (0, 0),      # Blue: top-left
                1: (0, 19),     # Yellow: top-right
                2: (19, 19),    # Red: bottom-right
                3: (19, 0),     # Green: bottom-left
            }[player]

            # Add the corner itself and surrounding positions (reduced radius)
            x, y = player_corner
            for dx in range(-3, 4):  # Check 7x7 area around corner (reduced from 11x11)
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 20 and 0 <= ny < 20:
                        candidate_positions.add((nx, ny))
        else:
            # Subsequent pieces: check expanded area around existing pieces
            # Find all existing pieces of this player
            player_pieces = []
            for x in range(20):
                for y in range(20):
                    if env.board[x, y, player] == 1:
                        player_pieces.append((x, y))

            if player_pieces:
                # Find the bounding box of existing pieces
                min_x = min(x for x, y in player_pieces)
                max_x = max(x for x, y in player_pieces)
                min_y = min(y for x, y in player_pieces)
                max_y = max(y for x, y in player_pieces)

                # Reduce the expansion radius for optimization
                expand_radius = 5  # Reduced from 8

                start_x = max(0, min_x - expand_radius)
                end_x = min(19, max_x + expand_radius)
                start_y = max(0, min_y - expand_radius)
                end_y = min(19, max_y + expand_radius)

                # Add all positions in the expanded bounding box
                for x in range(start_x, end_x + 1):
                    for y in range(start_y, end_y + 1):
                        candidate_positions.add((x, y))

        return candidate_positions

    def get_legal_actions_optimized(self, env: BlokusEnv, player: int) -> List[Tuple[int, int, int, int]]:
        """Optimized version of get_legal_actions with caching and reduced search space.

        Args:
            env: Blokus environment
            player: Current player

        Returns:
            List of legal actions as (piece_id, rotation, x, y) tuples
        """
        self._total_legal_action_calls += 1

        # Create cache key
        cache_key = (env.board.data.tobytes(), player)

        # Check cache first
        if cache_key in self._legal_actions_cache:
            self._cache_hits += 1
            return self._legal_actions_cache[cache_key].copy()

        self._cache_misses += 1

        legal_actions = []
        
        # Get available pieces for the player (optimized)
        available_pieces = [i for i, available in enumerate(env.available_pieces[player]) if available]

        # Get candidate positions (optimized)
        candidate_positions = self._get_candidate_positions_optimized(env, player)

        # For each piece, check candidate positions and rotations
        for piece_id in available_pieces:
            for rotation in range(4):  # 0, 1, 2, 3 rotations
                for x, y in candidate_positions:
                    # Check if placement is valid (use the environment's method)
                    if env._is_valid_placement(piece_id, x, y, rotation):
                        legal_actions.append((piece_id, rotation, x, y))

        # Cache the result
        self._legal_actions_cache[cache_key] = legal_actions.copy()

        return legal_actions

    def get_state_hash(self, state: np.ndarray, player: int) -> str:
        """Convert game state to hashable representation (wrapper for fast version).

        Args:
            state: Current game state (20x20x4 board)
            player: Current player (0 or 1)

        Returns:
            Hashable state representation
        """
        return self._get_state_hash_fast(state, player)

    def get_legal_actions(self, env: BlokusEnv, player: int) -> List[Tuple[int, int, int, int]]:
        """Get all legal actions for current state (wrapper for optimized version).

        Args:
            env: Blokus environment
            player: Current player

        Returns:
            List of legal actions as (piece_id, rotation, x, y) tuples
        """
        return self.get_legal_actions_optimized(env, player)

    def choose_action(
        self, env: BlokusEnv, state: np.ndarray, player: int, training: bool = True
    ) -> Tuple[int, int, int, int]:
        """Choose action using epsilon-greedy policy.

        Args:
            env: Blokus environment
            state: Current game state
            player: Current player
            training: Whether in training mode (affects exploration)

        Returns:
            Selected action as (piece_id, rotation, x, y) tuple
        """
        legal_actions = self.get_legal_actions(env, player)
        
        if not legal_actions:
            return None

        state_hash = self.get_state_hash(state, player)

        # Exploration: choose random action
        if training and random.random() < self.exploration_rate:
            return random.choice(legal_actions)

        # Exploitation: choose best action based on Q-values
        if state_hash in self.q_table:
            # Get Q-values for all legal actions
            q_values = {}
            for action in legal_actions:
                q_values[action] = self.q_table[state_hash].get(action, 0.0)
            
            # Choose action with highest Q-value
            if q_values:
                best_action = max(q_values.items(), key=lambda x: x[1])[0]
                return best_action

        # If no Q-values available, choose random action
        return random.choice(legal_actions)

    def update_q_table(
        self,
        state: np.ndarray,
        action: Tuple[int, int, int, int],
        reward: float,
        next_state: np.ndarray,
        next_player: int,
        done: bool,
    ) -> None:
        """Update Q-table using Q-learning update rule.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_player: Next player
            done: Whether episode is done
        """
        state_hash = self.get_state_hash(state, next_player)
        next_state_hash = self.get_state_hash(next_state, next_player)

        # Initialize Q-table entry if not exists
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}

        if action not in self.q_table[state_hash]:
            self.q_table[state_hash][action] = 0.0

        # Get current Q-value
        current_q = self.q_table[state_hash][action]

        # Calculate max Q-value for next state
        if done or next_state_hash not in self.q_table:
            max_next_q = 0.0
        else:
            next_q_values = self.q_table[next_state_hash]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        # Update Q-value
        self.q_table[state_hash][action] = new_q

    def train(self, episodes: int = 100, max_steps: int = 50) -> None:
        """Train the Q-learning agent.

        Args:
            episodes: Number of training episodes
            max_steps: Maximum steps per episode
        """
        env = BlokusEnv()

        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            step = 0

            while not done and step < max_steps:
                current_player = env.current_player
                
                # Choose action
                action = self.choose_action(env, state, current_player, training=True)
                
                if action is None:
                    # No legal moves, end episode
                    break

                # Execute action
                next_state, reward, done, _, _ = env.step(action)

                # Update Q-table
                self.update_q_table(state, action, reward, next_state, current_player, done)

                # Update state and statistics
                state = next_state
                step += 1

            # Update training statistics
            self.episodes += 1
            if reward > 0:
                self.wins += 1
            elif reward < 0:
                self.losses += 1
            else:
                self.ties += 1

            # Decay exploration rate
            self.exploration_rate = max(
                self.min_exploration, 
                self.exploration_rate * self.exploration_decay
            )

    def play_game(self, render: bool = False) -> None:
        """Play a demonstration game using the trained agent.

        Args:
            render: Whether to render the game
        """
        env = BlokusEnv()
        state, _ = env.reset()
        done = False

        if render:
            env.render()

        while not done:
            current_player = env.current_player
            action = self.choose_action(env, state, current_player, training=False)

            if action is None:
                break

            state, reward, done, _, _ = env.step(action)

            if render:
                env.render()
                print(f"Player {current_player} played: {action}")
                print(f"Reward: {reward}")

    def save(self, filename: str) -> None:
        """Save Q-learning agent to file.

        Args:
            filename: File to save to
        """
        data = {
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "exploration_rate": self.exploration_rate,
            "exploration_decay": self.exploration_decay,
            "min_exploration": self.min_exploration,
            "episodes": self.episodes,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "q_table": self.q_table,
            "_cache_hits": self._cache_hits,
            "_cache_misses": self._cache_misses,
            "_total_state_hashes": self._total_state_hashes,
            "_total_legal_action_calls": self._total_legal_action_calls,
        }

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename: str) -> bool:
        """Load Q-learning agent from file.

        Args:
            filename: File to load from

        Returns:
            True if loading successful, False otherwise
        """
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

            self.learning_rate = data["learning_rate"]
            self.discount_factor = data["discount_factor"]
            self.exploration_rate = data["exploration_rate"]
            self.exploration_decay = data["exploration_decay"]
            self.min_exploration = data["min_exploration"]
            self.episodes = data["episodes"]
            self.wins = data["wins"]
            self.losses = data["losses"]
            self.ties = data["ties"]
            self.q_table = data["q_table"]
            self._cache_hits = data.get("_cache_hits", 0)
            self._cache_misses = data.get("_cache_misses", 0)
            self._total_state_hashes = data.get("_total_state_hashes", 0)
            self._total_legal_action_calls = data.get("_total_legal_action_calls", 0)

            return True

        except (FileNotFoundError, pickle.PickleError, KeyError):
            return False

    def get_q_table_size(self) -> Tuple[int, int]:
        """Get size of Q-table.

        Returns:
            Tuple of (number of states, number of state-action pairs)
        """
        state_count = len(self.q_table)
        action_count = sum(len(actions) for actions in self.q_table.values())
        return state_count, action_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0.0,
            "total_state_hashes": self._total_state_hashes,
            "total_legal_action_calls": self._total_legal_action_calls,
            "state_hash_cache_size": len(self._state_cache),
            "legal_actions_cache_size": len(self._legal_actions_cache),
        }

    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self._state_cache.clear()
        self._legal_actions_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0