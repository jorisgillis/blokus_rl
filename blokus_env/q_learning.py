"""Q-Learning implementation for Blokus with performance improvements and parallel training."""

import hashlib
import multiprocessing
import pickle
import random
from typing import Any

import numpy as np
from tqdm import tqdm

from blokus_env.blokus_env import BlokusEnv


class QLearningAgent:
    """Q-Learning agent for Blokus game with performance improvements."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.9995,
        min_exploration: float = 0.01,
    ):
        """Initialize Q-Learning agent.

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
        self.q_table: dict[bytes, dict[tuple[int, int, int, int, bool, bool], float]] = {}

        # Performance optimization: caching and precomputation
        self._state_cache: dict[bytes, np.ndarray] = {}  # state_bytes -> canonical_state
        self._legal_actions_cache: dict[bytes, list[tuple[int, int, int, int, bool, bool]]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Statistics
        self._total_state_hashes = 0
        self._total_legal_action_calls = 0

    def _get_canonical_state(self, state: np.ndarray) -> np.ndarray:
        """Get the canonical representative of a state using symmetries with caching."""
        self._total_state_hashes += 1
        
        # Check cache (board bytes are enough since we don't care about player index here)
        state_bytes = state.tobytes()
        if state_bytes in self._state_cache:
            self._cache_hits += 1
            return self._state_cache[state_bytes]
            
        self._cache_misses += 1
        
        symmetries = []
        curr = state
        for _ in range(4):
            symmetries.append(curr)
            symmetries.append(np.flip(curr, axis=0))
            curr = np.rot90(curr, k=1, axes=(0, 1))
            
        canonical = min(symmetries, key=lambda s: s.tobytes())
        self._state_cache[state_bytes] = canonical
        return canonical

    def _get_state_hash_fast(self, state: np.ndarray, player: int) -> bytes:
        """Fast canonical state hashing."""
        canonical_state = self._get_canonical_state(state)
        state_bytes = canonical_state.tobytes()
        # Use raw bytes + player byte for hash key (no MD5 overhead)
        return state_bytes + bytes([player])

    def get_state_hash(self, state: np.ndarray, player: int) -> bytes:
        """Convert game state to hashable representation."""
        return self._get_state_hash_fast(state, player)

    def get_legal_actions(
        self, env: BlokusEnv, player: int
    ) -> list[tuple[int, int, int, int, bool, bool]]:
        """Get all legal actions for current state using environment's optimized checks."""
        self._total_legal_action_calls += 1
        
        # Create cache key
        cache_key = (env.board.tobytes(), player)
        if cache_key in self._legal_actions_cache:
            self._cache_hits += 1
            return self._legal_actions_cache[cache_key].copy()

        self._cache_misses += 1
        
        legal_actions = []
        available_pieces = [i for i, avail in enumerate(env.available_pieces[player]) if avail]
        
        # We still need to iterate orientations and positions, 
        # but env._is_valid_placement is now very fast.
        for piece_id in available_pieces:
            for rotation in range(4):
                for flip_h in [False, True]:
                    for flip_v in [False, True]:
                        # Optimization: only check near existing pieces (candidate positions)
                        # We use the existing environment's candidate logic
                        for x, y in range_20_20:
                            if env._is_valid_placement(piece_id, x, y, rotation, flip_h, flip_v):
                                legal_actions.append((piece_id, x, y, rotation, flip_h, flip_v))

        self._legal_actions_cache[cache_key] = legal_actions.copy()
        return legal_actions

    def choose_action(
        self, env: BlokusEnv, state: np.ndarray, player: int, training: bool = True
    ) -> tuple[int, int, int, int, bool, bool]:
        """Choose action using epsilon-greedy policy."""
        # Use simple coordinate iteration for speed if candidate logic is too complex
        legal_actions = []
        available_pieces = [i for i, avail in enumerate(env.available_pieces[player]) if avail]
        
        for piece_id in available_pieces:
            # Note: For faster selection during training, we could pre-filter using bitboards 
            # but let's stick to the environment's method for correctness first.
            # Q-learning optimized version used a bounding box, let's keep that.
            pass

        # Re-using the bounding box logic from before for efficiency
        candidate_positions = self._get_candidate_positions_simple(env, player)
        for piece_id in available_pieces:
            for rotation in range(4):
                for flip_h in [False, True]:
                    for flip_v in [False, True]:
                        for x, y in candidate_positions:
                            if env._is_valid_placement(piece_id, x, y, rotation, flip_h, flip_v):
                                legal_actions.append((piece_id, x, y, rotation, flip_h, flip_v))

        if not legal_actions:
            return None

        state_hash = self.get_state_hash(state, player)

        if training and random.random() < self.exploration_rate:
            return random.choice(legal_actions)

        if state_hash in self.q_table:
            q_values = {action: self.q_table[state_hash].get(action, 0.0) for action in legal_actions}
            return max(q_values.items(), key=lambda x: x[1])[0]

        return random.choice(legal_actions)

    def _get_candidate_positions_simple(self, env: BlokusEnv, player: int) -> list[tuple[int, int]]:
        """Simple candidate generation logic."""
        if not env.first_piece_placed[player]:
            corners = {0: (0,0), 1: (0,19), 2: (19,19), 3:(19,0)}
            cx, cy = corners[player]
            res = []
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < 20 and 0 <= ny < 20: res.append((nx, ny))
            return res
        
        # Bounding box of all player pieces
        coords = []
        for x in range(20):
            for y in range(20):
                if env.board[x,y,player]: coords.append((x,y))
        
        if not coords: return []
        min_x, max_x = min(x for x,y in coords), max(x for x,y in coords)
        min_y, max_y = min(y for x,y in coords), max(y for x,y in coords)
        
        res = []
        for x in range(max(0, min_x-5), min(20, max_x+6)):
            for y in range(max(0, min_y-5), min(20, max_y+6)):
                res.append((x,y))
        return res

    def update_q_table(
        self, state: np.ndarray, action: Any, reward: float, next_state: np.ndarray, next_player: int, done: bool
    ) -> None:
        """Standard Q-update."""
        state_hash = self.get_state_hash(state, next_player)
        next_state_hash = self.get_state_hash(next_state, next_player)

        if state_hash not in self.q_table: self.q_table[state_hash] = {}
        if action not in self.q_table[state_hash]: self.q_table[state_hash][action] = 0.0

        current_q = self.q_table[state_hash][action]
        max_next_q = max(self.q_table[next_state_hash].values()) if not done and next_state_hash in self.q_table and self.q_table[next_state_hash] else 0.0
        
        self.q_table[state_hash][action] = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

    def train(self, episodes: int = 100, max_steps: int = 50, parallel: bool = False) -> None:
        """Generic train method."""
        if parallel:
            self.train_parallel(episodes, max_steps)
            return

        env = BlokusEnv()
        pbar = tqdm(range(episodes), desc="Training")
        for _ in pbar:
            state, _ = env.reset()
            done = False
            step = 0
            while not done and step < max_steps:
                player = env.current_player
                action = self.choose_action(env, state, player, training=True)
                if action is None: break
                
                next_state, reward, done, truncated, _ = env.step(action)
                self.update_q_table(state, action, reward, next_state, player, done)
                state = next_state
                step += 1
            
            self._update_stats(truncated, env, reward)
            if self.episodes % 10 == 0:
                pbar.set_postfix({"Exp": f"{self.exploration_rate:.3f}", "States": len(self.q_table)})

    def _update_stats(self, truncated, env, reward):
        self.episodes += 1
        if truncated: self.ties += 1
        elif (env and env.current_player == 0 and reward > 100) or (reward > 100): 
            self.wins += 1
        else: self.losses += 1
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)

    def train_parallel(self, episodes: int, max_steps: int):
        """Parallel training entry point."""
        cpus = multiprocessing.cpu_count()
        batch_size = 50
        num_batches = (episodes + batch_size - 1) // batch_size
        
        pbar = tqdm(total=episodes, desc="Parallel Training")
        with multiprocessing.Pool(processes=cpus) as pool:
            for _ in range(num_batches):
                count = min(batch_size, episodes - self.episodes)
                if count <= 0: break
                
                jobs = [pool.apply_async(worker_loop, (self.q_table, self.exploration_rate, count // cpus, max_steps)) for _ in range(cpus)]
                for j in jobs:
                    for exp in j.get():
                        state, action, reward, next_state, player, done, truncated, win_flag = exp
                        self.update_q_table(state, action, reward, next_state, player, done)
                        self._update_stats(truncated, None, reward) 
                        pbar.update(1)
                pbar.set_postfix({"States": len(self.q_table)})

    def save(self, filename):
        with open(filename, "wb") as f: pickle.dump(self.__dict__, f)
    
    def load(self, filename):
        try:
            with open(filename, "rb") as f: self.__dict__.update(pickle.load(f))
            return True
        except: return False

    def get_cache_stats(self): return {"hit_rate": 0.0} # Placeholder for compatibility
    def get_q_table_size(self): return len(self.q_table), 0

def worker_loop(q_table, epsilon, n, max_steps):
    """External worker function for pickling."""
    agent = QLearningAgent(exploration_rate=epsilon)
    agent.q_table = q_table
    env = BlokusEnv()
    exps = []
    for _ in range(n):
        state, _ = env.reset()
        done = False
        step = 0
        while not done and step < max_steps:
            player = env.current_player
            action = agent.choose_action(env, state, player, training=True)
            if action is None: break
            n_state, r, d, t, _ = env.step(action)
            exps.append((state, action, r, n_state, player, d, t, env.winner == 0))
            state = n_state
            step += 1
    return exps

# Global constant for candidate search
range_20_20 = [(x, y) for x in range(20) for y in range(20)]
