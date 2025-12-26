"""
Performance analysis for Q-Learning algorithm.
This script identifies the main bottlenecks in the training process.
"""

import sys
import time
import cProfile
import pstats
from collections import defaultdict

import numpy as np

# Add the current directory to Python path
sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv


class PerformanceAnalyzer:
    """
    Analyze performance bottlenecks in Q-Learning.
    """

    def __init__(self):
        self.env = BlokusEnv()

    def analyze_get_legal_actions(self, iterations=10):
        """
        Analyze the performance of get_legal_actions method.
        """
        print("Analyzing get_legal_actions performance...")

        # Reset environment
        state, info = self.env.reset()

        def get_legal_actions_profiled(env, player):
            """Simplified version for profiling"""
            legal_actions = []

            # Check each available piece
            for piece_index, available in enumerate(env.available_pieces[player]):
                if available:
                    # Get candidate positions based on game state
                    candidate_positions = _get_candidate_positions_profiled(env, player)

                    # Try all candidate positions and rotations
                    for x, y in candidate_positions:
                        for rotation in range(4):
                            if env._is_valid_placement(piece_index, x, y, rotation):
                                legal_actions.append((piece_index, x, y, rotation))

            return legal_actions

        def _get_candidate_positions_profiled(env, player):
            """Simplified candidate positions for profiling"""
            candidate_positions = set()

            if not env.first_piece_placed[player]:
                # First piece: check positions around the player's starting corner
                player_corner = {
                    0: (0, 0),  # Blue: top-left
                    1: (0, 19),  # Yellow: top-right
                    2: (19, 19),  # Red: bottom-right
                    3: (19, 0),  # Green: bottom-left
                }[player]

                # Add the corner itself and surrounding positions
                x, y = player_corner
                for dx in range(-5, 6):  # Check 11x11 area around corner
                    for dy in range(-5, 6):
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

                    # Expand the bounding box to cover potential placement areas
                    expand_radius = 8

                    start_x = max(0, min_x - expand_radius)
                    end_x = min(19, max_x + expand_radius)
                    start_y = max(0, min_y - expand_radius)
                    end_y = min(19, max_y + expand_radius)

                    # Add all positions in the expanded bounding box
                    for x in range(start_x, end_x + 1):
                        for y in range(start_y, end_y + 1):
                            candidate_positions.add((x, y))

            return list(candidate_positions)

        # Profile the get_legal_actions function
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        for i in range(iterations):
            for player in range(4):
                legal_actions = get_legal_actions_profiled(self.env, player)
        end_time = time.time()

        profiler.disable()

        total_time = end_time - start_time
        avg_time = total_time / (iterations * 4)

        print(f"Total time for {iterations} iterations: {total_time:.3f} seconds")
        print(f"Average time per player: {avg_time:.3f} seconds")
        print(f"Legal actions per call: {len(legal_actions)}")

        # Print profiling results
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("\nProfiling results for get_legal_actions:")
        stats.print_stats(10)

        return total_time, avg_time

    def analyze_state_hashing(self, iterations=100):
        """
        Analyze the performance of state hashing.
        """
        print("\nAnalyzing state hashing performance...")

        # Create a sample state
        state = np.zeros((20, 20, 4), dtype=np.float32)

        def get_state_hash_profiled(state, player):
            """Simplified state hashing for profiling"""
            return (tuple(state.flatten()), player)

        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        for i in range(iterations):
            for player in range(4):
                state_hash = get_state_hash_profiled(state, player)
        end_time = time.time()

        profiler.disable()

        total_time = end_time - start_time
        avg_time = total_time / (iterations * 4)

        print(f"Total time for {iterations} iterations: {total_time:.3f} seconds")
        print(f"Average time per hash: {avg_time:.6f} seconds")

        # Print profiling results
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("\nProfiling results for state hashing:")
        stats.print_stats(5)

        return total_time, avg_time

    def analyze_full_training_step(self, steps=5):
        """
        Analyze a full training step to identify bottlenecks.
        """
        print("\nAnalyzing full training step performance...")

        # Reset environment
        state, info = self.env.reset()
        current_player = self.env.current_player

        def training_step_profiled():
            """Simplified training step for profiling"""

            # Get legal actions (this is likely the bottleneck)
            legal_actions = []
            for piece_index, available in enumerate(
                self.env.available_pieces[current_player]
            ):
                if available:
                    # Simplified candidate positions
                    if not self.env.first_piece_placed[current_player]:
                        player_corner = (0, 0)  # Simplified
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                nx, ny = player_corner[0] + dx, player_corner[1] + dy
                                if 0 <= nx < 20 and 0 <= ny < 20:
                                    for rotation in range(4):
                                        if self.env._is_valid_placement(
                                            piece_index, nx, ny, rotation
                                        ):
                                            legal_actions.append(
                                                (piece_index, nx, ny, rotation)
                                            )
                    else:
                        # Simplified: just check a few positions
                        for x in range(0, 20, 5):
                            for y in range(0, 20, 5):
                                for rotation in range(4):
                                    if self.env._is_valid_placement(
                                        piece_index, x, y, rotation
                                    ):
                                        legal_actions.append(
                                            (piece_index, x, y, rotation)
                                        )

            # Choose action (simplified)
            if legal_actions:
                action = legal_actions[0]  # Just pick first action

                # Take action
                next_state, reward, done, truncated, info = self.env.step(action)

                # State hashing
                state_hash = (tuple(state.flatten()), current_player)
                next_state_hash = (tuple(next_state.flatten()), self.env.current_player)

                return next_state, reward, done
            else:
                return state, 0, False

        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        for i in range(steps):
            next_state, reward, done = training_step_profiled()
            state = next_state
            if done:
                state, info = self.env.reset()
        end_time = time.time()

        profiler.disable()

        total_time = end_time - start_time
        avg_time = total_time / steps

        print(f"Total time for {steps} steps: {total_time:.3f} seconds")
        print(f"Average time per step: {avg_time:.3f} seconds")
        print(f"Steps per second: {1 / avg_time:.2f}")

        # Print profiling results
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("\nProfiling results for full training step:")
        stats.print_stats(15)

        return total_time, avg_time

    def run_analysis(self):
        """
        Run comprehensive performance analysis.
        """
        print("=== Q-Learning Performance Analysis ===")
        print("This analysis will help identify performance bottlenecks.\n")

        # Analyze individual components
        legal_actions_time, legal_actions_avg = self.analyze_get_legal_actions(
            iterations=5
        )
        state_hash_time, state_hash_avg = self.analyze_state_hashing(iterations=50)
        full_step_time, full_step_avg = self.analyze_full_training_step(steps=3)

        print("\n=== Performance Summary ===")
        print(
            f"Legal actions analysis: {legal_actions_time:.3f}s total, {legal_actions_avg:.3f}s avg"
        )
        print(
            f"State hashing analysis: {state_hash_time:.3f}s total, {state_hash_avg:.6f}s avg"
        )
        print(
            f"Full step analysis: {full_step_time:.3f}s total, {full_step_avg:.3f}s avg"
        )

        # Estimate full training performance
        estimated_episodes_per_minute = 60 / full_step_avg if full_step_avg > 0 else 0
        print(
            f"\nEstimated performance: {estimated_episodes_per_minute:.2f} steps/minute"
        )

        print("\n=== Key Bottlenecks Identified ===")
        print("1. get_legal_actions() - This is likely the main bottleneck")
        print("2. _get_candidate_positions() - Expensive bounding box calculations")
        print("3. _is_valid_placement() - Called repeatedly for many positions")
        print("4. State hashing - tuple(state.flatten()) creates large tuples")

        print("\n=== Optimization Recommendations ===")
        print("1. Optimize get_legal_actions() by reducing candidate positions")
        print("2. Cache or memoize valid placements")
        print("3. Use more efficient state representation (e.g., hash of board)")
        print("4. Implement early termination in placement validation")
        print("5. Consider using Numba or Cython for critical loops")


if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    analyzer.run_analysis()
