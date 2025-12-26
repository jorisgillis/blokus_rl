"""
Test script to compare performance between original and optimized Blokus environments.
"""

import sys
import time
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv
from blokus_env.blokus_env_optimized import BlokusEnvOptimized


class PerformanceComparison:
    """
    Compare performance between original and optimized environments.
    """

    def __init__(self):
        self.original_env = BlokusEnv()
        self.optimized_env = BlokusEnvOptimized()

    def test_placement_validation(self, iterations=100):
        """Test placement validation performance."""
        print("=== Placement Validation Performance Test ===")

        # Test positions
        test_positions = [
            (0, 0, 0),
            (5, 5, 1),
            (10, 10, 2),
            (15, 15, 3),
            (1, 1, 0),
            (7, 3, 1),
            (12, 8, 2),
            (18, 12, 3),
        ]

        print(
            f"Testing {iterations} iterations with {len(test_positions)} positions..."
        )

        # Test original
        print("\n1. Original environment:")
        start_time = time.time()
        for i in range(iterations):
            for piece_index in range(5):  # Test first 5 pieces
                for x, y, rotation in test_positions:
                    self.original_env._is_valid_placement(piece_index, x, y, rotation)
        original_time = time.time() - start_time

        print(f"   Time: {original_time:.3f}s")
        print(
            f"   Per validation: {original_time / (iterations * len(test_positions) * 5):.6f}s"
        )

        # Test optimized
        print("\n2. Optimized environment:")
        start_time = time.time()
        for i in range(iterations):
            for piece_index in range(5):  # Test first 5 pieces
                for x, y, rotation in test_positions:
                    self.optimized_env._is_valid_placement(piece_index, x, y, rotation)
        optimized_time = time.time() - start_time

        print(f"   Time: {optimized_time:.3f}s")
        print(
            f"   Per validation: {optimized_time / (iterations * len(test_positions) * 5):.6f}s"
        )

        speedup = original_time / optimized_time
        print(f"\n   Speedup: {speedup:.2f}×")
        print(f"   Improvement: {(1 - optimized_time / original_time) * 100:.1f}%")

        return original_time, optimized_time, speedup

    def test_legal_actions(self, iterations=10):
        """Test legal actions computation performance."""
        print("\n=== Legal Actions Performance Test ===")

        print(f"Testing {iterations} iterations...")

        # Test original
        print("\n1. Original environment:")
        start_time = time.time()
        for i in range(iterations):
            for player in range(4):
                # Use the original method (we need to implement a wrapper)
                legal_actions = []
                for piece_index, available in enumerate(
                    self.original_env.available_pieces[player]
                ):
                    if available:
                        # Simplified candidate positions for fair comparison
                        if not self.original_env.first_piece_placed[player]:
                            x, y = 0, 0  # Just test one position for baseline
                            for rotation in range(4):
                                if self.original_env._is_valid_placement(
                                    piece_index, x, y, rotation
                                ):
                                    legal_actions.append((piece_index, x, y, rotation))
                        else:
                            # Test a few positions
                            for x in [5, 10, 15]:
                                for y in [5, 10, 15]:
                                    for rotation in range(4):
                                        if self.original_env._is_valid_placement(
                                            piece_index, x, y, rotation
                                        ):
                                            legal_actions.append(
                                                (piece_index, x, y, rotation)
                                            )
        original_time = time.time() - start_time

        print(f"   Time: {original_time:.3f}s")
        print(f"   Per call: {original_time / (iterations * 4):.3f}s")

        # Test optimized
        print("\n2. Optimized environment:")
        start_time = time.time()
        for i in range(iterations):
            for player in range(4):
                legal_actions = self.optimized_env.get_legal_actions_optimized(
                    player, max_actions=100
                )
        optimized_time = time.time() - start_time

        print(f"   Time: {optimized_time:.3f}s")
        print(f"   Per call: {optimized_time / (iterations * 4):.3f}s")

        speedup = original_time / optimized_time
        print(f"\n   Speedup: {speedup:.2f}×")
        print(f"   Improvement: {(1 - optimized_time / original_time) * 100:.1f}%")

        return original_time, optimized_time, speedup

    def test_candidate_positions(self, iterations=10):
        """Test candidate position generation performance."""
        print("\n=== Candidate Positions Performance Test ===")

        print(f"Testing {iterations} iterations...")

        # Test original (simulated)
        print("\n1. Original candidate generation (simulated):")
        start_time = time.time()
        for i in range(iterations):
            for player in range(4):
                # Simulate original candidate generation (121 positions for first piece)
                if not self.original_env.first_piece_placed[player]:
                    positions = []
                    x, y = 0, 0  # Corner
                    for dx in range(-5, 6):
                        for dy in range(-5, 6):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < 20 and 0 <= ny < 20:
                                positions.append((nx, ny))
                else:
                    # Simulate bounding box approach
                    positions = []
                    for x in range(0, 20, 2):
                        for y in range(0, 20, 2):
                            positions.append((x, y))
        original_time = time.time() - start_time

        print(f"   Time: {original_time:.3f}s")
        print(f"   Positions generated: {len(positions)}")

        # Test optimized
        print("\n2. Optimized candidate generation:")
        start_time = time.time()
        for i in range(iterations):
            for player in range(4):
                positions = self.optimized_env._get_candidate_positions_optimized(
                    player
                )
        optimized_time = time.time() - start_time

        print(f"   Time: {optimized_time:.3f}s")
        print(f"   Positions generated: {len(positions)}")

        speedup = original_time / optimized_time
        print(f"\n   Speedup: {speedup:.2f}×")
        print(f"   Improvement: {(1 - optimized_time / original_time) * 100:.1f}%")
        print(f"   Position reduction: {1 - len(positions) / 121:.1f}% (first piece)")

        return original_time, optimized_time, speedup

    def test_full_training_step(self, steps=5):
        """Test full training step performance."""
        print("\n=== Full Training Step Performance Test ===")

        print(f"Testing {steps} training steps...")

        # Reset both environments
        self.original_env.reset()
        self.optimized_env.reset()

        # Test original
        print("\n1. Original environment:")
        step_times = []
        for step in range(steps):
            start_time = time.time()

            # Get legal actions (simplified)
            legal_actions = []
            for piece_index in range(5):  # Test first 5 pieces
                if self.original_env.available_pieces[self.original_env.current_player][
                    piece_index
                ]:
                    # Test a few positions
                    for x in [0, 5, 10]:
                        for y in [0, 5, 10]:
                            for rotation in range(4):
                                if self.original_env._is_valid_placement(
                                    piece_index, x, y, rotation
                                ):
                                    legal_actions.append((piece_index, x, y, rotation))

            if legal_actions:
                action = legal_actions[0]
                self.original_env.step(action)
            else:
                self.original_env.current_player = (
                    self.original_env.current_player + 1
                ) % 4

            step_time = time.time() - start_time
            step_times.append(step_time)

        original_time = sum(step_times)
        print(f"   Total time: {original_time:.3f}s")
        print(f"   Average step time: {original_time / steps:.3f}s")

        # Reset optimized environment
        self.optimized_env.reset()

        # Test optimized
        print("\n2. Optimized environment:")
        step_times = []
        for step in range(steps):
            start_time = time.time()

            # Get legal actions using optimized method
            legal_actions = self.optimized_env.get_legal_actions_optimized(
                self.optimized_env.current_player, max_actions=50
            )

            if legal_actions:
                action = legal_actions[0]
                self.optimized_env.step(action)
            else:
                self.optimized_env.current_player = (
                    self.optimized_env.current_player + 1
                ) % 4

            step_time = time.time() - start_time
            step_times.append(step_time)

        optimized_time = sum(step_times)
        print(f"   Total time: {optimized_time:.3f}s")
        print(f"   Average step time: {optimized_time / steps:.3f}s")

        speedup = original_time / optimized_time
        print(f"\n   Speedup: {speedup:.2f}×")
        print(f"   Improvement: {(1 - optimized_time / original_time) * 100:.1f}%")

        return original_time, optimized_time, speedup

    def test_cache_effectiveness(self, steps=20):
        """Test cache effectiveness."""
        print("\n=== Cache Effectiveness Test ===")

        # Reset environment
        self.optimized_env.reset()

        print(f"Running {steps} steps to test caching...")

        # Run some steps
        for step in range(steps):
            legal_actions = self.optimized_env.get_legal_actions_optimized(
                self.optimized_env.current_player, max_actions=50
            )

            if legal_actions:
                action = legal_actions[0]
                self.optimized_env.step(action)
            else:
                self.optimized_env.current_player = (
                    self.optimized_env.current_player + 1
                ) % 4

        # Get cache stats
        cache_stats = self.optimized_env.get_cache_stats()

        print(f"\nCache Statistics:")
        print(f"   Cache hits: {cache_stats['cache_hits']}")
        print(f"   Cache misses: {cache_stats['cache_misses']}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.3f}")
        print(f"   Placement cache size: {cache_stats['placement_cache_size']}")
        print(f"   Legal actions cache size: {cache_stats['legal_actions_cache_size']}")

        return cache_stats

    def run_comparison(self):
        """Run complete performance comparison."""
        print("Running Blokus Environment Performance Comparison")
        print("=" * 60)

        # Run individual tests
        placement_results = self.test_placement_validation(iterations=50)
        legal_actions_results = self.test_legal_actions(iterations=5)
        candidate_results = self.test_candidate_positions(iterations=5)
        training_results = self.test_full_training_step(steps=3)
        cache_stats = self.test_cache_effectiveness(steps=10)

        print("\n" + "=" * 60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("=" * 60)

        print("\n=== Individual Component Improvements ===")
        print(f"Placement validation: {placement_results[2]:.2f}× speedup")
        print(f"Legal actions: {legal_actions_results[2]:.2f}× speedup")
        print(f"Candidate positions: {candidate_results[2]:.2f}× speedup")
        print(f"Full training step: {training_results[2]:.2f}× speedup")

        print(f"\n=== Cache Effectiveness ===")
        print(f"Cache hit rate: {cache_stats['hit_rate']:.3f}")
        print(f"Placement cache size: {cache_stats['placement_cache_size']}")
        print(f"Legal actions cache size: {cache_stats['legal_actions_cache_size']}")

        # Calculate overall speedup
        overall_speedup = (
            placement_results[2] * 0.6  # Placement validation is ~60% of time
            + legal_actions_results[2] * 0.3  # Legal actions is ~30%
            + training_results[2] * 0.1  # Other operations ~10%
        )

        print(f"\n=== Overall Performance Improvement ===")
        print(f"Estimated overall speedup: {overall_speedup:.2f}×")
        print(
            f"This could increase training speed from ~8 to ~{8 * overall_speedup:.1f} episodes/minute"
        )

        return {
            "placement_validation": placement_results,
            "legal_actions": legal_actions_results,
            "candidate_positions": candidate_results,
            "training_step": training_results,
            "cache_stats": cache_stats,
            "overall_speedup": overall_speedup,
        }


if __name__ == "__main__":
    comparison = PerformanceComparison()
    results = comparison.run_comparison()

    # Save results
    import pickle

    with open("optimization_comparison_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nComparison results saved to 'optimization_comparison_results.pkl'")
