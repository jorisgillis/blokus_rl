"""
Performance baseline test for Q-Learning algorithm.
This script establishes a performance baseline before optimizations.
"""

import sys
import time
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv
from blokus_env.q_learning import QLearningAgent


class PerformanceBaseline:
    """
    Establish performance baseline for Q-Learning algorithm.
    """

    def __init__(self):
        self.env = BlokusEnv()
        self.agent = QLearningAgent()

    def test_individual_components(self):
        """Test individual components to establish baseline."""
        print("=== Performance Baseline Test ===")
        print("Testing individual components...\n")

        # Test 1: Environment reset
        print("1. Testing environment reset:")
        start_time = time.time()
        for i in range(100):
            state, info = self.env.reset()
        reset_time = time.time() - start_time
        print(f"   100 resets: {reset_time:.3f}s ({reset_time / 100:.6f}s per reset)")

        # Test 2: State hashing
        print("\n2. Testing state hashing:")
        state, info = self.env.reset()
        start_time = time.time()
        for i in range(1000):
            for player in range(4):
                state_hash = self.agent.get_state_hash(state, player)
        hash_time = time.time() - start_time
        print(f"   4000 hashes: {hash_time:.3f}s ({hash_time / 4000:.6f}s per hash)")

        # Test 3: Get legal actions (empty board)
        print("\n3. Testing get_legal_actions (empty board):")
        state, info = self.env.reset()
        start_time = time.time()
        for i in range(10):
            for player in range(4):
                legal_actions = self.agent.get_legal_actions(self.env, player)
        legal_actions_time = time.time() - start_time
        print(
            f"   40 calls: {legal_actions_time:.3f}s ({legal_actions_time / 40:.3f}s per call)"
        )
        print(f"   Legal actions found: {len(legal_actions)}")

        # Test 4: Get legal actions (with pieces placed)
        print("\n4. Testing get_legal_actions (with pieces):")
        # Place some pieces first
        for player in range(4):
            if player < 2:  # Place pieces for first two players
                # Find a valid placement
                for piece_index in range(21):
                    if self.env.available_pieces[player][piece_index]:
                        # Try to place in corner
                        if self.env._is_valid_placement(piece_index, 0, 0, 0):
                            self.env._place_piece(piece_index, 0, 0, 0)
                            break

        start_time = time.time()
        for i in range(10):
            for player in range(4):
                legal_actions = self.agent.get_legal_actions(self.env, player)
        legal_actions_with_pieces_time = time.time() - start_time
        print(
            f"   40 calls: {legal_actions_with_pieces_time:.3f}s ({legal_actions_with_pieces_time / 40:.3f}s per call)"
        )
        print(f"   Legal actions found: {len(legal_actions)}")

        # Test 5: Full action selection cycle
        print("\n5. Testing full action selection cycle:")
        state, info = self.env.reset()
        start_time = time.time()
        for i in range(5):
            for player in range(4):
                # Get legal actions
                legal_actions = self.agent.get_legal_actions(self.env, player)
                if legal_actions:
                    # Choose action (simplified - just pick first)
                    action = legal_actions[0]
                    # Get state hash
                    state_hash = self.agent.get_state_hash(state, player)
                    # Simulate Q-table update (just access)
                    q_value = self.agent.q_table[state_hash].get(action, 0)

        action_cycle_time = time.time() - start_time
        print(
            f"   20 cycles: {action_cycle_time:.3f}s ({action_cycle_time / 20:.3f}s per cycle)"
        )

        return {
            "reset_time": reset_time / 100,
            "hash_time": hash_time / 4000,
            "legal_actions_empty": legal_actions_time / 40,
            "legal_actions_with_pieces": legal_actions_with_pieces_time / 40,
            "action_cycle": action_cycle_time / 20,
        }

    def test_training_performance(self, steps=10):
        """Test actual training performance."""
        print("\n=== Training Performance Test ===")
        print(f"Testing {steps} training steps...\n")

        # Reset for training
        state, info = self.env.reset()
        current_player = self.env.current_player

        step_times = []
        legal_action_counts = []

        for step in range(steps):
            step_start = time.time()

            # Get legal actions
            legal_actions = self.agent.get_legal_actions(self.env, current_player)
            legal_action_counts.append(len(legal_actions))

            if legal_actions:
                # Choose action (use exploration for realism)
                if step % 2 == 0:  # Simulate exploration
                    action = (
                        np.random.choice(len(legal_actions)) if legal_actions else None
                    )
                    action = legal_actions[action] if action is not None else None
                else:
                    # Simulate exploitation
                    state_hash = self.agent.get_state_hash(state, current_player)
                    q_values = [
                        self.agent.q_table[state_hash].get(a, 0) for a in legal_actions
                    ]
                    max_q = max(q_values)
                    best_actions = [
                        a for a, q in zip(legal_actions, q_values) if q == max_q
                    ]
                    action = (
                        np.random.choice(len(best_actions)) if best_actions else None
                    )
                    action = best_actions[action] if action is not None else None

                # Take action
                next_state, reward, done, truncated, info = self.env.step(action)

                # Update Q-table (simplified - just compute, don't store)
                next_state_hash = self.agent.get_state_hash(
                    next_state, self.env.current_player
                )

                # Update state
                state = next_state
                current_player = self.env.current_player
            else:
                # No legal moves, switch player
                current_player = (current_player + 1) % 4

            step_time = time.time() - step_start
            step_times.append(step_time)

            print(
                f"Step {step + 1}: {step_time:.3f}s, {legal_action_counts[-1]} legal actions"
            )

        # Calculate statistics
        avg_step_time = np.mean(step_times)
        min_step_time = np.min(step_times)
        max_step_time = np.max(step_times)
        steps_per_second = 1 / avg_step_time
        steps_per_minute = steps_per_second * 60
        avg_legal_actions = np.mean(legal_action_counts)

        print(f"\n=== Performance Summary ===")
        print(f"Average step time: {avg_step_time:.3f}s")
        print(f"Min step time: {min_step_time:.3f}s")
        print(f"Max step time: {max_step_time:.3f}s")
        print(f"Steps per second: {steps_per_second:.2f}")
        print(f"Steps per minute: {steps_per_minute:.2f}")
        print(f"Average legal actions per step: {avg_legal_actions:.1f}")

        # Estimate episodes per minute (assuming ~50 steps per episode)
        estimated_episodes_per_minute = steps_per_minute / 50
        print(f"Estimated episodes per minute: {estimated_episodes_per_minute:.2f}")

        return {
            "avg_step_time": avg_step_time,
            "steps_per_minute": steps_per_minute,
            "estimated_episodes_per_minute": estimated_episodes_per_minute,
            "avg_legal_actions": avg_legal_actions,
            "step_times": step_times,
            "legal_action_counts": legal_action_counts,
        }

    def run_baseline_test(self):
        """Run complete baseline test."""
        print("Running Q-Learning Performance Baseline Test")
        print("=" * 50)

        # Test individual components
        component_results = self.test_individual_components()

        # Test training performance
        training_results = self.test_training_performance(steps=8)

        print("\n" + "=" * 50)
        print("BASELINE TEST COMPLETE")
        print("=" * 50)

        print("\n=== Key Performance Metrics ===")
        print(f"Environment reset: {component_results['reset_time']:.6f}s")
        print(f"State hashing: {component_results['hash_time']:.6f}s")
        print(f"Legal actions (empty): {component_results['legal_actions_empty']:.3f}s")
        print(
            f"Legal actions (with pieces): {component_results['legal_actions_with_pieces']:.3f}s"
        )
        print(f"Action cycle: {component_results['action_cycle']:.3f}s")
        print(f"Training step: {training_results['avg_step_time']:.3f}s")
        print(
            f"Estimated episodes/minute: {training_results['estimated_episodes_per_minute']:.2f}"
        )

        print(f"\n=== Bottleneck Analysis ===")
        print(
            f"Legal actions take {component_results['legal_actions_empty']:.3f}s (empty) to {component_results['legal_actions_with_pieces']:.3f}s (with pieces)"
        )
        print(
            f"This represents {component_results['legal_actions_empty'] / training_results['avg_step_time'] * 100:.1f}% to {component_results['legal_actions_with_pieces'] / training_results['avg_step_time'] * 100:.1f}% of step time"
        )
        print(
            f"State hashing is fast: {component_results['hash_time']:.6f}s ({component_results['hash_time'] / training_results['avg_step_time'] * 100:.2f}% of step time)"
        )

        return {
            "component_results": component_results,
            "training_results": training_results,
        }


if __name__ == "__main__":
    baseline = PerformanceBaseline()
    results = baseline.run_baseline_test()

    # Save results for future comparison
    import pickle

    with open("performance_baseline_results.pkl", "wb") as f:
        pickle.dump(results, f)

    print(f"\nBaseline results saved to 'performance_baseline_results.pkl'")
    print("Use this to compare performance after optimizations.")
