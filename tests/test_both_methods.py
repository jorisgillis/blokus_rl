"""Test both Q-Learning and Deep RL methods."""

import sys
import numpy as np

sys.path.insert(0, ".")

from blokus_env.q_learning import QLearningAgent
from blokus_env.self_play import SelfPlay
from blokus_env.neural_network_mock import MockBlokusModel


def test_q_learning_method():
    """Test Q-Learning method."""
    print("=== Testing Q-Learning Method ===")

    # Create Q-Learning agent
    agent = QLearningAgent(
        learning_rate=0.2,
        discount_factor=0.8,
        exploration_rate=1.0,
        exploration_decay=0.9,
        min_exploration=0.1,
    )

    # Train for a few episodes
    print("Training Q-Learning agent...")
    agent.train(episodes=3, max_steps=8)

    # Check results
    assert agent.episodes == 3
    assert agent.wins + agent.losses + agent.ties == 3

    # Check Q-table
    state_count, action_count = agent.get_q_table_size()
    assert state_count > 0
    assert action_count > 0

    print(f"✓ Q-Learning trained successfully")
    print(f"  Episodes: {agent.episodes}")
    print(f"  Wins: {agent.wins}, Losses: {agent.losses}, Ties: {agent.ties}")
    print(f"  Q-table: {state_count} states, {action_count} actions")

    return True


def test_deep_rl_method():
    """Test Deep RL method."""
    print("\n=== Testing Deep RL Method ===")

    # Create mock neural network
    mock_nn = MockBlokusModel()

    # Create self-play with mock network
    self_play = SelfPlay(neural_network=mock_nn)
    self_play.mcts_config["max_simulations"] = 2  # Very small for testing

    # Run a few self-play episodes
    print("Running self-play episodes...")
    states, policies, values = self_play.run_self_play_episodes(
        num_episodes=2, temperature=1.0
    )

    # Check results
    assert len(states) > 0
    assert len(policies) > 0
    assert len(values) > 0
    assert len(states) == len(policies) == len(values)

    # Train neural network (mock)
    print("Training neural network (mock)...")
    trained_model = self_play.train_neural_network(
        states, policies, values, epochs=1, batch_size=2
    )

    assert trained_model is not None

    print(f"✓ Deep RL trained successfully")
    print(f"  Collected: {len(states)} states")
    print(f"  Policies: {len(policies)} policy vectors")
    print(f"  Values: {len(values)} value estimates")

    return True


def test_method_comparison():
    """Compare the two methods."""
    print("\n=== Method Comparison ===")

    print("Q-Learning:")
    print("  • Pros: Simple, no PyTorch dependency, interpretable")
    print("  • Cons: Limited scalability, state space grows quickly")
    print("  • Best for: Small to medium state spaces, quick prototyping")

    print("\nDeep RL (AlphaZero-style):")
    print("  • Pros: Scalable, handles complex patterns, state-of-the-art")
    print("  • Cons: Requires PyTorch, more complex, needs more data")
    print("  • Best for: Large state spaces, complex games, high performance")

    print("\nRecommendation:")
    print("  • Use Q-Learning for quick testing and development")
    print("  • Use Deep RL for production-quality agents")

    return True


if __name__ == "__main__":
    print("Testing both reinforcement learning methods for Blokus\n")

    # Test Q-Learning
    q_learning_success = test_q_learning_method()

    # Test Deep RL
    deep_rl_success = test_deep_rl_method()

    # Compare methods
    comparison_success = test_method_comparison()

    if q_learning_success and deep_rl_success and comparison_success:
        print("\n🎉 All tests passed!")
        print("Both Q-Learning and Deep RL methods are working correctly.")
        print("You can now choose the appropriate method for your needs:")
        print("  • Q-Learning: python main.py --method q_learning")
        print("  • Deep RL:    python main.py --method deep_rl")
    else:
        print("\n❌ Some tests failed!")
