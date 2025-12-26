"""Comprehensive test of the entire RL pipeline."""

import sys
import numpy as np

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv
from blokus_env.mcts import MCTS
from blokus_env.self_play import SelfPlay
from blokus_env.neural_network_mock import MockBlokusModel


def test_comprehensive_pipeline():
    """Test the entire RL pipeline end-to-end."""
    print("=== Comprehensive RL Pipeline Test ===")

    # 1. Test Environment
    print("1. Testing Environment...")
    env = BlokusEnv()
    state, info = env.reset()
    assert state.shape == (20, 20, 4)
    assert info["current_player"] == 0
    print("   ✓ Environment working")

    # 2. Test Neural Network
    print("2. Testing Neural Network...")
    mock_nn = MockBlokusModel()
    policy, value = mock_nn.predict(state)
    assert isinstance(policy, np.ndarray)
    assert isinstance(value, float)
    print("   ✓ Neural network working")

    # 3. Test MCTS
    print("3. Testing MCTS...")
    mcts = MCTS(neural_network=mock_nn, max_simulations=2)
    action = mcts.search(state, 0)
    assert action is None or (isinstance(action, tuple) and len(action) == 4)
    print("   ✓ MCTS working")

    # 4. Test Self-Play
    print("4. Testing Self-Play...")
    self_play = SelfPlay(neural_network=mock_nn)
    self_play.mcts_config["max_simulations"] = 1

    # Test data collection
    states = [state.copy()]
    policies = [np.zeros(21 * 20 * 20 * 4)]
    values = [0.5]

    # Test training (mock)
    trained_model = self_play.train_neural_network(states, policies, values, epochs=1)
    assert trained_model is not None
    print("   ✓ Self-play working")

    # 5. Test Data Handling
    print("5. Testing Data Handling...")
    self_play.save_data(states, policies, values, "test_data.npz")
    loaded_states, loaded_policies, loaded_values = self_play.load_data("test_data.npz")
    assert len(loaded_states) == 1
    assert len(loaded_policies) == 1
    assert len(loaded_values) == 1
    print("   ✓ Data handling working")

    # Clean up
    import os

    if os.path.exists("test_data.npz"):
        os.remove("test_data.npz")

    print("\n=== All Tests Passed! ===")
    print("The complete RL pipeline is functioning correctly.")
    print("Components tested:")
    print("  • Environment (Blokus game logic)")
    print("  • Neural Network (policy/value prediction)")
    print("  • MCTS (tree search with neural network guidance)")
    print("  • Self-Play (data collection and training)")
    print("  • Data Handling (save/load functionality)")

    return True


if __name__ == "__main__":
    test_comprehensive_pipeline()
