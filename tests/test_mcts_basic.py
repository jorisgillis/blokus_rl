"""Basic MCTS tests without PyTorch dependency."""

import sys
import numpy as np

sys.path.insert(0, ".")

from blokus_env.mcts import MCTS, MCTSNode
from blokus_env.blokus_env import BlokusEnv


class MockNeuralNetwork:
    """Mock neural network for testing."""

    def predict(self, board_state):
        # Return dummy policy and value
        policy_size = 21 * 20 * 20 * 4
        policy = np.ones(policy_size) / policy_size  # Uniform distribution
        value = 0.5  # Neutral value
        return policy, value


def test_mcts_node_basic():
    """Test basic MCTS node functionality."""
    env = BlokusEnv()
    state, _ = env.reset()

    node = MCTSNode(state)

    # Test initial state
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.prior_probability == 0.0

    # Test update
    node.update(0.7)
    assert node.visit_count == 1
    assert node.total_value == 0.7

    print("MCTS node basic test passed!")


def test_mcts_legal_actions():
    """Test MCTS legal actions detection."""
    mcts = MCTS()
    env = BlokusEnv()
    state, _ = env.reset()

    legal_actions = mcts._get_legal_actions(env, 0)

    assert isinstance(legal_actions, list)
    assert len(legal_actions) > 0  # Should have legal first moves

    # Check action format
    if legal_actions:
        action = legal_actions[0]
        assert isinstance(action, tuple)
        assert len(action) == 4
        assert all(isinstance(x, int) for x in action)

    print("MCTS legal actions test passed!")


def test_mcts_search_basic():
    """Test basic MCTS search functionality."""
    mcts = MCTS(max_simulations=5)  # Small number for testing
    env = BlokusEnv()
    state, _ = env.reset()

    # This should work without errors
    action = mcts.search(state, 0)

    # Action could be None or a valid action tuple
    assert action is None or (isinstance(action, tuple) and len(action) == 4)

    print("MCTS basic search test passed!")


def test_mcts_with_mock_neural_network():
    """Test MCTS with mock neural network."""
    mock_nn = MockNeuralNetwork()
    mcts = MCTS(neural_network=mock_nn, max_simulations=3)

    env = BlokusEnv()
    state, _ = env.reset()

    # This should work without errors
    action = mcts.search(state, 0)

    assert action is None or (isinstance(action, tuple) and len(action) == 4)

    print("MCTS with mock neural network test passed!")


def test_mcts_node_expansion():
    """Test MCTS node expansion."""
    env = BlokusEnv()
    state, _ = env.reset()

    node = MCTSNode(state)
    node.untried_actions = [(0, 0, 0, 0)]  # Add a simple action

    # Test expansion
    child_node = node.expand()

    assert child_node is not None
    assert child_node in node.children
    assert child_node.parent == node
    assert child_node.action == (0, 0, 0, 0)

    print("MCTS node expansion test passed!")


if __name__ == "__main__":
    test_mcts_node_basic()
    test_mcts_legal_actions()
    test_mcts_search_basic()
    test_mcts_with_mock_neural_network()
    test_mcts_node_expansion()
    print("All MCTS basic tests passed!")
