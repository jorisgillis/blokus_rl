"""Tests for the MCTS implementation."""

import sys
import numpy as np
import pytest

sys.path.insert(0, ".")

from blokus_env.mcts import MCTS, MCTSNode
from blokus_env.blokus_env import BlokusEnv
from blokus_env.neural_network import BlokusModel


def test_mcts_node_initialization():
    """Test MCTSNode initialization."""
    env = BlokusEnv()
    state, _ = env.reset()

    node = MCTSNode(state)

    assert node.state is not None
    assert node.parent is None
    assert node.action is None
    assert node.children == []
    assert node.visit_count == 0
    assert node.total_value == 0.0
    assert node.untried_actions is None
    assert node.player is None
    assert node.prior_probability == 0.0


def test_mcts_initialization():
    """Test MCTS initialization."""
    mcts = MCTS()

    assert mcts.exploration_weight == 1.4
    assert mcts.max_simulations == 1000
    assert mcts.neural_network is None

    # Test with neural network
    model = BlokusModel()
    mcts_with_nn = MCTS(neural_network=model)

    assert mcts_with_nn.neural_network is not None


def test_mcts_legal_actions():
    """Test MCTS legal actions detection."""
    mcts = MCTS()
    env = BlokusEnv()
    state, _ = env.reset()

    # For initial state, there should be legal actions (first piece placements)
    legal_actions = mcts._get_legal_actions(env, 0)

    assert isinstance(legal_actions, list)
    # Should have at least some legal actions for the first move
    assert len(legal_actions) > 0

    # Check that actions are in the correct format
    if legal_actions:
        action = legal_actions[0]
        assert isinstance(action, tuple)
        assert len(action) == 4
        assert all(isinstance(x, int) for x in action)


def test_mcts_search():
    """Test MCTS search functionality."""
    mcts = MCTS(max_simulations=10)  # Small number for testing
    env = BlokusEnv()
    state, _ = env.reset()

    # Search should return an action or None
    action = mcts.search(state, 0)

    assert action is None or (isinstance(action, tuple) and len(action) == 4)


def test_mcts_with_neural_network():
    """Test MCTS with neural network integration."""
    model = BlokusModel()
    mcts = MCTS(neural_network=model, max_simulations=5)

    env = BlokusEnv()
    state, _ = env.reset()

    # This should work without errors
    action = mcts.search(state, 0)

    assert action is None or (isinstance(action, tuple) and len(action) == 4)


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


def test_mcts_node_update():
    """Test MCTS node update."""
    env = BlokusEnv()
    state, _ = env.reset()

    node = MCTSNode(state)

    # Initial state
    assert node.visit_count == 0
    assert node.total_value == 0.0

    # Update with a value
    node.update(0.5)

    assert node.visit_count == 1
    assert node.total_value == 0.5

    # Update again
    node.update(0.7)

    assert node.visit_count == 2
    assert node.total_value == 1.2  # 0.5 + 0.7


if __name__ == "__main__":
    pytest.main([__file__])
