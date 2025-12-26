"""Tests for the Q-Learning implementation."""

import sys
import numpy as np

sys.path.insert(0, ".")

from blokus_env.q_learning import QLearningAgent
from blokus_env.blokus_env import BlokusEnv


def test_q_learning_initialization():
    """Test Q-Learning agent initialization."""
    agent = QLearningAgent()

    assert agent.learning_rate == 0.1
    assert agent.discount_factor == 0.9
    assert agent.exploration_rate == 1.0
    assert agent.exploration_decay == 0.995
    assert agent.min_exploration == 0.01
    assert agent.episodes == 0
    assert agent.wins == 0
    assert agent.losses == 0
    assert agent.ties == 0

    print("Q-Learning initialization test passed!")


def test_state_hashing():
    """Test state hashing functionality."""
    agent = QLearningAgent()
    env = BlokusEnv()

    # Test with initial state
    state, _ = env.reset()
    state_hash = agent.get_state_hash(state, 0)

    assert isinstance(state_hash, tuple)
    assert len(state_hash) == 2  # (board_flattened, player)
    assert isinstance(state_hash[0], tuple)
    assert isinstance(state_hash[1], int)

    print("State hashing test passed!")


def test_legal_actions():
    """Test legal actions detection."""
    agent = QLearningAgent()
    env = BlokusEnv()
    state, _ = env.reset()

    legal_actions = agent.get_legal_actions(env, 0)

    assert isinstance(legal_actions, list)
    assert len(legal_actions) > 0  # Should have legal first moves

    # Check action format
    if legal_actions:
        action = legal_actions[0]
        assert isinstance(action, tuple)
        assert len(action) == 4
        assert all(isinstance(x, int) for x in action)

    print("Legal actions test passed!")


def test_action_selection():
    """Test action selection (exploration vs exploitation)."""
    agent = QLearningAgent()
    env = BlokusEnv()
    state, _ = env.reset()

    # Test exploration (should return random action)
    agent.exploration_rate = 1.0  # Force exploration
    action = agent.choose_action(env, state, 0, training=True)
    assert action is not None
    assert isinstance(action, tuple)

    # Test exploitation (should return best action or random if no Q-values)
    agent.exploration_rate = 0.0  # Force exploitation
    action = agent.choose_action(env, state, 0, training=False)
    assert action is not None
    assert isinstance(action, tuple)

    print("Action selection test passed!")


def test_q_table_update():
    """Test Q-table update functionality."""
    agent = QLearningAgent()
    env = BlokusEnv()

    # Create a simple state and action
    state, _ = env.reset()
    action = (0, 0, 0, 0)  # Simple action

    # Initialize Q-table entry for the state
    state_hash = agent.get_state_hash(state, 0)
    if state_hash not in agent.q_table:
        agent.q_table[state_hash] = {}

    # Initial Q-value should be 0
    initial_q = agent.q_table[state_hash].get(action, 0)
    assert initial_q == 0

    # Update Q-table
    reward = 1.0
    next_state = state.copy()  # Simple next state
    agent.update_q_table(state, action, reward, next_state, 0, done=True)

    # Q-value should be updated
    updated_q = agent.q_table[state_hash].get(action, 0)
    assert updated_q > initial_q

    print("Q-table update test passed!")


def test_save_load():
    """Test saving and loading Q-learning agent."""
    agent = QLearningAgent()

    # Train for a few steps to populate Q-table
    env = BlokusEnv()
    state, _ = env.reset()
    legal_actions = agent.get_legal_actions(env, 0)

    if legal_actions:
        action = legal_actions[0]
        agent.update_q_table(state, action, 1.0, state.copy(), 0, done=True)

    # Save agent
    filename = "test_q_agent.pkl"
    agent.save(filename)

    # Load agent
    new_agent = QLearningAgent()
    success = new_agent.load(filename)
    assert success

    # Check that data was loaded correctly
    assert new_agent.episodes == agent.episodes
    assert new_agent.wins == agent.wins

    # Clean up
    import os

    if os.path.exists(filename):
        os.remove(filename)

    print("Save/load test passed!")


def test_q_table_size():
    """Test Q-table size calculation."""
    agent = QLearningAgent()

    # Initially empty
    state_count, action_count = agent.get_q_table_size()
    assert state_count == 0
    assert action_count == 0

    # Add some entries
    env = BlokusEnv()
    state, _ = env.reset()
    legal_actions = agent.get_legal_actions(env, 0)

    if legal_actions:
        for action in legal_actions[:3]:  # Add first 3 actions
            agent.update_q_table(state, action, 1.0, state.copy(), 0, done=True)

    # Check size
    state_count, action_count = agent.get_q_table_size()
    assert state_count >= 1
    assert action_count >= 1

    print("Q-table size test passed!")


def test_simple_training():
    """Test a simple training run."""
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.9,
        min_exploration=0.1,
    )

    # Train for just 1 episode with very few steps
    agent.train(episodes=1, max_steps=5)

    assert agent.episodes == 1
    assert agent.wins + agent.losses + agent.ties == 1

    print("Simple training test passed!")


if __name__ == "__main__":
    test_q_learning_initialization()
    test_state_hashing()
    test_legal_actions()
    test_action_selection()
    test_q_table_update()
    test_save_load()
    test_q_table_size()
    test_simple_training()
    print("All Q-Learning tests passed!")
