"""Tests for the self-play implementation."""

import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv
from blokus_env.neural_network import BlokusModel
from blokus_env.self_play import BlokusDataset, SelfPlay


def test_blokus_dataset():
    """Test BlokusDataset functionality."""
    # Create dummy data
    states = [np.random.rand(20, 20, 4) for _ in range(5)]
    policies = [np.random.rand(21 * 20 * 20 * 4) for _ in range(5)]
    values = [0.5, 0.6, 0.7, 0.8, 0.9]

    dataset = BlokusDataset(states, policies, values)

    assert len(dataset) == 5

    # Test getting an item
    item = dataset[0]
    assert len(item) == 3
    assert isinstance(item[0], torch.FloatTensor)
    assert isinstance(item[1], torch.FloatTensor)
    assert isinstance(item[2], torch.FloatTensor)


def test_self_play_initialization():
    """Test SelfPlay initialization."""
    self_play = SelfPlay()

    assert self_play.neural_network is not None
    assert self_play.mcts_config is not None
    assert self_play.states == []
    assert self_play.policies == []
    assert self_play.values == []

    # Test with custom neural network
    model = BlokusModel()
    self_play_custom = SelfPlay(neural_network=model)

    assert self_play_custom.neural_network == model


def test_self_play_game():
    """Test self-play game execution."""
    self_play = SelfPlay()

    # Play a single game with very short MCTS
    self_play.mcts_config["max_simulations"] = 5

    game_data = self_play.play_game(temperature=1.0)

    assert "states" in game_data
    assert "policies" in game_data
    assert "values" in game_data

    # Should have collected some data
    assert len(game_data["states"]) > 0
    assert len(game_data["policies"]) > 0
    assert len(game_data["values"]) > 0


def test_self_play_episodes():
    """Test running multiple self-play episodes."""
    self_play = SelfPlay()
    self_play.mcts_config["max_simulations"] = 2  # Very small for testing

    states, policies, values = self_play.run_self_play_episodes(
        num_episodes=2, temperature=1.0
    )

    assert len(states) > 0
    assert len(policies) > 0
    assert len(values) > 0
    assert len(states) == len(policies) == len(values)


def test_final_values():
    """Test final values calculation."""
    self_play = SelfPlay()
    env = BlokusEnv()

    # Test with a fresh environment (all pieces available)
    # This should give equal values since no pieces are placed
    values = self_play._get_final_values(env)

    assert len(values) == 4
    assert all(isinstance(v, float) for v in values)
    assert all(0.0 <= v <= 1.0 for v in values)


def test_policy_from_mcts():
    """Test policy extraction from MCTS."""
    self_play = SelfPlay()
    env = BlokusEnv()
    state, _ = env.reset()

    # Create a simple MCTS instance
    from blokus_env.mcts import MCTS

    mcts = MCTS(max_simulations=1)

    policy = self_play._get_policy_from_mcts(mcts, state, 0)

    # Policy should be a numpy array
    assert isinstance(policy, np.ndarray)

    # Should have the correct size
    expected_size = 21 * 20 * 20 * 4
    assert policy.shape[0] == expected_size


def test_data_saving_and_loading():
    """Test data saving and loading."""
    self_play = SelfPlay()

    # Create dummy data
    states = [np.random.rand(20, 20, 4) for _ in range(3)]
    policies = [np.random.rand(100) for _ in range(3)]  # Smaller for testing
    values = [0.5, 0.6, 0.7]

    filename = "test_data.npz"

    # Save data
    self_play.save_data(states, policies, values, filename)

    # Load data
    loaded_states, loaded_policies, loaded_values = self_play.load_data(filename)

    assert len(loaded_states) == 3
    assert len(loaded_policies) == 3
    assert len(loaded_values) == 3

    # Clean up
    import os

    if os.path.exists(filename):
        os.remove(filename)


if __name__ == "__main__":
    pytest.main([__file__])
