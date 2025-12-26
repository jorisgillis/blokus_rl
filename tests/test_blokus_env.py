"""Tests for the Blokus environment."""

import sys

import numpy as np
import pytest
from gymnasium import spaces

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv


def test_blokus_env_initialization():
    """Test the initialization of the Blokus environment."""
    env = BlokusEnv()
    assert env.action_space is not None
    assert env.observation_space is not None
    assert env.board is not None
    assert env.pieces is not None
    assert env.current_player == 0


def test_blokus_env_reset():
    """Test the reset method of the Blokus environment."""
    env = BlokusEnv()
    state, info = env.reset()
    assert state.shape == (20, 20, 4)
    assert np.all(state == 0)
    # Info now contains useful information
    assert "current_player" in info
    assert "available_pieces" in info
    assert "game_over" in info
    assert info["current_player"] == 0
    assert info["available_pieces"] == 21
    assert not info["game_over"]


def test_blokus_env_step():
    """Test the step method of the Blokus environment."""
    env = BlokusEnv()
    state, info = env.reset()

    # Test with a valid first move (piece 0 at corner with no rotation)
    # Piece 0 is the monomino, which should be valid at the starting corner
    action = (0, 0, 0, 0)  # piece_index=0, x=0, y=0, rotation=0
    new_state, reward, done, truncated, info = env.step(action)

    assert new_state.shape == (20, 20, 4)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    # Info should contain current_player and available_pieces
    assert "current_player" in info
    assert "available_pieces" in info
    assert "game_over" in info


def test_blokus_env_pieces():
    """Test the initialization of the pieces in the Blokus environment."""
    env = BlokusEnv()
    assert len(env.pieces) == 21
    for piece in env.pieces:
        assert isinstance(piece, list)
        for coord in piece:
            assert isinstance(coord, tuple)
            assert len(coord) == 2


def test_blokus_env_action_space():
    """Test the action space of the Blokus environment."""
    env = BlokusEnv()
    # Action space is now a tuple of (piece_index, x, y, rotation)
    assert isinstance(env.action_space, spaces.Tuple)
    assert len(env.action_space.spaces) == 4
    assert env.action_space.spaces[0].n == 21  # piece_index
    assert env.action_space.spaces[1].n == 20  # x position
    assert env.action_space.spaces[2].n == 20  # y position
    assert env.action_space.spaces[3].n == 4  # rotation


def test_blokus_env_observation_space():
    """Test the observation space of the Blokus environment."""
    env = BlokusEnv()
    assert env.observation_space.shape == (20, 20, 4)


if __name__ == "__main__":
    pytest.main([__file__])
