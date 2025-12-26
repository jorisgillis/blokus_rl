"""Basic functionality tests that don't require PyTorch."""

import sys
import numpy as np

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv


def test_basic_env_functionality():
    """Test basic environment functionality."""
    env = BlokusEnv()

    # Test reset
    state, info = env.reset()
    assert state.shape == (20, 20, 4)
    assert info["current_player"] == 0
    assert not info["game_over"]

    # Test a simple move
    action = (0, 0, 0, 0)  # piece 0 (monomino) at (0,0) with rotation 0
    new_state, reward, done, truncated, info = env.step(action)

    assert new_state.shape == (20, 20, 4)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)

    print("Basic environment functionality test passed!")


def test_legal_actions_detection():
    """Test legal actions detection."""
    env = BlokusEnv()
    state, _ = env.reset()

    # Test first move - should be legal in corner
    assert env._is_valid_placement(0, 0, 0, 0)  # Monomino at blue corner

    # Test invalid placement (out of bounds)
    assert not env._is_valid_placement(0, -1, 0, 0)  # Negative x
    assert not env._is_valid_placement(0, 0, 20, 0)  # y = 20 (out of bounds)

    print("Legal actions detection test passed!")


def test_game_progression():
    """Test basic game progression."""
    env = BlokusEnv()
    state, _ = env.reset()

    # Make a few valid moves
    moves = [
        (0, 0, 0, 0),  # Blue: monomino at corner
        (0, 0, 19, 0),  # Yellow: monomino at corner
        (0, 19, 19, 0),  # Red: monomino at corner
        (0, 19, 0, 0),  # Green: monomino at corner
    ]

    for i, move in enumerate(moves):
        state, reward, done, truncated, info = env.step(move)
        assert reward > 0  # Should get positive reward for placing piece
        assert not done  # Game shouldn't be over yet

    print("Game progression test passed!")


if __name__ == "__main__":
    test_basic_env_functionality()
    test_legal_actions_detection()
    test_game_progression()
    print("All basic functionality tests passed!")
