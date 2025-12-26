"""Basic self-play tests without PyTorch dependency."""

import sys
import numpy as np

sys.path.insert(0, ".")

from blokus_env.self_play import SelfPlay
from blokus_env.neural_network_mock import MockBlokusModel


def test_self_play_initialization():
    """Test self-play initialization."""
    self_play = SelfPlay()

    assert self_play.neural_network is not None
    assert self_play.mcts_config is not None
    assert self_play.states == []
    assert self_play.policies == []
    assert self_play.values == []

    print("Self-play initialization test passed!")


def test_self_play_with_mock_nn():
    """Test self-play with mock neural network."""
    mock_nn = MockBlokusModel()
    self_play = SelfPlay(neural_network=mock_nn)

    # Configure for fast testing
    self_play.mcts_config["max_simulations"] = 2

    # Play a single game
    game_data = self_play.play_game(temperature=1.0)

    assert "states" in game_data
    assert "policies" in game_data
    assert "values" in game_data

    # Should have collected some data
    assert len(game_data["states"]) > 0
    assert len(game_data["policies"]) > 0
    assert len(game_data["values"]) > 0

    print(f"Self-play game test passed! Collected {len(game_data['states'])} states.")


def test_final_values():
    """Test final values calculation."""
    self_play = SelfPlay()

    # Create a mock environment state
    class MockEnv:
        def __init__(self):
            self.available_pieces = [
                [True] * 5 + [False] * 16 for _ in range(4)
            ]  # 5 pieces available per player
            self.pieces = [[(0, 0)] for _ in range(21)]  # Simple pieces
            self.winner = None

    mock_env = MockEnv()

    values = self_play._get_final_values(mock_env)

    assert len(values) == 4
    assert all(isinstance(v, float) for v in values)
    assert all(0.0 <= v <= 1.0 for v in values)

    print("Final values test passed!")


def test_policy_from_mcts():
    """Test policy extraction from MCTS."""
    self_play = SelfPlay()

    # Create a simple environment state
    from blokus_env.blokus_env import BlokusEnv

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

    print("Policy from MCTS test passed!")


if __name__ == "__main__":
    test_self_play_initialization()
    test_self_play_with_mock_nn()
    test_final_values()
    test_policy_from_mcts()
    print("All self-play basic tests passed!")
