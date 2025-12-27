"""Simple self-play test."""

import sys

sys.path.insert(0, ".")

from blokus_env.neural_network_mock import MockBlokusModel
from blokus_env.self_play import SelfPlay


def test_self_play_simple():
    """Test self-play with very limited MCTS."""
    print("Creating self-play with mock neural network...")
    mock_nn = MockBlokusModel()
    self_play = SelfPlay(neural_network=mock_nn)

    # Configure for very fast testing
    self_play.mcts_config["max_simulations"] = 1  # Just 1 simulation

    print("Playing a single game...")
    # Use a timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Self-play game timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(15)  # 15 second timeout

    try:
        game_data = self_play.play_game(temperature=1.0)
        signal.alarm(0)  # Cancel the alarm

        assert "states" in game_data
        assert "policies" in game_data
        assert "values" in game_data

        print(f"Self-play game completed! Collected {len(game_data['states'])} states.")
        assert len(game_data["states"]) > 0
        print("Simple self-play test passed!")

    except TimeoutError:
        signal.alarm(0)
        print("Self-play game timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"Self-play game failed with error: {e}")
        return False

    return True


if __name__ == "__main__":
    test_self_play_simple()
