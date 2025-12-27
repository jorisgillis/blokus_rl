"""Debug self-play to find the issue."""

import sys

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv
from blokus_env.mcts import MCTS
from blokus_env.neural_network_mock import MockBlokusModel
from blokus_env.self_play import SelfPlay


def debug_self_play():
    """Debug self-play step by step."""
    print("Creating self-play with mock neural network...")
    mock_nn = MockBlokusModel()
    self_play = SelfPlay(neural_network=mock_nn)
    self_play.mcts_config["max_simulations"] = 1

    print("Creating environment...")
    env = BlokusEnv()
    state, _ = env.reset()

    print("Testing MCTS search...")
    mcts = MCTS(neural_network=mock_nn, max_simulations=1)

    # Test with timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("MCTS search timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    try:
        action = mcts.search(state, 0)
        signal.alarm(0)
        print(f"MCTS search completed with action: {action}")

        if action:
            print("Applying action...")
            new_state, reward, done, truncated, info = env.step(action)
            print(f"Action applied. Reward: {reward}, Done: {done}")

        print("Debug completed successfully!")

    except TimeoutError:
        signal.alarm(0)
        print("MCTS search timed out")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"Debug failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    debug_self_play()
