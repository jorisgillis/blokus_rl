"""Simple MCTS test to debug issues."""

import sys

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv
from blokus_env.mcts import MCTS


def test_simple_mcts():
    """Test simple MCTS functionality."""
    print("Creating MCTS...")
    mcts = MCTS(max_simulations=2)  # Very small number

    print("Creating environment...")
    env = BlokusEnv()
    state, _ = env.reset()

    print("Testing legal actions...")
    legal_actions = mcts._get_legal_actions(env, 0)
    print(f"Found {len(legal_actions)} legal actions")

    print("Testing MCTS search...")
    # This might take a while, so let's limit it
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("MCTS search timed out")

    # Set a 10-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    try:
        action = mcts.search(state, 0)
        signal.alarm(0)  # Cancel the alarm

        print(f"MCTS search completed with action: {action}")
        assert action is None or (isinstance(action, tuple) and len(action) == 4)
        print("Simple MCTS test passed!")

    except TimeoutError:
        signal.alarm(0)  # Cancel the alarm
        print("MCTS search timed out - there might be an infinite loop")
        return False

    return True


if __name__ == "__main__":
    test_simple_mcts()
