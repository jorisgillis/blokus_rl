"""Debug MCTS to find infinite loop."""

import sys

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv
from blokus_env.mcts import MCTS, MCTSNode


def debug_mcts():
    """Debug MCTS step by step."""
    print("Creating MCTS...")
    mcts = MCTS(max_simulations=1)  # Just 1 simulation

    print("Creating environment...")
    env = BlokusEnv()
    state, _ = env.reset()

    print("Creating root node...")
    root = MCTSNode(state)
    root.player = 0

    print("Getting legal actions...")
    root.untried_actions = mcts._get_legal_actions(env, 0)
    print(f"Found {len(root.untried_actions)} legal actions")

    print("Testing node expansion...")
    if root.untried_actions:
        child_node = root.expand()
        print(f"Expanded to child node with action: {child_node.action}")
    else:
        print("No actions to expand")
        return

    print("Testing simulation...")
    # Test simulation with timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Simulation timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout

    try:
        result = mcts._simulate(child_node)
        signal.alarm(0)
        print(f"Simulation completed with result: {result}")
    except TimeoutError:
        signal.alarm(0)
        print("Simulation timed out")
        return
    except Exception as e:
        signal.alarm(0)
        print(f"Simulation failed with error: {e}")
        return

    print("Testing backpropagation...")
    mcts._backpropagate(child_node, result)
    print(f"Root visit count: {root.visit_count}")
    print(f"Child visit count: {child_node.visit_count}")

    print("Debug completed successfully!")


if __name__ == "__main__":
    debug_mcts()
