from argparse import Namespace

from blokus_env.q_learning import QLearningAgent


def main():
    print("Blokus Reinforcement Learning Training")
    print("=" * 50)

    args = parse_arguments()
    if args.method == "q_learning":
        q_learning(args)
    elif args.method == "deep_rl":
        deep_rl(args)

    print("Training complete!")


def q_learning(args: Namespace) -> None:
    import os

    print("Using Q-Learning method")

    # Initialize Q-Learning agent
    if args.load and os.path.exists(args.load):
        print(f"Loading Q-learning agent from {args.load}")
        agent = QLearningAgent()
        agent.load(args.load)
    else:
        print("Initializing new Q-learning agent")
        agent = QLearningAgent()

    # Train the agent
    print(f"Training Q-learning agent for {args.episodes} episodes...")
    agent.train(episodes=args.episodes, max_steps=50, parallel=args.parallel)

    # Save the trained agent
    if args.save:
        agent.save(args.save)
        print(f"Q-learning agent saved to {args.save}")

    # Show final statistics
    state_count, action_count = agent.get_q_table_size()
    print(f"Q-table size: {state_count} states, {action_count} state-action pairs")
    print(f"Final win rate: {agent.wins / agent.episodes:.2f}")

    # Play a demonstration game
    if args.render:
        print("\nPlaying demonstration game...")
        agent.play_game(render=True)


def deep_rl(args: Namespace) -> None:
    print("Using Deep Reinforcement Learning method")

    # Import the necessary modules
    import os

    from blokus_env.self_play import SelfPlay

    # Try to import the real neural network, fall back to mock if not available
    try:
        from blokus_env.neural_network import BlokusModel
    except ImportError:
        from blokus_env.neural_network_mock import MockBlokusModel as BlokusModel

        print("Using mock neural network (PyTorch not available)")

    # Initialize neural network
    if args.load and os.path.exists(args.load):
        print(f"Loading model from {args.load}")
        model = BlokusModel()
        model.load(args.load)
    else:
        print("Initializing new model")
        model = BlokusModel()

    # Initialize self-play
    self_play = SelfPlay(neural_network=model)

    # Run self-play episodes
    print(f"Running {args.episodes} self-play episodes...")
    states, policies, values = self_play.run_self_play_episodes(
        num_episodes=args.episodes, temperature=1.0
    )

    # Save the collected data
    self_play.save_data(states, policies, values, args.data)

    # Train the neural network
    print(f"Training neural network for {args.epochs} epochs...")
    trained_model = self_play.train_neural_network(
        states, policies, values, epochs=args.epochs, batch_size=32
    )

    # Save the trained model
    if args.save:
        trained_model.save(args.save)
        print(f"Model saved to {args.save}")


def parse_arguments():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Blokus RL Training")
    parser.add_argument(
        "--method",
        type=str,
        default="q_learning",
        choices=["q_learning", "deep_rl"],
        help="Training method: q_learning or deep_rl",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of training episodes (for Q-learning) or self-play episodes (for deep RL)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs (for deep RL)"
    )
    parser.add_argument("--load", type=str, help="Load model from file")
    parser.add_argument("--save", type=str, help="Save model to file")
    parser.add_argument(
        "--data", type=str, default="blokus_data.npz", help="Data file (for deep RL)"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render game during Q-learning training"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Use parallel Q-learning agent (faster but uses more memory)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
