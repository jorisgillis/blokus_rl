"""
Self-play implementation for Blokus reinforcement learning.
"""

import random

import numpy as np

from blokus_env.blokus_env import BlokusEnv
from blokus_env.mcts import MCTS

# Try to import the real neural network, fall back to mock if not available
try:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from blokus_env.neural_network import BlokusModel

    USE_REAL_NN = True
except ImportError:
    from blokus_env.neural_network_mock import MockBlokusModel as BlokusModel

    # Create mock Dataset and DataLoader
    class Dataset:
        def __init__(self, states, policies, values):
            self.states = states
            self.policies = policies
            self.values = values

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (self.states[idx], self.policies[idx], self.values[idx])

    class DataLoader:
        def __init__(self, dataset, batch_size=32, shuffle=False):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                batch = [self.dataset[idx] for idx in batch_indices]
                yield batch

    USE_REAL_NN = False



class BlokusDataset(Dataset):
    """
    Dataset for storing and loading Blokus game data.
    """

    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.policies[idx]),
            torch.FloatTensor([self.values[idx]]),
        )


class SelfPlay:
    """
    Self-play implementation for Blokus.
    """

    def __init__(self, neural_network=None, mcts_config=None):
        self.neural_network = neural_network or BlokusModel()
        self.mcts_config = mcts_config or {
            "exploration_weight": 1.4,
            "max_simulations": 100,
        }

        # Training data storage
        self.states = []
        self.policies = []
        self.values = []

    def play_game(self, temperature=1.0):
        """
        Play a single game of self-play.

        Args:
            temperature: Temperature parameter for action selection

        Returns:
            Game data (states, policies, values)
        """
        env = BlokusEnv()
        state, info = env.reset()

        game_data = {"states": [], "policies": [], "values": []}

        while not env.game_over:
            current_player = env.current_player

            # Create MCTS with the current neural network
            mcts = MCTS(
                exploration_weight=self.mcts_config["exploration_weight"],
                max_simulations=self.mcts_config["max_simulations"],
                neural_network=self.neural_network,
            )

            # Get MCTS policy
            action = mcts.search(state, current_player)

            if action is None:
                # No valid moves, skip turn
                env.current_player = (env.current_player + 1) % 4
                continue

            # Get visit counts from MCTS root node
            # For now, we'll use a simplified approach
            # In a real implementation, we would get the full policy from MCTS
            policy = self._get_policy_from_mcts(mcts, state, current_player)

            # Store the state and policy
            game_data["states"].append(state.copy())
            game_data["policies"].append(policy)

            # Apply the action
            new_state, reward, done, truncated, info = env.step(action)

            state = new_state

        # Game is over, determine final values
        final_values = self._get_final_values(env)

        # Assign values to each state
        for i in range(len(game_data["states"])):
            # For now, use the final value for all states
            # In a real implementation, this would be more sophisticated
            game_data["values"].append(final_values[env.current_player])

        return game_data

    def _get_policy_from_mcts(self, mcts, state, player):
        """
        Get policy from MCTS visit counts.
        """
        # This is a simplified version
        # In a real implementation, we would get the full policy from MCTS

        # For now, return a uniform policy
        env = BlokusEnv()
        env.state = state.copy()
        legal_actions = mcts._get_legal_actions(env, player)

        if not legal_actions:
            return []

        # Create a policy vector with uniform probabilities
        policy_size = 21 * 20 * 20 * 4  # Total possible actions
        policy = np.zeros(policy_size)

        # Set equal probability for all legal actions
        prob_per_action = 1.0 / len(legal_actions)

        for action in legal_actions:
            # Convert action to flat index
            piece_index, x, y, rotation = action
            flat_index = piece_index * 20 * 20 * 4 + x * 20 * 4 + y * 4 + rotation
            policy[flat_index] = prob_per_action

        return policy

    def _get_final_values(self, env):
        """
        Get final values for each player based on game outcome.
        """
        # Calculate remaining squares for each player
        remaining_squares = []
        for player in range(4):
            count = 0
            for piece_index, available in enumerate(env.available_pieces[player]):
                if available:
                    count += len(env.pieces[piece_index])
            remaining_squares.append(count)

        # Normalize to [0, 1] range
        min_squares = min(remaining_squares)
        max_squares = max(remaining_squares)

        if max_squares == min_squares:
            # All players have same score (tie)
            return [0.5 for _ in range(4)]

        # Convert to values (lower remaining squares = higher value)
        values = []
        for squares in remaining_squares:
            # Normalize: 1.0 for best, 0.0 for worst
            normalized = (max_squares - squares) / (max_squares - min_squares)
            values.append(normalized)

        return values

    def run_self_play_episodes(self, num_episodes=10, temperature=1.0):
        """
        Run multiple episodes of self-play.
        """
        all_states = []
        all_policies = []
        all_values = []

        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")

            game_data = self.play_game(temperature)

            all_states.extend(game_data["states"])
            all_policies.extend(game_data["policies"])
            all_values.extend(game_data["values"])

            # Print some stats
            print(f"  States collected: {len(game_data['states'])}")
            print(f"  Total states: {len(all_states)}")

        return all_states, all_policies, all_values

    def train_neural_network(self, states, policies, values, epochs=10, batch_size=32):
        """
        Train the neural network on collected data.
        """
        # Create dataset and dataloader
        dataset = BlokusDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Create optimizer (only if using real neural network)
        if USE_REAL_NN:
            optimizer = torch.optim.Adam(
                self.neural_network.model.parameters(), lr=0.001
            )
        else:
            # Mock optimizer for testing
            class MockOptimizer:
                def step(self):
                    pass

                def zero_grad(self):
                    pass

            optimizer = MockOptimizer()

        # Train
        print("Training neural network...")
        self.neural_network.train(dataloader, optimizer, epochs=epochs)

        return self.neural_network

    def save_data(self, states, policies, values, filename="blokus_data.npz"):
        """
        Save training data to file.
        """
        np.savez(
            filename,
            states=np.array(states, dtype=object),
            policies=np.array(policies, dtype=object),
            values=np.array(values),
        )
        print(f"Data saved to {filename}")

    def load_data(self, filename="blokus_data.npz"):
        """
        Load training data from file.
        """
        data = np.load(filename, allow_pickle=True)
        return data["states"], data["policies"], data["values"]
