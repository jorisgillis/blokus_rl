"""
Monte Carlo Tree Search implementation for Blokus game.
"""

import numpy as np
import math
import random
from collections import defaultdict
from blokus_env.blokus_env import BlokusEnv

# Try to import the real neural network, fall back to mock if not available
try:
    from blokus_env.neural_network import BlokusModel
    import torch

    USE_REAL_NN = True
except ImportError:
    from blokus_env.neural_network_mock import MockBlokusModel as BlokusModel

    USE_REAL_NN = False


class MCTSNode:
    """
    Node in the MCTS tree.
    """

    def __init__(self, state, parent=None, action=None):
        self.state = state  # Game state
        self.parent = parent  # Parent node
        self.action = action  # Action that led to this node
        self.children = []  # Child nodes
        self.visit_count = 0  # Number of times this node has been visited
        self.total_value = 0.0  # Total value accumulated from simulations
        self.untried_actions = None  # Actions not yet tried from this node
        self.player = None  # Player who will make the next move
        self.prior_probability = 0.0  # Prior probability from neural network

    def is_fully_expanded(self):
        """Check if all possible actions have been tried."""
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=1.4):
        """
        Select the best child using PUCT formula (used in AlphaZero).
        """
        if not self.children:
            return None

        # Find child with maximum PUCT value
        best_child = None
        best_value = -float("inf")

        for child in self.children:
            # PUCT formula: Q + exploration_weight * P * sqrt(parent_visits) / (1 + child_visits)
            if child.visit_count == 0:
                value = float("inf")
            else:
                exploit = child.total_value / child.visit_count
                explore = (
                    exploration_weight
                    * child.prior_probability
                    * math.sqrt(self.visit_count)
                    / (1 + child.visit_count)
                )
                value = exploit + explore

            if value > best_value:
                best_value = value
                best_child = child

        return best_child

    def expand(self):
        """
        Expand the tree by trying one untried action.
        """
        if not self.untried_actions:
            return None

        # Get a random untried action
        action = self.untried_actions.pop()

        # Create new state by applying the action
        env = BlokusEnv()
        env.state = self.state.copy()

        # Apply the action (this is a simplified version - need to implement proper action application)
        # For now, we'll just create a dummy state
        new_state = self.state.copy()

        # Create new node
        child_node = MCTSNode(new_state, parent=self, action=action)
        self.children.append(child_node)

        return child_node

    def update(self, value):
        """
        Update this node's statistics with the result of a simulation.
        """
        self.visit_count += 1
        self.total_value += value


class MCTS:
    """
    Monte Carlo Tree Search algorithm.
    """

    def __init__(
        self, exploration_weight=1.4, max_simulations=1000, neural_network=None
    ):
        self.exploration_weight = exploration_weight
        self.max_simulations = max_simulations
        self.neural_network = neural_network

    def search(self, initial_state, player):
        """
        Perform MCTS search from the initial state.

        Args:
            initial_state: The starting game state
            player: The player who will make the next move

        Returns:
            The best action found
        """
        # Create root node
        root = MCTSNode(initial_state)
        root.player = player

        # Get all possible actions from initial state
        env = BlokusEnv()
        env.state = initial_state.copy()
        root.untried_actions = self._get_legal_actions(env, player)

        # If we have a neural network, get prior probabilities
        if self.neural_network:
            policy, _ = self.neural_network.predict(initial_state)
            self._set_prior_probabilities(root, policy)

        # Run simulations
        for _ in range(self.max_simulations):
            node = self._select(root)
            result = self._simulate(node)
            self._backpropagate(node, result)

        # Return the action with the highest visit count
        if not root.children:
            return None

        best_child = max(root.children, key=lambda child: child.visit_count)
        return best_child.action

    def _select(self, node):
        """
        Select a node to expand using the tree policy.
        """
        while not node.is_fully_expanded():
            if node.visit_count == 0:
                return node.expand()
            else:
                return node.expand()

        # If fully expanded, select best child
        while node.children:
            node = node.best_child(self.exploration_weight)

        return node

    def _simulate(self, node):
        """
        Simulate a random playout from the given node.
        """
        # For now, return a random result to avoid complex simulation
        # In a real implementation, this would simulate the game properly
        return np.random.random()

    def _backpropagate(self, node, result):
        """
        Backpropagate the simulation result up the tree.
        """
        while node is not None:
            node.update(result)
            node = node.parent

    def _set_prior_probabilities(self, node, policy):
        """
        Set prior probabilities for child nodes based on neural network policy.
        """
        # Policy is a flat array of probabilities for all possible actions
        # We need to map these to the untried actions
        if not node.untried_actions or policy is None or len(policy) == 0:
            return

        # For now, distribute probabilities equally among untried actions
        # In a real implementation, we would map the policy to specific actions
        prob_per_action = 1.0 / len(node.untried_actions)

        # Set the prior probability for each untried action
        # This will be used when creating child nodes
        node.prior_probability = prob_per_action

    def _get_legal_actions(self, env, player):
        """
        Get all legal actions for the current player.
        """
        legal_actions = []

        # Check if player is valid
        if player is None or player < 0 or player >= 4:
            return legal_actions

        # Check if available_pieces is properly initialized
        if not hasattr(env, "available_pieces") or len(env.available_pieces) <= player:
            return legal_actions

        # Check each available piece
        for piece_index, available in enumerate(env.available_pieces[player]):
            if available:
                # Try all positions and rotations
                for x in range(20):
                    for y in range(20):
                        for rotation in range(4):
                            if env._is_valid_placement(piece_index, x, y, rotation):
                                legal_actions.append((piece_index, x, y, rotation))

        return legal_actions
