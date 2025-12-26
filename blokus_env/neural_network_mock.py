"""
Mock neural network implementation for testing without PyTorch.
"""

import numpy as np


class MockBlokusNet:
    """
    Mock neural network for Blokus game (no PyTorch dependency).
    """

    def __init__(self):
        # Mock parameters - not actually used
        pass

    def forward(self, x):
        """Mock forward pass."""
        # Return dummy policy and value
        batch_size = x.shape[0] if len(x.shape) == 4 else 1
        policy_size = 21 * 20 * 20 * 4

        # Uniform policy
        policy = np.ones((batch_size, policy_size)) / policy_size

        # Neutral value
        value = np.array([[0.5]] * batch_size)

        return policy, value


class MockBlokusModel:
    """
    Mock wrapper class for testing.
    """

    def __init__(self):
        self.model = MockBlokusNet()
        self.device = "cpu"

    def predict(self, board_state):
        """
        Mock prediction - returns uniform policy and neutral value.
        """
        # Convert to mock tensor format
        if len(board_state.shape) == 3:
            # Add batch dimension
            board_tensor = np.expand_dims(board_state, axis=0)
            # Permute to (batch, channels, height, width)
            board_tensor = np.transpose(board_tensor, (0, 3, 1, 2))
        else:
            board_tensor = board_state

        # Get mock predictions
        policy, value = self.model.forward(board_tensor)

        # Convert to numpy
        policy = policy.flatten()
        value = float(value.flatten()[0])

        return policy, value

    def train(self, data_loader, optimizer, epochs=10):
        """Mock training - does nothing."""
        print(f"Mock training for {epochs} epochs (no actual training)")

    def save(self, path):
        """Mock save - does nothing."""
        print(f"Mock save to {path}")

    def load(self, path):
        """Mock load - does nothing."""
        print(f"Mock load from {path}")
