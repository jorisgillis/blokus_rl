import torch
import torch.nn as nn
import torch.nn.functional as F


class BlokusNet(nn.Module):
    """
    Neural network for Blokus game.
    Input: 20x20x4 board state (4 players)
    Output: Policy (probability distribution over moves) and value (game outcome)
    """

    def __init__(self):
        super(BlokusNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(256, 4, kernel_size=1)
        self.policy_fc = nn.Linear(
            4 * 20 * 20, 21 * 20 * 20 * 4
        )  # 21 pieces, 20x20 positions, 4 rotations

        # Value head
        self.value_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * 20 * 20, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Input shape: (batch_size, 4, 20, 20)
        # x = x.permute(0, 3, 1, 2)  # Change from (batch, 20, 20, 4) to (batch, 4, 20, 20)

        # Common convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value


class BlokusModel:
    """
    Wrapper class for the Blokus neural network model.
    Handles input/output processing and model management.
    """

    def __init__(self):
        self.model = BlokusNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, board_state):
        """
        Predict policy and value for a given board state.

        Args:
            board_state: numpy array of shape (20, 20, 4)

        Returns:
            policy: numpy array of shape (21 * 20 * 20 * 4,) representing move probabilities
            value: float representing the estimated game outcome
        """
        # Convert to tensor and add batch dimension
        board_tensor = torch.FloatTensor(board_state).unsqueeze(0).to(self.device)

        # Permute dimensions to (batch, channels, height, width)
        board_tensor = board_tensor.permute(0, 3, 1, 2)

        # Get predictions
        with torch.no_grad():
            policy, value = self.model(board_tensor)

        # Convert to numpy
        policy = policy.cpu().numpy().flatten()
        value = value.cpu().numpy().item()

        return policy, value

    def train(self, data_loader, optimizer, epochs=10):
        """
        Train the model using the given data loader.

        Args:
            data_loader: PyTorch DataLoader providing training data
            optimizer: PyTorch optimizer
            epochs: Number of training epochs
        """
        self.model.train()
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                # Unpack batch
                board_states, target_policies, target_values = batch

                # Convert to tensors and move to device
                board_states = board_states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)

                # Forward pass
                policies, values = self.model(board_states)

                # Calculate losses
                policy_loss = criterion(policies, target_policies)
                value_loss = criterion(values, target_values)
                total_loss = policy_loss + value_loss

                # Backward pass and optimize
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")

    def save(self, path):
        """Save the model to the specified path."""
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """Load the model from the specified path."""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
