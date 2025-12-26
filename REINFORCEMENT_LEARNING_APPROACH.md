# Reinforcement Learning Approach for Blokus

## Overview

This document outlines the approach to building an AlphaZero-like reinforcement
learning program for the board game Blokus. The goal is to train an AI agent
that can play Blokus at a high level by learning from self-play and using neural
networks for move selection and evaluation.

## Game Description

Blokus is a strategic board game played by four players on a 400-square board (20x20 grid). Each player has a set of 21 differently shaped puzzle pieces, totaling 84 tiles. The objective is to place as many of these pieces on the board as possible, following specific rules:

1. **Starting Position**: Each player must start by placing their first piece in their designated corner of the board.
2. **Corner Contact**: Each new piece must touch at least one existing piece of the same color at a corner. Pieces cannot touch edges.
3. **Immutability**: Once placed, pieces cannot be moved.
4. **No Overlapping**: Pieces cannot be placed on top of each other.
5. **One Piece per Turn**: Each player places one piece per turn.
6. **Game End**: The game ends when no player can make a legal move. If one player can still move, they continue until they can no longer move.
7. **Different Colors**: Pieces of different colors can touch freely at corners or edges.

The winner is determined by the player with the fewest remaining squares. For example, a player with one tile of 5 squares has a score of 5, while a player with two tiles of 2 squares each has a score of 4. The latter wins.

### Game Details
- **Board Size**: 20x20 grid (400 squares).
- **Tiles**: 84 tiles in total, 21 per player.
- **Colors**: Blue, yellow, red, and green.
- **Shapes**: The 21 possible free polyominoes of one to five squares (one monomino, one domino, two trominoes/triominoes, five tetrominoes, and 12 pentominoes).
- **Order of Play**: Blue, yellow, red, green.

### Piece Shapes

#### Monomino (1 square)
```
■
```

#### Domino (2 squares)
```
■ ■
```

#### Trominoes (3 squares)
```
■
■ ■
```

```
■ ■ ■
```

#### Tetrominoes (4 squares)
```
■
■
■ ■
```

```
■
■
■
■
```

```
■ ■
  ■ ■
```

```
  ■
■ ■ ■
```

```
■ ■
■ ■
```

#### Pentominoes (5 squares)

**F Pentomino**
```
  ■ ■
■ ■
  ■
```

**L Pentomino**
```
■
■
■
■ ■
```

**N Pentomino**
```
■
■ ■
  ■
  ■
```

**P Pentomino**
```
■ ■
■ ■
■
```

**Y Pentomino**
```
  ■
■ ■ ■
  ■
```

**T Pentomino**
```
■ ■ ■
  ■
  ■
```

**U Pentomino**
```
■ ■
■   
■ ■
```

**V Pentomino**
```
■
■
■ ■ ■
```

**W Pentomino**
```
■
■ ■
  ■ ■
```

**Z Pentomino**
```
■ ■
  ■ ■
  ■
```

**I Pentomino**
```
■
■
■
■
■
```

**X Pentomino**
```
  ■
■ ■ ■
  ■
```

## Reinforcement Learning Approach

### AlphaZero Overview

AlphaZero is a reinforcement learning algorithm that combines deep neural
networks with Monte Carlo Tree Search (MCTS). It learns by playing games against
itself, starting from random moves and gradually improving through iterative
training.

### Key Components

1. **Neural Network**:
   - **Input**: The current state of the board, represented as a tensor.
   - **Output**: A policy (probability distribution over possible moves) and a value (estimated outcome of the game from the current state).
   - **Architecture**: A convolutional neural network (CNN) is suitable for processing the board state due to its spatial nature.

2. **Monte Carlo Tree Search (MCTS)**:
   - **Purpose**: To explore possible moves and evaluate their outcomes.
   - **Integration**: The neural network guides the MCTS by providing prior probabilities for moves and evaluating board states.

3. **Self-Play**:
   - **Process**: The AI plays games against itself, using the current neural network to guide its moves.
   - **Data Collection**: Each game generates training data (board states, move probabilities, and game outcomes).

4. **Training Loop**:
   - **Iteration**: The neural network is periodically updated using the data collected from self-play.
   - **Improvement**: Over time, the AI improves as it learns from its own experiences.

### Implementation Steps

1. **Environment Setup**:
   - **Board Representation**: Represent the board as a 2D array or tensor, where each cell can be empty or occupied by a piece from one of the four players.
   - **Piece Representation**: Represent each piece as a set of coordinates relative to a pivot point.
   - **Game Logic**: Implement the rules of Blokus, including move validation, piece placement, and game termination.

2. **Neural Network**:
   - **Input Layer**: Accepts the board state as a tensor.
   - **Hidden Layers**: Use convolutional layers to process the spatial data.
   - **Output Layers**: One layer for the policy (move probabilities) and one for the value (game outcome).

3. **Monte Carlo Tree Search (MCTS)**:
   - **Selection**: Traverse the tree using the Upper Confidence Bound (UCB) formula, which balances exploration and exploitation.
   - **Expansion**: Add new nodes to the tree for unexplored moves.
   - **Simulation**: Use the neural network to evaluate the outcome of the game from the new state.
   - **Backpropagation**: Update the tree with the results of the simulation.

4. **Self-Play**:
   - **Initialization**: Start with a randomly initialized neural network.
   - **Game Play**: Use MCTS to select moves during self-play games.
   - **Data Collection**: Store the board states, move probabilities, and game outcomes for training.

5. **Training**:
   - **Data Preparation**: Prepare the collected data for training.
   - **Model Training**: Train the neural network using the collected data.
   - **Iteration**: Repeat the self-play and training process to improve the model.

6. **Evaluation**:
   - **Testing**: Evaluate the trained model by playing against itself or other agents.
   - **Metrics**: Use metrics such as win rate, move quality, and game outcomes to assess performance.

### Tools and Libraries

- **Python**: The primary programming language for the project.
- **Uv**: The package manager for managing dependencies.
- **PyTorch**: For implementing the neural network.
- **NumPy**: For numerical operations and board representations.
- **Matplotlib**: For visualizing the board and game states.
- **Flask**: For creating a web-based graphical interface.
- **React**: For building the frontend of the web-based graphical interface.

### Project Structure

```
blokus/
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
├── uv.lock
└── REINFORCEMENT_LEARNING_APPROACH.md
```

### Implementation Status

The deep reinforcement learning approach has been implemented with the following components:

### ✅ Completed Components

1. **Game Environment**: Fully implemented in `blokus_env/blokus_env.py`
   - Complete board representation (20x20x4)
   - All 21 piece shapes implemented
   - Full game rules and validation
   - Proper action and observation spaces

2. **Neural Network**: Implemented in `blokus_env/neural_network.py`
   - Convolutional neural network architecture
   - Policy and value heads
   - Training and prediction methods
   - Model saving/loading functionality

3. **Monte Carlo Tree Search (MCTS)**: Implemented in `blokus_env/mcts.py`
   - Tree search with PUCT formula (AlphaZero-style)
   - Neural network integration for prior probabilities
   - Legal action detection
   - Simulation and backpropagation

4. **Self-Play**: Implemented in `blokus_env/self_play.py`
   - Complete self-play loop
   - Data collection (states, policies, values)
   - Training pipeline
   - Data saving/loading

5. **Training Pipeline**: Implemented in `main.py`
   - Command-line interface for training
   - Model loading/saving
   - Self-play episodes with configurable parameters

### 🔧 Current Implementation Details

The current implementation includes:

- **Mock Neural Network**: A mock implementation (`neural_network_mock.py`) for testing without PyTorch dependency
- **Fallback Mechanism**: Automatic fallback to mock components when PyTorch is not available
- **Comprehensive Testing**: Unit tests for all major components
- **Error Handling**: Robust error handling throughout the pipeline

### 🚀 Usage

To run the reinforcement learning training:

```bash
python main.py --episodes 10 --epochs 5 --save model.pth
```

Command-line arguments:
- `--episodes`: Number of self-play episodes (default: 5)
- `--epochs`: Number of training epochs (default: 10)
- `--load`: Load model from file
- `--save`: Save model to file
- `--data`: Data file name (default: blokus_data.npz)

### 📋 Next Steps for Improvement

1. **Enhanced Simulation**: Improve the MCTS simulation with proper game rollouts
2. **Advanced Training**: Implement more sophisticated training strategies
3. **Performance Optimization**: Optimize the neural network architecture
4. **Evaluation Metrics**: Add comprehensive evaluation metrics
5. **Visualization**: Add training progress visualization
6. **Hyperparameter Tuning**: Implement hyperparameter optimization
7. **Distributed Training**: Add support for distributed self-play and training

### 🎯 Future Enhancements

1. **Graphical Interface**: Web-based interface for human play and visualization
2. **Multi-agent Training**: Support for training multiple agents simultaneously
3. **Curriculum Learning**: Progressive difficulty training
4. **Transfer Learning**: Adapt models from similar games
5. **Explanation AI**: Interpretability features for understanding agent decisions

## Conclusion

This approach leverages the power of reinforcement learning and neural networks to create an AI agent capable of playing Blokus at a high level. By combining self-play, MCTS, and deep learning, the AI can learn complex strategies and improve over time.
