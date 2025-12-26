# Blokus Reinforcement Learning Environment

## Project Summary

This project implements a reinforcement learning environment for the board game
Blokus using the Gymnasium framework. The goal is to train an AI agent to play
Blokus at a high level by learning from self-play and using neural networks for
move selection and evaluation.

## How to Run

### Prerequisites

- Python 3.13 or higher
- UV package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blokus.git
   cd blokus
   ```

2. Install the dependencies:
   ```bash
   uv pip install -e .
   ```

### Running the Environment

To test the Blokus environment, run the following command:

```bash
uv run python blokus_env/test_env.py
```

## Development Approach

This project follows a **Test-Driven Development (TDD)** approach. This means
that tests are written before the actual implementation to ensure that the code
meets the specified requirements and behaves as expected.

## How to Develop

### Setting Up the Development Environment
1. Install the development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

### Running Linters and Type Checkers
- **Ruff**: A fast Python linter.
  ```bash
  uv run ruff check .
  ```

- **Mypy**: A static type checker for Python.
  ```bash
  uv run mypy .
  ```

### Code Formatting
- **Ruff**: Format the code using Ruff.
  ```bash
  uv run ruff format .
  ```

### Running Tests
To run the test script for the Blokus environment:
```bash
uv run python blokus_env/test_env.py
```

To run the unit tests for the Blokus environment:
```bash
uv run pytest tests/test_blokus_env.py -v
```

### Project Structure
```
blokus/
├── blokus_env/
│   ├── __init__.py
│   ├── blokus_env.py
│   ├── register_env.py
│   └── test_env.py
├── pyproject.toml
├── README.md
├── REINFORCEMENT_LEARNING_APPROACH.md
└── GAME_ENVIRONMENT_SETUP.md
```

### Key Files
- `blokus_env/blokus_env.py`: Contains the implementation of the Blokus environment.
- `blokus_env/register_env.py`: Registers the Blokus environment with Gymnasium.
- `blokus_env/test_env.py`: Tests the Blokus environment.
- `pyproject.toml`: Project configuration and dependencies.
- `REINFORCEMENT_LEARNING_APPROACH.md`: Documentation on the reinforcement learning approach.
- `GAME_ENVIRONMENT_SETUP.md`: Tutorial on setting up the game environment.

### Development Guidelines
- Follow the coding standards and guidelines specified in the `pyproject.toml` file.
- Use type hints for better code clarity and maintainability.
- Write docstrings for functions and classes to explain their purpose and usage.
- Run the linters and type checkers regularly to ensure code quality.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
