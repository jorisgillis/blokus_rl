import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BlokusEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        # Define the action and observation spaces
        self.action_space = spaces.Tuple(
            [
                spaces.Discrete(21),  # piece_index (0-20)
                spaces.Discrete(20),  # x position (0-19)
                spaces.Discrete(20),  # y position (0-19)
                spaces.Discrete(4),  # rotation (0-3)
            ]
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20, 20, 4), dtype=np.uint8
        )  # 20x20 board, 4 players

        # Initialize the board
        self.board = np.zeros((20, 20, 4), dtype=np.uint8)

        # Initialize the pieces
        self.pieces = self._initialize_pieces()

        # Initialize the current player
        self.current_player = 0

        # Track available pieces for each player (boolean array)
        self.available_pieces = [[True] * 21 for _ in range(4)]

        # Track if players have placed their first piece
        self.first_piece_placed = [False for _ in range(4)]

        # Track game state
        self.game_over = False
        self.winner: int | None = None

        # Player colors (for rendering)
        self.player_colors = ["blue", "yellow", "red", "green"]

    def _initialize_pieces(self) -> list[list[tuple[int, int]]]:
        # Define the 21 pieces for each player
        pieces = []
        # Monomino
        pieces.append([(0, 0)])
        # Domino
        pieces.append([(0, 0), (0, 1)])
        # Trominoes
        pieces.append([(0, 0), (0, 1), (1, 0)])
        pieces.append([(0, 0), (0, 1), (0, 2)])
        # Tetrominoes
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 0)])
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 2)])
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2)])
        pieces.append([(0, 0), (1, 0), (1, 1), (1, 2)])
        pieces.append([(0, 0), (0, 1), (1, 0), (1, 1)])
        # Pentominoes
        # F Pentomino
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)])
        # L Pentomino
        pieces.append([(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)])
        # N Pentomino
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])
        # P Pentomino
        pieces.append([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        # Y Pentomino
        pieces.append([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)])
        # T Pentomino
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)])
        # U Pentomino
        pieces.append([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)])
        # V Pentomino
        pieces.append([(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)])
        # W Pentomino
        pieces.append([(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)])
        # Z Pentomino
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])
        # I Pentomino
        pieces.append([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        # X Pentomino
        pieces.append([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)])
        return pieces

    def _get_piece_coordinates(
        self, piece_index: int, x: int, y: int, rotation: int
    ) -> list[tuple[int, int]]:
        """Get the absolute coordinates of a piece given its position and rotation."""
        piece = self.pieces[piece_index]

        # Apply rotation (0: 0°, 1: 90°, 2: 180°, 3: 270°)
        rotated_piece = []
        for px, py in piece:
            if rotation == 0:  # 0°
                rotated_piece.append((px, py))
            elif rotation == 1:  # 90°
                rotated_piece.append((py, -px))
            elif rotation == 2:  # 180°
                rotated_piece.append((-px, -py))
            elif rotation == 3:  # 270°
                rotated_piece.append((-py, px))

        # Translate to absolute position
        absolute_coords = []
        for px, py in rotated_piece:
            absolute_coords.append((x + px, y + py))

        return absolute_coords

    def _is_valid_placement(
        self, piece_index: int, x: int, y: int, rotation: int
    ) -> bool:
        """Check if a piece can be placed at the given position and rotation."""
        # Get the absolute coordinates of the piece
        coords = self._get_piece_coordinates(piece_index, x, y, rotation)

        # Check if the piece is within the board boundaries
        for x, y in coords:
            if x < 0 or x >= 20 or y < 0 or y >= 20:
                return False

        # Check if the piece overlaps with any existing pieces (all players)
        for x, y in coords:
            if np.any(self.board[x, y, :] == 1):
                return False

        # Check if the piece touches edges of same-color pieces (except for first piece)
        if self.first_piece_placed[self.current_player]:
            for x, y in coords:
                # Check adjacent cells (up, down, left, right)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 20 and 0 <= ny < 20:
                        if self.board[nx, ny, self.current_player] == 1:
                            return False

        # Check if the piece touches corners of same-color pieces (for non-first pieces)
        if self.first_piece_placed[self.current_player]:
            has_corner_contact = False
            for x, y in coords:
                # Check diagonal cells (corners)
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 20 and 0 <= ny < 20:
                        if self.board[nx, ny, self.current_player] == 1:
                            has_corner_contact = True
                            break
                if has_corner_contact:
                    break

            if not has_corner_contact:
                return False

        # For first piece, check if it's in the correct corner
        if not self.first_piece_placed[self.current_player]:
            player_corner = {
                0: (0, 0),  # Blue: top-left
                1: (0, 19),  # Yellow: top-right
                2: (19, 19),  # Red: bottom-right
                3: (19, 0),  # Green: bottom-left
            }[self.current_player]

            # Check if any part of the piece is in the corner
            corner_touched = False
            for x, y in coords:
                if (x, y) == player_corner:
                    corner_touched = True
                    break

            if not corner_touched:
                return False

        return True

    def _place_piece(self, piece_index: int, x: int, y: int, rotation: int) -> None:
        """Place a piece on the board."""
        coords = self._get_piece_coordinates(piece_index, x, y, rotation)

        for x, y in coords:
            self.board[x, y, self.current_player] = 1

        # Mark the piece as unavailable
        if self.available_pieces[self.current_player][piece_index]:
            self.available_pieces[self.current_player][piece_index] = False

        # Mark first piece as placed
        if not self.first_piece_placed[self.current_player]:
            self.first_piece_placed[self.current_player] = True

    def _calculate_reward(self, piece_index: int) -> float:
        """Calculate the reward for the current player."""
        # Reward is positive based on squares placed
        # Simple reward: 1.0 per square placed
        reward = float(len(self.pieces[piece_index]))

        return reward

    def _check_game_over(self) -> bool:
        """Check if the game is over."""
        # Game is over if no player can make a legal move
        for player in range(4):
            # Check if player has any available pieces
            if any(self.available_pieces[player]):
                # Check if any piece can be placed
                for piece_index, available in enumerate(self.available_pieces[player]):
                    if available:
                        # Try all possible positions and rotations
                        for x in range(20):
                            for y in range(20):
                                for rotation in range(4):
                                    if self._is_valid_placement(
                                        piece_index, x, y, rotation
                                    ):
                                        return False  # Game is not over

        # If we get here, no player can make a move
        self.game_over = True

        # Determine winner (player with fewest remaining squares)
        remaining_squares = []
        for player in range(4):
            count = 0
            for piece_index, available in enumerate(self.available_pieces[player]):
                if available:
                    count += len(self.pieces[piece_index])
            remaining_squares.append(count)

        min_squares = min(remaining_squares)
        winners = [
            i for i, count in enumerate(remaining_squares) if count == min_squares
        ]

        if len(winners) == 1:
            self.winner = winners[0]
        else:
            self.winner = None  # Tie

        return True

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        # Reset the board and pieces
        super().reset(seed=seed)
        self.board = np.zeros((20, 20, 4), dtype=np.uint8)
        self.pieces = self._initialize_pieces()
        self.current_player = 0
        self.available_pieces = [[True] * 21 for _ in range(4)]
        self.first_piece_placed = [False for _ in range(4)]
        self.game_over = False
        self.winner = None
        return self.board, {
            "current_player": self.current_player,
            "available_pieces": sum(self.available_pieces[self.current_player]),
            "game_over": self.game_over,
        }

    def get_legal_actions(self, player: int) -> list[tuple[int, int, int, int]]:
        """Get all legal actions for the current player."""
        legal_actions = []

        # Check each available piece
        for piece_index, available in enumerate(self.available_pieces[player]):
            if available:
                # Get candidate positions based on game state
                candidate_positions = self._get_candidate_positions(player)

                # Try all candidate positions and rotations
                for x, y in candidate_positions:
                    for rotation in range(4):
                        if self._is_valid_placement(piece_index, x, y, rotation):
                            legal_actions.append((piece_index, x, y, rotation))

        return legal_actions

    def _get_candidate_positions(self, player: int) -> list[tuple[int, int]]:
        """Get candidate positions to check for piece placement."""
        candidate_positions = set()

        if not self.first_piece_placed[player]:
            # First piece: check positions around the player's starting corner
            player_corner = {
                0: (0, 0),  # Blue: top-left
                1: (0, 19),  # Yellow: top-right
                2: (19, 19),  # Red: bottom-right
                3: (19, 0),  # Green: bottom-left
            }[player]

            # Add the corner itself and surrounding positions
            x, y = player_corner
            for dx in range(-5, 6):  # Check 11x11 area around corner
                for dy in range(-5, 6):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 20 and 0 <= ny < 20:
                        candidate_positions.add((nx, ny))
        else:
            # Subsequent pieces: check expanded area around existing pieces
            # Find all existing pieces of this player
            player_pieces = []
            for x in range(20):
                for y in range(20):
                    if self.board[x, y, player] == 1:
                        player_pieces.append((x, y))

            if player_pieces:
                # Find the bounding box of existing pieces
                min_x = min(x for x, y in player_pieces)
                max_x = max(x for x, y in player_pieces)
                min_y = min(y for x, y in player_pieces)
                max_y = max(y for x, y in player_pieces)

                # Expand the bounding box to cover potential placement areas
                # We need to cover areas where pieces can touch corners
                expand_radius = 8  # This covers most potential placements

                start_x = max(0, min_x - expand_radius)
                end_x = min(19, max_x + expand_radius)
                start_y = max(0, min_y - expand_radius)
                end_y = min(19, max_y + expand_radius)

                # Add all positions in the expanded bounding box
                for x in range(start_x, end_x + 1):
                    for y in range(start_y, end_y + 1):
                        candidate_positions.add((x, y))

        return list(candidate_positions)

    def step(
        self, action: tuple[int, int, int, int]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: A tuple of (piece_index, position_x, position_y, rotation)
                where piece_index is 0-20, position is (0-19, 0-19),
                and rotation is 0-3 (0°, 90°, 180°, 270°)

        Returns:
            observation: The new board state
            reward: The reward for the current player
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.game_over:
            return self.board, 0.0, True, False, {"message": "Game already over"}

        # Parse the action (piece_index, x, y, rotation)
        piece_index, x, y, rotation = action

        # Validate the action
        if piece_index < 0 or piece_index >= 21:
            return self.board, -10.0, False, False, {"message": "Invalid piece index"}

        if x < 0 or x >= 20 or y < 0 or y >= 20:
            return self.board, -10.0, False, False, {"message": "Invalid position"}

        if rotation < 0 or rotation >= 4:
            return self.board, -10.0, False, False, {"message": "Invalid rotation"}

        if not self.available_pieces[self.current_player][piece_index]:
            return self.board, -10.0, False, False, {"message": "Piece not available"}

        # Check if the placement is valid
        if not self._is_valid_placement(piece_index, x, y, rotation):
            return self.board, -5.0, False, False, {"message": "Invalid placement"}

        # Place the piece
        self._place_piece(piece_index, x, y, rotation)

        # Calculate reward
        reward = self._calculate_reward(piece_index)

        # Check if game is over
        game_over = self._check_game_over()

        # Switch to next player
        self.current_player = (self.current_player + 1) % 4

        # Prepare info dictionary
        info = {
            "current_player": self.current_player,
            "available_pieces": sum(self.available_pieces[self.current_player]),
            "game_over": game_over,
        }

        if game_over:
            info["winner"] = self.winner if self.winner is not None else "Tie"
            winner_msg = "Tie" if self.winner is None else str(self.winner)
            info["message"] = f"Game over. Winner: {winner_msg}"

        return self.board, reward, game_over, False, info

    def render(self, mode: str = "human") -> None | str:
        """Render the current state of the board."""
        if mode == "human":
            print("Current Board State:")
            print("Player colors: Blue (0), Yellow (1), Red (2), Green (3)")
            print("Current player:", self.player_colors[self.current_player])
            print()

            # Create a display grid
            for y in range(20):
                row = []
                for x in range(20):
                    cell = "·"
                    for player in range(4):
                        if self.board[x, y, player] == 1:
                            cell = str(player)
                            break
                    row.append(cell)
                print(" ".join(row))

            print()
            print(
                "Available pieces for current player:",
                sum(self.available_pieces[self.current_player]),
            )
            print("Game over:", self.game_over)
            if self.game_over:
                print("Winner:", self.winner if self.winner is not None else "Tie")

        elif mode == "ansi":
            # ANSI color rendering - return as string
            output_lines = []
            output_lines.append("Current Board State (ANSI colors):")
            output_lines.append(
                f"Current player: {self.player_colors[self.current_player]}"
            )
            output_lines.append("")

            # ANSI color codes
            colors = [
                "\033[94m",  # Blue
                "\033[93m",  # Yellow
                "\033[91m",  # Red
                "\033[92m",  # Green
            ]
            reset = "\033[0m"

            for y in range(20):
                row = []
                for x in range(20):
                    cell = "·"
                    color = ""
                    for player in range(4):
                        if self.board[x, y, player] == 1:
                            cell = "■"
                            color = colors[player]
                            break
                    row.append(f"{color}{cell}{reset}")
                output_lines.append(" ".join(row))

            output_lines.append("")
            output_lines.append(
                f"Available pieces: {sum(self.available_pieces[self.current_player])}"
            )
            output_lines.append(f"Game over: {self.game_over}")

            return "\n".join(str(line) for line in output_lines)

        else:
            raise ValueError(f"Unknown render mode: {mode}")

    def close(self) -> None:
        # Clean up any resources
        pass
