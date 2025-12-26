"""
Optimized Blokus Environment with performance improvements.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import hashlib
from collections import defaultdict


class BlokusEnvOptimized(gym.Env):
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

        # Pre-compute piece data for all rotations
        self._precompute_piece_data()

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

        # Caching for performance optimization
        self._placement_cache = {}
        self._legal_actions_cache = {}
        self._current_state_hash = None

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0

    def _initialize_pieces(self) -> list[list[tuple[int, int]]]:
        # Define the 21 pieces for each player
        pieces = []
        # Monomino
        pieces.append([(0, 0)])
        # Domino
        pieces.append([(0, 0), (0, 1)])
        # Trominoes
        pieces.append([(0, 0), (0, 1), (0, 2)])  # I
        pieces.append([(0, 0), (0, 1), (1, 0)])  # L
        # Tetrominoes
        pieces.append([(0, 0), (0, 1), (0, 2), (0, 3)])  # I
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 2)])  # L
        pieces.append([(0, 0), (0, 1), (0, 2), (-1, 2)])  # J
        pieces.append([(0, 0), (0, 1), (1, 0), (1, 1)])  # O
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 1)])  # T
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2)])  # S
        pieces.append([(0, 0), (0, 1), (-1, 1), (-1, 2)])  # Z
        # Pentominoes
        pieces.append([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])  # I
        pieces.append([(0, 0), (0, 1), (0, 2), (0, 3), (1, 3)])  # L
        pieces.append([(0, 0), (0, 1), (0, 2), (0, 3), (-1, 3)])  # J
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])  # T
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])  # S
        pieces.append([(0, 0), (0, 1), (-1, 1), (-1, 2), (-2, 2)])  # Z
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])  # P
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)])  # Q
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)])  # U
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)])  # V
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])  # W
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)])  # X

        return pieces

    def _precompute_piece_data(self):
        """Pre-compute piece data for all rotations to optimize placement validation."""
        self.piece_data = []  # piece_index -> rotation -> precomputed_data

        for piece_index, piece in enumerate(self.pieces):
            rotations_data = []

            for rotation in range(4):
                # Apply rotation
                rotated_coords = []
                for px, py in piece:
                    if rotation == 0:  # 0°
                        rotated_coords.append((px, py))
                    elif rotation == 1:  # 90°
                        rotated_coords.append((py, -px))
                    elif rotation == 2:  # 180°
                        rotated_coords.append((-px, -py))
                    elif rotation == 3:  # 270°
                        rotated_coords.append((-py, px))

                # Calculate bounding box and dimensions
                if rotated_coords:
                    xs, ys = zip(*rotated_coords)
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    width = max_x - min_x + 1
                    height = max_y - min_y + 1

                    # Normalize coordinates to start from (0,0)
                    normalized_coords = [
                        (x - min_x, y - min_y) for x, y in rotated_coords
                    ]

                    # Create mask for vectorized operations
                    mask = np.zeros((width, height), dtype=bool)
                    for x, y in normalized_coords:
                        mask[x, y] = True

                    # Store corner positions for adjacency checking
                    corners = []
                    for x, y in normalized_coords:
                        is_corner = True
                        # Check if this is a corner (has at most 2 neighbors)
                        neighbors = 0
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height and mask[nx, ny]:
                                neighbors += 1
                        if neighbors <= 2:
                            corners.append((x, y))

                    rotations_data.append(
                        {
                            "coords": rotated_coords,
                            "normalized_coords": normalized_coords,
                            "min_x": min_x,
                            "max_x": max_x,
                            "min_y": min_y,
                            "max_y": max_y,
                            "width": width,
                            "height": height,
                            "mask": mask,
                            "corners": corners,
                            "size": len(rotated_coords),
                        }
                    )
                else:
                    # Empty piece (shouldn't happen)
                    rotations_data.append(
                        {
                            "coords": [],
                            "normalized_coords": [],
                            "min_x": 0,
                            "max_x": 0,
                            "min_y": 0,
                            "max_y": 0,
                            "width": 0,
                            "height": 0,
                            "mask": np.array([]),
                            "corners": [],
                            "size": 0,
                        }
                    )

            self.piece_data.append(rotations_data)

    def _compute_state_hash(self):
        """Compute a hash of the current board state for caching."""
        if self._current_state_hash is None:
            # Use a fast hash of the board
            board_bytes = self.board.tobytes()
            self._current_state_hash = hashlib.md5(board_bytes).hexdigest()
        return self._current_state_hash

    def _get_piece_coordinates_optimized(
        self, piece_index: int, x: int, y: int, rotation: int
    ) -> list[tuple[int, int]]:
        """Get the absolute coordinates of a piece using pre-computed data."""
        piece_data = self.piece_data[piece_index][rotation]
        absolute_coords = []
        for px, py in piece_data["coords"]:
            absolute_coords.append((x + px, y + py))
        return absolute_coords

    def _is_valid_placement_optimized(
        self, piece_index: int, x: int, y: int, rotation: int
    ) -> bool:
        """Optimized version of placement validation using pre-computed data and vectorization."""

        # Use caching if available
        cache_key = (
            piece_index,
            x,
            y,
            rotation,
            self._compute_state_hash(),
            self.current_player,
        )
        if cache_key in self._placement_cache:
            self._cache_hits += 1
            return self._placement_cache[cache_key]

        self._cache_misses += 1
        piece_data = self.piece_data[piece_index][rotation]

        # Early boundary check using pre-computed bounds
        if (
            x + piece_data["min_x"] < 0
            or x + piece_data["max_x"] >= 20
            or y + piece_data["min_y"] < 0
            or y + piece_data["max_y"] >= 20
        ):
            result = False
        else:
            # Vectorized collision check
            board_slice = self.board[
                x + piece_data["min_x"] : x + piece_data["max_x"] + 1,
                y + piece_data["min_y"] : y + piece_data["max_y"] + 1,
                :,
            ]

            # Check if any cell in the piece position is already occupied
            if np.any(board_slice[piece_data["mask"]] == 1):
                result = False
            else:
                # Check edge adjacency for non-first pieces
                if self.first_piece_placed[self.current_player]:
                    # Check if piece touches edges of same-color pieces
                    edge_touch = False
                    for nx, ny in piece_data["coords"]:
                        abs_x, abs_y = x + nx, y + ny
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            check_x, check_y = abs_x + dx, abs_y + dy
                            if (
                                0 <= check_x < 20
                                and 0 <= check_y < 20
                                and self.board[check_x, check_y, self.current_player]
                                == 1
                            ):
                                edge_touch = True
                                break
                        if edge_touch:
                            break

                    if edge_touch:
                        result = False
                    else:
                        # Check corner adjacency
                        corner_touch = False
                        for nx, ny in piece_data["corners"]:
                            abs_x, abs_y = x + nx, y + ny
                            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                check_x, check_y = abs_x + dx, abs_y + dy
                                if (
                                    0 <= check_x < 20
                                    and 0 <= check_y < 20
                                    and self.board[
                                        check_x, check_y, self.current_player
                                    ]
                                    == 1
                                ):
                                    corner_touch = True
                                    break
                            if corner_touch:
                                break

                        result = corner_touch
                else:
                    # First piece - check if it touches the player's corner
                    player_corner = {
                        0: (0, 0),  # Blue: top-left
                        1: (0, 19),  # Yellow: top-right
                        2: (19, 19),  # Red: bottom-right
                        3: (19, 0),  # Green: bottom-left
                    }[self.current_player]

                    corner_touched = False
                    for nx, ny in piece_data["coords"]:
                        if (x + nx, y + ny) == player_corner:
                            corner_touched = True
                            break

                    result = corner_touched

        # Cache the result
        self._placement_cache[cache_key] = result
        return result

    def _is_valid_placement(
        self, piece_index: int, x: int, y: int, rotation: int
    ) -> bool:
        """Check if a piece can be placed at the given position and rotation."""
        # Use the optimized version
        return self._is_valid_placement_optimized(piece_index, x, y, rotation)

    def _get_piece_coordinates(
        self, piece_index: int, x: int, y: int, rotation: int
    ) -> list[tuple[int, int]]:
        """Get the absolute coordinates of a piece given its position and rotation."""
        # Use the optimized version
        return self._get_piece_coordinates_optimized(piece_index, x, y, rotation)

    def get_legal_actions_optimized(
        self, player: int, max_actions: int = 100
    ) -> list[tuple]:
        """Optimized version of get_legal_actions with caching and early termination."""

        # Use caching for legal actions
        cache_key = (player, self._compute_state_hash(), max_actions)
        if cache_key in self._legal_actions_cache:
            return self._legal_actions_cache[cache_key]

        legal_actions = []

        # Check each available piece
        for piece_index, available in enumerate(self.available_pieces[player]):
            if available:
                # Get candidate positions based on game state
                candidate_positions = self._get_candidate_positions_optimized(player)

                # Try all candidate positions and rotations
                for x, y in candidate_positions:
                    for rotation in range(4):
                        if self._is_valid_placement_optimized(
                            piece_index, x, y, rotation
                        ):
                            legal_actions.append((piece_index, x, y, rotation))

                            # Early termination if we have enough actions
                            if len(legal_actions) >= max_actions:
                                break

                    if len(legal_actions) >= max_actions:
                        break

                if len(legal_actions) >= max_actions:
                    break

        # Cache the result
        self._legal_actions_cache[cache_key] = legal_actions
        return legal_actions

    def _get_candidate_positions_optimized(self, player: int) -> list[tuple[int, int]]:
        """Optimized candidate position generation."""
        candidate_positions = set()

        if not self.first_piece_placed[player]:
            # First piece: only check the corner and immediate surroundings
            player_corner = {
                0: (0, 0),  # Blue: top-left
                1: (0, 19),  # Yellow: top-right
                2: (19, 19),  # Red: bottom-right
                3: (19, 0),  # Green: bottom-left
            }[player]

            # Add the corner and 8 surrounding positions
            x, y = player_corner
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 20 and 0 <= ny < 20:
                        candidate_positions.add((nx, ny))
        else:
            # Subsequent pieces: check only around corners of existing pieces
            # Find corner positions of existing pieces
            corners = set()
            for x in range(20):
                for y in range(20):
                    if self.board[x, y, player] == 1:
                        # Check if this is a corner piece
                        neighbors = 0
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < 20
                                and 0 <= ny < 20
                                and self.board[nx, ny, player] == 1
                            ):
                                neighbors += 1
                        # If it's a corner (2 or fewer neighbors), add it
                        if neighbors <= 2:
                            corners.add((x, y))

            # Expand around corners
            for cx, cy in corners:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < 20 and 0 <= ny < 20:
                            candidate_positions.add((nx, ny))

        return list(candidate_positions)

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0,
            "placement_cache_size": len(self._placement_cache),
            "legal_actions_cache_size": len(self._legal_actions_cache),
        }

    def clear_caches(self):
        """Clear all caches."""
        self._placement_cache.clear()
        self._legal_actions_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._current_state_hash = None

    # Rest of the methods would be inherited or implemented similarly to the original
    # For now, let's implement the essential methods needed for testing

    def reset(self):
        """Reset the environment."""
        self.board = np.zeros((20, 20, 4), dtype=np.uint8)
        self.current_player = 0
        self.available_pieces = [[True] * 21 for _ in range(4)]
        self.first_piece_placed = [False for _ in range(4)]
        self.game_over = False
        self.winner = None
        self._current_state_hash = None
        self.clear_caches()

        return self.board.copy(), {}

    def step(self, action):
        """Take a step in the environment."""
        piece_index, x, y, rotation = action

        # Check if placement is valid
        if self._is_valid_placement(piece_index, x, y, rotation):
            # Place the piece
            coords = self._get_piece_coordinates(piece_index, x, y, rotation)
            for cx, cy in coords:
                self.board[cx, cy, self.current_player] = 1

            # Mark piece as used
            self.available_pieces[self.current_player][piece_index] = False

            # Mark first piece as placed
            if not self.first_piece_placed[self.current_player]:
                self.first_piece_placed[self.current_player] = True

            # Switch to next player
            self.current_player = (self.current_player + 1) % 4

            # Invalidate state hash
            self._current_state_hash = None

            return self.board.copy(), 1.0, False, False, {}
        else:
            # Invalid move - switch player
            self.current_player = (self.current_player + 1) % 4
            return self.board.copy(), -1.0, False, False, {}

    def render(self):
        """Simple text-based rendering."""
        print("Current player:", self.player_colors[self.current_player])
        print("Board:")
        for x in range(20):
            row = ""
            for y in range(20):
                cell = "."
                for player in range(4):
                    if self.board[x, y, player] == 1:
                        cell = self.player_colors[player][0].upper()
                        break
                row += cell
            print(row)
        print()
