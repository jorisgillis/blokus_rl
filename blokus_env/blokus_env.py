import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BlokusEnv(gym.Env):
    # Class-level cache for masks
    piece_masks = None
    piece_adj_masks = None
    piece_corner_masks = None

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
        
        # Bitboard representation (20x20 = 400 bits)
        # Each integer is a 400-bit bitmask
        self.player_bitboards = [0] * 4
        self.total_bitboard = 0

        # Track where same-color pieces cannot touch (edges) and where they MUST touch (corners)
        self.player_edge_masks = [0] * 4
        self.player_corner_masks = [0] * 4

        # Initialize the pieces
        self.pieces = self._initialize_pieces()
        
        # Pre-compute piece bitmasks for all orientations and positions
        # Use class-level cache if available
        if BlokusEnv.piece_masks is None:
            BlokusEnv.piece_masks, BlokusEnv.piece_adj_masks, BlokusEnv.piece_corner_masks = self._precompute_piece_masks()
            
        self.piece_masks = BlokusEnv.piece_masks
        self.piece_adj_masks = BlokusEnv.piece_adj_masks
        self.piece_corner_masks = BlokusEnv.piece_corner_masks

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
        pieces.append([(0, 0), (0, 1), (0, 2), (0, 3)])
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2)])
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 1)])
        pieces.append([(0, 0), (0, 1), (1, 0), (1, 1)])
        # Pentominoes
        # F Pentomino
        pieces.append([(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)])
        # L Pentomino
        pieces.append([(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)])
        # N Pentomino
        pieces.append([(0, 1), (1, 0), (1, 1), (2, 0), (3, 0)])
        # P Pentomino
        pieces.append([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)])
        # Y Pentomino
        pieces.append([(0, 0), (1, 0), (2, 0), (3, 0), (1, 1)])
        # T Pentomino
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)])
        # U Pentomino
        pieces.append([(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)])
        # V Pentomino
        pieces.append([(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)])
        # W Pentomino
        pieces.append([(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)])
        # Z Pentomino
        pieces.append([(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)])
        # I Pentomino
        pieces.append([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])
        # X Pentomino
        pieces.append([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)])
        return pieces

        pieces.append([(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)])
        return pieces

    def _precompute_piece_masks(self):
        """Pre-compute bitmasks for all pieces, orientations, and positions."""
        # 21 pieces, 16 orientations (4 rotations * 2 flip_h * 2 flip_v)
        masks = [[[[0] * 20 for _ in range(20)] for _ in range(16)] for _ in range(21)]
        adj_masks = [[[[0] * 20 for _ in range(20)] for _ in range(16)] for _ in range(21)]
        corner_masks = [[[[0] * 20 for _ in range(20)] for _ in range(16)] for _ in range(21)]

        for piece_id in range(21):
            for rotation in range(4):
                for flip_h in [0, 1]:
                    for flip_v in [0, 1]:
                        orientation_idx = (rotation << 2) | (flip_h << 1) | flip_v
                        
                        # Get base coordinates relative to (0,0)
                        base_coords = self._get_piece_coordinates(piece_id, 0, 0, rotation, bool(flip_h), bool(flip_v))
                        
                        # Determine valid placement range
                        min_x = min(x for x, y in base_coords)
                        max_x = max(x for x, y in base_coords)
                        min_y = min(y for x, y in base_coords)
                        max_y = max(y for x, y in base_coords)
                        
                        # Calculate relative sets ONCE
                        piece_cells_rel = set(base_coords)
                        adj_cells_rel = set()
                        corner_cells_rel = set()
                        
                        for px, py in piece_cells_rel:
                            # Edges
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ax, ay = px + dx, py + dy
                                if (ax, ay) not in piece_cells_rel:
                                    adj_cells_rel.add((ax, ay))
                            # Corners
                            for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                                cx, cy = px + dx, py + dy
                                if (cx, cy) not in piece_cells_rel:
                                    corner_cells_rel.add((cx, cy))
                        
                        # Iterate only over valid board positions
                        # valid x: 0 <= x + min_x  AND  x + max_x < 20
                        # => -min_x <= x < 20 - max_x
                        start_x = max(0, -min_x)
                        end_x = 20 - max_x
                        
                        start_y = max(0, -min_y)
                        end_y = 20 - max_y
                        
                        for x in range(start_x, end_x):
                            for y in range(start_y, end_y):
                                # Logic: mask |= 1 << ((y+py)*20 + (x+px))
                                # Optimized: shift = y*20 + x. 
                                # But we typically use (y+py)*20 ... 
                                # Let's stick to accumulating bits to be safe and clear
                                
                                piece_mask = 0
                                for px, py in piece_cells_rel:
                                    piece_mask |= (1 << ((y + py) * 20 + (x + px)))
                                masks[piece_id][orientation_idx][x][y] = piece_mask

                                adj_mask = 0
                                for ax, ay in adj_cells_rel:
                                    nx, ny = x + ax, y + ay
                                    # Adjacency can fall off board, ignore those bits
                                    if 0 <= nx < 20 and 0 <= ny < 20:
                                        adj_mask |= (1 << (ny * 20 + nx))
                                adj_masks[piece_id][orientation_idx][x][y] = adj_mask
                                
                                corner_mask = 0
                                for cx, cy in corner_cells_rel:
                                    nx, ny = x + cx, y + cy
                                    if 0 <= nx < 20 and 0 <= ny < 20:
                                        corner_mask |= (1 << (ny * 20 + nx))
                                corner_masks[piece_id][orientation_idx][x][y] = corner_mask

        return masks, adj_masks, corner_masks

    def _get_piece_coordinates(
        self, piece_index: int, x: int, y: int, rotation: int, 
        flip_horizontal: bool = False, flip_vertical: bool = False
    ) -> list[tuple[int, int]]:
        """Get the absolute coordinates of a piece given its position, rotation, and flips."""
        piece = self.pieces[piece_index]

        # Start with base coordinates
        coords = [(px, py) for px, py in piece]

        # Apply flips first (before rotation, matching frontend)
        if flip_horizontal:
            coords = [(-px, py) for px, py in coords]
        if flip_vertical:
            coords = [(px, -py) for px, py in coords]

        # Apply rotation (0: 0°, 1: 90°, 2: 180°, 3: 270°)
        rotated_piece = []
        for px, py in coords:
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

    def _is_valid_placement(self, piece_index: int, x: int, y: int, rotation: int, 
                           flip_h: bool = False, flip_v: bool = False, player: int = None) -> bool:
        """Check if a piece placement is valid using bitboards."""
        if player is None:
            player = self.current_player
            
        orientation_idx = (rotation << 2) | (int(flip_h) << 1) | int(flip_v)
        piece_mask = self.piece_masks[piece_index][orientation_idx][x][y]
        
        # 1. Check if out of bounds (mask is 0 if any part was OOB during pre-computation)
        if piece_mask == 0:
            return False

        # 2. Check if overlapping with ANY existing pieces
        if (piece_mask & self.total_bitboard) != 0:
            return False

        # 3. Check official Blokus rules
        if self.first_piece_placed[player]:
            # Rule: Must NOT touch edges of same-color pieces.
            # Implementation: piece_mask & self.player_edge_masks[player] == 0
            if (piece_mask & self.player_edge_masks[player]) != 0:
                return False
                
            # Rule: Must touch at least one corner of a same-color piece.
            # Implementation: piece_mask & self.player_corner_masks[player] != 0
            if (piece_mask & self.player_corner_masks[player]) == 0:
                return False
        else:
            # First piece: must touch the starting corner
            # Map (x,y) to bit index: y * 20 + x
            player_corner_bit_idx = {
                0: 0 * 20 + 0,    # (0,0)   - Top-Left (Blue)
                1: 19 * 20 + 0,   # (0,19)  - Top-Right (Yellow) -> Bit 380 (19*20 + 0)
                2: 19 * 20 + 19,  # (19,19) - Bottom-Right (Red)
                3: 0 * 20 + 19,   # (19,0)  - Bottom-Left (Green) -> Bit 19 (0*20 + 19)
            }[player]
            corner_bit = 1 << player_corner_bit_idx
            if (piece_mask & corner_bit) == 0:
                return False

        return True

    def _place_piece(self, piece_index: int, x: int, y: int, rotation: int,
                     flip_horizontal: bool = False, flip_vertical: bool = False) -> None:
        """Place a piece on the board and update bitboards."""
        orientation_idx = (rotation << 2) | (int(flip_horizontal) << 1) | int(flip_vertical)
        piece_mask = self.piece_masks[piece_index][orientation_idx][x][y]
        
        # Update bitboards
        self.player_bitboards[self.current_player] |= piece_mask
        self.total_bitboard |= piece_mask
        
        # Update adjacency and corner bitboards for this player
        self.player_edge_masks[self.current_player] |= self.piece_adj_masks[piece_index][orientation_idx][x][y]
        self.player_corner_masks[self.current_player] |= self.piece_corner_masks[piece_index][orientation_idx][x][y]

        # Sync with numpy board for rendering/other logic
        coords = self._get_piece_coordinates(piece_index, x, y, rotation, flip_horizontal, flip_vertical)
        for px, py in coords:
            self.board[px, py, self.current_player] = 1

        # Mark the piece as unavailable
        if self.available_pieces[self.current_player][piece_index]:
            self.available_pieces[self.current_player][piece_index] = False

        # Mark first piece as placed
        if not self.first_piece_placed[self.current_player]:
            self.first_piece_placed[self.current_player] = True

    def _calculate_reward(self, piece_index: int, game_over: bool) -> float:
        """Calculate the reward for the current player."""
        # Base reward: 1.0 per square placed
        reward = float(len(self.pieces[piece_index]))

        # Full clearance bonus
        if sum(self.available_pieces[self.current_player]) == 0:
            reward += 15.0
            # Monomino bonus (piece index 0)
            if piece_index == 0:
                reward += 5.0

        # Terminal rewards
        if game_over:
            if self.winner == self.current_player:
                reward += 100.0  # Big win bonus
            elif self.winner is None:
                reward += 20.0   # Tie bonus
            else:
                reward -= 50.0   # Loss penalty

        return reward

    def has_legal_moves(self, player: int) -> bool:
        """Check if the player has any legal moves available."""
        # Check if player has any available pieces
        if not any(self.available_pieces[player]):
            return False

        # Get candidate positions based on game state
        candidate_positions = self._get_candidate_positions(player)
        if not candidate_positions:
            return False

        # Try all available pieces
        for piece_index, available in enumerate(self.available_pieces[player]):
            if available:
                # Try all candidate positions
                for x, y in candidate_positions:
                    # Try all rotations and flips
                    for rotation in range(4):
                        for flip_h in [False, True]:
                            for flip_v in [False, True]:
                                if self._is_valid_placement(piece_index, x, y, rotation, flip_h, flip_v, player):
                                    return True
        return False

    def _check_game_over(self) -> bool:
        """Check if the game is over."""
        # Game is over if no player can make a legal move
        for player in range(4):
            if self.has_legal_moves(player):
                return False  # At least one player can still move

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

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        self.board = np.zeros((20, 20, 4), dtype=np.uint8)
        self.player_bitboards = [0] * 4
        self.total_bitboard = 0
        
        # Track where same-color pieces cannot touch (edges) and where they MUST touch (corners)
        self.player_edge_masks = [0] * 4
        self.player_corner_masks = [0] * 4

        self.available_pieces = [[True] * 21 for _ in range(4)]
        self.first_piece_placed = [False] * 4
        self.current_player = 0
        self.game_over = False
        self.winner = None
        
        # Skip players if they have no moves at start
        attempts = 0
        while not self.has_legal_moves(self.current_player) and attempts < 4:
            self.current_player = (self.current_player + 1) % 4
            attempts += 1
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
                        if self._is_valid_placement(piece_index, x, y, rotation, player=player):
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
        self, action: tuple[int, int, int, int] | tuple[int, int, int, int, bool, bool]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: A tuple of (piece_index, position_x, position_y, rotation) or
                (piece_index, position_x, position_y, rotation, flip_horizontal, flip_vertical)
                where piece_index is 0-20, position is (0-19, 0-19),
                rotation is 0-3 (0°, 90°, 180°, 270°), and flips are boolean

        Returns:
            observation: The new board state
            reward: The reward for the current player
            terminated: Whether the episode has terminated
            truncated: Whether the episode was truncated
            info: Additional information
        """
        if self.game_over:
            return self.board, 0.0, True, False, {"message": "Game already over"}

        # Parse the action with optional flip parameters
        if len(action) == 6:
            piece_index, x, y, rotation, flip_h, flip_v = action
        else:
            piece_index, x, y, rotation = action
            flip_h, flip_v = False, False

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
        if not self._is_valid_placement(piece_index, x, y, rotation, flip_h, flip_v):
            return self.board, -5.0, False, False, {"message": "Invalid placement"}

        # Place the piece
        self._place_piece(piece_index, x, y, rotation, flip_h, flip_v)

        # Check if game is over BEFORE calculating reward to include terminal bonus
        game_over = self._check_game_over()

        # Calculate reward
        reward = self._calculate_reward(piece_index, game_over)
        if not game_over:
            # Advance to the next player who has legal moves
            self.current_player = (self.current_player + 1) % 4
            attempts = 0
            while not self.has_legal_moves(self.current_player) and attempts < 4:
                self.current_player = (self.current_player + 1) % 4
                attempts += 1
        
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
