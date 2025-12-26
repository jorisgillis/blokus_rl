"""Tests for the piece shapes in the Blokus environment."""

import sys

sys.path.insert(0, ".")

from blokus_env.blokus_env import BlokusEnv


def piece_to_ascii(piece):
    """Convert a piece to ASCII art."""
    if not piece:
        return ""

    min_x = min(coord[0] for coord in piece)
    max_x = max(coord[0] for coord in piece)
    min_y = min(coord[1] for coord in piece)
    max_y = max(coord[1] for coord in piece)

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    grid = [["  " for _ in range(width)] for _ in range(height)]

    for coord in piece:
        x, y = coord
        grid[y - min_y][x - min_x] = "X "

    # Remove trailing spaces from each line
    lines = []
    for row in grid:
        line = "".join(row)
        line = line.rstrip()  # Remove trailing spaces
        lines.append(line)

    ascii_art = "\n".join(lines)
    return ascii_art


def test_piece_shapes():
    """Test the shapes of the pieces in the Blokus environment."""
    env = BlokusEnv()

    # Expected ASCII representations of the pieces (stripped versions)
    expected_pieces = [
        "X",  # Piece 1: [(0, 0)]
        "X\nX",  # Piece 2: [(0, 0), (0, 1)]
        "X X\nX",  # Piece 3: [(0, 0), (0, 1), (1, 0)]
        "X\nX\nX",  # Piece 4: [(0, 0), (0, 1), (0, 2)]
        "X X\nX\nX",  # Piece 5: [(0, 0), (0, 1), (0, 2), (1, 0)]
        "X\nX\nX X",  # Piece 6: [(0, 0), (0, 1), (0, 2), (1, 2)]
        "X\nX X\n  X",  # Piece 7: [(0, 0), (0, 1), (1, 1), (1, 2)]
        "X X\n  X\n  X",  # Piece 8: [(0, 0), (1, 0), (1, 1), (1, 2)]
        "X X\nX X",  # Piece 9: [(0, 0), (0, 1), (1, 0), (1, 1)]
        "X\nX X X\n  X",  # Piece 10: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)]
        "X X X X\n      X",  # Piece 11: [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)]
        "X\nX X\n  X X",  # Piece 12: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        "X X X\nX X",  # Piece 13: [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        "  X\nX X X\n  X",  # Piece 14: [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
        "X\nX X X\nX",  # Piece 15: [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)]
        "X X X\nX X X\n    X",  # Piece 16: 2x3 U shape
        "X X X\n    X X",  # Piece 17: [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1)]
        "X X\n  X X\n    X",  # Piece 18: [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]
        "X\nX X\n  X X",  # Piece 19: [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
        "X X X X X",  # Piece 20: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        "  X\nX X X\n  X",  # Piece 21: [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    ]

    for i, piece in enumerate(env.pieces):
        ascii_art = piece_to_ascii(piece)
        assert ascii_art.strip() == expected_pieces[i].strip(), (
            f"Piece {i + 1} does not match expected shape."
        )


if __name__ == "__main__":
    test_piece_shapes()
