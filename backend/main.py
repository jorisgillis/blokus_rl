"""
Backend for playing a game of Blokus!
"""

import logging
import os
import sys
import uuid

import numpy as np
import uvicorn
import json
import pickle
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add the root directory to Python path to import blokus_env
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from blokus_env import BlokusEnv
from blokus_env.q_learning import QLearningAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Blokus Game API",
    description="Backend API for the Blokus board game",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# FRONTEND
###############################################################################
# Mount static files on /static path
frontend_path = "backend/static"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")
logger.info(f"Mounted static files from: {frontend_path}")


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join("backend/html", "index.html"))


# # Catch-all route for frontend routing (for single-page applications)
# # This will serve index.html for any non-API routes
# @app.get("/{full_path:path}")
# async def serve_frontend_routes(full_path: str, request: Request):
#     # Serve index.html for all non-API routes (for frontend routing)
#     return FileResponse(os.path.join("backend/html", "index.html"))


###############################################################################
# BACKEND ROUTER
###############################################################################
# Data models
class GameState(BaseModel):
    board: list[list[list[int]]]  # 20x20x4 board
    current_player: int
    players: list[dict[str, str]]  # Player info
    remaining_pieces: list[list[bool]]  # Available pieces per player
    scores: list[int]  # Current scores
    game_over: bool
    winner: int | None
    message: str | None


class PlayerMove(BaseModel):
    player_id: int
    piece_id: int
    rotation: int
    x: int
    y: int
    flip_horizontal: bool = False
    flip_vertical: bool = False


class GameCreateResponse(BaseModel):
    game_id: str
    message: str


class PlayerConfig(BaseModel):
    player_id: int
    type: str  # "human" or "ai"


class GameCreateRequest(BaseModel):
    player_configs: list[PlayerConfig] | None = None


class ErrorResponse(BaseModel):
    error: str
    details: str | None


# Game Logger
class GameLogger:
    def __init__(self, log_file: str = "backend/games_log.txt"):
        self.log_file = log_file

    def log_move(self, game_id: str, move: PlayerMove):
        """Log a move to the game log file."""
        log_entry = {
            "game_id": game_id,
            "player_id": move.player_id,
            "piece_id": move.piece_id,
            "rotation": move.rotation,
            "x": move.x,
            "y": move.y,
            "flip_h": move.flip_horizontal,
            "flip_v": move.flip_vertical
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# Global AI Agent
AI_AGENT = QLearningAgent()
if os.path.exists("q_agent.pkl"):
    success = AI_AGENT.load("q_agent.pkl")
    if success:
        logger.info("Successfully loaded AI agent from q_agent.pkl")
    else:
        logger.warning("Failed to load AI agent from q_agent.pkl")
else:
    logger.warning("No q_agent.pkl found, AI moves will be random.")


# Game manager
class GameManager:
    def __init__(self):
        self.games = {}  # game_id -> game_data
        self.player_colors = ["blue", "yellow", "red", "green"]
        self.player_names = [
            "Blue Player",
            "Yellow Player",
            "Red Player",
            "Green Player",
        ]
        self.logger = GameLogger()

    def create_game(self, player_configs: list[PlayerConfig] | None = None) -> str:
        """Create a new game and return the game ID."""
        game_id = str(uuid.uuid4())
 
        # Initialize game environment
        env = BlokusEnv()
        state, _ = env.reset()

        # Default: all human
        player_types = ["human"] * 4
        if player_configs:
            for config in player_configs:
                if 0 <= config.player_id < 4:
                    player_types[config.player_id] = config.type
 
        # Store game data
        self.games[game_id] = {
            "env": env,
            "state": state,
            "players": [
                {"id": "0", "name": self.player_names[0], "color": self.player_colors[0], "type": player_types[0]},
                {"id": "1", "name": self.player_names[1], "color": self.player_colors[1], "type": player_types[1]},
                {"id": "2", "name": self.player_names[2], "color": self.player_colors[2], "type": player_types[2]},
                {"id": "3", "name": self.player_names[3], "color": self.player_colors[3], "type": player_types[3]},
            ],
            "remaining_pieces": env.available_pieces,
            "scores": [0, 0, 0, 0],
            "game_over": False,
            "winner": None,
            "winner": None,
            "current_player": 0,
            "history": [],  # List of game states
        }

        # methods to capture history
        self._capture_history(game_id)
 
        return game_id

    def get_game_state(self, game_id: str) -> dict:
        """Get the current state of a game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")

        game_data = self.games[game_id]
        env = game_data["env"]

        # Calculate scores (number of squares placed on the board)
        scores = []
        for player_id in range(4):
            # Count all squares placed by this player on the board
            placed_squares = int(np.sum(env.board[:, :, player_id]))
            scores.append(placed_squares)

        # Update game data
        game_data["scores"] = scores
        game_data["remaining_pieces"] = env.available_pieces
        game_data["current_player"] = env.current_player
        game_data["game_over"] = env.game_over
        game_data["winner"] = env.winner

        return {
            "board": env.board.tolist(),
            "current_player": env.current_player,
            "players": game_data["players"],
            "remaining_pieces": env.available_pieces,
            "scores": scores,
            "game_over": env.game_over,
            "winner": env.winner,
            "message": f"Player {env.current_player + 1}'s turn"
            if not env.game_over
            else "Game Over!",
        }

    def make_move(self, game_id: str, move: PlayerMove) -> dict:
        """Make a move in the game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")

        game_data = self.games[game_id]
        env = game_data["env"]

        # Check if it's the player's turn
        if move.player_id != env.current_player:
            raise HTTPException(
                status_code=400, detail=f"Not player {move.player_id}'s turn"
            )

        # Check if the piece is available
        if not env.available_pieces[move.player_id][move.piece_id]:
            raise HTTPException(status_code=400, detail="Piece not available")

        # Check if the move is valid
        if not env._is_valid_placement(move.piece_id, move.x, move.y, move.rotation, 
                                       move.flip_horizontal, move.flip_vertical):
            raise HTTPException(status_code=400, detail="Invalid move")

        # Make the move
        try:
            next_state, reward, done, _, _ = env.step(
                (move.piece_id, move.x, move.y, move.rotation, 
                 move.flip_horizontal, move.flip_vertical)
            )
            game_data["state"] = next_state

            # Log the move
            self.logger.log_move(game_id, move)

            # Update game state
            if done:
                game_data["game_over"] = True
                game_data["winner"] = env.winner

            # Update game state
            if done:
                game_data["game_over"] = True
                game_data["winner"] = env.winner

            # Capture history
            self._capture_history(game_id)

            return self.get_game_state(game_id)

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Move failed: {str(e)}") from e

    def make_ai_move(self, game_id: str) -> dict:
        """Let the AI make a move for the current player."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")

        game_data = self.games[game_id]
        env = game_data["env"]

        if env.game_over:
            raise HTTPException(status_code=400, detail="Game already over")

        # Verify that the current player is an AI
        current_player_info = game_data["players"][env.current_player]
        if current_player_info["type"] != "ai":
             raise HTTPException(status_code=400, detail="Current player is not an AI")

        # Choose action
        action = AI_AGENT.choose_action(env, game_data["state"], env.current_player, training=False)
        
        # If no valid action, skip (though Blokus Envoy handles this in step, we should be explicit)
        if action is None:
             raise HTTPException(status_code=400, detail="AI found no legal moves")

        piece_id, x, y, rotation, flip_h, flip_v = action
        
        move = PlayerMove(
            player_id=env.current_player,
            piece_id=piece_id,
            rotation=rotation,
            x=x,
            y=y,
            flip_horizontal=flip_h,
            flip_vertical=flip_v
        )

        return self.make_move(game_id, move)

    def skip_turn(self, game_id: str) -> dict:
        """Skip the current player's turn if they have no moves."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")

        game_data = self.games[game_id]
        env = game_data["env"]

        if env.game_over:
            raise HTTPException(status_code=400, detail="Game already over")

        # Check if they really have no moves
        if env.has_legal_moves(env.current_player):
             raise HTTPException(status_code=400, detail="Player still has legal moves")

        # Manual skip
        env.current_player = (env.current_player + 1) % 4
        
        # Advance to the next player who has legal moves
        attempts = 0
        while not env.has_legal_moves(env.current_player) and attempts < 4:
            env.current_player = (env.current_player + 1) % 4
            attempts += 1
            
        # Check if game is over after skipping
        env._check_game_over()
        
        if env.game_over:
             game_data["winner"] = env.winner

        # Capture history
        self._capture_history(game_id)

        return self.get_game_state(game_id)

    def get_piece_shapes(self) -> list[list[tuple[int, int]]]:
        """Get the shapes of all pieces."""
        env = BlokusEnv()
        return env.pieces

    def _capture_history(self, game_id: str):
        """Capture the current game state into history."""
        game_data = self.games[game_id]
        env = game_data["env"]
        
        # Deep copy the board to avoid reference issues
        board_copy = env.board.copy().tolist()
        
        # Calculate current scores
        scores = []
        for player_id in range(4):
            scores.append(int(np.sum(env.board[:, :, player_id])))

        snapshot = {
            "board": board_copy,
            "current_player": env.current_player,
            "scores": scores,
            "remaining_pieces": [row[:] for row in env.available_pieces], # Deep copy
            "game_over": env.game_over,
            "winner": env.winner,
            "prev_action": None # Could add last action details here if needed
        }
        
        game_data["history"].append(snapshot)

    def get_game_history(self, game_id: str) -> list[dict]:
        """Get the history of a game."""
        if game_id not in self.games:
            raise HTTPException(status_code=404, detail="Game not found")
        return self.games[game_id]["history"]


# Game manager instance
game_manager = GameManager()


# API endpoints
@app.get("/api", tags=["general"])
async def api_root():
    """API root endpoint."""
    return {
        "message": "Welcome to Blokus Game API!",
        "version": "1.0.0",
        "endpoints": {
            "/games": "Create and manage games",
            "/games/{game_id}": "Get game state",
            "/games/{game_id}/move": "Make a move",
            "/pieces": "Get piece shapes",
            "/games/{game_id}/history": "Get game history",
        },
    }


@app.post("/games", response_model=GameCreateResponse, tags=["games"])
async def create_game(request: GameCreateRequest | None = None):
    """Create a new game."""
    player_configs = request.player_configs if request else None
    game_id = game_manager.create_game(player_configs)
    return {
        "game_id": game_id,
        "message": f"Game created successfully. Use this ID: {game_id}",
    }


@app.get("/games/{game_id}", response_model=GameState, tags=["games"])
async def get_game_state(game_id: str):
    """Get the current state of a game."""
    return game_manager.get_game_state(game_id)


@app.post("/games/{game_id}/move", response_model=GameState, tags=["games"])
async def make_move(game_id: str, move: PlayerMove):
    """Make a move in the game."""
    state = game_manager.make_move(game_id, move)
    await broadcast_game_update(game_id, "Board updated")
    return state


@app.post("/games/{game_id}/ai-move", response_model=GameState, tags=["games"])
async def make_ai_move(game_id: str):
    """Trigger an AI move for the current player."""
    state = game_manager.make_ai_move(game_id)
    await broadcast_game_update(game_id, "AI moved")
    return state


@app.post("/games/{game_id}/skip", response_model=GameState, tags=["games"])
async def skip_turn(game_id: str):
    """Skip the current player's turn."""
    state = game_manager.skip_turn(game_id)
    await broadcast_game_update(game_id, "Player skipped")
    return state


@app.get("/games/{game_id}/history", tags=["games"])
async def get_game_history(game_id: str):
    """Get the full history of a game."""
    return game_manager.get_game_history(game_id)


@app.get("/pieces", tags=["pieces"])
async def get_piece_shapes():
    """Get the shapes of all pieces."""
    pieces = game_manager.get_piece_shapes()

    # Convert to a more JSON-friendly format
    piece_shapes = []
    for piece in pieces:
        piece_shapes.append([{"x": x, "y": y} for x, y in piece])

    return {
        "pieces": piece_shapes,
        "count": len(piece_shapes),
        "description": "All 21 Blokus piece shapes in relative coordinates",
    }


# WebSocket for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.game_connections = {}  # game_id -> list of connections

    async def connect(self, websocket: WebSocket, game_id: str | None = None):
        await websocket.accept()
        self.active_connections.append(websocket)

        if game_id:
            if game_id not in self.game_connections:
                self.game_connections[game_id] = []
            self.game_connections[game_id].append(websocket)

    def disconnect(self, websocket: WebSocket, game_id: str | None = None):
        self.active_connections.remove(websocket)

        if game_id and game_id in self.game_connections:
            if websocket in self.game_connections[game_id]:
                self.game_connections[game_id].remove(websocket)
            if not self.game_connections[game_id]:
                del self.game_connections[game_id]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_to_game(self, message: str, game_id: str):
        if game_id in self.game_connections:
            for connection in self.game_connections[game_id]:
                try:
                    await connection.send_text(message)
                except WebSocketDisconnect:
                    self.disconnect(connection, game_id)


websocket_manager = ConnectionManager()


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """WebSocket endpoint for real-time game updates."""
    await websocket_manager.connect(websocket, game_id)

    try:
        while True:
            # Just keep the connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket, game_id)


# Helper function to broadcast game updates
async def broadcast_game_update(game_id: str, message: str):
    """Broadcast a game update to all connected clients."""
    await websocket_manager.broadcast_to_game(message, game_id)


if __name__ == "__main__":
    logger.info("Starting Blokus backend...")

    # Try running with different configuration
    try:
        uvicorn.run(
            "main:app",  # Use just the filename
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False,
        )
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        raise
