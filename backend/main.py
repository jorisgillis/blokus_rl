"""
FastAPI backend for Blokus game.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Optional


# Create FastAPI app
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
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Data models
class HelloResponse(BaseModel):
    message: str
    status: str
    version: str


class GameState(BaseModel):
    board: list
    current_player: int
    game_over: bool
    winner: Optional[int]


# API endpoints
@app.get("/", response_model=HelloResponse)
async def root():
    """
    Root endpoint - Hello World
    """
    return {"message": "Hello, Blokus World!", "status": "success", "version": "1.0.0"}


@app.get("/api/hello", response_model=HelloResponse)
async def hello_world():
    """
    Hello World endpoint
    """
    return {
        "message": "Hello from Blokus API!",
        "status": "success",
        "version": "1.0.0",
    }


@app.get("/api/hello/{name}", response_model=HelloResponse)
async def hello_name(name: str):
    """
    Personalized hello endpoint
    """
    return {
        "message": f"Hello, {name}! Welcome to Blokus!",
        "status": "success",
        "version": "1.0.0",
    }


@app.get("/api/status")
async def status():
    """
    API status endpoint
    """
    return {
        "status": "running",
        "message": "Blokus API is up and running!",
        "endpoints": [
            {"path": "/", "description": "Root endpoint"},
            {"path": "/api/hello", "description": "Hello World"},
            {"path": "/api/hello/{name}", "description": "Personalized hello"},
            {"path": "/api/status", "description": "API status"},
            {"path": "/api/docs", "description": "Swagger docs"},
            {"path": "/api/redoc", "description": "ReDoc docs"},
        ],
    }


# Game-related endpoints (will be implemented later)
@app.get("/api/game/new")
async def new_game():
    """
    Create a new game
    """
    # This will be implemented when we integrate the game logic
    return {
        "message": "New game endpoint (to be implemented)",
        "status": "not_implemented",
    }


@app.get("/api/game/state")
async def get_game_state():
    """
    Get current game state
    """
    # This will be implemented when we integrate the game logic
    return {
        "message": "Game state endpoint (to be implemented)",
        "status": "not_implemented",
    }


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload for development
        log_level="info",
        access_log=True,
    )
