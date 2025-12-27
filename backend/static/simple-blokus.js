/**
 * Minimal Blokus React Application
 * Simple 20x20 board with title
 */

class SimpleBlokusApp extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });

        // Game state
        this.gameId = null;
        this.players = [
            { name: 'Blue', color: '#3b82f6', tilesLeft: 21, score: 0 },
            { name: 'Yellow', color: '#eab308', tilesLeft: 21, score: 0 },
            { name: 'Red', color: '#ef4444', tilesLeft: 21, score: 0 },
            { name: 'Green', color: '#22c55e', tilesLeft: 21, score: 0 }
        ];
        this.currentPlayerIndex = 0;

        // Pieces data
        this.pieces = [];
        this.totalSquaresInPieces = 0;
    }

    connectedCallback() {
        this.render();
        this.fetchPieces();
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }

                .app-container {
                    text-align: center;
                }

                h1 {
                    color: #2563eb;
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }

                .game-info {
                    background-color: #f0f9ff;
                    border: 2px solid #3b82f6;
                    border-radius: 8px;
                    padding: 15px;
                    margin-bottom: 20px;
                    text-align: center;
                }

                .game-id {
                    font-family: monospace;
                    font-weight: bold;
                    color: #1e40af;
                    font-size: 1.1rem;
                    margin-top: 10px;
                    padding: 8px 12px;
                    background-color: #e0f2fe;
                    border-radius: 4px;
                    display: inline-block;
                }

                .player-area {
                    display: flex;
                    margin: 20px 0;
                    align-items: flex-start;
                    position: relative;
                    min-width: 800px;
                    padding-right: 150px; /* Space for yellow player */
                }

                .player-blue-side {
                    width: 120px;
                    padding: 10px;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                    background-color: #dbeafe;
                    border: 2px solid #3b82f6;
                    margin-top: 0;
                    margin-right: 20px; /* Equal spacing */
                }

                .board-container {
                    border: 3px solid #3b82f6;
                    border-radius: 8px;
                    overflow: hidden;
                    flex: 1;
                }

                .player-yellow-top {
                    position: absolute;
                    top: 0;
                    right: 20px;
                    width: 120px;
                    padding: 10px;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                    background-color: #fef9c3;
                    border: 2px solid #eab308;
                    z-index: 10;
                }

                .board {
                    display: grid;
                    grid-template-columns: repeat(20, 30px);
                    grid-template-rows: repeat(20, 30px);
                    gap: 1px;
                    background-color: #e5e7eb;
                }

                .cell {
                    width: 30px;
                    height: 30px;
                    background-color: white;
                    border: 1px solid #f3f4f6;
                    box-sizing: border-box;
                }

                .cell:hover {
                    background-color: #f0f9ff;
                    cursor: pointer;
                }

                .players-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr 1fr;
                    gap: 20px;
                    margin-top: 20px;
                    max-width: 800px;
                    margin-left: auto;
                    margin-right: auto;
                }



                .player-info {
                    width: 150px;
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 0.9rem;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                    text-align: center;
                }

                .player-blue { 
                    background-color: #dbeafe; 
                    border: 2px solid #3b82f6;
                    grid-column: 1;
                    grid-row: 1;
                }

                .player-yellow { 
                    background-color: #fef9c3; 
                    border: 2px solid #eab308;
                    grid-column: 2;
                    grid-row: 1;
                }

                .player-red { 
                    background-color: #fee2e2; 
                    border: 2px solid #ef4444;
                    grid-column: 1;
                    grid-row: 2;
                }

                .player-green { 
                    background-color: #d1fae5; 
                    border: 2px solid #22c55e;
                    grid-column: 2;
                    grid-row: 2;
                }

                .player-name {
                    font-weight: bold;
                    margin-bottom: 5px;
                }

                .player-color {
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    margin-right: 5px;
                    display: inline-block;
                }

                .player-stats {
                    margin-top: 8px;
                    font-size: 0.8rem;
                    text-align: left;
                }

                .stat-row {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 3px;
                }

                .new-game-btn {
                    background-color: #3b82f6;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 6px;
                    font-size: 1rem;
                    cursor: pointer;
                    margin-top: 15px;
                    transition: background-color 0.2s;
                }

                .new-game-btn:hover {
                    background-color: #2563eb;
                }
            </style>

            <div class="app-container">
                <h1>Blokus</h1>
                
                <div class="game-info">
                    <div>Welcome to Blokus!</div>
                    <div style="margin: 10px 0; font-size: 0.9rem;">
                        Click "New Game" to start playing
                    </div>
                    ${this.gameId ? `
                        <div>
                            Game ID: <span class="game-id">${this.gameId}</span>
                        </div>
                    ` : `
                        <div style="font-style: italic; color: #6b7280;">
                            No active game
                        </div>
                    `}
                    
                    <button class="new-game-btn" id="newGameBtn">New Game</button>
                </div>
                
                <!-- Player area with board and blue player side by side -->
                <div class="player-area">
                    <!-- Blue player on the left side -->
                    <div class="player-blue-side">
                        <div class="player-name">
                            <span class="player-color" style="background-color: #3b82f6;"></span>
                            Blue
                        </div>
                        <div class="player-stats">
                            <div class="stat-row">
                                <span>Tiles left:</span>
                                <span id="blueTiles">21</span>
                            </div>
                            <div class="stat-row">
                                <span>Score:</span>
                                <span id="blueScore">0</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Board on the right side -->
                    <div class="board-container">
                        <div class="board" id="board"></div>
                    </div>
                    
                    <!-- Yellow player in top-right corner -->
                    <div class="player-yellow-top">
                        <div class="player-name">
                            <span class="player-color" style="background-color: #eab308;"></span>
                            Yellow
                        </div>
                        <div class="player-stats">
                            <div class="stat-row">
                                <span>Tiles left:</span>
                                <span id="yellowTiles">21</span>
                            </div>
                            <div class="stat-row">
                                <span>Score:</span>
                                <span id="yellowScore">0</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Player info boxes - moved outside the board -->
                <div class="players-container">
                    <div class="player-info player-red">
                        <div class="player-name">
                            <span class="player-color" style="background-color: #ef4444;"></span>
                            Red
                        </div>
                        <div class="player-stats">
                            <div class="stat-row">
                                <span>Tiles left:</span>
                                <span id="redTiles">21</span>
                            </div>
                            <div class="stat-row">
                                <span>Score:</span>
                                <span id="redScore">0</span>
                            </div>
                        </div>
                    </div>

                    <div class="player-info player-green">
                        <div class="player-name">
                            <span class="player-color" style="background-color: #22c55e;"></span>
                            Green
                        </div>
                        <div class="player-stats">
                            <div class="stat-row">
                                <span>Tiles left:</span>
                                <span id="greenTiles">21</span>
                            </div>
                            <div class="stat-row">
                                <span>Score:</span>
                                <span id="greenScore">0</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Create the board cells
        this.setupBoard();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Set up new game button
        const newGameBtn = this.shadowRoot.getElementById('newGameBtn');
        if (newGameBtn) {
            newGameBtn.addEventListener('click', () => {
                this.createNewGame();
            });
        }
    }

    setupBoard() {
        const board = this.shadowRoot.getElementById('board');

        // Create 20x20 = 400 cells
        for (let row = 0; row < 20; row++) {
            for (let col = 0; col < 20; col++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = row;
                cell.dataset.col = col;

                // Add click handler
                cell.addEventListener('click', () => {
                    this.handleCellClick(cell, row, col);
                });

                board.appendChild(cell);
            }
        }
    }

    async fetchPieces() {
        try {
            console.log('Fetching pieces from backend...');

            // Call backend to get pieces
            const response = await fetch('http://localhost:8000/pieces');

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const piecesData = await response.json();
            this.pieces = piecesData.pieces || [];

            // Count total squares in all pieces
            // Pieces are in format: [{"x": 0, "y": 0}, {"x": 1, "y": 0}, ...]
            // Each object represents a square in the piece
            this.totalSquaresInPieces = 0;
            this.pieces.forEach((piece, index) => {
                const squareCount = piece.length; // Each array element is a square
                this.totalSquaresInPieces += squareCount;
                console.log(`Piece ${index + 1}: ${squareCount} squares`);
            });

            console.log(`Total squares in all pieces: ${this.totalSquaresInPieces}`);
            console.log(`Fetched ${this.pieces.length} pieces from backend`);

        } catch (error) {
            console.error('Error fetching pieces:', error);
            // Don't show alert to user as this is background loading
        }
    }

    async createNewGame() {
        try {
            console.log('Creating new game...');

            // Call backend to create new game
            const response = await fetch('http://localhost:8000/games', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    players: 4
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const gameData = await response.json();
            this.gameId = gameData.game_id;

            console.log('New game created:', this.gameId);

            // Update UI with game ID
            this.updateGameInfo();

        } catch (error) {
            console.error('Error creating new game:', error);
            alert('Failed to create new game. See console for details.');
        }
    }

    updateGameInfo() {
        // Update player stats
        this.players.forEach(player => {
            const tilesElement = this.shadowRoot.getElementById(`${player.name.toLowerCase()}Tiles`);
            const scoreElement = this.shadowRoot.getElementById(`${player.name.toLowerCase()}Score`);

            if (tilesElement) tilesElement.textContent = player.tilesLeft;
            if (scoreElement) scoreElement.textContent = player.score;
        });

        // Re-render to show game ID
        this.render();
    }

    handleCellClick(cell, row, col) {
        // Cell click handler - just log for now
        console.log(`Clicked cell at row ${row}, col ${col}`);
    }
}

// Register the custom element
customElements.define('simple-blokus', SimpleBlokusApp);

// Auto-load the app when the page loads
document.addEventListener('DOMContentLoaded', function () {
    console.log('Simple Blokus app loaded!');

    // Create and append the app
    const app = document.createElement('simple-blokus');
    document.body.appendChild(app);
});