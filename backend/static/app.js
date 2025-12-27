/**
 * Blokus Game Client
 * Complete game logic and UI management
 */

class BlokusGame {
    constructor() {
        // API Configuration
        this.API_BASE = window.location.origin;

        // Game State
        this.gameId = null;
        this.currentPlayer = 0;
        this.players = [];
        this.board = null;
        this.pieces = [];
        this.availablePieces = [[], [], [], []]; // For each player
        this.gameOver = false;

        // UI State
        this.selectedPieceId = null;
        this.rotation = 0;
        this.flipHorizontal = false;
        this.flipVertical = false;
        this.previewCells = [];

        // Player colors
        this.playerColors = ['#3b82f6', '#eab308', '#ef4444', '#22c55e'];
        this.playerNames = ['Blue', 'Yellow', 'Red', 'Green'];

        // Initialize
        this.init();
    }

    async init() {
        console.log('Initializing Blokus game...');

        // Setup event listeners
        this.setupEventListeners();

        // Create the board
        this.createBoard();

        // Load pieces from backend
        await this.loadPieces();

        console.log('Game initialized!');
    }

    setupEventListeners() {
        // New game button
        document.getElementById('newGameBtn').addEventListener('click', () => {
            this.createNewGame();
        });

        // Skip turn button
        document.getElementById('skipTurnBtn').addEventListener('click', () => {
            this.skipTurn();
        });

        // Rotation buttons
        document.getElementById('rotateLeftBtn').addEventListener('click', () => {
            this.rotate(-1);
        });

        document.getElementById('rotateRightBtn').addEventListener('click', () => {
            this.rotate(1);
        });

        // Flip buttons
        document.getElementById('flipHorizontalBtn').addEventListener('click', () => {
            this.flipHorizontal = !this.flipHorizontal;
            this.updateFlipDisplay();
        });

        document.getElementById('flipVerticalBtn').addEventListener('click', () => {
            this.flipVertical = !this.flipVertical;
            this.updateFlipDisplay();
        });
    }

    createBoard() {
        const board = document.getElementById('gameBoard');
        board.innerHTML = '';

        // Create 20x20 grid
        for (let row = 0; row < 20; row++) {
            for (let col = 0; col < 20; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell bg-white rounded-sm cursor-pointer hover:bg-blue-50';
                cell.dataset.row = row;
                cell.dataset.col = col;

                cell.addEventListener('mouseenter', () => this.handleCellHover(row, col));
                cell.addEventListener('mouseleave', () => this.clearPreview());
                cell.addEventListener('click', () => this.handleCellClick(row, col));

                board.appendChild(cell);
            }
        }
    }

    async loadPieces() {
        try {
            const response = await fetch(`${this.API_BASE}/pieces`);
            if (!response.ok) throw new Error('Failed to load pieces');

            const data = await response.json();
            this.pieces = data.pieces;

            console.log(`Loaded ${this.pieces.length} pieces`);
            this.renderPiecesPalette();
        } catch (error) {
            console.error('Error loading pieces:', error);
            this.showError('Failed to load game pieces');
        }
    }

    renderPiecesPalette() {
        const container = document.getElementById('piecesContainer');
        container.innerHTML = '';

        this.pieces.forEach((piece, index) => {
            // Only show pieces that are still available for current player
            if (this.availablePieces[this.currentPlayer] &&
                this.availablePieces[this.currentPlayer][index]) {
                const pieceDiv = this.createPiecePreview(piece, index);
                container.appendChild(pieceDiv);
            }
        });
    }

    createPiecePreview(piece, pieceId) {
        // Calculate bounding box
        const coords = piece.map(p => ({ x: p.x, y: p.y }));
        const minX = Math.min(...coords.map(c => c.x));
        const maxX = Math.max(...coords.map(c => c.x));
        const minY = Math.min(...coords.map(c => c.y));
        const maxY = Math.max(...coords.map(c => c.y));

        const width = maxX - minX + 1;
        const height = maxY - minY + 1;

        // Create container
        const container = document.createElement('div');
        container.className = 'piece-preview bg-white rounded-lg border-2 border-gray-200 hover:border-blue-400 transition-all';
        container.dataset.pieceId = pieceId;
        container.style.gridTemplateColumns = `repeat(${width}, 12px)`;
        container.style.gridTemplateRows = `repeat(${height}, 12px)`;

        // Create grid
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const square = document.createElement('div');

                // Check if this position has a piece square
                const hasPiece = coords.some(c =>
                    c.x - minX === x && c.y - minY === y
                );

                if (hasPiece) {
                    square.className = 'piece-square bg-blue-500';
                    square.style.backgroundColor = this.playerColors[this.currentPlayer];
                } else {
                    square.className = 'piece-square bg-transparent';
                }

                container.appendChild(square);
            }
        }

        // Add piece count badge
        const badge = document.createElement('div');
        badge.className = 'absolute -top-1 -right-1 bg-gray-800 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center font-bold';
        badge.textContent = coords.length;
        container.style.position = 'relative';
        container.appendChild(badge);

        // Click handler
        container.addEventListener('click', () => {
            this.selectPiece(pieceId);
        });

        return container;
    }

    selectPiece(pieceId) {
        // Check if piece is available for current player
        if (this.availablePieces[this.currentPlayer] &&
            !this.availablePieces[this.currentPlayer][pieceId]) {
            this.showError('This piece has already been used');
            return;
        }

        // Update selection
        this.selectedPieceId = pieceId;

        // Update UI
        document.querySelectorAll('.piece-preview').forEach(el => {
            el.classList.remove('selected');
        });

        const selectedEl = document.querySelector(`[data-piece-id="${pieceId}"]`);
        if (selectedEl) {
            selectedEl.classList.add('selected');
        }

        console.log(`Selected piece ${pieceId}`);
    }

    rotate(direction) {
        // Direction: 1 for clockwise, -1 for counter-clockwise
        this.rotation = (this.rotation + direction + 4) % 4;

        // Update display
        const degrees = this.rotation * 90;
        document.getElementById('rotationDisplay').textContent = `${degrees}°`;

        console.log(`Rotated to ${degrees}°`);
    }

    updateFlipDisplay() {
        // Update button styles to show active state
        const hBtn = document.getElementById('flipHorizontalBtn');
        const vBtn = document.getElementById('flipVerticalBtn');

        if (this.flipHorizontal) {
            hBtn.classList.add('bg-blue-100', 'border-blue-400');
            hBtn.classList.remove('bg-white');
        } else {
            hBtn.classList.remove('bg-blue-100', 'border-blue-400');
            hBtn.classList.add('bg-white');
        }

        if (this.flipVertical) {
            vBtn.classList.add('bg-blue-100', 'border-blue-400');
            vBtn.classList.remove('bg-white');
        } else {
            vBtn.classList.remove('bg-blue-100', 'border-blue-400');
            vBtn.classList.add('bg-white');
        }
    }

    getRotatedPiece(piece, rotation) {
        // Match backend rotation exactly!
        // Backend: 0=(x,y), 1=(y,-x), 2=(-x,-y), 3=(-y,x)
        let coords = piece.map(p => ({ x: p.x, y: p.y }));

        // Apply flips first (before rotation)
        if (this.flipHorizontal) {
            coords = coords.map(({ x, y }) => ({ x: -x, y: y }));
        }
        if (this.flipVertical) {
            coords = coords.map(({ x, y }) => ({ x: x, y: -y }));
        }

        // Then apply rotation
        if (rotation === 1) {
            coords = coords.map(({ x, y }) => ({ x: y, y: -x }));
        } else if (rotation === 2) {
            coords = coords.map(({ x, y }) => ({ x: -x, y: -y }));
        } else if (rotation === 3) {
            coords = coords.map(({ x, y }) => ({ x: -y, y: x }));
        }

        return coords;
    }

    handleCellHover(row, col) {
        if (!this.selectedPieceId && this.selectedPieceId !== 0) return;
        if (this.gameOver) return;

        // Clear previous preview
        this.clearPreview();

        // Get rotated piece
        const piece = this.pieces[this.selectedPieceId];
        const rotated = this.getRotatedPiece(piece, this.rotation);

        // Preview placement
        this.previewCells = [];
        let isValid = true;

        rotated.forEach(offset => {
            // Backend uses x=row, y=col (note: swapped from typical!)
            const r = row + offset.x;
            const c = col + offset.y;

            if (r < 0 || r >= 20 || c < 0 || c >= 20) {
                isValid = false;
                return;
            }

            const cell = document.querySelector(`[data-row="${r}"][data-col="${c}"]`);
            if (cell) {
                this.previewCells.push(cell);
                cell.classList.add(isValid ? 'bg-blue-200' : 'bg-red-200');
                cell.style.opacity = '0.7';
            }
        });
    }

    clearPreview() {
        this.previewCells.forEach(cell => {
            cell.classList.remove('bg-blue-200', 'bg-red-200');
            cell.style.opacity = '1';
        });
        this.previewCells = [];
    }

    async handleCellClick(row, col) {
        if (!this.gameId) {
            this.showError('Please create a new game first');
            return;
        }

        if (this.gameOver) {
            this.showError('Game is over');
            return;
        }

        if (this.selectedPieceId === null) {
            this.showError('Please select a piece first');
            return;
        }

        // Attempt to make move
        await this.makeMove(this.selectedPieceId, row, col, this.rotation, this.flipHorizontal, this.flipVertical);
    }

    async createNewGame() {
        try {
            const btn = document.getElementById('newGameBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="loading-dots">Creating game</span>';

            const response = await fetch(`${this.API_BASE}/games`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) throw new Error('Failed to create game');

            const data = await response.json();
            this.gameId = data.game_id;

            console.log('New game created:', this.gameId);

            // Update UI
            document.getElementById('gameIdDisplay').classList.remove('hidden');
            document.getElementById('gameIdDisplay').querySelector('code').textContent = this.gameId;

            // Load game state
            await this.refreshGameState();

            // Reset UI
            btn.disabled = false;
            btn.innerHTML = `
                <svg class="inline-block w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                </svg>
                New Game
            `;

            this.showSuccess('Game created! Start playing.');

        } catch (error) {
            console.error('Error creating game:', error);
            this.showError('Failed to create game');

            const btn = document.getElementById('newGameBtn');
            btn.disabled = false;
            btn.innerHTML = 'New Game';
        }
    }

    async refreshGameState() {
        if (!this.gameId) return;

        try {
            const response = await fetch(`${this.API_BASE}/games/${this.gameId}`);
            if (!response.ok) throw new Error('Failed to fetch game state');

            const state = await response.json();

            // Update game state
            this.currentPlayer = state.current_player;
            this.players = state.players;
            this.board = state.board;
            this.availablePieces = state.remaining_pieces;
            this.gameOver = state.game_over;

            // Update UI
            this.updateBoard();
            this.updatePlayerInfo(state);
            this.updateStatus(state);
            this.renderPiecesPalette(); // Update piece colors

            // Check for game over
            if (this.gameOver) {
                this.showGameOver(state);
            }

        } catch (error) {
            console.error('Error refreshing game state:', error);
        }
    }

    updateBoard() {
        if (!this.board) return;

        // Update all cells
        for (let row = 0; row < 20; row++) {
            for (let col = 0; col < 20; col++) {
                const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                if (!cell) continue;

                // Check which player occupies this cell
                let occupied = false;
                for (let player = 0; player < 4; player++) {
                    if (this.board[row][col][player] === 1) {
                        cell.classList.add('occupied');
                        cell.style.color = this.playerColors[player];
                        occupied = true;
                        break;
                    }
                }

                if (!occupied) {
                    cell.classList.remove('occupied');
                    cell.style.color = '';
                }
            }
        }
    }

    updatePlayerInfo(state) {
        state.players.forEach((player, index) => {
            const playerDiv = document.getElementById(`player-${index}`);

            // Highlight current player
            if (index === this.currentPlayer && !this.gameOver) {
                playerDiv.classList.add('border-blue-500', 'shadow-xl', 'scale-105');
                playerDiv.classList.remove('border-transparent');
            } else {
                playerDiv.classList.remove('border-blue-500', 'shadow-xl', 'scale-105');
                playerDiv.classList.add('border-transparent');
            }

            // Update pieces count
            const piecesLeft = state.remaining_pieces[index].filter(p => p).length;
            document.getElementById(`pieces-${index}`).textContent = piecesLeft;

            // Update score
            document.getElementById(`score-${index}`).textContent = state.scores[index];
        });
    }

    updateStatus(state) {
        const statusEl = document.getElementById('gameStatus');

        if (this.gameOver) {
            statusEl.textContent = state.message || 'Game Over!';
            statusEl.className = 'text-red-600 font-bold';
        } else {
            const playerName = this.playerNames[this.currentPlayer];
            const playerColor = this.playerColors[this.currentPlayer];
            statusEl.innerHTML = `
                <span style="color: ${playerColor}">●</span>
                <span class="font-semibold">${playerName}'s Turn</span>
            `;
        }
    }

    async makeMove(pieceId, x, y, rotation, flipHorizontal, flipVertical) {
        try {
            const response = await fetch(`${this.API_BASE}/games/${this.gameId}/move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    player_id: this.currentPlayer,
                    piece_id: pieceId,
                    x: x,
                    y: y,
                    rotation: rotation,
                    flip_horizontal: flipHorizontal,
                    flip_vertical: flipVertical
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Invalid move');
            }

            const state = await response.json();

            // Update game state
            this.currentPlayer = state.current_player;
            this.board = state.board;
            this.availablePieces = state.remaining_pieces;
            this.gameOver = state.game_over;

            // Clear selection and reset transformations
            this.selectedPieceId = null;
            this.rotation = 0;
            this.flipHorizontal = false;
            this.flipVertical = false;
            document.getElementById('rotationDisplay').textContent = '0°';
            this.updateFlipDisplay();

            // Update UI
            this.updateBoard();
            this.updatePlayerInfo(state);
            this.updateStatus(state);
            this.renderPiecesPalette();

            // Check for game over
            if (this.gameOver) {
                this.showGameOver(state);
            }

        } catch (error) {
            console.error('Error making move:', error);
            this.showError(error.message);
        }
    }

    async skipTurn() {
        // Implementation for skipping turn
        // This would need backend support
        console.log('Skip turn not yet implemented');
    }

    showGameOver(state) {
        const modal = document.getElementById('gameOverModal');
        const content = document.getElementById('gameOverContent');

        // Find winner
        let winnerIndex = state.winner;
        if (winnerIndex === null || winnerIndex === undefined) {
            // Find player with highest score
            const maxScore = Math.max(...state.scores);
            winnerIndex = state.scores.indexOf(maxScore);
        }

        const winnerName = this.playerNames[winnerIndex];
        const winnerColor = this.playerColors[winnerIndex];

        content.innerHTML = `
            <div class="mb-4">
                <div class="text-6xl mb-4">🏆</div>
                <div class="text-2xl font-bold mb-2" style="color: ${winnerColor}">
                    ${winnerName} Wins!
                </div>
            </div>
            <div class="space-y-2">
                ${state.scores.map((score, i) => `
                    <div class="flex justify-between items-center text-lg">
                        <span class="flex items-center gap-2">
                            <span class="w-3 h-3 rounded-full" style="background-color: ${this.playerColors[i]}"></span>
                            ${this.playerNames[i]}
                        </span>
                        <span class="font-bold">${score}</span>
                    </div>
                `).join('')}
            </div>
        `;

        modal.classList.remove('hidden');
    }

    showError(message) {
        // Simple error notification
        const toast = document.createElement('div');
        toast.className = 'fixed top-4 right-4 bg-red-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-up';
        toast.textContent = message;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }

    showSuccess(message) {
        const toast = document.createElement('div');
        toast.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-up';
        toast.textContent = message;

        document.body.appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
}

// Initialize game when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('Starting Blokus game...');
    window.game = new BlokusGame();
});
