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
        this.autoAi = false;
        this.autoAiInterval = null;
        this.socket = null;
        this.pollingInterval = null;

        // Replay State
        this.history = [];
        this.historyIndex = -1;
        this.isReplaying = false;
        this.replayInterval = null;

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

        // Close Modal Button
        const closeBtn = document.getElementById('closeModalBtn');
        if (closeBtn) {
            closeBtn.onclick = () => {
                console.log('Close button clicked');
                document.getElementById('gameOverModal').classList.add('hidden');
                this.startReplay();
            };
        }

        // Replay Controls
        document.getElementById('replayStartBtn').addEventListener('click', () => this.setReplayIndex(0));
        document.getElementById('replayBackBtn').addEventListener('click', () => this.stepReplay(-1));
        document.getElementById('replayPlayBtn').addEventListener('click', () => this.toggleReplay());
        document.getElementById('replayForwardBtn').addEventListener('click', () => this.stepReplay(1));
        document.getElementById('replayEndBtn').addEventListener('click', () => this.setReplayIndex(this.history.length - 1));

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (this.isReplaying) {
                if (e.key === 'ArrowLeft') this.stepReplay(-1);
                if (e.key === 'ArrowRight') this.stepReplay(1);
                if (e.key === ' ') {
                    e.preventDefault(); // Prevent scroll
                    this.toggleReplay();
                }
            }
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

        // AI buttons
        document.getElementById('aiMoveBtn').addEventListener('click', () => {
            this.makeAiMove();
        });

        document.getElementById('autoAiBtn').addEventListener('click', () => {
            this.toggleAutoAi();
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

            // Collect player configurations
            const playerConfigs = [];
            for (let i = 0; i < 4; i++) {
                const typeEl = document.getElementById(`player-${i}-type`);
                if (typeEl) {
                    playerConfigs.push({
                        player_id: i,
                        type: typeEl.value
                    });
                }
            }

            const response = await fetch(`${this.API_BASE}/games`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ player_configs: playerConfigs })
            });

            if (!response.ok) throw new Error('Failed to create game');

            const data = await response.json();
            this.gameId = data.game_id;

            console.log('New game created:', this.gameId);

            // Update UI
            document.getElementById('gameIdDisplay').classList.remove('hidden');
            document.getElementById('gameIdDisplay').querySelector('code').textContent = this.gameId;

            // Setup Websocket
            this.initWebsocket();

            // Show AI buttons
            document.getElementById('aiMoveBtn').classList.remove('hidden');
            document.getElementById('autoAiBtn').classList.remove('hidden');

            this.showSuccess('Game created! Start playing.');

            // Load initial state
            await this.refreshGameState();

            // Reset UI
            btn.disabled = false;
            btn.innerHTML = `
                <svg class="inline-block w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/>
                </svg>
                New Game
            `;

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

            // Automatically make AI move if current player is AI
            const currentPlayerObj = state.players[this.currentPlayer];
            if (currentPlayerObj && currentPlayerObj.type === 'ai' && !this.gameOver) {
                // Delay slightly for better UX
                setTimeout(() => this.makeAiMove(), 500);
            }

            // Check for game over
            if (this.gameOver) {
                this.showGameOver(state);
            }

        } catch (error) {
            console.error('Error refreshing game state:', error);
        }
    }

    initWebsocket() {
        if (this.socket) {
            this.socket.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.gameId}`;

        console.log(`Connecting to websocket: ${wsUrl}`);
        this.socket = new WebSocket(wsUrl);

        this.socket.onmessage = (event) => {
            console.log('Websocket message received:', event.data);
            this.refreshGameState();
        };

        this.socket.onclose = () => {
            console.log('Websocket closed. Falling back to polling.');
            if (!this.gameOver && this.gameId) {
                // If websocket fails, use polling as fallback
                if (!this.pollingInterval) {
                    this.pollingInterval = setInterval(() => {
                        if (!this.gameOver) {
                            this.refreshGameState();
                        } else {
                            clearInterval(this.pollingInterval);
                            this.pollingInterval = null;
                        }
                    }, 2000); // Poll every 2 seconds
                }
            }
        };

        this.socket.onerror = (error) => {
            console.error('Websocket error:', error);
        };
    }

    async makeAiMove() {
        if (!this.gameId || this.gameOver) return;

        try {
            const btn = document.getElementById('aiMoveBtn');
            btn.disabled = true;

            const response = await fetch(`${this.API_BASE}/games/${this.gameId}/ai-move`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'AI move failed');
            }

            // State will be updated via websocket, but we refresh anyway for safety
            await this.refreshGameState();

            btn.disabled = false;
        } catch (error) {
            // Ignore "Game already over" error as it's expected at the end
            if (error.message.includes('Game already over')) {
                console.log('AI Move stopped: Game is over');
                this.gameOver = true;
                if (this.autoAi) {
                    this.toggleAutoAi();
                }
                return;
            }

            console.error('AI move error:', error);
            this.showError(error.message);
            const btn = document.getElementById('aiMoveBtn');
            btn.disabled = false;

            // If AI failed (e.g. no moves), stop auto-ai
            if (this.autoAi) {
                this.toggleAutoAi();
            }
        }

    }


    async startReplay() {
        if (!this.gameId) return;

        try {
            // Fetch history
            const response = await fetch(`${this.API_BASE}/games/${this.gameId}/history`);
            if (!response.ok) throw new Error('Failed to fetch history');

            this.history = await response.json();
            this.isReplaying = true;
            this.historyIndex = this.history.length - 1; // Start at end

            // Show controls
            document.getElementById('replayControls').classList.remove('hidden');

            this.updateReplayUI();

        } catch (error) {
            console.error('Error starting replay:', error);
            this.showError(`Replay Failed: ${error.message}`);
        }
    }

    setReplayIndex(index) {
        if (!this.isReplaying || !this.history.length) return;

        // Clamp index
        this.historyIndex = Math.max(0, Math.min(index, this.history.length - 1));

        this.updateReplayState();
    }

    stepReplay(delta) {
        this.setReplayIndex(this.historyIndex + delta);
    }

    toggleReplay() {
        if (this.replayInterval) {
            // Pause
            clearInterval(this.replayInterval);
            this.replayInterval = null;
        } else {
            // Play
            if (this.historyIndex >= this.history.length - 1) {
                this.historyIndex = -1; // Loop back to start if at end
            }

            this.stepReplay(1); // Make first move immediately

            this.replayInterval = setInterval(() => {
                if (this.historyIndex >= this.history.length - 1) {
                    this.toggleReplay(); // Stop at end
                } else {
                    this.stepReplay(1);
                }
            }, 1000); // 1 state per second
        }
        this.updateReplayUI();
    }

    updateReplayState() {
        if (!this.history[this.historyIndex]) return;

        const state = this.history[this.historyIndex];

        // Update Board using history state
        this.updateBoard(state.board);

        // Update UI elements from history state
        this.updatePlayerInfo(state); // Ensure this method handles the history state format
        // Scores are in state.scores, players... we assume players don't change names/types
        // But history state might have slightly different structure? 
        // Backend history snapshot: {board, current_player, scores, remaining_pieces...}
        // Our updatePlayerInfo expects {players, remaining_pieces, scores}
        // HISTORY SNAPSHOT has 'remaining_pieces' and 'scores'. It might miss 'players' list.
        // We can reuse this.players if not present.

        const uiState = {
            ...state,
            players: this.players, // Use current static player info
        };
        this.updatePlayerInfo(uiState);
    }

    updateReplayUI() {
        const playIcon = document.getElementById('playIcon');
        const pauseIcon = document.getElementById('pauseIcon');

        if (this.replayInterval) {
            playIcon.classList.add('hidden');
            pauseIcon.classList.remove('hidden');
        } else {
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
        }
    }

    updateBoard(boardData = null) {
        const boardToRender = boardData || this.board;
        if (!boardToRender) return;

        // Update all cells
        for (let row = 0; row < 20; row++) {
            for (let col = 0; col < 20; col++) {
                const cell = document.querySelector(`[data-row="${row}"][data-col="${col}"]`);
                if (!cell) continue;

                // Check which player occupies this cell
                let occupied = false;
                for (let player = 0; player < 4; player++) {
                    if (boardToRender[row][col][player] === 1) {
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

            // Update player type badge
            const nameEl = playerDiv.querySelector('.font-bold');
            if (nameEl) {
                const existingBadge = nameEl.querySelector('.ai-badge');
                if (existingBadge) existingBadge.remove();

                if (state.players[index].type === 'ai') {
                    const badge = document.createElement('span');
                    badge.className = 'ai-badge ml-2 px-1.5 py-0.5 bg-purple-100 text-purple-700 text-[10px] rounded uppercase font-bold';
                    badge.textContent = 'AI';
                    nameEl.appendChild(badge);
                }
            }
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
            // Update skip button visibility
            const skipBtn = document.getElementById('skipTurnBtn');
            if (skipBtn) {
                // If the player has no moves but it's their turn, show skip
                // Actually, the backend auto-skips, so this button is mostly a fallback
                // or useful if the player thinks they are stuck.
                skipBtn.classList.remove('hidden');
            }

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
        if (!this.gameId || this.gameOver) return;

        try {
            const btn = document.getElementById('skipTurnBtn');
            btn.disabled = true;

            const response = await fetch(`${this.API_BASE}/games/${this.gameId}/skip`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Skip failed');
            }

            // State will be updated via websocket/polling
            await this.refreshGameState();
            this.showSuccess('Turn skipped');

            btn.disabled = false;
        } catch (error) {
            console.error('Skip turn error:', error);
            this.showError(error.message);
            const btn = document.getElementById('skipTurnBtn');
            btn.disabled = false;
        }
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
