// 五子棋小游戏核心函数

// 1. 绘制棋盘
function drawBoard(ctx, size) {
    const cell = ctx.canvas.width / size;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.strokeStyle = '#333';
    for (let i = 0; i < size; i++) {
        // 横线
        ctx.beginPath();
        ctx.moveTo(cell / 2, cell / 2 + i * cell);
        ctx.lineTo(ctx.canvas.width - cell / 2, cell / 2 + i * cell);
        ctx.stroke();
        // 竖线
        ctx.beginPath();
        ctx.moveTo(cell / 2 + i * cell, cell / 2);
        ctx.lineTo(cell / 2 + i * cell, ctx.canvas.height - cell / 2);
        ctx.stroke();
    }
}

// 2. 绘制棋子
function drawPieces(ctx, board, size) {
    const cell = ctx.canvas.width / size;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            if (board[y][x] !== 0) {
                ctx.beginPath();
                ctx.arc(cell / 2 + x * cell, cell / 2 + y * cell, cell * 0.4, 0, 2 * Math.PI);
                ctx.fillStyle = board[y][x] === 1 ? '#111' : '#fff';
                ctx.fill();
                ctx.strokeStyle = '#555';
                ctx.stroke();
            }
        }
    }
}

// 3. 初始化棋盘
function initBoard(size) {
    return Array.from({ length: size }, () => Array(size).fill(0));
}

// 4. 处理用户点击落子
function handleClick(x, y, board, currentPlayer) {
    if (board[y][x] !== 0) {
        return { board, winner: null };
    }
    const newBoard = board.map(row => row.slice());
    newBoard[y][x] = currentPlayer;
    const winner = checkWin(x, y, newBoard, currentPlayer) ? currentPlayer : null;
    return { board: newBoard, winner };
}

// 5. 检查当前玩家是否胜利
function checkWin(x, y, board, player) {
    const size = board.length;
    const dirs = [[1,0],[0,1],[1,1],[1,-1]];
    for (let [dx, dy] of dirs) {
        let count = 1;
        for (let i = 1; i < 5; i++) {
            let nx = x + dx * i, ny = y + dy * i;
            if (nx >= 0 && nx < size && ny >= 0 && ny < size && board[ny][nx] === player) count++; else break;
        }
        for (let i = 1; i < 5; i++) {
            let nx = x - dx * i, ny = y - dy * i;
            if (nx >= 0 && nx < size && ny >= 0 && ny < size && board[ny][nx] === player) count++; else break;
        }
        if (count >= 5) return true;
    }
    return false;
}

// 6. 重置游戏
function restartGame(ctx, size, statusDiv) {
    const board = initBoard(size);
    const currentPlayer = 1;
    const winner = null;
    drawBoard(ctx, size);
    drawPieces(ctx, board, size);
    updateStatus(statusDiv, currentPlayer, winner);
    return { board, currentPlayer, winner };
}

// 7. 更新状态栏内容
function updateStatus(statusDiv, currentPlayer, winner) {
    if (winner) {
        statusDiv.textContent = (winner === 1 ? '黑棋' : '白棋') + '胜利！';
    } else {
        statusDiv.textContent = (currentPlayer === 1 ? '黑棋' : '白棋') + '回合';
    }
}
