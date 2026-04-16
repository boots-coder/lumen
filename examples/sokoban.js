// 简单推箱子实现
const LEVEL = [
    ' #####',
    ' #  .#',
    ' #$$ #',
    ' # @ #',
    ' #####'
];

let map = [];
let player = {x: 0, y: 0};

function resetGame() {
    map = LEVEL.map(row => row.split(''));
    for (let y = 0; y < map.length; y++) {
        for (let x = 0; x < map[y].length; x++) {
            if (map[y][x] === '@') {
                player.x = x;
                player.y = y;
            }
        }
    }
    render();
}

function render() {
    let html = '';
    for (let y = 0; y < map.length; y++) {
        html += '<div class="row">';
        for (let x = 0; x < map[y].length; x++) {
            let cell = map[y][x];
            let classList = 'cell ';
            let text = '';
            switch (cell) {
                case '#': classList += 'wall'; text = '■'; break;
                case ' ': classList += 'floor'; text = ''; break;
                case '.': classList += 'target'; text = '●'; break;
                case '$': classList += 'box'; text = '□'; break;
                case '*': classList += 'box-on-target'; text = '★'; break;
                case '@': classList += 'player'; text = '☺'; break;
                case '+': classList += 'player target'; text = '☺'; break;
            }
            html += `<span class="${classList}">${text}</span>`;
        }
        html += '</div>';
    }
    document.getElementById('game').innerHTML = html;
}

function move(dx, dy) {
    let x = player.x;
    let y = player.y;
    let tx = x + dx;
    let ty = y + dy;
    let tcell = map[ty][tx];
    // 判断目标格
    if (tcell === '#' ) return;
    if (tcell === ' ' || tcell === '.') {
        updatePlayer(x, y, tx, ty);
    } else if (tcell === '$' || tcell === '*') {
        let bx = tx + dx;
        let by = ty + dy;
        let bcell = map[by][bx];
        if (bcell === ' ' || bcell === '.') {
            // 推箱子
            map[by][bx] = (bcell === '.' ? '*' : '$');
            map[ty][tx] = (tcell === '*' ? '+' : '@');
            updatePlayer(x, y, tx, ty);
        }
    }
    render();
    checkWin();
}

function updatePlayer(px, py, nx, ny) {
    if (map[py][px] === '+') {
        map[py][px] = '.';
    } else {
        map[py][px] = ' ';
    }
    if (map[ny][nx] === '.') {
        map[ny][nx] = '+';
    } else {
        map[ny][nx] = '@';
    }
    player.x = nx;
    player.y = ny;
}

function checkWin() {
    for (let y = 0; y < map.length; y++) {
        for (let x = 0; x < map[y].length; x++) {
            if (map[y][x] === '$') return; // 还有箱子没到目标点
        }
    }
    setTimeout(() => alert('恭喜通关！'), 100);
}

document.addEventListener('keydown', e => {
    switch (e.key) {
        case 'ArrowUp': move(0, -1); break;
        case 'ArrowDown': move(0, 1); break;
        case 'ArrowLeft': move(-1, 0); break;
        case 'ArrowRight': move(1, 0); break;
    }
});

resetGame();
