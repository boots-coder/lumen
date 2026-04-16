const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

const box = 20; // 每格大小
let snake = [
    { x: 9 * box, y: 10 * box }
];
let direction = 'RIGHT';
let food = {
    x: Math.floor(Math.random() * 20) * box,
    y: Math.floor(Math.random() * 20) * box
};
let score = 0;
let gameInterval;

// 监听键盘
window.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft' && direction !== 'RIGHT') direction = 'LEFT';
    else if (e.key === 'ArrowUp' && direction !== 'DOWN') direction = 'UP';
    else if (e.key === 'ArrowRight' && direction !== 'LEFT') direction = 'RIGHT';
    else if (e.key === 'ArrowDown' && direction !== 'UP') direction = 'DOWN';
});

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 画蛇
    for (let i = 0; i < snake.length; i++) {
        ctx.fillStyle = i === 0 ? '#4287f5' : '#7fc8f8';
        ctx.fillRect(snake[i].x, snake[i].y, box, box);
    }
    // 画食物
    ctx.fillStyle = '#e74c3c';
    ctx.fillRect(food.x, food.y, box, box);

    // 文字分数
    ctx.fillStyle = '#222';
    ctx.font = '20px Arial';
    ctx.fillText('分数: ' + score, 10, 390);

    // 新蛇头位置
    let headX = snake[0].x;
    let headY = snake[0].y;
    if (direction === 'LEFT') headX -= box;
    if (direction === 'RIGHT') headX += box;
    if (direction === 'UP') headY -= box;
    if (direction === 'DOWN') headY += box;

    // 吃到食物
    if (headX === food.x && headY === food.y) {
        score++;
        food = {
            x: Math.floor(Math.random() * 20) * box,
            y: Math.floor(Math.random() * 20) * box
        };
    } else {
        snake.pop();
    }

    // 新蛇头
    const newHead = { x: headX, y: headY };

    // 撞墙或撞自己
    if (
        headX < 0 || headX >= 400 || headY < 0 || headY >= 400 ||
        snake.some(seg => seg.x === headX && seg.y === headY)
    ) {
        clearInterval(gameInterval);
        ctx.fillStyle = 'rgba(0,0,0,0.5)';
        ctx.fillRect(0, 180, 400, 40);
        ctx.fillStyle = '#fff';
        ctx.font = '24px Arial';
        ctx.fillText('游戏结束', 145, 208);
        return;
    }

    snake.unshift(newHead);
}

gameInterval = setInterval(draw, 120);
