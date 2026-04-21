// Pomodoro Timer core logic

class PomodoroTimer {
  /**
   * @param {number} duration - Session duration in seconds
   * @param {(mm: number, ss: number, progress: number) => void} onTick - Called every tick
   * @param {(isWork: boolean) => void} onSessionSwitch - Called when session switches
   */
  constructor(duration, onTick, onSessionSwitch) {
    this.workDuration = 25 * 60;
    this.breakDuration = 5 * 60;
    this.duration = duration; // in seconds
    this.onTick = onTick;
    this.onSessionSwitch = onSessionSwitch;
    this.isWorkSession = true;
    this.remaining = duration;
    this.timer = null;
    this.startTimestamp = null;
  }

  start() {
    if (this.timer) return;
    this.startTimestamp = Date.now() - (this.duration - this.remaining) * 1000;
    this.timer = setInterval(() => this._tick(), 1000);
  }

  pause() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  reset(isWorkSession = this.isWorkSession) {
    this.isWorkSession = isWorkSession;
    this.duration = isWorkSession ? this.workDuration : this.breakDuration;
    this.remaining = this.duration;
    this.pause();
    this._emitTick();
  }

  switchSession() {
    this.isWorkSession = !this.isWorkSession;
    this.duration = this.isWorkSession ? this.workDuration : this.breakDuration;
    this.remaining = this.duration;
    this.pause();
    this._emitTick();
    if (this.onSessionSwitch) this.onSessionSwitch(this.isWorkSession);
  }

  isRunning() {
    return !!this.timer;
  }

  getTimeLeft() {
    const mm = Math.floor(this.remaining / 60);
    const ss = this.remaining % 60;
    return { mm, ss };
  }

  getProgress() {
    return 1 - this.remaining / this.duration;
  }

  _tick() {
    this.remaining--;
    if (this.remaining <= 0) {
      if (this.isWorkSession && this.onSessionSwitch) {
        this.onSessionSwitch(false);
      }
      this.switchSession();
    } else {
      this._emitTick();
    }
  }

  _emitTick() {
    if (this.onTick) {
      const { mm, ss } = this.getTimeLeft();
      this.onTick(mm, ss, this.getProgress());
    }
  }
}

// Task list logic
class TaskList {
  /**
   * @param {string} storageKey - localStorage key
   */
  constructor(storageKey) {
    this.storageKey = storageKey;
    this.tasks = [];
  }

  addTask(text) {
    this.tasks.push({ text, completed: false });
    this.save();
  }

  removeTask(index) {
    if (index >= 0 && index < this.tasks.length) {
      this.tasks.splice(index, 1);
      this.save();
    }
  }

  toggleComplete(index) {
    if (index >= 0 && index < this.tasks.length) {
      this.tasks[index].completed = !this.tasks[index].completed;
      this.save();
    }
  }

  getTasks() {
    return this.tasks.slice();
  }

  load() {
    const raw = localStorage.getItem(this.storageKey);
    if (raw) {
      try {
        this.tasks = JSON.parse(raw);
      } catch {
        this.tasks = [];
      }
    } else {
      this.tasks = [];
    }
  }

  save() {
    localStorage.setItem(this.storageKey, JSON.stringify(this.tasks));
  }
}

/**
 * Update MM:SS display and animate SVG ring
 * @param {number} mm - Minutes left
 * @param {number} ss - Seconds left
 * @param {number} progress - 0..1 progress for ring
 */
function renderTimer(mm, ss, progress) {
  // Assumes elements: #timer-display, #timer-ring exist
  const display = document.getElementById('timer-display');
  const ring = document.getElementById('timer-ring');
  display.textContent = `${mm.toString().padStart(2, '0')}:${ss.toString().padStart(2, '0')}`;
  if (ring) {
    const radius = ring.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    ring.style.strokeDasharray = `${circumference}`;
    ring.style.strokeDashoffset = `${circumference * (1 - progress)}`;
  }
}

/**
 * Update UI colors/text for session
 * @param {boolean} isWork - True if work session, false if break
 */
function updateSessionUI(isWork) {
  // Assumes root element and session label
  const root = document.documentElement;
  const sessionLabel = document.getElementById('session-label');
  root.style.setProperty('--accent', '#00d9ff');
  root.style.setProperty('--bg', '#181c23');
  root.style.setProperty('--font', 'JetBrains Mono, monospace');
  if (isWork) {
    sessionLabel.textContent = 'Work';
    root.style.setProperty('--session-bg', '#191e29');
  } else {
    sessionLabel.textContent = 'Break';
    root.style.setProperty('--session-bg', '#0a1722');
  }
}

/**
 * Play subtle beep when work session completes
 */
function playBeep() {
  // Subtle beep using WebAudio API
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const oscillator = ctx.createOscillator();
  const gain = ctx.createGain();
  oscillator.type = 'sine';
  oscillator.frequency.value = 880;
  gain.gain.value = 0.08;
  oscillator.connect(gain);
  gain.connect(ctx.destination);
  oscillator.start();
  oscillator.stop(ctx.currentTime + 0.15);
  oscillator.onended = () => ctx.close();
}

/**
 * Render task list in DOM
 * @param {Array<{text: string, completed: boolean}>} tasks
 */
function renderTaskList(tasks) {
  // Assumes element: #task-list exists
  const listEl = document.getElementById('task-list');
  listEl.innerHTML = '';
  tasks.forEach((task, idx) => {
    const item = document.createElement('div');
    item.className = 'task-item';
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.checked = task.completed;
    checkbox.addEventListener('click', () => {
      window.taskList.toggleComplete(idx);
      renderTaskList(window.taskList.getTasks());
    });
    const span = document.createElement('span');
    span.textContent = task.text;
    span.className = task.completed ? 'completed' : '';
    const del = document.createElement('button');
    del.textContent = '✕';
    del.className = 'del-btn';
    del.addEventListener('click', () => {
      window.taskList.removeTask(idx);
      renderTaskList(window.taskList.getTasks());
    });
    item.appendChild(checkbox);
    item.appendChild(span);
    item.appendChild(del);
    listEl.appendChild(item);
  });
}

/**
 * Bind UI event listeners for timer/task controls
 */
function bindUIEvents() {
  // Timer controls
  document.getElementById('start-btn').addEventListener('click', () => {
    window.pomodoro.start();
  });
  document.getElementById('pause-btn').addEventListener('click', () => {
    window.pomodoro.pause();
  });
  document.getElementById('reset-btn').addEventListener('click', () => {
    window.pomodoro.reset();
    renderTimer(...Object.values(window.pomodoro.getTimeLeft()), window.pomodoro.getProgress());
  });
  // Task input
  document.getElementById('add-task-btn').addEventListener('click', () => {
    const input = document.getElementById('task-input');
    const text = input.value.trim();
    if (text) {
      window.taskList.addTask(text);
      input.value = '';
      renderTaskList(window.taskList.getTasks());
    }
  });
}

/**
 * Load monospace font and dark theme
 */
function loadFontsAndTheme() {
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = 'https://fonts.googleapis.com/css?family=JetBrains+Mono:400,700&display=swap';
  document.head.appendChild(link);
  document.documentElement.style.setProperty('--font', 'JetBrains Mono, monospace');
  document.documentElement.style.setProperty('--accent', '#00d9ff');
  document.documentElement.style.setProperty('--bg', '#181c23');
}
