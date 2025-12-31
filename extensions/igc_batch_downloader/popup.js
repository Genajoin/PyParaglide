// Storage keys
const STORAGE_KEYS = {
  URL_LIST: 'urlList',
  DELAY_MS: 'delayMs',
  DOWNLOAD_STATE: 'downloadState',
  CURRENT_INDEX: 'currentIndex',
  IS_PAUSED: 'isPaused',
  START_TIME: 'startTime'
};

// Default values
const DEFAULTS = {
  DELAY_MS: 2000
};

// DOM elements
const urlListInput = document.getElementById('urlList');
const delayInput = document.getElementById('delay');
const startBtn = document.getElementById('startBtn');
const pauseResumeBtn = document.getElementById('pauseResumeBtn');
const clearBtn = document.getElementById('clearBtn');
const totalEl = document.getElementById('total');
const downloadedEl = document.getElementById('downloaded');
const failedEl = document.getElementById('failed');
const pendingEl = document.getElementById('pending');
const statusTextEl = document.getElementById('statusText');
const timeElapsedEl = document.getElementById('timeElapsed');
const currentFileEl = document.getElementById('currentFile');
const logEl = document.getElementById('log');

// Load saved state
async function loadState() {
  const result = await chrome.storage.local.get(Object.values(STORAGE_KEYS));
  return {
    urlList: result[STORAGE_KEYS.URL_LIST] || [],
    delayMs: result[STORAGE_KEYS.DELAY_MS] || DEFAULTS.DELAY_MS,
    downloadState: result[STORAGE_KEYS.DOWNLOAD_STATE] || {},
    currentIndex: result[STORAGE_KEYS.CURRENT_INDEX] || 0,
    isPaused: result[STORAGE_KEYS.IS_PAUSED] !== undefined ? result[STORAGE_KEYS.IS_PAUSED] : true,
    startTime: result[STORAGE_KEYS.START_TIME] || null
  };
}

// Save state
async function saveState(state) {
  await chrome.storage.local.set({
    [STORAGE_KEYS.URL_LIST]: state.urlList,
    [STORAGE_KEYS.DELAY_MS]: state.delayMs,
    [STORAGE_KEYS.DOWNLOAD_STATE]: state.downloadState,
    [STORAGE_KEYS.CURRENT_INDEX]: state.currentIndex,
    [STORAGE_KEYS.IS_PAUSED]: state.isPaused,
    [STORAGE_KEYS.START_TIME]: state.startTime
  });
}

// Parse URL list from textarea
function parseUrlList(text) {
  return text
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0 && (line.startsWith('http://') || line.startsWith('https://')));
}

// Format time elapsed
function formatTimeElapsed(ms) {
  if (!ms) return '00:00:00';
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// Update statistics display
async function updateStats() {
  const state = await loadState();

  // Count by status
  const stats = {
    total: state.urlList.length,
    downloaded: 0,
    failed: 0,
    pending: 0,
    downloading: 0
  };

  for (const url of state.urlList) {
    const itemState = state.downloadState[url] || { status: 'pending' };
    stats[itemState.status] = (stats[itemState.status] || 0) + 1;
  }

  totalEl.textContent = stats.total;
  downloadedEl.textContent = stats.downloaded;
  failedEl.textContent = stats.failed;
  pendingEl.textContent = stats.pending;

  // Update time elapsed
  if (state.startTime && !state.isPaused) {
    const elapsed = Date.now() - state.startTime;
    timeElapsedEl.textContent = formatTimeElapsed(elapsed);
  } else if (state.startTime) {
    const elapsed = state.pauseTime || state.startTime;
    timeElapsedEl.textContent = formatTimeElapsed(elapsed - state.startTime);
  }

  // Update current file
  if (state.currentIndex < state.urlList.length && !state.isPaused) {
    const currentUrl = state.urlList[state.currentIndex];
    const itemState = state.downloadState[currentUrl];
    if (itemState?.status === 'downloading') {
      currentFileEl.textContent = extractFileName(currentUrl);
    }
  }

  // Update status text and pause button
  if (state.isPaused) {
    if (stats.pending > 0) {
      statusTextEl.textContent = 'Paused';
      pauseResumeBtn.textContent = 'Resume';
    } else {
      statusTextEl.textContent = stats.downloaded > 0 ? 'Completed' : 'Ready';
      pauseResumeBtn.textContent = 'Pause';
    }
    pauseResumeBtn.disabled = stats.pending === 0;
  } else {
    statusTextEl.textContent = 'Downloading...';
    pauseResumeBtn.textContent = 'Pause';
    pauseResumeBtn.disabled = false;
  }

  // Update start button
  startBtn.textContent = stats.pending > 0 ? 'Restart' : 'Start Download';
}

// Extract filename from URL for display
function extractFileName(url) {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    const filename = pathname.split('/').pop();
    return filename || url;
  } catch {
    return url;
  }
}

// Show log message
function showLog(message) {
  logEl.textContent = message;
  setTimeout(() => {
    if (logEl.textContent === message) {
      logEl.textContent = '';
    }
  }, 5000);
}

// Start downloads
async function startDownloads() {
  const state = await loadState();
  const urls = parseUrlList(urlListInput.value);

  if (urls.length === 0) {
    showLog('Please enter at least one URL');
    return;
  }

  // Reset state for URLs not in download state
  const newDownloadState = { ...state.downloadState };
  for (const url of urls) {
    if (!newDownloadState[url]) {
      newDownloadState[url] = { status: 'pending' };
    }
  }

  const newState = {
    ...state,
    urlList: urls,
    downloadState: newDownloadState,
    delayMs: parseInt(delayInput.value, 10) || DEFAULTS.DELAY_MS,
    currentIndex: 0,
    isPaused: false,
    startTime: state.startTime || Date.now()
  };

  await saveState(newState);

  // Send start command to background
  chrome.runtime.sendMessage({ type: 'start', urls, delayMs: newState.delayMs }, (response) => {
    if (response && response.ok) {
      showLog('Download started');
    } else {
      showLog('Failed to start download');
    }
  });

  updateStats();
}

// Pause/Resume toggle
async function togglePauseResume() {
  const state = await loadState();

  if (state.isPaused) {
    // Resume
    const newState = {
      ...state,
      isPaused: false
    };
    // Adjust start time if we have a pause time
    if (state.pauseTime) {
      const pausedDuration = Date.now() - state.pauseTime;
      newState.startTime = state.startTime + pausedDuration;
      delete newState.pauseTime;
    }
    await saveState(newState);

    chrome.runtime.sendMessage({ type: 'resume' }, (response) => {
      if (response && response.ok) {
        showLog('Resumed');
      }
    });
  } else {
    // Pause
    const newState = {
      ...state,
      isPaused: true,
      pauseTime: Date.now()
    };
    await saveState(newState);

    chrome.runtime.sendMessage({ type: 'pause' }, (response) => {
      if (response && response.ok) {
        showLog('Paused');
      }
    });
  }

  updateStats();
}

// Clear all state
async function clearAll() {
  if (!confirm('Clear all data? This will reset the download state.')) {
    return;
  }

  await chrome.storage.local.clear();
  urlListInput.value = '';
  delayInput.value = DEFAULTS.DELAY_MS;

  chrome.runtime.sendMessage({ type: 'clear' }, (response) => {
    if (response && response.ok) {
      showLog('Cleared');
    }
  });

  updateStats();
}

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  // Load saved values
  const state = await loadState();
  urlListInput.value = state.urlList.join('\n');
  delayInput.value = state.delayMs;

  // Event listeners
  startBtn.addEventListener('click', startDownloads);
  pauseResumeBtn.addEventListener('click', togglePauseResume);
  clearBtn.addEventListener('click', clearAll);

  // Save delay on change
  delayInput.addEventListener('change', async () => {
    const state = await loadState();
    state.delayMs = parseInt(delayInput.value, 10) || DEFAULTS.DELAY_MS;
    await saveState(state);
  });

  // Save URL list on change
  urlListInput.addEventListener('change', async () => {
    const urls = parseUrlList(urlListInput.value);
    const state = await loadState();
    state.urlList = urls;
    // Reset state for new URLs
    for (const url of urls) {
      if (!state.downloadState[url]) {
        state.downloadState[url] = { status: 'pending' };
      }
    }
    await saveState(state);
    updateStats();
  });

  // Initial stats update
  updateStats();

  // Update stats every second
  setInterval(updateStats, 1000);
});
