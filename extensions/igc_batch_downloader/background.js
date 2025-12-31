// Storage keys (must match popup.js)
const STORAGE_KEYS = {
  URL_LIST: 'urlList',
  DELAY_MS: 'delayMs',
  DOWNLOAD_STATE: 'downloadState',
  CURRENT_INDEX: 'currentIndex',
  IS_PAUSED: 'isPaused',
  START_TIME: 'startTime'
};

// Download state
let isRunning = false;
let isPaused = true;
let currentUrl = null;
let processingQueue = false;

// Load state from storage
async function loadState() {
  const result = await chrome.storage.local.get(Object.values(STORAGE_KEYS));
  return {
    urlList: result[STORAGE_KEYS.URL_LIST] || [],
    delayMs: result[STORAGE_KEYS.DELAY_MS] || 2000,
    downloadState: result[STORAGE_KEYS.DOWNLOAD_STATE] || {},
    currentIndex: result[STORAGE_KEYS.CURRENT_INDEX] || 0,
    isPaused: result[STORAGE_KEYS.IS_PAUSED] !== undefined ? result[STORAGE_KEYS.IS_PAUSED] : true,
    startTime: result[STORAGE_KEYS.START_TIME] || null
  };
}

// Save state to storage
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

// Extract track ID from URL based on site pattern
function extractTrackId(url) {
  try {
    const urlObj = new URL(url);

    // paraplan.ru: extract number after /leonardo/flights/
    if (urlObj.hostname.includes('paraplan.ru')) {
      const match = url.match(/\/leonardo\/flights\/(\d+)/);
      if (match) return match[1];
    }

    // sky.gr: extract flightID from query parameter
    if (urlObj.hostname.includes('sky.gr')) {
      const flightId = urlObj.searchParams.get('flightID');
      if (flightId) return flightId;
    }

    // Generic: try to extract from filename
    const pathname = urlObj.pathname;
    const filename = pathname.split('/').pop();
    if (filename) {
      // Remove .igc extension and extract potential ID
      const nameWithoutExt = filename.replace(/\.igc$/i, '');
      // Look for patterns like 2018_07_21 or numbers
      const numberMatch = nameWithoutExt.match(/\d{4,}/);
      if (numberMatch) return numberMatch[0];
      return nameWithoutExt.substring(0, 30); // Truncate long names
    }

    return 'unknown';
  } catch (e) {
    console.error('Error extracting track ID:', e);
    return 'unknown';
  }
}

// Extract domain for folder structure
function extractDomain(url) {
  try {
    const urlObj = new URL(url);
    return urlObj.hostname.replace(/^www\./, '');
  } catch (e) {
    return 'unknown';
  }
}

// Extract filename from URL
function extractFilename(url) {
  try {
    const urlObj = new URL(url);
    const pathname = urlObj.pathname;
    let filename = pathname.split('/').pop();

    if (!filename || filename === '') {
      const trackId = extractTrackId(url);
      filename = `${trackId}.igc`;
    }

    if (!filename.toLowerCase().endsWith('.igc')) {
      filename += '.igc';
    }

    return filename;
  } catch (e) {
    return 'track.igc';
  }
}

// Build download path: domain/track_id/filename.igc
function buildDownloadPath(url) {
  const domain = extractDomain(url);
  const trackId = extractTrackId(url);
  const filename = extractFilename(url);

  // Sanitize components
  const safeDomain = domain.replace(/[^a-zA-Z0-9._-]/g, '_');
  const safeTrackId = String(trackId).replace(/[^a-zA-Z0-9._-]/g, '_');
  const safeFilename = filename.replace(/[^a-zA-Z0-9._-]/g, '_');

  return `${safeDomain}/${safeTrackId}/${safeFilename}`;
}

// Download single file
async function downloadFile(url) {
  const state = await loadState();
  const downloadPath = buildDownloadPath(url);

  return new Promise((resolve, reject) => {
    chrome.downloads.download(
      {
        url: url,
        filename: downloadPath,
        conflictAction: 'uniquify',
        saveAs: false
      },
      (downloadId) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }

        // Update state to downloading
        state.downloadState[url] = {
          status: 'downloading',
          downloadId: downloadId,
          downloadPath: downloadPath,
          startedAt: new Date().toISOString()
        };
        saveState(state);

        // Wait for download to complete
        const checkDownload = () => {
          chrome.downloads.search({ id: downloadId }, (items) => {
            if (items && items.length > 0) {
              const item = items[0];

              if (item.state === 'complete') {
                state.downloadState[url] = {
                  status: 'downloaded',
                  downloadId: downloadId,
                  downloadPath: item.filename || downloadPath,
                  fileSize: item.fileSize,
                  downloadedAt: item.endTime || new Date().toISOString()
                };
                saveState(state);
                resolve({ downloadId, filename: item.filename });
              } else if (item.error) {
                state.downloadState[url] = {
                  status: 'failed',
                  error: item.error,
                  failedAt: new Date().toISOString()
                };
                saveState(state);
                reject(new Error(item.error));
              } else if (item.state === 'interrupted') {
                state.downloadState[url] = {
                  status: 'failed',
                  error: item.error || 'Download interrupted',
                  failedAt: new Date().toISOString()
                };
                saveState(state);
                reject(new Error(item.error || 'Download interrupted'));
              } else {
                // Still in progress, check again
                setTimeout(checkDownload, 500);
              }
            } else {
              reject(new Error('Download not found'));
            }
          });
        };

        checkDownload();
      }
    );
  });
}

// Process download queue
async function processQueue() {
  if (processingQueue) return;
  processingQueue = true;

  try {
    while (isRunning && !isPaused) {
      const state = await loadState();

      // Find next pending URL
      let nextIndex = -1;
      for (let i = state.currentIndex; i < state.urlList.length; i++) {
        const url = state.urlList[i];
        const itemState = state.downloadState[url];
        if (!itemState || itemState.status === 'pending' || itemState.status === 'failed') {
          nextIndex = i;
          break;
        }
      }

      if (nextIndex === -1) {
        // No more URLs to process
        isRunning = false;
        break;
      }

      state.currentIndex = nextIndex;
      await saveState(state);

      const url = state.urlList[nextIndex];
      currentUrl = url;

      try {
        await downloadFile(url);
        console.log('Downloaded:', url);
      } catch (error) {
        console.error('Download failed:', url, error);
        // State is already updated in downloadFile
      }

      currentUrl = null;

      // Check if paused before delay
      if (isPaused) break;

      // Apply delay
      const freshState = await loadState();
      if (freshState.delayMs > 0 && !isPaused) {
        await new Promise(resolve => setTimeout(resolve, freshState.delayMs));
      }
    }
  } finally {
    processingQueue = false;
  }
}

// Start downloads
async function startDownloads(urls, delayMs) {
  const state = await loadState();

  // Initialize download state
  for (const url of urls) {
    if (!state.downloadState[url]) {
      state.downloadState[url] = { status: 'pending' };
    }
  }

  state.urlList = urls;
  state.delayMs = delayMs;
  state.currentIndex = 0;
  state.isPaused = false;
  state.startTime = state.startTime || Date.now();

  await saveState(state);

  isRunning = true;
  isPaused = false;

  // Start processing queue
  processQueue();

  return { ok: true };
}

// Pause downloads
async function pauseDownloads() {
  isPaused = true;
  const state = await loadState();
  state.isPaused = true;
  await saveState(state);
  return { ok: true };
}

// Resume downloads
async function resumeDownloads() {
  isPaused = false;
  const state = await loadState();
  state.isPaused = false;
  await saveState(state);

  if (!processingQueue) {
    processQueue();
  }

  return { ok: true };
}

// Clear all state
async function clearAll() {
  isRunning = false;
  isPaused = true;
  currentUrl = null;
  processingQueue = false;
  return { ok: true };
}

// Listen for download completion to handle edge cases
chrome.downloads.onChanged.addListener((delta) => {
  if (delta.id && delta.state && delta.state.current === 'complete') {
    // We handle completion in downloadFile, but this is a safety net
    console.log('Download completed:', delta.id);
  }
});

// Message handler
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    try {
      switch (message.type) {
        case 'start':
          const startResult = await startDownloads(message.urls, message.delayMs);
          sendResponse(startResult);
          break;

        case 'pause':
          const pauseResult = await pauseDownloads();
          sendResponse(pauseResult);
          break;

        case 'resume':
          const resumeResult = await resumeDownloads();
          sendResponse(resumeResult);
          break;

        case 'clear':
          const clearResult = await clearAll();
          sendResponse(clearResult);
          break;

        default:
          sendResponse({ ok: false, error: 'Unknown command' });
      }
    } catch (error) {
      sendResponse({ ok: false, error: error.message });
    }
  })();

  return true; // Keep message channel open for async response
});

// On service worker startup, check if there's an active download
chrome.runtime.onStartup.addListener(async () => {
  const state = await loadState();
  if (!state.isPaused && state.urlList.length > 0) {
    isRunning = true;
    isPaused = false;
    processQueue();
  }
});

// On extension install or update
chrome.runtime.onInstalled.addListener(async () => {
  const state = await loadState();
  if (!state.isPaused && state.urlList.length > 0) {
    isRunning = true;
    isPaused = false;
    processQueue();
  }
});
