const SERVER_URL_KEY = "paraplan_server_url";
const DEFAULT_SERVER_URL = "http://127.0.0.1:8787";

function setLog(message) {
  const log = document.getElementById("log");
  log.textContent = message || "";
}

function renderStats(stats) {
  const el = document.getElementById("stats");
  const total = stats.total || 0;
  const withIgc = stats.with_igc_url || 0;
  const status = stats.status_counts || {};
  const queued = status.queued || 0;
  const downloaded = status.downloaded || 0;
  const failed = status.failed || 0;
  const resolving = status.resolving || 0;
  const downloading = status.downloading || 0;
  const extra = [];
  if (resolving) {
    extra.push(`resolving ${resolving}`);
  }
  if (downloading) {
    extra.push(`downloading ${downloading}`);
  }
  const suffix = extra.length ? ` | ${extra.join(" | ")}` : "";
  el.textContent = `total: ${total} | with igc: ${withIgc} | queued: ${queued} | downloaded: ${downloaded} | failed: ${failed}${suffix}`;
}

function renderProcessStatus(status) {
  const el = document.getElementById("processStatus");
  if (!status) {
    el.textContent = "resolve: unknown | download: unknown";
    return;
  }
  const resolveState = status.resolveRunning ? "running" : "idle";
  const downloadState = status.downloadRunning ? "running" : "idle";
  el.textContent = `resolve: ${resolveState} | download: ${downloadState}`;
}

function getDelay() {
  const value = parseInt(document.getElementById("delay").value, 10);
  return Number.isNaN(value) ? 0 : value;
}

function getBackfillLimit() {
  const value = parseInt(document.getElementById("backfillLimit").value, 10);
  return Number.isNaN(value) ? 0 : value;
}

async function loadServerUrl() {
  return new Promise((resolve) => {
    chrome.storage.local.get([SERVER_URL_KEY], (result) => {
      resolve(result[SERVER_URL_KEY] || DEFAULT_SERVER_URL);
    });
  });
}

async function saveServerUrl(value) {
  return new Promise((resolve) => {
    chrome.storage.local.set({ [SERVER_URL_KEY]: value }, () => resolve());
  });
}

async function refresh() {
  chrome.runtime.sendMessage({ type: "getStats" }, (response) => {
    if (response?.ok) {
      renderStats(response.stats || {});
    } else {
      setLog(`stats failed: ${response?.error || "unknown error"}`);
    }
  });
}

async function refreshProcessStatus() {
  chrome.runtime.sendMessage({ type: "getProcessStatus" }, (response) => {
    if (response?.ok) {
      renderProcessStatus(response.status);
    } else {
      renderProcessStatus(null);
    }
  });
}

async function collectLinksFromPage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    setLog("No active tab");
    return;
  }
  const result = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => Array.from(document.querySelectorAll("a[href]"), (a) => a.href),
  });
  const links = result[0]?.result || [];
  chrome.runtime.sendMessage({ type: "collectLinks", links }, (response) => {
    if (response?.ok) {
      setLog(`links added: ${response.added}`);
      refresh();
    } else {
      setLog("failed to collect links");
    }
  });
}

async function collectIgcFromPage() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    setLog("No active tab");
    return;
  }
  const result = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => Array.from(document.querySelectorAll("a[href]"), (a) => a.href),
  });
  const links = result[0]?.result || [];
  chrome.runtime.sendMessage({ type: "collectIgc", links, pageUrl: tab.url }, (response) => {
    if (response?.ok) {
      setLog(`igc added: ${response.added}, failed: ${response.failed}`);
      refresh();
    } else {
      setLog("failed to collect igc");
    }
  });
}

function resolveIgcLinks() {
  setLog("resolve started");
  refreshProcessStatus();
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tabId = tabs && tabs[0] ? tabs[0].id : null;
    chrome.runtime.sendMessage({ type: "resolveIgc", delayMs: getDelay(), tabId }, (response) => {
      if (response?.ok) {
        if (response.result.alreadyRunning) {
          setLog("resolve already running");
          return;
        }
        const stopped = response.result.stopped ? " (stopped)" : "";
        setLog(`resolved igc: +${response.result.added} failed ${response.result.failures}${stopped}`);
        refresh();
        refreshProcessStatus();
      } else {
        setLog("failed to resolve igc links");
      }
    });
  });
}

function stopResolve() {
  chrome.runtime.sendMessage({ type: "stopResolve" }, (response) => {
    if (response?.ok) {
      setLog("stop requested");
      refreshProcessStatus();
    } else {
      setLog("failed to stop resolve");
    }
  });
}

function downloadIgcLinks() {
  setLog("download started");
  refreshProcessStatus();
  chrome.runtime.sendMessage({ type: "downloadIgc", delayMs: getDelay() }, (response) => {
    if (response?.ok) {
      if (response.result.alreadyRunning) {
        setLog("downloads already running");
        return;
      }
      const stopped = response.result.stopped ? " (stopped)" : "";
      setLog(`downloads: started ${response.result.started}/${response.result.total}${stopped}`);
      refresh();
      refreshProcessStatus();
    } else {
      setLog("failed to start downloads");
    }
  });
}

function stopDownload() {
  chrome.runtime.sendMessage({ type: "stopDownload" }, (response) => {
    if (response?.ok) {
      setLog("download stop requested");
      refreshProcessStatus();
    } else {
      setLog("failed to stop download");
    }
  });
}

function backfillDownloads() {
  const limit = getBackfillLimit();
  chrome.runtime.sendMessage({ type: "backfillDownloads", limit }, (response) => {
    if (response?.ok) {
      setLog(`backfill synced: ${response.result.synced}`);
      refresh();
    } else {
      setLog("backfill failed");
    }
  });
}

function clearLocal() {
  chrome.storage.local.clear(() => {
    setLog("local settings cleared");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("collectLinks").addEventListener("click", collectLinksFromPage);
  document.getElementById("collectIgc").addEventListener("click", collectIgcFromPage);
  document.getElementById("resolveIgc").addEventListener("click", resolveIgcLinks);
  document.getElementById("stopResolve").addEventListener("click", stopResolve);
  document.getElementById("downloadIgc").addEventListener("click", downloadIgcLinks);
  document.getElementById("stopDownload").addEventListener("click", stopDownload);
  document.getElementById("backfillDownloads").addEventListener("click", backfillDownloads);
  document.getElementById("clearAll").addEventListener("click", clearLocal);
  loadServerUrl().then((value) => {
    document.getElementById("serverUrl").value = value;
  });
  document.getElementById("serverUrl").addEventListener("change", (event) => {
    saveServerUrl(event.target.value);
  });
  refresh();
  refreshProcessStatus();
  setInterval(() => {
    refresh();
    refreshProcessStatus();
  }, 2000);
});
