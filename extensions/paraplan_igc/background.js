const SERVER_URL_KEY = "paraplan_server_url";
const DEFAULT_SERVER_URL = "http://127.0.0.1:8787";
const DEFAULT_SOURCE = "paraplan";

let resolveRunning = false;
let resolveStopRequested = false;
let downloadRunning = false;
let downloadStopRequested = false;
const inProgressDownloads = new Set();

async function loadServerUrl() {
  return new Promise((resolve) => {
    chrome.storage.local.get([SERVER_URL_KEY], (result) => {
      resolve(result[SERVER_URL_KEY] || DEFAULT_SERVER_URL);
    });
  });
}

function normalizeServerUrl(url) {
  return url.replace(/\/+$/, "");
}

async function apiFetch(path, options = {}) {
  const serverUrl = await loadServerUrl();
  const url = `${normalizeServerUrl(serverUrl)}${path}`;
  const resp = await fetch(url, options);
  if (!resp.ok) {
    throw new Error(`server ${resp.status}`);
  }
  return resp.json();
}

function extractFlightLinks(links) {
  return links.filter((link) =>
    link.includes("op=show_flight") ||
    link.includes("flightID=") ||
    link.includes("/leonardo/flight/")
  );
}

function isFlightLink(url) {
  return (
    url &&
    (url.includes("op=show_flight") || url.includes("flightID=") || url.includes("/leonardo/flight/"))
  );
}

function extractIgcLinks(links) {
  return links.filter((link) => link.toLowerCase().includes(".igc"));
}

function extractIgcFromHtml(html, baseUrl) {
  const match = html.match(/href=["']([^"']+\.igc[^"']*)["']/i);
  if (match) {
    const url = match[1];
    return new URL(url, baseUrl).toString();
  }
  return null;
}

function extractFlightIdFromUrl(urlString) {
  if (!urlString) {
    return null;
  }
  try {
    const url = new URL(urlString);
    const flightId = url.searchParams.get("flightID") || url.searchParams.get("flightId");
    if (flightId) {
      return flightId;
    }
    const pathMatch = url.pathname.match(/\/flight\/(\d+)/i);
    if (pathMatch) {
      return pathMatch[1];
    }
  } catch {
    return null;
  }
  return null;
}

function buildDownloadPath(urlString, flightId, showUrl) {
  try {
    const url = new URL(urlString);
    const host = url.hostname.replace(/^www\./, "");
    const name = url.pathname.split("/").pop() || "track.igc";
    let year = "unknown";
    const nameMatch = name.match(/^(19|20)\d{2}/);
    if (nameMatch) {
      year = nameMatch[0];
    } else {
      const pathMatch = url.pathname.match(/(19|20)\d{2}/);
      if (pathMatch) {
        year = pathMatch[0];
      }
    }
    const safeHost = host.replace(/[^a-zA-Z0-9._-]/g, "_");
    const rawId =
      flightId || extractFlightIdFromUrl(showUrl) || extractFlightIdFromUrl(urlString) || "unknown";
    const safeId = String(rawId).replace(/[^a-zA-Z0-9._-]/g, "_");
    return `${safeHost}/${year}/${safeId}/${name}`;
  } catch {
    return null;
  }
}

async function fetchHtml(url) {
  const resp = await fetch(url, { credentials: "include", cache: "no-store" });
  if (!resp.ok) {
    throw new Error(`fetch failed ${resp.status} for ${url}`);
  }
  return resp.text();
}

async function fetchHtmlViaTab(tabId, url) {
  return new Promise((resolve, reject) => {
    chrome.scripting.executeScript(
      {
        target: { tabId },
        func: async (targetUrl) => {
          const resp = await fetch(targetUrl, { credentials: "include", cache: "no-store" });
          if (!resp.ok) {
            throw new Error(`fetch failed ${resp.status} for ${targetUrl}`);
          }
          return resp.text();
        },
        args: [url],
      },
      (results) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (!results || !results[0] || typeof results[0].result !== "string") {
          reject(new Error("no html result"));
          return;
        }
        resolve(results[0].result);
      }
    );
  });
}

async function fetchHtmlWithFallback(url, tabId) {
  try {
    return await fetchHtml(url);
  } catch (err) {
    if (tabId) {
      return await fetchHtmlViaTab(tabId, url);
    }
    throw err;
  }
}

async function resolveIgcLoop(delayMs, tabId, batchSize) {
  if (resolveRunning) {
    return { added: 0, failures: 0, alreadyRunning: true };
  }
  resolveRunning = true;
  resolveStopRequested = false;
  let added = 0;
  let failures = 0;
  let batches = 0;
  try {
    while (!resolveStopRequested) {
      const data = await apiFetch(
        `/resolve/next?source=${encodeURIComponent(DEFAULT_SOURCE)}&limit=${batchSize}`
      );
      const items = data.items || [];
      if (!items.length) {
        break;
      }
      batches += 1;
      for (const item of items) {
        if (resolveStopRequested) {
          break;
        }
        let html = null;
        try {
          html = await fetchHtmlWithFallback(item.show_url, tabId);
        } catch (err) {
          failures += 1;
          await apiFetch("/resolve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: DEFAULT_SOURCE,
              items: [
                {
                  show_url: item.show_url,
                  flight_id: item.flight_id,
                  igc_url: null,
                  error_msg: "fetch failed",
                },
              ],
            }),
          });
          continue;
        }
        const igc = extractIgcFromHtml(html, item.show_url);
        if (igc) {
          added += 1;
          await apiFetch("/resolve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: DEFAULT_SOURCE,
              items: [
                { show_url: item.show_url, flight_id: item.flight_id, igc_url: igc },
              ],
            }),
          });
        } else {
          failures += 1;
          await apiFetch("/resolve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              source: DEFAULT_SOURCE,
              items: [
                {
                  show_url: item.show_url,
                  flight_id: item.flight_id,
                  igc_url: null,
                  error_msg: "no igc url found",
                },
              ],
            }),
          });
        }
        if (delayMs > 0) {
          await new Promise((r) => setTimeout(r, delayMs));
        }
      }
    }
  } finally {
    resolveRunning = false;
  }
  return { added, failures, batches, stopped: resolveStopRequested };
}

async function downloadIgcLinks(delayMs, batchSize) {
  if (downloadRunning) {
    return { started: 0, total: 0, batches: 0, alreadyRunning: true };
  }
  downloadRunning = true;
  downloadStopRequested = false;
  let started = 0;
  let total = 0;
  let batches = 0;
  try {
    while (!downloadStopRequested) {
      const data = await apiFetch(
        `/downloads/next?source=${encodeURIComponent(DEFAULT_SOURCE)}&limit=${batchSize}`
      );
      const items = data.items || [];
      if (!items.length) {
        break;
      }
      batches += 1;
      total += items.length;
      for (const item of items) {
        if (downloadStopRequested) {
          break;
        }
        const url = item.igc_url;
        if (!url || inProgressDownloads.has(url)) {
          continue;
        }
        await new Promise((resolve) => {
          const filename = buildDownloadPath(url, item.flight_id || item.flightId, item.show_url);
          const options = { url, conflictAction: "uniquify", saveAs: false };
          if (filename) {
            options.filename = filename;
          }
          chrome.downloads.download(options, (downloadId) => {
            if (chrome.runtime.lastError || !downloadId) {
              resolve();
              return;
            }
            inProgressDownloads.add(url);
            started += 1;
            resolve();
          });
        });
        if (delayMs > 0) {
          await new Promise((r) => setTimeout(r, delayMs));
        }
      }
    }
  } finally {
    downloadRunning = false;
  }
  return { started, total, batches, stopped: downloadStopRequested };
}

function searchDownloadById(id) {
  return new Promise((resolve) => {
    chrome.downloads.search({ id }, (items) => {
      resolve(items && items.length ? items[0] : null);
    });
  });
}

async function handleDownloadComplete(id) {
  const item = await searchDownloadById(id);
  if (!item || !item.url) {
    return;
  }
  inProgressDownloads.delete(item.url);
  try {
    await apiFetch("/downloads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: DEFAULT_SOURCE,
        items: [
          {
            igc_url: item.url,
            igc_path: item.filename,
            file_size: item.fileSize,
            downloaded_at: item.endTime,
          },
        ],
      }),
    });
  } catch (err) {
    console.warn("download sync failed", err);
  }
}

async function backfillDownloads(limit) {
  const query = { state: "complete" };
  if (limit && limit > 0) {
    query.limit = limit;
  }
  const items = await new Promise((resolve) => {
    chrome.downloads.search(query, (results) => resolve(results || []));
  });
  const payload = [];
  for (const item of items) {
    if (!item || !item.url) {
      continue;
    }
    if (!item.url.toLowerCase().includes(".igc")) {
      continue;
    }
    payload.push({
      igc_url: item.url,
      igc_path: item.filename,
      file_size: item.fileSize,
      downloaded_at: item.endTime,
    });
  }
  if (!payload.length) {
    return { totalFound: 0, synced: 0 };
  }
  await apiFetch("/downloads", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ source: DEFAULT_SOURCE, items: payload }),
  });
  return { totalFound: payload.length, synced: payload.length };
}

chrome.downloads.onChanged.addListener((delta) => {
  if (delta && delta.state && delta.state.current === "complete") {
    handleDownloadComplete(delta.id);
  }
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  (async () => {
    if (message.type === "getStats") {
      try {
        const stats = await apiFetch(`/stats?source=${encodeURIComponent(DEFAULT_SOURCE)}`);
        sendResponse({ ok: true, stats });
      } catch (err) {
        sendResponse({ ok: false, error: err ? String(err) : "unknown error" });
      }
      return;
    }
    if (message.type === "collectLinks") {
      const flightLinks = extractFlightLinks(message.links || []);
      try {
        const items = flightLinks.map((url) => ({ show_url: url }));
        const result = await apiFetch("/links", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: DEFAULT_SOURCE, items }),
        });
        sendResponse({ ok: true, added: result.processed || items.length });
      } catch (err) {
        sendResponse({ ok: false, error: err ? String(err) : "unknown error" });
      }
      return;
    }
    if (message.type === "collectIgc") {
      try {
        const links = message.links || [];
        const igcLinks = extractIgcLinks(links);
        const pageUrl = message.pageUrl;
        let items = [];
        if (igcLinks.length) {
          const item = { igc_url: igcLinks[0] };
          if (isFlightLink(pageUrl)) {
            item.show_url = pageUrl;
          }
          items = [item];
        } else if (pageUrl && isFlightLink(pageUrl)) {
          items = [{ show_url: pageUrl, igc_url: null, error_msg: "no igc url found" }];
        }
        const result = await apiFetch("/resolve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ source: DEFAULT_SOURCE, items }),
        });
        sendResponse({ ok: true, added: result.queued || 0, failed: result.failed || 0 });
      } catch (err) {
        sendResponse({ ok: false, error: err ? String(err) : "unknown error" });
      }
      return;
    }
    if (message.type === "resolveIgc") {
      const delayMs = message.delayMs || 0;
      const tabId = message.tabId;
      const batchSize = message.batchSize || 50;
      const result = await resolveIgcLoop(delayMs, tabId, batchSize);
      sendResponse({ ok: true, result });
      return;
    }
    if (message.type === "stopResolve") {
      resolveStopRequested = true;
      sendResponse({ ok: true });
      return;
    }
    if (message.type === "downloadIgc") {
      try {
        const delayMs = message.delayMs || 0;
        const batchSize = message.batchSize || 50;
        const result = await downloadIgcLinks(delayMs, batchSize);
        sendResponse({ ok: true, result });
      } catch (err) {
        sendResponse({ ok: false, error: err ? String(err) : "unknown error" });
      }
      return;
    }
    if (message.type === "stopDownload") {
      downloadStopRequested = true;
      sendResponse({ ok: true });
      return;
    }
    if (message.type === "getProcessStatus") {
      sendResponse({
        ok: true,
        status: {
          resolveRunning,
          downloadRunning,
        },
      });
      return;
    }
    if (message.type === "backfillDownloads") {
      try {
        const limit = message.limit || 0;
        const result = await backfillDownloads(limit);
        sendResponse({ ok: true, result });
      } catch (err) {
        sendResponse({ ok: false, error: err ? String(err) : "unknown error" });
      }
      return;
    }
    sendResponse({ ok: false, error: "unknown action" });
  })();
  return true;
});
