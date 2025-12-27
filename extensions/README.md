# Extensions

## Paraplan Leonardo IGC Helper

Location: `extensions/paraplan_igc`

Purpose: collect paraplan.ru Leonardo flight links in the browser and send them to
the local backend (bridge server), then resolve IGC links and download tracks.
All state lives in the backend Postgres database.

### Load in Brave/Chrome

1. Open `chrome://extensions`
2. Enable Developer mode
3. Click "Load unpacked" and select `extensions/paraplan_igc`

### Prerequisites

- Backend running: `python scripts/igc_bridge_server.py --db-url ...`
- Any page from paraplan.ru open in a tab for "Resolve IGC links"

### Settings

- Server URL: defaults to `http://127.0.0.1:8787`
- Delay (ms): pause between requests/downloads
- Backfill limit: number of download history items to sync

### Usage flow

1. Open a list page and click "Collect flight links (page)"
2. Click "Resolve IGC links" to crawl flight pages and store `igc_url`
3. Click "Start downloads" to download `queued` tracks
4. Click "Stop" buttons to pause each process

The status line shows whether resolve/download loops are running.

### Download path

Files are stored under the default Downloads folder using:

```
host/year/flight_id/filename.igc
```

Example:

```
paraplan.ru/2025/130302/2025-12-27-XCS-AAA-01.igc
```

### Notes

- Disable "Ask where to save each file" in the browser to avoid prompts.
- The extension can run resolve and download in parallel.
- If the backend restarts, re-open the popup and click Start again.
