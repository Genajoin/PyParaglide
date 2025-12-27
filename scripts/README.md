# Scripts

## To be executed once

```bash
python download_data.py             # Download training weather and flights data (200MB)
python download_elevation_tiles.py  # Download elevation data (260MB)
python download_background_tiles.py # Download background tiles (facultative) (180MB)
sh build_tiler.sh                   # Build the C++ tiler

python download_GFS.py # Optional, download the source .grib weather data files from GFS
```

## To be executed once per session

```bash
sh start_server.sh  # Start Apache server
sh start_jupyter.sh # Start Jupyter server for the neural network documentation
```

## To be executed if needed

```bash
sh update_nn_README.sh # If you have modified the neural network documentation
```

## IGC ingestion (sky.gr)

```bash
# Postgres only; set --db-url or IGC_DB_URL
# Example:
# export IGC_DB_URL="postgresql://paraglidable:paraglidable@localhost:5432/paraglidable"

# Collect IGC links for a year range (resumable per year)
python igc_ingest_skygr.py --links-only --years 2010-2025 --continue --max-pages 0 --max-links 0

# Download IGC files slowly (resumes automatically)
python igc_ingest_skygr.py --download-only --max-flights 0 --min-delay 2 --max-delay 3

# Reparse metadata for already downloaded files
python igc_ingest_skygr.py --reparse-only --max-reparse 0
```

## IGC bridge server (browser -> DB)

```bash
# Postgres
python igc_bridge_server.py --db-url postgresql://paraglidable:paraglidable@localhost:5432/paraglidable
```

## Browser extension (paraplan.ru)

```bash
# Start the local backend
python igc_bridge_server.py --db-url postgresql://paraglidable:paraglidable@localhost:5432/paraglidable
```

Then load the extension from `extensions/paraplan_igc` and configure the server
URL in the popup. See `extensions/README.md` for the full usage flow.
