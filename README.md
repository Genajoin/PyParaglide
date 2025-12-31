<img src="www/imgs/logo/logo.svg" width="80" align="right" />

# Paraglidable

Paraglidable is an A.I.-based flying conditions forecasting program for paragliding.<br/>
You can find it live here: https://paraglidable.com

This repository contains:
* Scripts for setting and training the neural network, downloading +10 days forecasts data from third parties and running a prediction in `/neural_network/`
* Program for generating the map tiles from a prediction in `/tiler/`
* Complete web site in `/www/`

## Requirements

The easiest way to start playing with Paraglidable is to use [Docker](https://www.docker.com). I will only provide support for this workflow. But you can also check the [Dockerfile](docker/Dockerfile) and install dependencies on your own.

The main dependencies are:
* [Python 3](https://www.python.org/)
* [TensorFlow 2](https://www.tensorflow.org/)
* [Qt 5](https://www.qt.io/)
* [Apache HTTP Server](https://httpd.apache.org/) with [PHP](https://www.php.net/)

## Installation

### Docker Compose (Recommended)

```bash
git clone https://github.com/Genajoin/Paraglidable.git
cd Paraglidable

# Configure environment (optional - edit .env for custom paths/bbox)
cp .env.example .env

# Start container
docker compose up -d
```

**Access:**
- Web interface: http://localhost:8001
- Jupyter: http://localhost:8888 (after running `sh scripts/start_jupyter.sh` inside container)

**One-time setup inside container:**
```bash
docker exec -it paraglidable bash

cd /workspaces/Paraglidable/scripts/
python download_data.py             # Download training data (200MB)
python download_elevation_tiles.py  # Download elevation data (260MB)
python download_background_tiles.py # Download background tiles (optional) (180MB)
sh build_tiler.sh                   # Build the C++ tiler
```

## Usage

**Generate forecast:**
```bash
cd /workspaces/Paraglidable/neural_network/
python forecast.py  # Downloads GFS, runs ML prediction, generates tiles
```

**Training:**
```bash
cd /workspaces/Paraglidable/neural_network/
python train.py
```

**Start web server:**
```bash
sh /workspaces/Paraglidable/scripts/start_server.sh  # Visualize on localhost:8001
```

## Documentation

- **[Deployment Guide](specs/DEPLOYMENT.md)** — Complete Docker deployment and data preparation
- **[Training Process](specs/TRAINING_PROCESS.md)** — Neural network training workflow
- **[Neural Network](neural_network/)** — Architecture and API documentation

## Contributing

Contributions on any subject are welcome by doing a [pull request from a fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)!

## License

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
