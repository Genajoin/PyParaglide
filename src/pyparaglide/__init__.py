"""
PyParaglide - AI-based paragliding flyability forecasting.

Modern Python CLI application using TensorFlow 2.15+ for predicting
paragliding conditions based on GFS weather data.

Based on Paraglidable by Antoine Bourgois (https://paraglidable.com)
Original project: https://github.com/Genajoin/Paraglidable

This is a modernized fork focusing on:
- TensorFlow 2.15+ (migrated from 1.15)
- Python 3.12+ only
- CLI-first approach (no web interface)
- Simplified architecture

Licensed under GPL v3 (same as original Paraglidable)
"""

# Suppress TensorFlow logging BEFORE any TensorFlow imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['ABSL_LOG_MIN_LEVEL'] = '3'
# Disable CUDA plugin loading to suppress "Unable to register" errors
os.environ['TF_CUDA_PLUGGABLE_DEVICE_LIBRARY_PATH'] = ''
os.environ['TF_GPU_ALLOCATOR'] = 'cpu'

__version__ = "2.0.0"
__author__ = "Evgeny Istomin"
__original_author__ = "Antoine Bourgois"
__original_project__ = "Paraglidable"
__license__ = "GPL-3.0-or-later"

# Version info
VERSION = __version__
