"""
Downloads module for PyParaglide.

Handles downloading of GFS weather data from NOAA and elevation data.
"""

from pyparaglide.downloads.elevation_downloader import ElevationDownloader
from pyparaglide.downloads.gfs_downloader import GFSDownloader

__all__ = ["ElevationDownloader", "GFSDownloader"]
