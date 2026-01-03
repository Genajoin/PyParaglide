"""
GRIB file caching infrastructure for incremental dataset building.

This module provides caching for extracted GRIB data to avoid reprocessing
unchanged files when rebuilding the dataset with modified date ranges.
"""

from pyparaglide.preprocessing.cache.grib_cache import GribCache

__all__ = ["GribCache"]
