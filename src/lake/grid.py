"""Global 0.1 degree rain grid shared by IMERG and GSMaP NOW.

Index convention (used everywhere in the met dataset):
  lat_idx 0..1799 : centre lat = -89.95 + 0.1 * lat_idx   (south -> north)
  lon_idx 0..3599 : centre lon = -179.95 + 0.1 * lon_idx  (west -> east)
  pixel_id        = lat_idx * 3600 + lon_idx

IMERG V07 HDF5 stores precipitation as (time, lon, lat) on exactly these
centres; GSMaP NOW stores (time, lat, lon) on the same centres. Each reader
must still confirm this against the file's own coordinate arrays.
"""
from __future__ import annotations

import math

import numpy as np

RES = 0.1
N_LAT = 1800
N_LON = 3600
LAT0 = -89.95
LON0 = -179.95


def lat_centre(idx):
    return np.round(LAT0 + RES * np.asarray(idx), 2)


def lon_centre(idx):
    return np.round(LON0 + RES * np.asarray(idx), 2)


def pixel_id(lat_idx, lon_idx):
    return np.asarray(lat_idx, dtype=np.int64) * N_LON + np.asarray(lon_idx, dtype=np.int64)


def lat_index_of(lat):
    """Index of the 0.1 degree cell containing latitude ``lat``."""
    return np.floor((np.asarray(lat, dtype=float) + 90.0) / RES + 1e-9).astype(np.int64)


def lon_index_of(lon):
    return np.floor((np.asarray(lon, dtype=float) + 180.0) / RES + 1e-9).astype(np.int64)


def index_range_for_bbox(lat_min, lat_max, lon_min, lon_max):
    """Inclusive index ranges of cells intersecting a bbox (closed cells)."""
    i0 = int(math.floor((lat_min + 90.0) / RES))
    i1 = int(math.ceil((lat_max + 90.0) / RES)) - 1
    j0 = int(math.floor((lon_min + 180.0) / RES))
    j1 = int(math.ceil((lon_max + 180.0) / RES)) - 1
    return (max(i0, 0), min(i1, N_LAT - 1), max(j0, 0), min(j1, N_LON - 1))
