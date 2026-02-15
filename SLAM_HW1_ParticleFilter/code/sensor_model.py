'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.ndimage import distance_transform_edt

from map_reader import MapReader


class SensorModel:
    """
    Likelihood Field Range Finder Model.
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox.
    Probabilistic robotics. MIT press, 2005. [Section 6.4]
    """
    def __init__(self, occupancy_map):
        # --- Mixture weights ---
        self._z_hit = 0.9
        self._z_rand = 0.05
        self._z_max = 0.05

        # --- Gaussian std dev for p_hit (in cm) ---
        self._sigma_hit = 150.0

        # --- Sensor parameters ---
        self._max_range = 4000.0       # max laser range (cm)
        self._min_probability = 0.35   # obstacle occupancy threshold
        self._subsampling = 2          # use every Nth beam

        # --- Laser offset from robot center (cm, in robot frame) ---
        self._laser_loc_x = 25.0

        # --- Map ---
        self._occupancy_map = occupancy_map
        self._map_resolution = 10.0    # cm per cell
        self._map_rows, self._map_cols = occupancy_map.shape

        # --- Precompute distance field ---
        # Obstacle mask: occupied cells (>= threshold) or unknown cells (< 0)
        occupied = (occupancy_map >= self._min_probability) | (occupancy_map < 0)
        # distance_transform_edt on ~occupied: for each non-obstacle cell,
        # gives Euclidean distance (in grid cells) to nearest obstacle cell
        self._dist_field = distance_transform_edt(~occupied)

        # --- Precompute beam angles (radians, relative to robot heading) ---
        # Beam k spans -90 to +89 degrees (right to left, CCW)
        self._beam_indices = np.arange(0, 180, self._subsampling)
        self._beam_angles = np.radians(-90.0 + self._beam_indices)

        # --- Precompute sigma in grid cell units ---
        self._sigma_cells = self._sigma_hit / self._map_resolution

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t (cm)
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        """
        x, y, theta = x_t1[0], x_t1[1], x_t1[2]

        # Reject particles inside obstacles or outside the map
        col_p = int(x / self._map_resolution)
        row_p = int(y / self._map_resolution)
        if (col_p < 0 or col_p >= self._map_cols or
                row_p < 0 or row_p >= self._map_rows or
                self._occupancy_map[row_p, col_p] >= self._min_probability or
                self._occupancy_map[row_p, col_p] < 0):
            return 1e-100

        # Laser position in world frame (25cm forward offset)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_laser = x + self._laser_loc_x * cos_theta
        y_laser = y + self._laser_loc_x * sin_theta

        # Subsample measurements
        z_actual = z_t1_arr[self._beam_indices]
        num_beams = len(self._beam_indices)

        # Beam endpoint positions in world frame (vectorized)
        angles = theta + self._beam_angles
        x_endpoints = x_laser + z_actual * np.cos(angles)
        y_endpoints = y_laser + z_actual * np.sin(angles)

        # Convert to grid indices: col = x/resolution, row = y/resolution
        col_idx = (x_endpoints / self._map_resolution).astype(np.int32)
        row_idx = (y_endpoints / self._map_resolution).astype(np.int32)

        # Validity mask: in-bounds
        in_bounds = (
            (col_idx >= 0) & (col_idx < self._map_cols) &
            (row_idx >= 0) & (row_idx < self._map_rows)
        )
        not_max_range = z_actual < (self._max_range - 1.0)

        # Look up distances for valid, non-max-range beams
        dist = np.zeros(num_beams)
        lookup_mask = in_bounds & not_max_range
        if np.any(lookup_mask):
            dist[lookup_mask] = self._dist_field[
                row_idx[lookup_mask], col_idx[lookup_mask]
            ]

        # p_hit: unnormalized Gaussian (max=1 at dist=0, decays with distance)
        # Using unnormalized form keeps per-beam probs close to 1, avoiding underflow
        p_hit = np.exp(-0.5 * (dist ** 2) / (self._sigma_cells ** 2))

        # p_rand: uniform over [0, z_max]
        p_rand = 1.0 / self._max_range

        # p_max: indicator for max-range readings
        p_max_val = np.where(~not_max_range, 1.0, 0.0)

        # Mixture probability per beam
        p = self._z_hit * p_hit + self._z_rand * p_rand + self._z_max * p_max_val

        # Out-of-bounds non-max-range beams get a small probability
        out_of_bounds = ~in_bounds & not_max_range
        p[out_of_bounds] = 1e-5

        # Clamp to avoid zero
        p = np.maximum(p, 1e-12)

        # Raw product — per-beam values are close to 1 so this stays representable
        prob_zt1 = np.prod(p)

        return prob_zt1
