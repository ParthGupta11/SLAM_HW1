"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import os
import numpy as np
import math
import time
from matplotlib import pyplot as plt
from scipy.stats import norm

from map_reader import MapReader


class SensorModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 6.3]
    """

    def __init__(self, occupancy_map):
        """
        TODO : Tune Sensor Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._z_hit = 5
        self._z_short = 0.2
        self._z_max = 1
        self._z_rand = 250

        self._sigma_hit = 55
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.5

        # Used in sampling angles in ray casting
        self._subsampling = 5

        # offset of the laser from the robot center
        self._laser_offset = 25.0

        # occupancy map resolution
        self._map_resolution = 10.0

        # step size for raycasting
        self._ray_step_size = 5

        self._occupancy_map = occupancy_map

        # Precomputed ray casting lookup table
        self._ray_cast_table = self._load_or_compute_ray_table()

    def _load_or_compute_ray_table(self):
        cache_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "ray_cast_table.npy"
        )
        if os.path.exists(cache_path):
            print("Loading precomputed ray cast table...")
            table = np.load(cache_path)
            print(
                f"Loaded ray cast table: shape={table.shape}, size={table.nbytes/1e6:.1f} MB"
            )
            return table

        print("Precomputing ray cast table (one-time, ~1-2 min)...")
        table = self._precompute_ray_table()
        np.save(cache_path, table)
        print(f"Saved ray cast table to {cache_path}")
        return table

    def _precompute_ray_table(self):
        map_rows, map_cols = self._occupancy_map.shape
        table = np.full((map_rows, map_cols, 360), self._max_range, dtype=np.float16)

        resolution = self._map_resolution
        step_size = self._ray_step_size_size
        max_range = self._max_range
        num_steps = int(max_range / step_size)
        min_prob = self._min_probability

        # Only compute for free-space cells (not obstacles or unknown)
        # This reduces work from ~640K cells to ~10K cells (~64x speedup)
        free_mask = (self._occupancy_map >= 0) & (self._occupancy_map < min_prob)
        free_rows, free_cols = np.where(free_mask)
        num_free = len(free_rows)
        print(
            f"  Computing rays for {num_free} free-space cells "
            f"(skipping {map_rows * map_cols - num_free} obstacle/unknown cells)"
        )

        # World coordinates of free cell centers
        x_free = free_cols.astype(np.float64) * resolution + resolution / 2.0
        y_free = free_rows.astype(np.float64) * resolution + resolution / 2.0

        for angle_deg in range(360):
            cos_a = math.cos(math.radians(angle_deg))
            sin_a = math.sin(math.radians(angle_deg))

            active = np.ones(num_free, dtype=bool)
            distances = np.full(num_free, max_range, dtype=np.float16)

            for s in range(1, num_steps + 1):
                if not np.any(active):
                    break
                dist = s * step_size

                # Only compute for still-active cells
                idx = np.where(active)[0]
                x_curr = x_free[idx] + dist * cos_a
                y_curr = y_free[idx] + dist * sin_a

                c = (x_curr / resolution).astype(int)
                r = (y_curr / resolution).astype(int)

                oob = (r < 0) | (r >= map_rows) | (c < 0) | (c >= map_cols)
                r_s = np.clip(r, 0, map_rows - 1)
                c_s = np.clip(c, 0, map_cols - 1)
                hit = self._occupancy_map[r_s, c_s] > min_prob

                done = oob | hit
                distances[idx[done]] = np.float16(dist)
                active[idx[done]] = False

            # Write results back to full table
            table[free_rows, free_cols, angle_deg] = distances

            if angle_deg % 60 == 0:
                print(f"  Progress: {angle_deg}/360 angles")

        print("  Progress: 360/360 angles")
        return table

    def beam_range_finder_model_vectorized(self, z_t1_arr, X_t1):
        """
        Vectorized beam range finder for all particles at once.
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] X_t1 : (N, 3) array of [x, y, theta] for all particles
        param[out] weights : (N,) array of likelihoods
        """
        N = X_t1.shape[0]
        beam_indices = np.arange(0, 180, self._subsampling)

        # --- Vectorized ray casting for all particles ---
        thetas = X_t1[:, 2]
        x_lasers = X_t1[:, 0] + self._laser_offset * np.cos(thetas)
        y_lasers = X_t1[:, 1] + self._laser_offset * np.sin(thetas)

        cols = (x_lasers / self._map_resolution).astype(int)
        rows = (y_lasers / self._map_resolution).astype(int)
        map_rows, map_cols = self._occupancy_map.shape

        # Beam angles for all particles: (N, num_beams)
        beam_angles_deg = (
            np.rad2deg(thetas[:, None] + np.deg2rad(-90 + beam_indices[None, :]))
        ).astype(int) % 360

        # Clamp out-of-bounds particles
        valid = (rows >= 0) & (rows < map_rows) & (cols >= 0) & (cols < map_cols)
        rows_safe = np.clip(rows, 0, map_rows - 1)
        cols_safe = np.clip(cols, 0, map_cols - 1)

        # Look up expected ranges: (N, num_beams)
        z_star = self._ray_cast_table[
            rows_safe[:, None], cols_safe[:, None], beam_angles_deg
        ]
        z_star = z_star.astype(np.float64)
        z_star[~valid] = self._max_range

        # Actual measurements: (1, num_beams) for broadcasting
        z_k = z_t1_arr[beam_indices][None, :]

        z_max = self._max_range
        sigma = self._sigma_hit
        lam = self._lambda_short

        # --- Vectorized probability computation ---
        # p_hit: inline Gaussian
        in_range = (z_k >= 0) & (z_k <= z_max)
        gauss = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((z_k - z_star) / sigma) ** 2
        )
        p_hit = np.where(in_range, gauss, 0.0)

        # p_short: exponential
        short_valid = (z_k >= 0) & (z_k <= z_star) & (z_star > 0)
        eta_short = np.where(
            z_star > 0,
            1.0 / np.maximum(1.0 - np.exp(-lam * z_star), 1e-10),
            0.0,
        )
        p_short = np.where(short_valid, eta_short * lam * np.exp(-lam * z_k), 0.0)

        # p_max
        p_max = np.where(z_k >= z_max, 1.0, 0.0)

        # p_rand
        p_rand = np.where((z_k >= 0) & (z_k < z_max), 1.0 / z_max, 0.0)

        # Combined probability per beam: (N, num_beams)
        p = (
            self._z_hit * p_hit
            + self._z_short * p_short
            + self._z_max * p_max
            + self._z_rand * p_rand
        )

        # Log-space sum across beams, then exp
        log_prob = np.sum(np.log(np.maximum(p, 1e-10)), axis=1)
        return np.exp(log_prob)
