'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

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

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 1000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 5

        # offset of the laser from the robot center
        self._laser_offset = 25.0

        # occupancy map resolution
        self._map_resolution = 10.0

        # step size for raycasting
        self._ray_step_size = 10

        self._occupancy_map = occupancy_map

        # Precomputed ray casting lookup table
        self._ray_cast_table = self._load_or_compute_ray_table()

    def _load_or_compute_ray_table(self):
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ray_cast_table.npy')
        if os.path.exists(cache_path):
            print("Loading precomputed ray cast table...")
            table = np.load(cache_path)
            print(f"Loaded ray cast table: shape={table.shape}, size={table.nbytes/1e6:.1f} MB")
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
        step = self._ray_step_size
        max_range = self._max_range
        num_steps = int(max_range / step)
        min_prob = self._min_probability

        # Grid of cell centers (world coords)
        cols_arr = np.arange(map_cols) * resolution + resolution / 2.0
        rows_arr = np.arange(map_rows) * resolution + resolution / 2.0
        x_origins, y_origins = np.meshgrid(cols_arr, rows_arr)

        for angle_deg in range(360):
            angle_rad = np.deg2rad(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)

            # Track which cells still haven't hit an obstacle
            active = np.ones((map_rows, map_cols), dtype=bool)

            for s in range(1, num_steps + 1):
                dist = s * step
                x_curr = x_origins + dist * cos_a
                y_curr = y_origins + dist * sin_a

                c = (x_curr / resolution).astype(int)
                r = (y_curr / resolution).astype(int)

                # Bounds check
                out_of_bounds = (r < 0) | (r >= map_rows) | (c < 0) | (c >= map_cols)

                # Safe indexing for occupancy check
                r_safe = np.clip(r, 0, map_rows - 1)
                c_safe = np.clip(c, 0, map_cols - 1)
                hit_obstacle = self._occupancy_map[r_safe, c_safe] > min_prob

                # Cells that just terminated this step
                newly_done = active & (out_of_bounds | hit_obstacle)
                table[:, :, angle_deg] = np.where(
                    newly_done, np.float16(dist), table[:, :, angle_deg]
                )
                active &= ~newly_done

                if not np.any(active):
                    break

            if angle_deg % 60 == 0:
                print(f"  Progress: {angle_deg}/360 angles")

        print("  Progress: 360/360 angles")
        return table

    def ray_casting(self, x_t1):
        """
        Perform ray casting using precomputed lookup table.
        For each laser beam, look up the expected range from the table.
        """
        theta = x_t1[2]
        x_laser = x_t1[0] + self._laser_offset * math.cos(theta)
        y_laser = x_t1[1] + self._laser_offset * math.sin(theta)

        beam_indices = np.arange(0, 180, self._subsampling)
        num_beams = len(beam_indices)
        z_t_star = np.zeros(num_beams)

        map_rows, map_cols = self._occupancy_map.shape
        resolution = self._map_resolution

        # Convert laser position to cell coordinates
        col = int(x_laser / resolution)
        row = int(y_laser / resolution)

        # Bounds check — if laser is off map, return max_range for all beams
        if row < 0 or row >= map_rows or col < 0 or col >= map_cols:
            z_t_star[:] = self._max_range
            return z_t_star

        for i, k in enumerate(beam_indices):
            beam_angle = theta + math.radians(-90 + k)
            # Convert to degree index [0, 360)
            angle_deg = int(math.degrees(beam_angle)) % 360
            z_t_star[i] = self._ray_cast_table[row, col, angle_deg]

        return z_t_star

    def beam_range_finder_model(self, z_t1_arr, x_t1):
        """
        param[in] z_t1_arr : laser range readings [array of 180 values] at time t
        param[in] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        param[out] prob_zt1 : likelihood of a range scan zt1 at time t
        Implement the beam range finder model
        """
        # expected range measurements from raycasting
        z_t_star = self.ray_casting(x_t1)

        # subsampling the actual measurements
        beam_indices = np.arange(0, 180, self._subsampling)
        z_actual = z_t1_arr[beam_indices]

        z_max = self._max_range
        sigma = self._sigma_hit
        lam = self._lambda_short

        # Compute probability in log-space to avoid underflow
        log_prob = 0.0

        for j in range(len(beam_indices)):
            z_k = z_actual[j]      # actual measurement for beam j
            z_k_star = z_t_star[j]  # expected measurement for beam j

            # p_hit - it is the probability of the measurement given the expected range
            if 0 <= z_k <= z_max:
                p_hit = norm.pdf(z_k, loc=z_k_star, scale=sigma)
            else:
                p_hit = 0.0


            if 0 <= z_k <= z_k_star and z_k_star > 0:
                eta_short = 1.0 / (1.0 - math.exp(-lam * z_k_star))
                p_short = eta_short * lam * math.exp(-lam * z_k)
            else:
                p_short = 0.0

            #
            p_max = 1.0 if z_k >= z_max else 0.0

            if 0 <= z_k < z_max:
                p_rand = 1.0 / z_max
            else:
                p_rand = 0.0

            p = (self._z_hit * p_hit
                 + self._z_short * p_short
                 + self._z_max * p_max
                 + self._z_rand * p_rand)

            log_prob += math.log(max(p, 1e-10))

        return math.exp(log_prob)
