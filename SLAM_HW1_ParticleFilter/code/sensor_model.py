'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

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
        self._z_hit = 1
        self._z_short = 0.1
        self._z_max = 0.1
        self._z_rand = 100

        self._sigma_hit = 50
        self._lambda_short = 0.1

        # Used in p_max and p_rand, optionally in ray casting
        self._max_range = 8000

        # Used for thresholding obstacles of the occupancy map
        self._min_probability = 0.35

        # Used in sampling angles in ray casting
        self._subsampling = 2

        # offset of the laser from the robot center
        self._laser_offset = 25.0

        # occupancy map resolution
        self._map_resolution = 10.0

        # step size for raycasting
        self._ray_step_size = 6.0

        self._occupancy_map = occupancy_map

    def ray_casting(self, x_t1):
        """
        Perform ray casting from particle pose to get expected range measurements.
        Basically for each laser beam, cast a ray from the laser's position
        """
        # Compute laser position in world frame
        # offset of the laster
        theta = x_t1[2]
        x_laser = x_t1[0] + self._laser_offset * math.cos(theta)
        y_laser = x_t1[1] + self._laser_offset * math.sin(theta)

        # subsampling the beams
        beam_indices = np.arange(0, 180, self._subsampling)
        num_beams = len(beam_indices)
        z_t_star = np.zeros(num_beams)

        map_rows, map_cols = self._occupancy_map.shape
        step = self._ray_step_size
        max_range = self._max_range
        min_prob = self._min_probability
        resolution = self._map_resolution

        for i, k in enumerate(beam_indices):
            # angle of the beam
            # convert to to range of -90 to 90 degrees
            beam_angle = theta + math.radians(-90 + k)
            cos_angle = math.cos(beam_angle)
            sin_angle = math.sin(beam_angle)

            # raycasting
            dist = 0.0
            while dist < max_range:
                dist += step
                x_curr = x_laser + dist * cos_angle
                y_curr = y_laser + dist * sin_angle

                # converting to map grid indices
                col = int(x_curr / resolution)
                row = int(y_curr / resolution)

                # bound check
                if row < 0 or row >= map_rows or col < 0 or col >= map_cols:
                    break

                # check occupancy
                cell = self._occupancy_map[row, col]
                if cell >= min_prob or cell < 0:
                    break

            z_t_star[i] = min(dist, max_range)

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

        # Compute probability as a product over all beams
        prob_zt1 = 1.0

        for j in range(len(beam_indices)):
            z_k = z_actual[j]      # actual measurement for beam j
            z_k_star = z_t_star[j]  # expected measurement for beam j

            # p_hit - it is the probability of the measurement given the expected range
            if 0 <= z_k <= z_max:
                eta = 1.0 / (norm.cdf(z_max, loc=z_k_star, scale=sigma)
                             - norm.cdf(0.0, loc=z_k_star, scale=sigma))
                p_hit = eta * norm.pdf(z_k, loc=z_k_star, scale=sigma)
            else:
                p_hit = 0.0


            if 0 <= z_k <= z_k_star and z_k_star > 0:
                eta_short = 1.0 / (1.0 - math.exp(-lam * z_k_star))
                p_short = eta_short * lam * math.exp(-lam * z_k)
            else:
                p_short = 0.0

            #
            p_max = 1.0 if abs(z_k - z_max) < 1e-3 else 0.0

            if 0 <= z_k < z_max:
                p_rand = 1.0 / z_max
            else:
                p_rand = 0.0

            p = (self._z_hit * p_hit
                 + self._z_short * p_short
                 + self._z_max * p_max
                 + self._z_rand * p_rand)

            # avoiding numerical issues
            p= max(p,1e-9)
            prob_zt1 *= p

        return prob_zt1
