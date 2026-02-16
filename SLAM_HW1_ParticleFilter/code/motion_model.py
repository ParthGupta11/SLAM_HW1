"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """

    def __init__(self):
        """
        TODO : Tune Motion Model parameters here
        The original numbers are for reference but HAVE TO be tuned.
        """
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.005
        self._alpha4 = 0.005

    def _wrap_angle(self, a):
        """
        Helper function to wrap angles to the range [-pi, pi]
        """
        while a <= -np.pi:
            a += 2 * np.pi
        while a > np.pi:
            a -= 2 * np.pi
        return a

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : particle state odometry reading [x, y, theta] at time (t-1) [odometry_frame]
        param[in] u_t1 : particle state odometry reading [x, y, theta] at time t [odometry_frame]
        param[in] x_t0 : particle state belief [x, y, theta] at time (t-1) [world_frame]
        param[out] x_t1 : particle state belief [x, y, theta] at time t [world_frame]
        """
        """
        TODO : Add your code here
        """
        del_rot1 = math.atan2(u_t1[1] - u_t0[1], u_t1[0] - u_t0[0]) - u_t0[2]
        del_trans = math.sqrt((u_t1[0] - u_t0[0]) ** 2 + (u_t1[1] - u_t0[1]) ** 2)
        del_rot2 = u_t1[2] - u_t0[2] - del_rot1

        # Wrap angles
        del_rot1 = self._wrap_angle(del_rot1)
        del_rot2 = self._wrap_angle(del_rot2)

        del_rot1_hat = del_rot1 - np.random.normal(
            0, np.sqrt(self._alpha1 * del_rot1**2 + self._alpha2 * del_trans**2)
        )
        del_trans_hat = del_trans - np.random.normal(
            0,
            np.sqrt(
                self._alpha3 * del_trans**2
                + self._alpha4 * del_rot1**2
                + self._alpha4 * del_rot2**2
            ),
        )
        del_rot2_hat = del_rot2 - np.random.normal(
            0, np.sqrt(self._alpha1 * del_rot2**2 + self._alpha2 * del_trans**2)
        )

        x_t1 = np.zeros_like(x_t0)
        x_t1[0] = x_t0[0] + del_trans_hat * math.cos(x_t0[2] + del_rot1_hat)
        x_t1[1] = x_t0[1] + del_trans_hat * math.sin(x_t0[2] + del_rot1_hat)
        x_t1[2] = x_t0[2] + del_rot1_hat + del_rot2_hat

        # Angle wrapping to keep angles in range [-pi, pi]
        x_t1[2] = self._wrap_angle(x_t1[2])

        return x_t1
