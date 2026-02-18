"""
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

import numpy as np


class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 4.3]
    """

    def __init__(self):
        """
        TODO : Initialize resampling process parameters here
        """
        self.resampling_calls = 0

    def multinomial_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """
        X_bar_resampled = np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        param[in] X_bar : [num_particles x 4] sized array containing [x, y, theta, wt] values for all particles
        param[out] X_bar_resampled : [num_particles x 4] sized array containing [x, y, theta, wt] values for resampled set of particles
        """
        """
        TODO : Add your code here
        """

        # Throttle resampling when robot is moving
        self.resampling_calls += 1
        if self.resampling_calls % 5 != 0:
            print("Throttling resampling. Call count: {}".format(self.resampling_calls))
            return X_bar

        X_bar_resampled = np.zeros_like(X_bar)

        # Normalize weights to sum to 1
        X_bar[:, 3] = X_bar[:, 3] / np.sum(X_bar[:, 3])

        M = X_bar.shape[0]
        r = np.random.uniform(0, 1.0 / M)
        c = X_bar[0, 3]
        i = 0
        for m in range(M):
            U = r + m * (1.0 / M)
            while U > c:
                i += 1
                # Handle float point approximation issues
                if i == M:
                    i = M - 1
                    break
                c += X_bar[i, 3]
            X_bar_resampled[m] = X_bar[i]
            X_bar_resampled[m, 3] = 1.0 / M

        return X_bar_resampled

    def low_variance_sampler_vectorised(self, X_bar):
        """
        Vectorized version of low-variance sampler.
        Uses cumulative-sum + `np.searchsorted` to avoid Python loops.
        Returns a resampled `X_bar` with uniform weights (1/M).
        """
        self.resampling_calls += 1

        # Throttle resampling
        if self.resampling_calls % 5 != 0:
            print(
                "Throttling resampling (vectorised). Call count: {}".format(
                    self.resampling_calls
                )
            )
            return X_bar

        M = X_bar.shape[0]

        weights = X_bar[:, 3].astype(np.float64)
        w_sum = np.sum(weights)
        weights = weights / w_sum

        r = np.random.uniform(0, 1.0 / M)
        positions = r + np.arange(M) * (1.0 / M)

        cumulative = np.cumsum(weights)
        indices = np.searchsorted(cumulative, positions, side="left")
        indices = np.clip(indices, 0, M - 1)

        X_bar_resampled = X_bar[indices].copy()
        X_bar_resampled[:, 3] = 1.0 / M

        return X_bar_resampled
