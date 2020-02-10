"""
Functions to compute SVD quality meassurements
"""
import numpy as np


def svd_quality_measurements(svd, delta=1e-3):
    """
    Compute quality of singular values

    Parameters
    ----------
    svd : array
        1d-array containing singular values of Jacobian matrix
    """
    quality_measurements = []
    # Theta_0 = \sum \frac{-1}{\lambda_i + \delta}
    quality_measurements.append(np.sum(-1 / (svd + delta)))
    # Theta_1 = \sum \lambda_i
    quality_measurements.append(np.sum(svd))
    # Theta_2 = \sum \frac{\lambda_i}{max(\lambda)}
    quality_measurements.append(np.sum(svd) / np.max(svd))
    # Theta_3 = \prod \lambda_i
    quality_measurements.append(np.prod(svd))
    return quality_measurements
