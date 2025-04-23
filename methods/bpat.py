"""
BPAT: Bi-phase Adaptive Thresholding for Marker Intensity Analysis in Multiplexed Immunofluorescence (mIF) Images.

This script implements the BPAT method, which combines Otsu thresholding with Bayesian Gaussian Mixture Modeling (BGMM)
to identify a threshold for separating marker-positive and marker-negative cells in mIF image cell data.

Steps:
1. Phase 1 - Otsu: A global threshold is applied to extract foreground (positive candidate) cells.
2. Phase 2 - BGMM: A Bayesian GMM is fitted to the foreground intensities to identify an optimal intersection point
   between two Gaussian components, which is used as the final threshold.

The script includes optional plotting of the fitted components and selected threshold.

"""


import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import re


def find_intersections(pdf1, pdf2, x):
    intersections = []
    for i in range(1, len(x)):
        if (pdf1[i] - pdf2[i]) * (pdf1[i-1] - pdf2[i-1]) < 0:
            if pdf1[i] != 0 and pdf2[i] != 0:
                intersections.append(x[i][0])
    return intersections


def plot_bpat_results(data, x, sum_pdf, pdfs, threshold, save_path):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=100, density=True, alpha=0.5, label='Foreground Data')
    for i, pdf in enumerate(pdfs):
        plt.plot(x, pdf, label=f'Component {i+1}')
    plt.plot(x, sum_pdf, linestyle='--', label='GMM Sum')
    plt.axvline(threshold, color='red', linestyle=':', label='BPAT Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}_BPAT_plot.png')
    plt.close()


def BPAT(data, save_plot_path):
    """
    BPAT: Bi-phase Adaptive Thresholding using Otsu followed by Bayesian GMM.

    Parameters:
        data (np.ndarray or pd.Series): 1D array of marker intensities.
        save_plot_path (str): Path (no extension) to save the threshold plot.

    Returns:
        float: Final BPAT threshold.
    """
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.values

    if data.size == 0:
        print("Warning: Empty input data.")
        return float('inf')

    # Phase 1: Otsu to get a foreground mask
    otsu_thresh = threshold_otsu(data)
    foreground = data[data > otsu_thresh]

    if foreground.size == 0:
        print("Warning: No foreground after Otsu.")
        return float('inf')

    # Phase 2: Apply Bayesian GMM to foreground
    gmm = BayesianGaussianMixture(
        n_components=2,
        max_iter=3000,
        weight_concentration_prior=np.exp(-0.5)
    ).fit(foreground.reshape(-1, 1))

    x = np.linspace(foreground.min() - 0.1 * foreground.ptp(),
                    foreground.max() + 0.1 * foreground.ptp(), 1000).reshape(-1, 1)

    pdfs = []
    sum_pdf = np.zeros_like(x)
    for i in range(2):
        mean = gmm.means_[i]
        std = np.sqrt(gmm.covariances_[i])
        pdf = gmm.weights_[i] * stats.norm.pdf(x, mean, std)
        pdfs.append(pdf)
        sum_pdf += pdf

    intersections = find_intersections(pdfs[0], pdfs[1], x)
    mean1, mean2 = sorted(gmm.means_.flatten())

    threshold = None
    for pt in intersections:
        if mean1 <= pt <= mean2:
            threshold = pt
            break

    if threshold is None:
        idx = np.argmax(gmm.weights_)
        threshold = gmm.means_[idx] + 3 * np.sqrt(gmm.covariances_[idx])
        threshold = float(threshold)

    plot_bpat_results(foreground, x, sum_pdf, pdfs, threshold, save_plot_path)

    return threshold
