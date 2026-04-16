"""Reusable statistical plots for notebook reports."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


def add_significance_bracket(
    x1: float,
    x2: float,
    y: float,
    *,
    h: float = 2.0,
    text: str = "*",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Draw a significance bracket between two x positions."""
    if ax is None:
        ax = plt.gca()
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2.0, y + h + 0.5, text, ha="center", va="bottom", fontsize=11)
    return ax


def plot_gaussian_mixture_fit(
    df: pd.DataFrame,
    column: str,
    *,
    n_components: int = 3,
    bins: int = 30,
    n_init: int = 100,
    random_state: int = 42,
    grid_points: int = 1000,
    covariance_type: str = "full",
    ax: Optional[plt.Axes] = None,
    reset_cycle: bool = True,
) -> tuple[GaussianMixture, plt.Axes]:
    """Fit a Gaussian mixture model to one column and visualize the components."""
    x_values = df[column].to_numpy(dtype=float)
    x_values = x_values[np.isfinite(x_values)]
    if x_values.size < max(3, n_components):
        raise ValueError(f"Need at least max(3, n_components) valid samples; got {x_values.size}.")

    x_matrix = x_values.reshape(-1, 1)
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    if x_min == x_max:
        epsilon = max(1e-6, 0.01 * max(1.0, abs(x_min)))
        x_min, x_max = x_min - epsilon, x_max + epsilon
    x_range = np.linspace(x_min, x_max, grid_points).reshape(-1, 1)

    mixture = GaussianMixture(
        n_components=n_components,
        n_init=n_init,
        random_state=random_state,
        covariance_type=covariance_type,
    ).fit(x_matrix)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    ax.hist(x_values, bins=bins, density=True, color="skyblue", alpha=0.6, label="Data Histogram")
    log_prob = mixture.score_samples(x_range)
    density = np.exp(log_prob)
    ax.plot(x_range, density, color="black", linestyle="--", label="Overall GMM Fit")

    if reset_cycle:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
        if default_colors:
            ax.set_prop_cycle(cycler(color=default_colors))

    sorted_indices = np.argsort(mixture.means_.ravel())
    for label_index, component_index in enumerate(sorted_indices, start=1):
        if covariance_type == "full":
            variance = mixture.covariances_[component_index][0, 0]
        elif covariance_type == "diag":
            variance = mixture.covariances_[component_index][0]
        elif covariance_type == "spherical":
            variance = mixture.covariances_[component_index]
        elif covariance_type == "tied":
            variance = mixture.covariances_[0, 0]
        else:
            raise ValueError(f"Unsupported covariance_type: {covariance_type}")

        weight = mixture.weights_[component_index]
        mean = mixture.means_[component_index, 0]
        std = np.sqrt(variance)
        component_density = weight * norm.pdf(x_range.ravel(), loc=mean, scale=std)
        ax.plot(x_range, component_density, label=f"Component {label_index}")

    ax.set_xlabel("Log Power", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xticks([-5, -4, -3, -2])
    ax.tick_params(axis="both", labelsize=14)
    return mixture, ax


# Backward-compatible aliases.
add_sig_bracket = add_significance_bracket
plot_gmm_single = plot_gaussian_mixture_fit
