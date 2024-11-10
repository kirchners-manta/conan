import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def load_cdf_data(file_path):
    """
    Load the CDF data from a CSV file and update column names.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with the loaded data.
    """
    cdf_data = pd.read_csv(file_path, skiprows=1, delimiter=";")
    cdf_data.columns = ["Distance from Plane / pm", "Angle / Degree", "Occurrence"]
    return cdf_data


def create_pivot_table(cdf_data):
    """
    Create a pivot table for contour plotting.

    Parameters
    ----------
    cdf_data : pd.DataFrame
        Pandas DataFrame with the CDF data.

    Returns
    -------
    tuple of np.ndarray
        Meshgrid arrays X, Y and Z values for contour plotting.
    """
    pivot_table = cdf_data.pivot_table(
        index="Angle / Degree", columns="Distance from Plane / pm", values="Occurrence", fill_value=0
    )
    X = pivot_table.columns.values
    Y = pivot_table.index.values
    X, Y = np.meshgrid(X, Y)
    Z = pivot_table.values
    return X, Y, Z


def smooth_data(Z, sigma=1.0):
    """
    Smooth the Z data using a Gaussian filter.

    Parameters
    ----------
    Z : np.ndarray
        2D array of values to be smoothed.
    sigma : float, optional
        Smoothing factor for the Gaussian filter (default is 1.0).

    Returns
    -------
    np.ndarray
        Smoothed Z values.
    """
    return gaussian_filter(Z, sigma=sigma)


def plot_contour(X, Y, Z, title, cmap="jet", levels_filled=50, xlim=(-5200, 5200)):
    """
    Plot a contour plot with filled contours and contour lines.

    Parameters
    ----------
    X : np.ndarray
        Meshgrid array for the x-axis.
    Y : np.ndarray
        Meshgrid array for the y-axis.
    Z : np.ndarray
        2D array of values for contour plotting.
    title : str
        Title of the plot.
    cmap : str, optional
        Colormap for filled contours (default is 'jet').
    levels_filled : int, optional
        Number of levels for filled contours (default is 50).
    xlim : tuple, optional
        Limits for the x-axis (default is (-5000, 5000)).
    """
    plt.figure(figsize=(10, 8))
    contour_filled = plt.contourf(X, Y, Z, levels=levels_filled, cmap=cmap, alpha=0.8)
    plt.colorbar(contour_filled, label="Occurrence")
    plt.xlabel("Distance from Plane (pm)")
    plt.ylabel("Angle (Degree)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xticks(np.arange(-5200, 5200, 1000))
    plt.yticks(np.arange(0, 190, 45))
    plt.xlim(xlim)
    plt.show()


def main():
    """
    Main function to load data, create plots, and display them.
    """
    # File path to the CSV data
    file_path = (
        "/home/sarah/Desktop/Master AMP/Masterarbeit/mnt/marie/simulations/sim_master_thesis/prod1/output/Travis/"
        "cdf/cdf_2_pldf[C306r_C307r_C305r]_#2o_adf[C306r_C307r_C305r]-[C2o_C1o]_triples.csv"
    )

    # Load data
    cdf_data = load_cdf_data(file_path)

    # Create pivot table and meshgrid for plotting
    X, Y, Z = create_pivot_table(cdf_data)

    # Plot without smoothing
    plot_contour(
        X,
        Y,
        Z,
        title="Combined Distribution Function Contour Plot (Raw Data)",
        cmap="jet",
        levels_filled=50,
    )

    # Smooth data and plot
    Z_smoothed = smooth_data(Z, sigma=1.0)
    plot_contour(
        X,
        Y,
        Z_smoothed,
        title="Combined Distribution Function Contour Plot (Smoothed Data)",
        cmap="jet",
        levels_filled=50,
    )


if __name__ == "__main__":
    main()
