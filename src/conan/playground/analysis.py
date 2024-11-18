import os

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


def plot_contour(X, Y, Z, title, cmap="jet", levels_filled=50, xlim=(-5200, 5200), wall_positions=None, save_path=None):
    """
    Plot a contour plot with filled contours and vertical lines or rectangles indicating wall positions.

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
        Limits for the x-axis (default is (-5200, 5200)).
    wall_positions : list of float, optional
        Positions of the walls to be indicated on the plot.
    save_path : str, optional
        Path to save the plot as an image (default is None, which does not save the plot).
    """
    plt.figure(figsize=(10, 8))
    contour_filled = plt.contourf(X, Y, Z, levels=levels_filled, cmap=cmap, alpha=0.8)
    plt.colorbar(contour_filled, label="Occurrence")
    plt.xlabel("Distance from Plane (pm)")
    plt.ylabel("Angle (Degree)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xticks(np.arange(xlim[0], xlim[1] + 1000, 1000))
    plt.yticks(np.arange(0, 190, 45))
    plt.xlim(xlim)

    # Add wall indicators
    if wall_positions:
        for wall in wall_positions:
            plt.axvline(x=wall, color="k", linestyle="--", linewidth=1.5, alpha=0.8)
            plt.axvline(x=-wall, color="k", linestyle="--", linewidth=1.5, alpha=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot with walls saved to {save_path}")
    else:
        plt.show()


def plot_zoomed_contour(
    X, Y, Z, title, center, zoom_range=2000, cmap="jet", levels_filled=50, wall_positions=None, save_path=None
):
    """
    Plot a zoomed contour plot around a specified center.

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
    center : float
        The center of the zoomed region on the x-axis.
    zoom_range : int, optional
        Range of the zoom around the center (default is 2000).
    cmap : str, optional
        Colormap for filled contours (default is 'jet').
    levels_filled : int, optional
        Number of levels for filled contours (default is 50).
    wall_positions : list of float, optional
        Positions of the walls to be indicated on the plot.
    save_path : str, optional
        Path to save the plot as an image (default is None, which does not save the plot).
    """
    x_min, x_max = center - zoom_range, center + zoom_range
    plt.figure(figsize=(10, 8))
    contour_filled = plt.contourf(X, Y, Z, levels=levels_filled, cmap=cmap, alpha=0.8)
    plt.colorbar(contour_filled, label="Occurrence")
    plt.xlabel("Distance from Plane (pm)")
    plt.ylabel("Angle (Degree)")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.xticks(np.arange(x_min, x_max + 500, 500))
    plt.yticks(np.arange(0, 190, 45))
    plt.xlim(x_min, x_max)

    # Add wall indicators
    if wall_positions:
        for wall in wall_positions:
            plt.axvline(x=wall, color="k", linestyle="--", linewidth=2.5, alpha=0.8)
            plt.axvline(x=-wall, color="k", linestyle="--", linewidth=2.5, alpha=0.8)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Zoomed plot saved to {save_path}")
    else:
        plt.show()


def shift_data_to_center(cdf_data, center):
    """
    Shift the 'Distance from Plane / pm' values so that the specified center becomes zero.

    Parameters
    ----------
    cdf_data : pd.DataFrame
        Pandas DataFrame containing the CDF data.
    center : float
        The value in 'Distance from Plane / pm' to be shifted to zero.

    Returns
    -------
    pd.DataFrame
        Updated CDF data with shifted 'Distance from Plane / pm' values.
    """
    cdf_data["Distance from Plane / pm"] -= center
    return cdf_data


def determine_x_limits(cdf_data, padding=200):
    """
    Determine dynamic x-axis limits based on the data range.

    Parameters
    ----------
    cdf_data : pd.DataFrame
        Pandas DataFrame containing the CDF data.
    padding : int, optional
        Extra range to add to the min and max values (default is 200).

    Returns
    -------
    tuple
        The x-axis limits as (min_x, max_x).
    """
    min_x = cdf_data["Distance from Plane / pm"].min() - padding
    max_x = cdf_data["Distance from Plane / pm"].max() + padding
    return min_x, max_x


def process_cdf(file_path, center=None, wall_positions=None, output_dir=None):
    """
     Process a CDF file, optionally shift data to center around the specified value, and generate plots with wall
     indicators.

    Parameters
    ----------
    file_path : str
        Path to the CDF CSV file.
    center : float, optional
        The value to be shifted to zero in the 'Distance from Plane / pm' column.
    wall_positions : list of float, optional
        Positions of the walls to be indicated on the plots.
    output_dir : str, optional
        Directory to save the resulting plots. If None, plots are shown but not saved.
    """
    # Create output directory if it does not exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load data
    cdf_data = load_cdf_data(file_path)

    # Optionally shift data so that the specified center becomes zero
    if center is not None:
        cdf_data = shift_data_to_center(cdf_data, center)

    # Determine dynamic x-axis limits
    xlim = determine_x_limits(cdf_data)

    # Create pivot table and meshgrid for plotting
    X, Y, Z = create_pivot_table(cdf_data)

    # Plot shifted raw data
    raw_plot_path = os.path.join(output_dir, "cdf_raw.png") if output_dir else None
    plot_contour(
        X,
        Y,
        Z,
        title="Combined Distribution Function Contour Plot (Raw Data)",
        cmap="viridis",
        levels_filled=50,
        xlim=xlim,
        wall_positions=wall_positions,
        save_path=raw_plot_path,
    )

    # Smooth data and plot shifted smoothed data
    Z_smoothed = smooth_data(Z, sigma=1.0)
    smoothed_plot_path = os.path.join(output_dir, "cdf_smoothed.png") if output_dir else None
    plot_contour(
        X,
        Y,
        Z_smoothed,
        title="Combined Distribution Function Contour Plot (Smoothed Data)",
        cmap="viridis",
        levels_filled=50,
        xlim=xlim,
        wall_positions=wall_positions,
        save_path=smoothed_plot_path,
    )

    # Generate zoomed plot
    zoomed_plot_path = os.path.join(output_dir, "cdf_zoomed.png") if output_dir else None
    plot_zoomed_contour(
        X,
        Y,
        Z,
        title="Combined Distribution Function Contour Plot (Zoomed-In)",
        center=0,  # After shifting, the center is at 0
        zoom_range=2200,
        cmap="viridis",
        levels_filled=50,
        wall_positions=wall_positions,
        save_path=zoomed_plot_path,
    )


def load_density_profile(file_path):
    """
    Load the z-density profile from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the z-density profile CSV file.

    Returns
    -------
    tuple
        A tuple (z_values, density_values) where z_values are the z-coordinates
        and density_values are the corresponding density values.
    """
    data = pd.read_csv(file_path, delimiter=";")
    z_values = data.iloc[:, 0]
    density_values = data.iloc[:, 2]
    return z_values, density_values


def find_analysis_types(base_dir):
    """
    Dynamically find all analysis folders in the CONAN directory.

    Parameters
    ----------
    base_dir : str
        Base directory containing the production runs (e.g., prod1, prod2, prod3).

    Returns
    -------
    list of str
        List of analysis folder names found in the CONAN directory.
    """
    prod1_conan_dir = os.path.join(base_dir, "prod1", "output", "CONAN", "axial_density_analysis")
    if not os.path.exists(prod1_conan_dir):
        print(f"CONAN directory not found: {prod1_conan_dir}")
        return []

    return [folder for folder in os.listdir(prod1_conan_dir) if os.path.isdir(os.path.join(prod1_conan_dir, folder))]


def plot_z_density_profiles(base_dir, analysis_type, output_dir):
    """
    Plot z-density profiles for all production runs (prod1, prod2, prod3) for a specific analysis type.

    Parameters
    ----------
    base_dir : str
        Base directory containing the production runs (e.g., prod1, prod2, prod3).
    analysis_type : str
        The specific analysis folder name (e.g., 'analysis_all').
    output_dir : str
        Directory to save the resulting plot.
    """
    prod_dirs = ["prod1", "prod2", "prod3"]
    colors = ["blue", "green", "pink"]
    plt.figure(figsize=(10, 6))

    for prod, color in zip(prod_dirs, colors):
        analysis_path = os.path.join(
            base_dir, prod, "output", "CONAN", "axial_density_analysis", analysis_type, "z_dens_profile.csv"
        )
        if not os.path.exists(analysis_path):
            print(f"File not found: {analysis_path}")
            continue

        z_values, density_values = load_density_profile(analysis_path)
        plt.plot(z_values, density_values, label=f"{prod}", color=color, alpha=0.7)

    plt.title(f"z-Density Profile - {analysis_type}", fontsize=14)
    plt.xlabel("z [Å]", fontsize=12)
    plt.ylabel("Density [g/cm³]", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Save the plot to the output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{analysis_type}_z_density_profile.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Plot saved to {output_file}")


def cdf_analysis():
    """
    Perform CDF analysis with zoomed plots for prod1, prod2, and prod3.

    Centers are defined as:
    - prod1: z = 670 pm
    - prod2: z = -670 pm
    - prod3: z = -670 pm
    """
    wall_positions = [0, 335, 670]  # Wall positions in pm if the center is at 0
    file_paths_and_centers = [
        (
            "/home/sarah/Desktop/Master AMP/Masterarbeit/mnt/marie/simulations/sim_master_thesis/prod1/output/Travis/"
            "cdf/cdf_2_pldf[C306r_C307r_C305r]_#2o_adf[C306r_C307r_C305r]-[C2o_C1o]_triples.csv",
            670,
        ),
        (
            "/home/sarah/Desktop/Master AMP/Masterarbeit/mnt/marie/simulations/sim_master_thesis/prod2/output/Travis/"
            "cdf/cdf_2_pldf[C306r_C307r_C305r]_#2o_adf[C306r_C307r_C305r]-[C2o_C1o]_triples.csv",
            -670,
        ),
        (
            "/home/sarah/Desktop/Master AMP/Masterarbeit/mnt/marie/simulations/sim_master_thesis/prod3/output/Travis/"
            "cdf/cdf_2_pldf[C306r_C307r_C305r]_#2o_adf[C306r_C307r_C305r]-[C2o_C1o]_triples.csv",
            -670,
        ),
    ]
    output_base_dir = "analysis_outputs"

    for idx, (file_path, center) in enumerate(file_paths_and_centers, start=1):
        output_dir = os.path.join(output_base_dir, f"prod{idx}")
        process_cdf(file_path, center, wall_positions, output_dir)


def axial_density_analysis():
    base_dir = "/home/sarah/Desktop/Master AMP/Masterarbeit/mnt/marie/simulations/sim_master_thesis"
    output_dir = "analysis_outputs"

    # Dynamically find analysis folders in the CONAN directory
    analysis_types = find_analysis_types(base_dir)

    if not analysis_types:
        print("No analysis types found. Exiting.")
        return

    for analysis_type in analysis_types:
        plot_z_density_profiles(base_dir, analysis_type, output_dir)


def main():
    """
    Main function to process and visualize data to analyze.
    """

    cdf_analysis()

    # axial_density_analysis()


if __name__ == "__main__":
    main()
