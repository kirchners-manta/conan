import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from pymol import cmd


# Viridis color scheme function
def generate_linear_color_map(maxval: float):
    cmap = plt.cm.viridis
    norm = Normalize(vmin=0, vmax=maxval)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return cmap, cbar


max_density_value = 3  # maximum value in cube file
step_size = 0.2  # Increased step size for less isosurfaces

# Generate the Viridis colormap
viridis_cmap, viridis_bar = generate_linear_color_map(max_density_value)

# Load the cube file
cmd.load("velocity.cube")

# Create isosurfaces and color them
current_value = 0
while current_value <= max_density_value:
    color = viridis_cmap(current_value / max_density_value)[:3]
    color_name = f"viridis{current_value:.1f}".replace(".", "_")
    cmd.set_color(color_name, color)
    cmd.isomesh(f"level_{current_value:.1f}", "velocity", level=current_value)
    cmd.color(color_name, f"level_{current_value:.1f}")
    # cmd.set('transparency', '0.5', f'level_{current_value:.1f}')
    current_value += step_size

# Adjust visual style as needed
cmd.show("surface")

# make the background transparent
cmd.set("ray_opaque_background", "off")

# turn the simulation box
cmd.turn("y", "110")
cmd.turn("x", "10")

# Save the image at a reduced resolution
# cmd.png('3D_density_plot.png', width=4000, height=2000, ray=True)

# Create a standalone colorbar
fig = plt.figure(figsize=(2, 6))  # figure size
cbar_ax = fig.add_axes([0.05, 0.05, 0.15, 0.9])  # axes dimensions
plt.colorbar(viridis_bar, cax=cbar_ax, orientation="vertical")
# plt.title('Colorbar', y=1.02, fontsize=20)
plt.ylabel("Density [g/cm\u00B3]", fontsize=20)
cbar_ax.yaxis.set_label_position("left")
cbar_ax.set_yticks(np.linspace(0, max_density_value, 10))
cbar_ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))  # One digit after the comma
cbar_ax.tick_params(axis="x", which="both", length=0, labelsize=16)
cbar_ax.tick_params(axis="y", which="both", length=0, labelsize=16)
plt.savefig("density_colorbar.png", dpi=300, bbox_inches="tight")
