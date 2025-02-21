import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.axial_dens as axdens
import conan.analysis_modules.traj_an as traj_an
import conan.analysis_modules.utils as ut
import conan.defdict as ddict


def mol_velocity_analysis(traj_file, molecules, an):
    velocity_analyzer = VelocityAnalysis(traj_file, molecules)

    velocity_analyzer.velocity_prep()

    traj_an.process_trajectory(traj_file, molecules, an, velocity_analyzer)

    velocity_analyzer.velocity_processing()


class COMCalculation:
    def __init__(self, traj_file):
        self.traj_file = traj_file

    @staticmethod
    def unwrap_coordinates(group, box_size):
        coords = group[["X", "Y", "Z"]].values
        unwrapped_coords = coords.copy()
        for i in range(1, len(coords)):
            delta = coords[i] - coords[i - 1]
            delta -= box_size * np.round(delta / box_size)
            unwrapped_coords[i] = unwrapped_coords[i - 1] + delta
        group[["X", "Y", "Z"]] = unwrapped_coords
        return group

    def calculate_COM(self, frame):
        box_size = self.traj_file.box_size
        frame["X"] = frame["X"].astype(float)
        frame["Y"] = frame["Y"].astype(float)
        frame["Z"] = frame["Z"].astype(float)
        frame["Mass"] = frame["Mass"].astype(float)

        frame = frame.groupby("Molecule").apply(self.unwrap_coordinates, box_size=box_size).reset_index(drop=True)

        frame["X_COM"] = frame.groupby("Molecule").apply(lambda x: (x["X"] * x["Mass"]).sum() / x["Mass"].sum())
        frame["Y_COM"] = frame.groupby("Molecule").apply(lambda x: (x["Y"] * x["Mass"]).sum() / x["Mass"].sum())
        frame["Z_COM"] = frame.groupby("Molecule").apply(lambda x: (x["Z"] * x["Mass"]).sum() / x["Mass"].sum())

        mol_com = (
            frame.groupby("Molecule")
            .agg(Species=("Species", "first"), X_COM=("X_COM", "sum"), Y_COM=("Y_COM", "sum"), Z_COM=("Z_COM", "sum"))
            .reset_index()
        )

        mol_com["X_COM"] = mol_com["X_COM"] % box_size[0]
        mol_com["Y_COM"] = mol_com["Y_COM"] % box_size[1]
        mol_com["Z_COM"] = mol_com["Z_COM"] % box_size[2]

        return mol_com


class VelocityAnalysis:
    def __init__(self, traj_file, molecules):
        self.traj_file = traj_file
        self.molecules = molecules
        self.dt = None
        self.old_frame = None
        self.grid_points_velocities = None
        self.grid_point_occurrences = None
        self.analysis_counter = 0
        self.velocity_choice = None

    def velocity_prep(self):
        args = self.traj_file.args
        self.dt = ddict.get_input("What is the time step in the trajectory? [fs]  ", args, "float")

        # Initialize the DensityAnalysis class
        grid_setup = axdens.DensityAnalysis(self.traj_file, self.molecules)
        grid_setup.density_analysis_prep()

        self.x_incr = grid_setup.x_incr
        self.y_incr = grid_setup.y_incr
        self.z_incr = grid_setup.z_incr
        self.x_grid = grid_setup.x_grid
        self.y_grid = grid_setup.y_grid
        self.z_grid = grid_setup.z_grid

        first_frame = self.traj_file.frame0.copy()
        first_frame.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)

        element_masses = ddict.dict_mass()
        first_frame["Mass"] = first_frame["Element"].map(element_masses)

        # ddict.printLog(
        #    "Warning: This analysis potentially yields erroneous results, if the trajectory is wrapped atomwise!\n",
        #    color="red",
        # )
        com_calc = COMCalculation(self.traj_file)
        self.old_frame = com_calc.calculate_COM(first_frame)

        self.grid_points_velocities = grid_setup.grid_point_atom_labels
        self.grid_point_chunk_atom_labels = grid_setup.grid_point_chunk_atom_labels
        self.grid_points_tree = grid_setup.grid_points_tree

        self.grid_point_occurrences = [0] * len(self.grid_points_velocities)

    def velocity_calc_molecule(self, split_frame):
        com_calc = COMCalculation(self.traj_file)

        new_frame = com_calc.calculate_COM(split_frame)

        old_frame = self.old_frame
        old_frame.rename(columns={"X_COM": "X", "Y_COM": "Y", "Z_COM": "Z"}, inplace=True)
        new_frame.rename(columns={"X_COM": "X", "Y_COM": "Y", "Z_COM": "Z"}, inplace=True)

        old_frame = ut.wrapping_coordinates(self.traj_file.box_size, old_frame)
        new_frame = ut.wrapping_coordinates(self.traj_file.box_size, new_frame)

        all_coords = pd.concat(
            [old_frame[["X", "Y", "Z"]].add_prefix("old_"), new_frame[["X", "Y", "Z"]].add_prefix("new_")], axis=1
        )

        for i, axis in enumerate(["X", "Y", "Z"]):
            distance = np.abs(all_coords[f"new_{axis}"] - all_coords[f"old_{axis}"])
            boundary_adjusted = np.where(
                distance > self.traj_file.box_size[i] / 2, distance - self.traj_file.box_size[i], distance
            )
            all_coords[f"distance_{axis.lower()}"] = boundary_adjusted

        all_coords["distance"] = np.linalg.norm(all_coords[["distance_x", "distance_y", "distance_z"]], axis=1)
        all_coords["velocity"] = all_coords["distance"] / self.dt

        new_frame["velocity"] = all_coords["velocity"]
        self.old_frame = new_frame

    def analyze_frame(self, split_frame, frame_counter):
        self.velocity_calc_molecule(split_frame)

        old_frame_coords = np.array(self.old_frame[["X", "Y", "Z"]]).astype(float)

        closest_grid_point_dist, closest_grid_point_idx = self.grid_points_tree.query(old_frame_coords)

        velocities = self.old_frame["velocity"].values

        for i in range(len(self.old_frame)):
            self.grid_points_velocities[closest_grid_point_idx[i]].append(velocities[i])

        self.analysis_counter += 1
        if self.analysis_counter == 100:
            self.velocity_analysis_chunk_processing()
            self.analysis_counter = 0

    def velocity_analysis_chunk_processing(self):
        chunk_occurances = [len(self.grid_points_velocities[i]) for i in range(len(self.grid_points_velocities))]
        chunk_mean_velocities = [
            np.mean(self.grid_points_velocities[i]) if len(self.grid_points_velocities[i]) != 0 else 0
            for i in range(len(self.grid_points_velocities))
        ]

        # Initialize or update grid_point_average_velocities and grid_point_occurrences
        if not hasattr(self, "grid_point_average_velocities"):
            # First time initialization
            self.grid_point_average_velocities = chunk_mean_velocities.copy()
            self.grid_point_occurrences = chunk_occurances.copy()
        else:
            # Update existing values
            for i in range(len(self.grid_point_average_velocities)):
                total_occurrances = self.grid_point_occurrences[i] + chunk_occurances[i]
                if total_occurrances == 0:
                    self.grid_point_average_velocities[i] = 0
                else:
                    self.grid_point_average_velocities[i] = (
                        self.grid_point_average_velocities[i] * self.grid_point_occurrences[i]
                        + chunk_mean_velocities[i] * chunk_occurances[i]
                    ) / total_occurrances
                    self.grid_point_occurrences[i] = total_occurrances

        # Reset the grid_points_velocities for the next chunk
        self.grid_points_velocities = [[] for _ in range(len(self.grid_points_velocities))]

    def velocity_processing(self):
        self.velocity_analysis_chunk_processing()

        grid_points_average_velocities = self.grid_point_average_velocities

        ddict.printLog(f"The maximum velocity is: {max(grid_points_average_velocities):.5f} \u00c5/fs\n")

        # Prepare data for writing cube file
        self.grid_point_densities = grid_points_average_velocities

        # Write cube file
        ut.write_cube_file(self, filename="velocity.cube")

        # Extract velocity profiles along x, y, and z axes
        self.extract_velocity_profiles()

    def extract_velocity_profiles(self):
        velocities = np.array(self.grid_point_average_velocities).reshape(self.x_incr, self.y_incr, self.z_incr)

        # Extract velocity profiles along x, y, z axes
        x_vel_profile = np.sum(velocities, axis=(1, 2))
        y_vel_profile = np.sum(velocities, axis=(0, 2))
        z_vel_profile = np.sum(velocities, axis=(0, 1))

        # Sum grid points along each axis
        sum_gp_x = int(self.y_incr * self.z_incr)
        sum_gp_y = int(self.x_incr * self.z_incr)
        sum_gp_z = int(self.x_incr * self.y_incr)

        # Normalize velocity profiles by number of grid points
        x_vel_profile /= sum_gp_x
        y_vel_profile /= sum_gp_y
        z_vel_profile /= sum_gp_z

        # Save and plot velocity profiles
        self.save_velocity_profiles(x_vel_profile, y_vel_profile, z_vel_profile)

    def save_velocity_profiles(self, x_vel_profile, y_vel_profile, z_vel_profile):
        # Save profiles to CSV and plot
        self.save_profile("x", x_vel_profile, "Velocity [\u00c5/fs]")
        self.save_profile("y", y_vel_profile, "Velocity [\u00c5/fs]")
        self.save_profile("z", z_vel_profile, "Velocity [\u00c5/fs]")

    def save_profile(self, axis, profile, ylabel):
        df = pd.DataFrame(
            {
                axis: getattr(self, f"{axis}_grid"),
                ylabel: profile,
            }
        )
        df.to_csv(f"{axis}_velocity_profile.csv", sep=";", index=False, header=True, float_format="%.5f")

        # Plot profile
        fig, ax = plt.subplots()
        ax.plot(df[axis], df[ylabel], "-", label=f"{axis.upper()} Velocity Profile", color="black")
        ax.set(xlabel=f"{axis} [\u00c5]", ylabel=ylabel, title=f"Velocity Profile along {axis.upper()}")
        ax.grid()
        fig.savefig(f"{axis}_velocity_profile.pdf")
