Axial density distribution
--------------------------

Calculate the density within the simulation box along the x-, y-, and z-axes.
The analysis also generates a Gaussian cube file to provide a 3D density map of the system, named ``density.cube``.
The density profiles are saved as ``X_dens_profile.csv``, ``Y_dens_profile.csv``, and ``Z_dens_profile.csv``.
The analysis is performed on a user-defined grid, where the increments of the simulation box in the x, y, and z directions are defined by the user.
Each atom found in the trajectory is then assigned to the corresponding 3D grid point, weighted by its mass.
