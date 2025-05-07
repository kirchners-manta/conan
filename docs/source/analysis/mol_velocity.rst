Velocity distribution
---------------------

Calculate the velocity distribution of all liquid species in the system.
The user needs to define the timestep between consecutive frames and set the increments in x,y and z directions to set up the 3D grid.
The tool works as such, that the center of mass of a given molecule is calculated for each frame.
The center of mass is then used to calculate the velocity of the molecule by taking the difference between the center of mass positions the studied frames and dividing by the time between subsequent analyzed frames.
The velocity is then stored in the 3D grid point closest to the center of mass position of the molecule in its initial position.
The average velocity for a given grid point is calculated by summing the velocities of the molecules sorted in the grid point and dividing by the number of  entries for that grid point.
Periodic boundary conditions are taken into account when calculating the velocity of the molecules.
To calculate the velocity distribution for a specific species, the user can restrain the analysis to a specific molecule type.

The velocity distribution is calculated for all species in the system and saved in a CSV file, named ``<x,y,z>_velocity_profile.csv``.
Additionally, the velocity distributions are plotted in a separate figure ``<x,y,z>_velocity_profile.png``.
A Gaussian cube file is generated for the 3D grid of the velocity distribution, named ``velocity.cube``.

.. note::
    The velocity of a given species is calculated just for two consecutive time steps.
    The analysis thus describes, depending on the time between two analysed frames, the rattling of the molecule and thus shall more be seen as a measure of the active motion of the molecule and not its diffusion.
    The analysis is not suitable for diffusion studies.
